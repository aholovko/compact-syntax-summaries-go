from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from go_ast_assistant.paper4.prepared_study import PreparedSummaryLine, PreparedSummaryRecord
from go_ast_assistant.paper4.training.conditions import SummaryRender


DEFAULT_SKIP_THRESHOLD = 40


def _render_lines(lines: Iterable[PreparedSummaryLine]) -> str:
    return "\n".join(f"{'  ' * line.depth}{line.text}" for line in lines)


def _compress_runs(lines: list[PreparedSummaryLine]) -> list[PreparedSummaryLine]:
    compressed: list[PreparedSummaryLine] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        end = index + 1
        while (
            end < len(lines)
            and lines[end].tier == current.tier
            and lines[end].depth == current.depth
            and lines[end].text == current.text
        ):
            end += 1
        run_length = end - index
        if run_length > 1:
            current = current.model_copy(update={"text": f"{current.text} (×{run_length})"})
        compressed.append(current)
        index = end
    return compressed


class SerializedSummaryStore:
    def __init__(
        self,
        records: Mapping[str, PreparedSummaryRecord],
        count_tokens: Callable[[str], int],
        *,
        skip_threshold: int = DEFAULT_SKIP_THRESHOLD,
    ) -> None:
        self._records = records
        self._count_tokens = count_tokens
        self._skip_threshold = skip_threshold

    def render_for_main(self, snippet_id: str, code_tokens: int) -> SummaryRender:
        try:
            record = self._records[snippet_id]
        except KeyError:
            raise KeyError(
                f"no summary for snippet id {snippet_id!r}; summaries must cover every task snippet"
            ) from None

        if not record.ok:
            return SummaryRender(text="", attached="failed")
        if code_tokens < self._skip_threshold or not record.lines:
            return SummaryRender(text="", attached="skipped")

        text = _render_lines(record.lines)
        if self._count_tokens(text) <= code_tokens:
            return SummaryRender(text=text, attached="present")
        return SummaryRender(
            text=self._truncate(record.lines, code_tokens),
            attached="present_truncated",
        )

    def _truncate(self, lines: tuple[PreparedSummaryLine, ...], budget: int) -> str:
        work = [line for line in lines if line.tier != 2]
        text = _render_lines(work)
        if self._count_tokens(text) <= budget:
            return text

        work = [line for line in work if line.tier != 1]
        text = _render_lines(work)
        if self._count_tokens(text) <= budget:
            return text

        work = _compress_runs(work)
        text = _render_lines(work)
        if self._count_tokens(text) <= budget:
            return text

        while self._count_tokens(text) > budget and any(line.depth > 0 for line in work):
            deepest = max(line.depth for line in work)
            work = [line for line in work if line.depth != deepest]
            text = _render_lines(work)
        return text
