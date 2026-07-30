from __future__ import annotations

from go_ast_assistant.paper4.config import CHECK_NAMES
from go_ast_assistant.paper4.fences import fenced_go


TASK_TYPE = "rule_identification"

_CHECK_LIST = ", ".join(CHECK_NAMES)


def render_user(
    *,
    code: str,
    summary: str | None = None,
    target_checks: tuple[str, ...] | None = None,
) -> str:
    prompt = (
        "TASK: rule_identification\n"
        "Identify which of these go-critic checks the Go code violates. Answer with a\n"
        "JSON array of violated check names, drawn only from:\n"
        f"{_CHECK_LIST}.\n\n"
        f"{fenced_go(code)}"
    )
    if summary:
        prompt += f"\n\nSYNTAX SUMMARY:\n{summary}"
    return prompt
