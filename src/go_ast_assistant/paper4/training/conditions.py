from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from go_ast_assistant.paper4.prompts import registry


Attached = Literal["present", "present_truncated", "skipped", "failed"]


@dataclass(frozen=True)
class SummaryRender:
    text: str
    attached: Attached


@runtime_checkable
class SummaryStore(Protocol):
    def render_for_main(self, snippet_id: str, code_tokens: int) -> SummaryRender: ...


class NullSummaryStore:
    def render_for_main(self, snippet_id: str, code_tokens: int) -> SummaryRender:
        return SummaryRender(text="", attached="skipped")


@dataclass(frozen=True)
class Condition:
    name: str
    use_summary: bool
    aux: Literal["syntax", "main_dup"] | None
    aux_ratio: float


CONDITIONS: dict[str, Condition] = {
    "C0": Condition("C0", use_summary=False, aux=None, aux_ratio=0.0),
    "C1": Condition("C1", use_summary=True, aux=None, aux_ratio=0.0),
    "C2": Condition("C2", use_summary=True, aux="syntax", aux_ratio=0.2),
    "C2-control": Condition("C2-control", use_summary=True, aux="main_dup", aux_ratio=0.2),
}

AUX_ENCODING = Condition("aux_encoding", use_summary=False, aux=None, aux_ratio=0.0)


def assemble_main_user_content(
    task_type: str,
    code: str,
    summary: str | None,
    target_checks: tuple[str, ...] | None = None,
) -> str:
    fields: dict[str, object] = {"code": code, "target_checks": target_checks}
    if summary:
        fields["summary"] = summary
    return registry.render_user(task_type, **fields)
