from __future__ import annotations

from go_ast_assistant.paper4.fences import fenced_go


TASK_TYPE = "syntax_summary"


def render_user(*, code: str, target_checks: tuple[str, ...] | None = None) -> str:
    return (
        f"TASK: syntax_summary\nProduce the compact deterministic Go syntax summary for this code.\n\n{fenced_go(code)}"
    )
