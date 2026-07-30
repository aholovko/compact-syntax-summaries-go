from __future__ import annotations

from go_ast_assistant.paper4.fences import fenced_go


TASK_TYPE = "correction"


def render_user(
    *,
    code: str,
    summary: str | None = None,
    target_checks: tuple[str, ...] | None = None,
) -> str:
    checks = ", ".join(target_checks or ())
    prompt = (
        "TASK: correction\n"
        f"The following Go code violates these go-critic checks: {checks}. Rewrite the "
        "file to fix all of them while changing as little else as possible; do not alter "
        "behavior. Respond with the complete corrected file in a single fenced Go code "
        "block.\n\n"
        f"{fenced_go(code)}"
    )
    if summary:
        prompt += f"\n\nSYNTAX SUMMARY:\n{summary}"
    return prompt
