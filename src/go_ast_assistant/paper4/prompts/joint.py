from __future__ import annotations

from go_ast_assistant.paper4.fences import fenced_go


TASK_TYPE = "joint"


def render_user(
    *,
    code: str,
    summary: str | None = None,
    target_checks: tuple[str, ...] | None = None,
) -> str:
    prompt = (
        "TASK: joint\n"
        "Identify which go-critic style checks the following Go code violates, explain "
        "why each is problematic, and provide the corrected Go code in a single fenced Go "
        "code block.\n\n"
        f"{fenced_go(code)}"
    )
    if summary:
        prompt += f"\n\nSYNTAX SUMMARY:\n{summary}"
    return prompt
