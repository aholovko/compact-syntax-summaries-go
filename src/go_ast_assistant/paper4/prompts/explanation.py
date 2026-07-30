from __future__ import annotations

from go_ast_assistant.paper4.fences import fenced_go


TASK_TYPE = "explanation"


def render_user(
    *,
    code: str,
    summary: str | None = None,
    target_checks: tuple[str, ...] | None = None,
) -> str:
    checks = ", ".join(target_checks or ())
    prompt = (
        "TASK: explanation\n"
        f"The following Go code violates these go-critic checks: {checks}. Explain, in "
        "two to four sentences, why each is problematic, where it occurs, and how to fix "
        "it.\n\n"
        f"{fenced_go(code)}"
    )
    if summary:
        prompt += f"\n\nSYNTAX SUMMARY:\n{summary}"
    return prompt
