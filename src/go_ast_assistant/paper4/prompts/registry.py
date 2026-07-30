from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from go_ast_assistant.paper4.prompts import correction, explanation, joint, rule_identification, syntax_summary


SYSTEM_PROMPT = "You are a Go code style reviewer."

_RENDERERS: Mapping[str, Callable[..., str]] = MappingProxyType(
    {
        rule_identification.TASK_TYPE: rule_identification.render_user,
        correction.TASK_TYPE: correction.render_user,
        joint.TASK_TYPE: joint.render_user,
        explanation.TASK_TYPE: explanation.render_user,
        syntax_summary.TASK_TYPE: syntax_summary.render_user,
    }
)


def render_user(task_type: str, **fields: object) -> str:
    return _RENDERERS[task_type](**fields)
