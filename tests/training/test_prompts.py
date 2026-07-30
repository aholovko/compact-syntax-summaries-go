from __future__ import annotations

import pytest

from go_ast_assistant.paper4.config import CHECK_NAMES
from go_ast_assistant.paper4.prompts import registry


def test_rule_identification_prompt_uses_the_canonical_checks_and_safe_go_fence() -> None:
    code = 'package p\nvar raw = "```"'

    user = registry.render_user("rule_identification", code=code, target_checks=("elseif",))

    assert f"{', '.join(CHECK_NAMES)}." in user
    assert "````go\n" in user
    assert user.endswith('var raw = "```"\n````')


def test_syntax_summary_prompt_uses_raw_code_and_cannot_receive_its_target_summary() -> None:
    user = registry.render_user("syntax_summary", code="package p\n", target_checks=())

    assert user.endswith("```go\npackage p\n```")
    assert "SYNTAX SUMMARY" not in user
    with pytest.raises(TypeError):
        registry.render_user("syntax_summary", code="package p\n", summary="must not leak")


def test_unknown_prompt_task_is_rejected() -> None:
    with pytest.raises(KeyError):
        registry.render_user("unknown", code="package p\n")
