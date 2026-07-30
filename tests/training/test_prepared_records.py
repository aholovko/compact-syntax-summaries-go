from __future__ import annotations

import pytest

from go_ast_assistant.paper4.fences import fence, fenced_go


@pytest.mark.parametrize("run_length", [0, 1, 2, 3, 7])
def test_fence_is_longer_than_the_longest_arbitrary_backtick_run(run_length: int) -> None:
    assert fence("`" * run_length) == "`" * max(3, run_length + 1)


def test_fenced_go_handles_newlines_and_embedded_backticks() -> None:
    assert fenced_go("package p") == "```go\npackage p\n```"
    assert fenced_go('var raw = "```"\n') == '````go\nvar raw = "```"\n````'
