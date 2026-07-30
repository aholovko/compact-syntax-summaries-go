from __future__ import annotations

from go_ast_assistant.paper4.training.conditions import (
    AUX_ENCODING,
    CONDITIONS,
    Condition,
    NullSummaryStore,
    SummaryRender,
    SummaryStore,
    assemble_main_user_content,
)


def test_conditions_are_the_exact_four_fixed_experiment_assemblies() -> None:
    assert CONDITIONS == {
        "C0": Condition("C0", use_summary=False, aux=None, aux_ratio=0.0),
        "C1": Condition("C1", use_summary=True, aux=None, aux_ratio=0.0),
        "C2": Condition("C2", use_summary=True, aux="syntax", aux_ratio=0.2),
        "C2-control": Condition("C2-control", use_summary=True, aux="main_dup", aux_ratio=0.2),
    }
    assert AUX_ENCODING == Condition("aux_encoding", use_summary=False, aux=None, aux_ratio=0.0)


def test_null_store_is_a_summary_store_that_skips_main() -> None:
    store = NullSummaryStore()

    assert isinstance(store, SummaryStore)
    assert store.render_for_main("unused", 100) == SummaryRender("", "skipped")


def test_main_content_places_summary_after_the_fence_and_threads_target_checks() -> None:
    content = assemble_main_user_content(
        "correction",
        "package p\n",
        summary="func f()",
        target_checks=("captLocal", "elseif"),
    )

    assert "captLocal, elseif" in content
    assert content.rindex("```") < content.index("SYNTAX SUMMARY:\nfunc f()")
    plain = assemble_main_user_content("rule_identification", "package p\n", summary=None)
    assert "SYNTAX SUMMARY" not in plain


def test_auxiliary_syntax_content_cannot_attach_the_summary_it_predicts() -> None:
    content = assemble_main_user_content("syntax_summary", "package p\n", summary=None, target_checks=())

    assert content.startswith("TASK: syntax_summary\n")
    assert "```go" in content
    assert "SYNTAX SUMMARY" not in content
