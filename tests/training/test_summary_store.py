from __future__ import annotations

import json

import pytest

from go_ast_assistant.paper4.prepared_study import PreparedSummaryRecord
from go_ast_assistant.paper4.training.conditions import SummaryRender
from go_ast_assistant.paper4.training.summary_store import DEFAULT_SKIP_THRESHOLD, SerializedSummaryStore


def _id(character: str = "a") -> str:
    return f"sha256:{character * 64}"


def _record(
    lines: tuple[tuple[int, int, str], ...],
    *,
    character: str = "a",
    ok: bool = True,
    excluded: tuple[str, ...] = (),
) -> PreparedSummaryRecord:
    payload = {
        "id": _id(character),
        "ok": ok,
        "parse_strategy": "file" if ok else None,
        "type_facts_available": False,
        "lines": [
            {
                "tier": tier,
                "depth": depth,
                "text": text,
                "segments": [{"a": text}],
            }
            for tier, depth, text in lines
        ],
        "excluded_constructs": list(excluded),
        "parse_error": None if ok else "synthetic parse failure",
    }
    return PreparedSummaryRecord.model_validate_json(json.dumps(payload))


def _words(text: str) -> int:
    return len(text.split())


def _store(
    record: PreparedSummaryRecord,
    *,
    skip_threshold: int = DEFAULT_SKIP_THRESHOLD,
) -> SerializedSummaryStore:
    return SerializedSummaryStore({_id(): record}, _words, skip_threshold=skip_threshold)


def test_default_threshold_and_main_delivery_statuses() -> None:
    assert DEFAULT_SKIP_THRESHOLD == 40
    present = _store(_record(((0, 0, "func f()"), (0, 1, "return: 1")), excluded=("generics",)))
    assert present.render_for_main(_id(), code_tokens=100) == SummaryRender(
        text="func f()\n  return: 1",
        attached="present",
    )

    skipped = _store(_record(((0, 0, "func f()"),)))
    assert skipped.render_for_main(_id(), code_tokens=39) == SummaryRender("", "skipped")
    assert skipped.render_for_main(_id(), code_tokens=40) == SummaryRender("func f()", "present")
    empty = _store(_record(()))
    assert empty.render_for_main(_id(), code_tokens=100) == SummaryRender("", "skipped")
    failed = _store(_record((), ok=False, excluded=("cgo",)))
    assert failed.render_for_main(_id(), code_tokens=1) == SummaryRender("", "failed")

    exact_budget = _store(_record(((0, 0, "a b"),)), skip_threshold=0)
    assert exact_budget.render_for_main(_id(), code_tokens=2) == SummaryRender("a b", "present")


def test_main_delivery_rejects_a_missing_id() -> None:
    with pytest.raises(KeyError):
        SerializedSummaryStore({}, _words).render_for_main(_id(), code_tokens=100)


def test_main_delivery_uses_the_injected_non_whitespace_token_counter() -> None:
    calls: list[str] = []

    def injected_counter(text: str) -> int:
        calls.append(text)
        return 2

    store = SerializedSummaryStore(
        {_id(): _record(((0, 0, "one"),))},
        injected_counter,
        skip_threshold=0,
    )

    assert store.render_for_main(_id(), code_tokens=1) == SummaryRender("one", "present_truncated")
    assert calls and set(calls) == {"one"}


@pytest.mark.parametrize(
    ("lines", "budget", "expected"),
    [
        (((0, 0, "a b c"), (2, 0, "t u v w x")), 5, "a b c"),
        (((0, 0, "a b"), (1, 0, "c d e"), (2, 0, "f g h")), 4, "a b"),
        (
            ((0, 0, "x y"), (0, 1, "dup w"), (0, 1, "dup w"), (0, 1, "dup w")),
            6,
            "x y\n  dup w (×3)",
        ),
        (((0, 0, "a"), (0, 1, "b"), (0, 2, "c"), (0, 2, "d")), 2, "a\n  b"),
        (((0, 0, "a b c d e"),), 2, "a b c d e"),
    ],
)
def test_main_summary_truncation_uses_the_expected_priority(
    lines: tuple[tuple[int, int, str], ...],
    budget: int,
    expected: str,
) -> None:
    rendered = _store(_record(lines), skip_threshold=0).render_for_main(_id(), code_tokens=budget)

    assert rendered == SummaryRender(expected, "present_truncated")
