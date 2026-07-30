from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

from analysis.inputs import CheckName

from go_ast_assistant.paper4.config import CHECK_NAMES
from go_ast_assistant.paper4.eval.labels import normalize


@dataclass(frozen=True)
class ParsedLabels:
    normalization_status: Literal["recognized_array", "no_recognized_array"]
    pred: tuple[CheckName, ...]
    n_emitted: int
    rejected_label_count: int


def _balanced_array_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                return None
    return None


def _last_json_list(text: str) -> list[object] | None:
    selected: list[object] | None = None
    cursor = 0
    while cursor < len(text):
        start = text.find("[", cursor)
        if start < 0:
            break
        end = _balanced_array_end(text, start)
        if end is None:
            # A truncated outer candidate must not hide a later complete opening.
            cursor = start + 1
            continue
        try:
            candidate = json.loads(text[start:end])
        except json.JSONDecodeError:
            # The balanced span may be prose around a nested valid answer.
            cursor = start + 1
            continue
        if isinstance(candidate, list):
            selected = candidate
            # A valid outer list owns its nested arrays; do not reinterpret one as
            # a later top-level answer.
            cursor = end
        else:  # pragma: no cover - JSON beginning with `[` is necessarily a list
            cursor = start + 1
    return selected


def parse_rule_id_output(text: str, checks: frozenset[str]) -> ParsedLabels:
    selected = _last_json_list(text)
    if selected is None:
        return ParsedLabels(
            normalization_status="no_recognized_array",
            pred=(),
            n_emitted=0,
            rejected_label_count=0,
        )

    recognized: set[str] = set()
    rejected = 0
    for member in selected:
        normalized = normalize(member) if isinstance(member, str) else None
        if normalized is None or normalized not in checks:
            rejected += 1
        else:
            recognized.add(normalized)
    ordered = tuple(check for check in CHECK_NAMES if check in recognized)
    return ParsedLabels(
        normalization_status="recognized_array",
        pred=ordered,
        n_emitted=len(selected),
        rejected_label_count=rejected,
    )


def _paired_sets(
    predictions: Sequence[Collection[str]],
    gold: Sequence[Collection[str]],
) -> tuple[tuple[set[str], set[str]], ...]:
    if len(predictions) != len(gold):
        raise ValueError("prediction and gold counts differ")
    return tuple((set(prediction), set(expected)) for prediction, expected in zip(predictions, gold, strict=True))


def macro_f1(
    predictions: Sequence[Collection[str]],
    gold: Sequence[Collection[str]],
    checks: Collection[str],
) -> float:
    pairs = _paired_sets(predictions, gold)
    ordered_checks = tuple(checks)
    if not ordered_checks:
        return 0.0
    total = 0.0
    for check in ordered_checks:
        true_positive = sum(check in prediction and check in expected for prediction, expected in pairs)
        false_positive = sum(check in prediction and check not in expected for prediction, expected in pairs)
        false_negative = sum(check not in prediction and check in expected for prediction, expected in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        total += 2 * true_positive / denominator if denominator else 0.0
    return total / len(ordered_checks)


def micro_f1(
    predictions: Sequence[Collection[str]],
    gold: Sequence[Collection[str]],
    checks: Collection[str],
) -> float:
    pairs = _paired_sets(predictions, gold)
    check_set = set(checks)
    true_positive = sum(len((prediction & expected) & check_set) for prediction, expected in pairs)
    false_positive = sum(len((prediction - expected) & check_set) for prediction, expected in pairs)
    false_negative = sum(len((expected - prediction) & check_set) for prediction, expected in pairs)
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def exact_match_rate(
    predictions: Sequence[Collection[str]],
    gold: Sequence[Collection[str]],
) -> float:
    pairs = _paired_sets(predictions, gold)
    if not pairs:
        return 0.0
    return sum(prediction == expected for prediction, expected in pairs) / len(pairs)
