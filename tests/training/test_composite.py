from __future__ import annotations

import hashlib

import pytest

from go_ast_assistant.paper4.records import TaskExample
from go_ast_assistant.paper4.training.composite import (
    ComponentScoreFn,
    CorrectionFixRateScorer,
    JointFixRateScorer,
    RuleIdMacroF1Scorer,
    ValidationComposite,
)


EXPECTED_COMPONENT_NAMES = (
    "rule_id_macro_f1",
    "correction_fix_rate",
    "joint_fix_rate",
)


def _example(task_type: str) -> TaskExample:
    digest = hashlib.sha256(task_type.encode()).hexdigest()
    return TaskExample(
        id=f"sha256:{digest}",
        split="validation",
        task_type=task_type,  # type: ignore[arg-type]
        target_checks=("assignOp",),
        code=f"package p\n// {task_type}",
        target="target",
        meta={},
    )


def _scorers(calls: list[tuple[str, tuple[TaskExample, ...], tuple[str, ...]]]):
    def score(name: str, value: float) -> ComponentScoreFn:
        def callback(examples: tuple[TaskExample, ...], outputs: tuple[str, ...]) -> float:
            calls.append((name, examples, outputs))
            return value

        return callback

    return (
        RuleIdMacroF1Scorer(score_outputs_fn=score("rule_id_macro_f1", 0.3)),
        CorrectionFixRateScorer(score_outputs_fn=score("correction_fix_rate", 0.6)),
        JointFixRateScorer(score_outputs_fn=score("joint_fix_rate", 0.9)),
    )


class _ExtraScorer:
    name = "unexpected_metric"
    task_type = "rule_identification"
    max_new_tokens = 64

    def score_outputs(self, _examples: tuple[TaskExample, ...], _outputs: tuple[str, ...]) -> float:
        return 1.0


def test_full_composite_batches_by_task_and_averages_exact_three_components() -> None:
    score_calls: list[tuple[str, tuple[TaskExample, ...], tuple[str, ...]]] = []
    generation_calls: list[tuple[tuple[str, ...], int]] = []
    examples = tuple(_example(task) for task in ("joint", "explanation", "correction", "rule_identification"))

    def generate(selected: tuple[TaskExample, ...], max_new_tokens: int) -> tuple[str, ...]:
        generation_calls.append((tuple(example.task_type for example in selected), max_new_tokens))
        return tuple(f"output:{example.task_type}" for example in selected)

    result = ValidationComposite(_scorers(score_calls)).evaluate(object(), examples, generate)

    assert result.composite == pytest.approx(0.6)
    assert result.components == {
        "rule_id_macro_f1": 0.3,
        "correction_fix_rate": 0.6,
        "joint_fix_rate": 0.9,
    }
    assert generation_calls == [
        (("rule_identification",), 64),
        (("correction",), 512),
        (("joint",), 512),
    ]
    assert [name for name, _, _ in score_calls] == list(EXPECTED_COMPONENT_NAMES)
    assert all(
        isinstance(examples_arg, tuple) and isinstance(outputs, tuple) for _, examples_arg, outputs in score_calls
    )


@pytest.mark.parametrize(
    "scorers",
    [
        (),
        (RuleIdMacroF1Scorer(score_outputs_fn=lambda _examples, _outputs: 1.0),),
        (
            RuleIdMacroF1Scorer(score_outputs_fn=lambda _examples, _outputs: 1.0),
            RuleIdMacroF1Scorer(score_outputs_fn=lambda _examples, _outputs: 1.0),
            CorrectionFixRateScorer(score_outputs_fn=lambda _examples, _outputs: 1.0),
            JointFixRateScorer(score_outputs_fn=lambda _examples, _outputs: 1.0),
        ),
        (
            RuleIdMacroF1Scorer(score_outputs_fn=lambda _examples, _outputs: 1.0),
            CorrectionFixRateScorer(score_outputs_fn=lambda _examples, _outputs: 1.0),
            JointFixRateScorer(score_outputs_fn=lambda _examples, _outputs: 1.0),
            _ExtraScorer(),
        ),
    ],
)
def test_composite_rejects_missing_duplicate_or_extra_shape(scorers: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="component"):
        ValidationComposite(scorers)  # type: ignore[arg-type]


def test_required_component_cannot_return_none() -> None:
    scorers = (
        RuleIdMacroF1Scorer(score_outputs_fn=lambda _examples, _outputs: None),
        CorrectionFixRateScorer(score_outputs_fn=lambda _examples, _outputs: 1.0),
        JointFixRateScorer(score_outputs_fn=lambda _examples, _outputs: 1.0),
    )

    with pytest.raises(ValueError, match="rule_id_macro_f1"):
        ValidationComposite(scorers).evaluate(
            object(),
            (_example("rule_identification"),),
            lambda examples, _cap: tuple("[]" for _ in examples),
        )


def test_composite_rejects_generation_cardinality_mismatch() -> None:
    examples = tuple(_example(task) for task in ("rule_identification", "correction", "joint"))

    with pytest.raises(ValueError, match="output|count|number"):
        ValidationComposite(_scorers([])).evaluate(
            object(),
            examples,
            lambda _selected, _cap: (),
        )
