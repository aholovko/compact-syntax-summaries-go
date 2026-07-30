from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import pytest

from analysis.inputs import (
    CHECKS,
    Condition,
    CorrectionRecord,
    JointRecord,
    RepairOutcome,
    RuleIdentificationRecord,
)
from analysis.metrics import (
    DistributionSummary,
    Interval,
    TaskContrast,
    cluster_bootstrap_interval,
    distribution_summary,
    exact_match_rate,
    macro_f1,
    macro_f1_difference_interval,
    micro_f1,
    per_check_prf,
    rate,
    seed_averaged_snippet_differences,
    seed_mean_sd,
    task_contrast,
)

TestSeed = Literal[42, 43, 44]


def _snippet(index: int) -> str:
    return f"sha256:{index:064x}"


def _rule_record(
    index: int,
    condition: Condition,
    seed: TestSeed,
    *,
    pred: tuple[str, ...] = (),
    gold: tuple[str, ...] = (),
    exact_match: bool | None = None,
) -> RuleIdentificationRecord:
    if exact_match is None:
        exact_match = set(pred) == set(gold)
    return RuleIdentificationRecord(
        base_snippet_id=_snippet(index),
        condition=condition,
        seed=seed,
        task_type="rule_identification",
        target_checks=("assignOp",),
        summary_status="not_applicable",
        prompt_tokens=10,
        retokenized_response_token_proxy=2,
        latency_ms=1.0,
        pred=pred,
        gold=gold,
        rejected_label_count=0,
        exact_match=exact_match,
        n_emitted=len(pred),
        normalization_status="recognized_array",
    )


def _outcome(*, overall_fixed: bool) -> RepairOutcome:
    return RepairOutcome(
        target_fixed=not overall_fixed,
        overall_fixed=overall_fixed,
        studied_regression=False,
        enabled_regression=False,
        extracted=True,
        extraction_status="go_block",
        parse_ok=True,
        lint_ok=True,
        original_tool_status="ok",
        output_tool_status="ok",
        build_status="OK",
        category="A",
        introduced_checks=(),
        residual_findings=(),
    )


def _repair_record(
    index: int,
    condition: Condition,
    seed: TestSeed,
    task: Literal["correction", "joint"],
    *,
    overall_fixed: bool,
) -> CorrectionRecord | JointRecord:
    common = {
        "base_snippet_id": _snippet(index),
        "condition": condition,
        "seed": seed,
        "target_checks": ("assignOp",),
        "summary_status": "not_applicable",
        "prompt_tokens": 10,
        "retokenized_response_token_proxy": 2,
        "latency_ms": 1.0,
        "outcome": _outcome(overall_fixed=overall_fixed),
        "extracted_similarity": 1.0,
        "sensitivity_class": None,
    }
    if task == "correction":
        return CorrectionRecord(task_type=task, **common)
    return JointRecord(task_type=task, **common)


def _rule_grid() -> list[RuleIdentificationRecord]:
    return [
        _rule_record(index, condition, seed, pred=("assignOp",), gold=("assignOp",))
        for condition in ("C1", "C0")
        for seed in (42, 43)
        for index in (1, 2)
    ]


def _nonlinear_macro_grid(*, heterogeneous: bool = False) -> list[RuleIdentificationRecord]:
    records: list[RuleIdentificationRecord] = []
    for condition in ("C0", "C1"):
        for seed in (42, 43, 44):
            for index, label in ((1, "assignOp"), (2, "builtinShadow")):
                hit = condition == "C1" and index == 1 and (not heterogeneous or seed == 42)
                pred = (label,) if hit else ()
                records.append(_rule_record(index, condition, seed, pred=pred, gold=(label,)))
    return records


def _seed_difference(records: list[RuleIdentificationRecord]) -> object:
    return seed_averaged_snippet_differences(
        records,
        condition_a="C1",
        condition_b="C0",
        task="rule_identification",
    )


def _macro_difference(records: list[RuleIdentificationRecord]) -> object:
    return macro_f1_difference_interval(records, condition_a="C1", condition_b="C0", n_boot=4, seed=1)


def test_rule_metrics_cover_zero_support_checks_and_validate_pair_lengths() -> None:
    preds = [{"a"}, set()]
    golds = [{"a"}, {"b"}]
    checks = ("a", "b")

    assert exact_match_rate(preds, golds) == pytest.approx(0.5)
    assert micro_f1(preds, golds, checks) == pytest.approx(2.0 / 3.0)
    assert macro_f1(preds, golds, checks) == pytest.approx(0.5)
    assert per_check_prf(preds, golds, checks) == {
        "a": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "b": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    }

    for metric in (exact_match_rate, micro_f1, macro_f1, per_check_prf):
        args = (preds, golds[:1]) if metric is exact_match_rate else (preds, golds[:1], checks)
        with pytest.raises(ValueError, match="length"):
            metric(*args)


def test_macro_f1_is_mean_over_all_eight_checks() -> None:
    preds = [{"assignOp"}, set()]
    golds = [{"assignOp"}, {"builtinShadow"}]

    assert macro_f1(preds, golds, CHECKS) == pytest.approx(1.0 / 8.0)
    assert set(per_check_prf(preds, golds, CHECKS)) == set(CHECKS)


def test_rate_rejects_an_empty_population() -> None:
    assert rate([1, 2, 3], lambda value: value >= 2) == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError, match="at least one"):
        rate([], bool)


def test_seed_mean_uses_sample_standard_deviation_and_one_value_has_no_sd() -> None:
    mean, sd = seed_mean_sd([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert sd == pytest.approx(1.0)
    assert seed_mean_sd([4.25]) == (4.25, None)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [True]])
def test_seed_mean_rejects_empty_or_nonfinite_numeric_inputs(values: list[float]) -> None:
    with pytest.raises(ValueError):
        seed_mean_sd(values)


def test_distribution_uses_python_nearest_index_and_preserves_maximum() -> None:
    result = distribution_summary([20, 10])

    assert result == DistributionSummary(p50=10.0, p90=20.0, p95=20.0, p99=20.0, max=20.0, n=2)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("-inf")], [False]])
def test_distribution_rejects_empty_or_nonfinite_numeric_inputs(values: list[float]) -> None:
    with pytest.raises(ValueError):
        distribution_summary(values)


def test_bootstrap_is_order_independent_and_repeatable() -> None:
    values = {"b": 1.0, "a": -1.0, "c": 2.0}
    first = cluster_bootstrap_interval(values, n_boot=1_000, seed=42)
    second = cluster_bootstrap_interval(dict(reversed(list(values.items()))), n_boot=1_000, seed=42)

    assert first == second
    assert first.point == pytest.approx(2.0 / 3.0)
    assert (first.n_boot, first.seed, first.alpha, first.n_units) == (1_000, 42, 0.05, 3)


@pytest.mark.parametrize(
    ("values", "n_boot", "seed", "alpha"),
    [
        ({}, 10, 1, 0.05),
        ({"a": float("nan")}, 10, 1, 0.05),
        ({"a": float("inf")}, 10, 1, 0.05),
        ({"a": True}, 10, 1, 0.05),
        ({1: 1.0}, 10, 1, 0.05),
        ({"a": 1.0}, 0, 1, 0.05),
        ({"a": 1.0}, True, 1, 0.05),
        ({"a": 1.0}, 1.0, 1, 0.05),
        ({"a": 1.0}, 10, True, 0.05),
        ({"a": 1.0}, 10, 1.0, 0.05),
        ({"a": 1.0}, 10, 1, 0.0),
        ({"a": 1.0}, 10, 1, 1.0),
        ({"a": 1.0}, 10, 1, float("nan")),
    ],
)
def test_bootstrap_rejects_invalid_data_or_parameters(
    values: dict[object, object],
    n_boot: object,
    seed: object,
    alpha: object,
) -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap_interval(values, n_boot=n_boot, seed=seed, alpha=alpha)


def test_seed_averaging_uses_exact_match_and_overall_fixed() -> None:
    rule_records = [
        _rule_record(1, condition, seed, exact_match=hit)
        for condition, hits in (("C1", (True, False)), ("C0", (False, False)))
        for seed, hit in zip((42, 43), hits, strict=True)
    ]
    correction_records = [
        _repair_record(1, condition, seed, "correction", overall_fixed=hit)
        for condition, hits in (("C1", (True, False)), ("C0", (False, False)))
        for seed, hit in zip((42, 43), hits, strict=True)
    ]
    joint_records = [
        _repair_record(1, condition, seed, "joint", overall_fixed=hit)
        for condition, hits in (("C1", (False, True)), ("C0", (True, True)))
        for seed, hit in zip((42, 43), hits, strict=True)
    ]

    assert seed_averaged_snippet_differences(
        rule_records, condition_a="C1", condition_b="C0", task="rule_identification"
    ) == {_snippet(1): 0.5}
    assert seed_averaged_snippet_differences(
        correction_records, condition_a="C1", condition_b="C0", task="correction"
    ) == {_snippet(1): 0.5}
    assert seed_averaged_snippet_differences(joint_records, condition_a="C1", condition_b="C0", task="joint") == {
        _snippet(1): -0.5
    }


@pytest.mark.parametrize("runner", [_seed_difference, _macro_difference])
def test_paired_grids_reject_duplicate_cells(runner) -> None:
    records = _rule_grid()
    with pytest.raises(ValueError, match="duplicate"):
        runner([*records, records[0]])


@pytest.mark.parametrize("runner", [_seed_difference, _macro_difference])
def test_paired_grids_reject_incomplete_cells(runner) -> None:
    records = [
        record
        for record in _rule_grid()
        if not (record.condition == "C0" and record.seed == 43 and record.base_snippet_id == _snippet(1))
    ]
    with pytest.raises(ValueError, match="incomplete"):
        runner(records)


@pytest.mark.parametrize("runner", [_seed_difference, _macro_difference])
def test_paired_grids_reject_snippet_set_mismatches(runner) -> None:
    records = [
        record for record in _rule_grid() if not (record.condition == "C0" and record.base_snippet_id == _snippet(2))
    ]
    with pytest.raises(ValueError, match="snippet"):
        runner(records)


@pytest.mark.parametrize("runner", [_seed_difference, _macro_difference])
def test_paired_grids_reject_seed_set_mismatches(runner) -> None:
    records = [record for record in _rule_grid() if not (record.condition == "C0" and record.seed == 43)]
    with pytest.raises(ValueError, match="seed"):
        runner(records)


@pytest.mark.parametrize("runner", [_seed_difference, _macro_difference])
def test_paired_grids_reject_empty_selected_data(runner) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        runner([])


def test_paired_grids_reject_non_boolean_success_values() -> None:
    records: list[object] = list(_rule_grid())
    records[0] = SimpleNamespace(
        base_snippet_id=_snippet(1),
        condition="C1",
        seed=42,
        task_type="rule_identification",
        exact_match=float("nan"),
        pred=("assignOp",),
        gold=("assignOp",),
    )
    with pytest.raises(ValueError, match="success"):
        seed_averaged_snippet_differences(
            records,
            condition_a="C1",
            condition_b="C0",
            task="rule_identification",
        )


def test_macro_f1_interval_recomputes_set_metric() -> None:
    records = _nonlinear_macro_grid()
    result = macro_f1_difference_interval(records, condition_a="C1", condition_b="C0", n_boot=4, seed=1)
    reversed_result = macro_f1_difference_interval(
        list(reversed(records)), condition_a="C1", condition_b="C0", n_boot=4, seed=1
    )

    assert result == reversed_result
    assert result.point == pytest.approx(1.0 / 8.0)
    assert result.ci_high == pytest.approx(1.0 / 8.0)
    assert result.ci_high != pytest.approx(0.1203125)
    assert result.n_units == 2


def test_macro_f1_interval_averages_seed_metrics_instead_of_pooling_seeds() -> None:
    result = macro_f1_difference_interval(
        _nonlinear_macro_grid(heterogeneous=True),
        condition_a="C1",
        condition_b="C0",
        n_boot=4,
        seed=1,
    )

    assert result.point == pytest.approx(1.0 / 24.0)
    assert result.point != pytest.approx(1.0 / 16.0)


def test_task_contrast_resamples_base_snippets() -> None:
    records = [
        _repair_record(index, condition, seed, "joint", overall_fixed=condition == "C0")
        for condition in ("C1", "C0")
        for seed in (42, 43)
        for index in (1, 2)
    ]
    result = task_contrast(
        records,
        name="joint_fix_rate",
        condition_a="C1",
        condition_b="C0",
        task="joint",
        n_boot=20,
        seed=42,
    )

    assert result == TaskContrast(
        name="joint_fix_rate",
        condition_a="C1",
        condition_b="C0",
        task="joint",
        interval=Interval(
            point=-1.0,
            ci_low=-1.0,
            ci_high=-1.0,
            p_value=0.0,
            n_boot=20,
            seed=42,
            alpha=0.05,
            n_units=2,
        ),
    )
    assert result.interval.excludes_zero()


def test_rq1_rq2_contrasts_preserve_fixed_bootstrap_seeds_42_through_45() -> None:
    task_records = []
    for task in ("joint", "correction"):
        task_records.extend(
            _repair_record(1, condition, 42, task, overall_fixed=condition == "C1") for condition in ("C1", "C0")
        )
    task_records.extend(_rule_record(1, condition, 42, exact_match=condition == "C1") for condition in ("C1", "C0"))

    contrasts = [
        task_contrast(
            task_records,
            name=name,
            condition_a="C1",
            condition_b="C0",
            task=task,
            n_boot=10,
            seed=seed,
        )
        for name, task, seed in (
            ("joint_fix_rate", "joint", 42),
            ("correction_fix_rate", "correction", 43),
            ("rule_id_exact_match", "rule_identification", 44),
        )
    ]
    macro = macro_f1_difference_interval(
        _nonlinear_macro_grid(), condition_a="C1", condition_b="C0", n_boot=10, seed=45
    )

    assert [contrast.interval.seed for contrast in contrasts] == [42, 43, 44]
    assert macro.seed == 45
