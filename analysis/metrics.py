from __future__ import annotations

import statistics
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Literal, TypeVar, cast

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from analysis.inputs import CHECKS, Condition, ReleasedRecord

ComparisonTask = Literal["rule_identification", "correction", "joint"]
T = TypeVar("T")
V = TypeVar("V")

__all__ = [
    "DistributionSummary",
    "Interval",
    "Rq3Result",
    "Rq3TaskContrast",
    "TaskContrast",
    "build_rq3_frame",
    "cluster_bootstrap_interval",
    "distribution_summary",
    "exact_match_rate",
    "macro_f1",
    "macro_f1_difference_interval",
    "micro_f1",
    "per_check_prf",
    "rate",
    "rq3_task_contrasts",
    "seed_averaged_snippet_differences",
    "seed_mean_sd",
    "task_contrast",
    "fit_rq3_interaction",
]

_RQ3_CONDITIONS = ("C0", "C1")
_RQ3_TASKS = ("rule_identification", "correction", "joint")
_RQ3_COLUMNS = ("base_snippet_id", "seed", "condition", "task", "success")
_RQ3_FORMULA = "success ~ C(condition, Treatment('C0')) * C(task, Treatment('rule_identification'))"
_RQ3_INTERACTION_TERM = "C(condition, Treatment('C0'))[T.C1]:C(task, Treatment('rule_identification'))[T.joint]"


@dataclass(frozen=True)
class Interval:
    point: float
    ci_low: float
    ci_high: float
    p_value: float
    n_boot: int
    seed: int
    alpha: float
    n_units: int

    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


@dataclass(frozen=True)
class DistributionSummary:
    p50: float
    p90: float
    p95: float
    p99: float
    max: float
    n: int


@dataclass(frozen=True)
class TaskContrast:
    name: str
    condition_a: Condition
    condition_b: Condition
    task: Literal["rule_identification", "correction", "joint"]
    interval: Interval


@dataclass(frozen=True)
class Rq3Result:
    coefficient: float
    sd: float
    ci_low: float
    ci_high: float
    excludes_zero: bool
    reject: bool
    term_name: str

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "coefficient": self.coefficient,
            "sd": self.sd,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "excludes_zero": self.excludes_zero,
            "reject": self.reject,
            "term_name": self.term_name,
        }


@dataclass(frozen=True)
class Rq3TaskContrast:
    name: str
    condition_a: Literal["C1"]
    condition_b: Literal["C0"]
    focal_task: Literal["joint"]
    reference_task: Literal["correction", "rule_identification"]
    interval: Interval
    reject: bool


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_values(values: Collection[object], *, name: str) -> list[float]:
    if not values:
        raise ValueError(f"{name} requires at least one value")
    return [_finite_number(value, name=name) for value in values]


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _alpha(value: object) -> float:
    result = _finite_number(value, name="alpha")
    if not 0.0 < result < 1.0:
        raise ValueError("alpha must be between zero and one")
    return result


def rate(records: Sequence[T], predicate: Callable[[T], bool]) -> float:
    if not records:
        raise ValueError("rate requires at least one record")
    return sum(1 for record in records if predicate(record)) / len(records)


def _paired_labels(
    preds: Sequence[Collection[str]],
    golds: Sequence[Collection[str]],
) -> tuple[tuple[Collection[str], Collection[str]], ...]:
    if len(preds) != len(golds):
        raise ValueError("prediction and gold lengths must match")
    return tuple(zip(preds, golds, strict=True))


def _checks(checks: Sequence[str]) -> tuple[str, ...]:
    result = tuple(checks)
    if not result:
        raise ValueError("checks must contain at least one name")
    if any(not isinstance(check, str) or not check for check in result):
        raise ValueError("checks must be nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("checks must be unique")
    return result


def exact_match_rate(preds: Sequence[Collection[str]], golds: Sequence[Collection[str]]) -> float:
    pairs = _paired_labels(preds, golds)
    if not pairs:
        return 0.0
    return sum(1 for pred, gold in pairs if set(pred) == set(gold)) / len(pairs)


def per_check_prf(
    preds: Sequence[Collection[str]],
    golds: Sequence[Collection[str]],
    checks: Sequence[str],
) -> dict[str, dict[str, float]]:
    pairs = _paired_labels(preds, golds)
    check_names = _checks(checks)
    result: dict[str, dict[str, float]] = {}
    for check in check_names:
        true_positive = sum(1 for pred, gold in pairs if check in pred and check in gold)
        false_positive = sum(1 for pred, gold in pairs if check in pred and check not in gold)
        false_negative = sum(1 for pred, gold in pairs if check not in pred and check in gold)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[check] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def micro_f1(
    preds: Sequence[Collection[str]],
    golds: Sequence[Collection[str]],
    checks: Sequence[str],
) -> float:
    pairs = _paired_labels(preds, golds)
    check_names = set(_checks(checks))
    true_positive = sum(len((set(pred) & set(gold)) & check_names) for pred, gold in pairs)
    false_positive = sum(len((set(pred) - set(gold)) & check_names) for pred, gold in pairs)
    false_negative = sum(len((set(gold) - set(pred)) & check_names) for pred, gold in pairs)
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 0.0


def macro_f1(
    preds: Sequence[Collection[str]],
    golds: Sequence[Collection[str]],
    checks: Sequence[str],
) -> float:
    per_check = per_check_prf(preds, golds, checks)
    return sum(metrics["f1"] for metrics in per_check.values()) / len(per_check)


def seed_mean_sd(values: Collection[object]) -> tuple[float, float | None]:
    finite = _finite_values(values, name="seed_mean_sd")
    mean = float(statistics.mean(finite))
    if len(finite) == 1:
        return mean, None
    return mean, float(statistics.stdev(finite))


def distribution_summary(values: Collection[object]) -> DistributionSummary:
    finite = sorted(_finite_values(values, name="distribution_summary"))
    n = len(finite)

    def quantile(probability: float) -> float:
        index = min(n - 1, max(0, round(probability * (n - 1))))
        return finite[index]

    return DistributionSummary(
        p50=quantile(0.50),
        p90=quantile(0.90),
        p95=quantile(0.95),
        p99=quantile(0.99),
        max=finite[-1],
        n=n,
    )


def _bootstrap_interval(
    unit_ids: Collection[str],
    statistic: Callable[[list[str]], float],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
) -> Interval:
    boot_count = _positive_integer(n_boot, name="n_boot")
    random_seed = _integer(seed, name="seed")
    interval_alpha = _alpha(alpha)
    if not unit_ids:
        raise ValueError("bootstrap requires at least one unit")
    if any(not isinstance(unit_id, str) or not unit_id for unit_id in unit_ids):
        raise ValueError("bootstrap unit IDs must be nonempty strings")
    ordered_ids = sorted(unit_ids)
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("bootstrap unit IDs must be unique")

    point = _finite_number(statistic(ordered_ids), name="bootstrap point")
    generator = np.random.default_rng(random_seed)
    draws = np.empty(boot_count, dtype=float)
    for index in range(boot_count):
        positions = generator.integers(0, len(ordered_ids), size=len(ordered_ids))
        sample = [ordered_ids[position] for position in positions]
        draws[index] = _finite_number(statistic(sample), name="bootstrap draw")
    low, high = np.quantile(draws, [interval_alpha / 2.0, 1.0 - interval_alpha / 2.0])
    probability_le_zero = float(np.mean(draws <= 0.0))
    probability_ge_zero = float(np.mean(draws >= 0.0))
    p_value = min(1.0, 2.0 * min(probability_le_zero, probability_ge_zero))
    return Interval(
        point=point,
        ci_low=float(low),
        ci_high=float(high),
        p_value=p_value,
        n_boot=boot_count,
        seed=random_seed,
        alpha=interval_alpha,
        n_units=len(ordered_ids),
    )


def cluster_bootstrap_interval(
    values: Mapping[str, float],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> Interval:
    normalized: dict[str, float] = {}
    for unit_id, value in values.items():
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError("bootstrap unit IDs must be nonempty strings")
        normalized[unit_id] = _finite_number(value, name=f"bootstrap value for {unit_id}")

    def statistic(sample_ids: list[str]) -> float:
        return float(np.mean([normalized[unit_id] for unit_id in sample_ids]))

    return _bootstrap_interval(normalized, statistic, n_boot=n_boot, seed=seed, alpha=alpha)


def _comparison_task(task: object) -> ComparisonTask:
    if task not in {"rule_identification", "correction", "joint"}:
        raise ValueError("task must be rule_identification, correction, or joint")
    return cast(ComparisonTask, task)


def _record_success(record: object, task: ComparisonTask) -> float:
    if task == "rule_identification":
        value = getattr(record, "exact_match", None)
    else:
        outcome = getattr(record, "outcome", None)
        value = getattr(outcome, "overall_fixed", None)
    if not isinstance(value, bool):
        raise ValueError(f"{task} success must be boolean")
    return 1.0 if value else 0.0


def _normalize_rq3_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != _RQ3_COLUMNS:
        raise ValueError(f"RQ3 frame columns must be {_RQ3_COLUMNS!r}")
    normalized = frame.copy(deep=True)
    if normalized.empty:
        raise ValueError("RQ3 frame requires nonempty data")

    cells: list[tuple[str, int, str, str]] = []
    successes: list[int] = []
    for snippet_id, seed, condition, task, success in normalized.itertuples(index=False, name=None):
        if not isinstance(snippet_id, str) or not snippet_id:
            raise ValueError("RQ3 base snippet IDs must be nonempty strings")
        normalized_seed = _integer(seed, name="RQ3 record seed")
        if condition not in _RQ3_CONDITIONS:
            raise ValueError("RQ3 conditions must be C0 or C1")
        if task not in _RQ3_TASKS:
            raise ValueError("RQ3 tasks must be rule-identification, correction, or joint")
        if isinstance(success, bool):
            normalized_success = int(success)
        elif isinstance(success, Integral) and success in (0, 1):
            normalized_success = int(success)
        else:
            raise ValueError("RQ3 success must be binary")
        cells.append((snippet_id, normalized_seed, condition, task))
        successes.append(normalized_success)

    if len(cells) != len(set(cells)):
        raise ValueError("RQ3 frame contains a duplicate ID/seed/condition/task cell")
    snippet_ids = {cell[0] for cell in cells}
    seeds = {cell[1] for cell in cells}
    expected = {
        (snippet_id, seed, condition, task)
        for snippet_id in snippet_ids
        for seed in seeds
        for condition in _RQ3_CONDITIONS
        for task in _RQ3_TASKS
    }
    if set(cells) != expected:
        raise ValueError("RQ3 frame has an incomplete ID-by-seed-by-condition-by-task grid")

    normalized["seed"] = [cell[1] for cell in cells]
    normalized["success"] = successes
    normalized["condition"] = pd.Categorical(normalized["condition"], categories=list(_RQ3_CONDITIONS), ordered=True)
    normalized["task"] = pd.Categorical(normalized["task"], categories=list(_RQ3_TASKS), ordered=True)
    return normalized.sort_values(["base_snippet_id", "seed", "condition", "task"], kind="mergesort").reset_index(
        drop=True
    )


def build_rq3_frame(records: Sequence[ReleasedRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        condition = getattr(record, "condition", None)
        task = getattr(record, "task_type", None)
        if condition not in _RQ3_CONDITIONS or task not in _RQ3_TASKS:
            continue
        comparison_task = _comparison_task(task)
        rows.append(
            {
                "base_snippet_id": getattr(record, "base_snippet_id", None),
                "seed": getattr(record, "seed", None),
                "condition": condition,
                "task": comparison_task,
                "success": int(_record_success(record, comparison_task)),
            }
        )
    return _normalize_rq3_frame(pd.DataFrame.from_records(rows, columns=_RQ3_COLUMNS))


def fit_rq3_interaction(frame: pd.DataFrame) -> Rq3Result:
    normalized = _normalize_rq3_frame(frame)
    model = BinomialBayesMixedGLM.from_formula(
        _RQ3_FORMULA,
        {"snippet": "0 + C(base_snippet_id)"},
        normalized,
    )
    n_parameters = model.k_fep + model.k_vcp + model.k_vc
    result = model.fit_vb(
        mean=np.zeros(n_parameters, dtype=float),
        sd=np.full(n_parameters, np.exp(-0.5), dtype=float),
    )
    exog_names = result.model.exog_names
    if exog_names is None or _RQ3_INTERACTION_TERM not in exog_names:
        raise ValueError(f"RQ3 interaction term {_RQ3_INTERACTION_TERM!r} is absent")
    index = list(exog_names).index(_RQ3_INTERACTION_TERM)
    coefficient = _finite_number(result.fe_mean[index], name="RQ3 coefficient")
    sd = _finite_number(result.fe_sd[index], name="RQ3 standard deviation")
    z_value = float(norm.ppf(0.975))
    ci_low = coefficient - z_value * sd
    ci_high = coefficient + z_value * sd
    return Rq3Result(
        coefficient=coefficient,
        sd=sd,
        ci_low=ci_low,
        ci_high=ci_high,
        excludes_zero=ci_low > 0.0 or ci_high < 0.0,
        reject=ci_low > 0.0,
        term_name=_RQ3_INTERACTION_TERM,
    )


def _rule_labels(record: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    pred = getattr(record, "pred", None)
    gold = getattr(record, "gold", None)
    if (
        isinstance(pred, (str, bytes))
        or isinstance(gold, (str, bytes))
        or not isinstance(pred, Collection)
        or not isinstance(gold, Collection)
        or any(not isinstance(label, str) for label in pred)
        or any(not isinstance(label, str) for label in gold)
    ):
        raise ValueError("rule-identification pred and gold must be string collections")
    return tuple(pred), tuple(gold)


def _paired_grid(
    records: Sequence[ReleasedRecord],
    *,
    condition_a: Condition,
    condition_b: Condition,
    task: ComparisonTask,
    value: Callable[[object], V],
) -> tuple[tuple[str, ...], tuple[int, ...], dict[str, dict[tuple[str, int], V]]]:
    if condition_a == condition_b:
        raise ValueError("paired comparison conditions must differ")
    grids: dict[str, dict[tuple[str, int], V]] = {condition_a: {}, condition_b: {}}
    for record in records:
        record_condition = getattr(record, "condition", None)
        if record_condition not in grids or getattr(record, "task_type", None) != task:
            continue
        snippet_id = getattr(record, "base_snippet_id", None)
        seed = getattr(record, "seed", None)
        if not isinstance(snippet_id, str) or not snippet_id:
            raise ValueError("base snippet IDs must be nonempty strings")
        seed = _integer(seed, name="record seed")
        key = (snippet_id, seed)
        if key in grids[record_condition]:
            raise ValueError(f"duplicate paired cell for {record_condition}/{snippet_id}/{seed}")
        grids[record_condition][key] = value(record)

    if not grids[condition_a] or not grids[condition_b]:
        raise ValueError("paired comparison requires nonempty data for both conditions")
    snippet_sets = {condition: {snippet_id for snippet_id, _ in grid} for condition, grid in grids.items()}
    if snippet_sets[condition_a] != snippet_sets[condition_b]:
        raise ValueError("paired comparison snippet sets differ")
    seed_sets = {condition: {seed for _, seed in grid} for condition, grid in grids.items()}
    if seed_sets[condition_a] != seed_sets[condition_b]:
        raise ValueError("paired comparison seed sets differ")
    snippets = tuple(sorted(snippet_sets[condition_a]))
    seeds = tuple(sorted(seed_sets[condition_a]))
    expected = {(snippet_id, seed) for snippet_id in snippets for seed in seeds}
    for condition, grid in grids.items():
        if set(grid) != expected:
            raise ValueError(f"paired comparison has incomplete cells for {condition}")
    return snippets, seeds, grids


def seed_averaged_snippet_differences(
    records: Sequence[ReleasedRecord],
    *,
    condition_a: Condition,
    condition_b: Condition,
    task: Literal["rule_identification", "correction", "joint"],
) -> dict[str, float]:
    comparison_task = _comparison_task(task)
    snippets, seeds, grids = _paired_grid(
        records,
        condition_a=condition_a,
        condition_b=condition_b,
        task=comparison_task,
        value=lambda record: _record_success(record, comparison_task),
    )
    return {
        snippet_id: float(
            np.mean([grids[condition_a][(snippet_id, seed)] - grids[condition_b][(snippet_id, seed)] for seed in seeds])
        )
        for snippet_id in snippets
    }


def rq3_task_contrasts(
    records: Sequence[ReleasedRecord],
    *,
    n_boot: int,
    seed: int,
) -> tuple[Rq3TaskContrast, ...]:
    build_rq3_frame(records)
    focal = seed_averaged_snippet_differences(
        records,
        condition_a="C1",
        condition_b="C0",
        task="joint",
    )
    results: list[Rq3TaskContrast] = []
    for name, reference_task in (
        ("joint_minus_correction", "correction"),
        ("joint_minus_rule_identification", "rule_identification"),
    ):
        reference = seed_averaged_snippet_differences(
            records,
            condition_a="C1",
            condition_b="C0",
            task=reference_task,
        )
        if set(focal) != set(reference):
            raise ValueError(f"RQ3 focal and {reference_task} snippet sets differ")
        differences = {snippet_id: focal[snippet_id] - reference[snippet_id] for snippet_id in sorted(focal)}
        interval = cluster_bootstrap_interval(differences, n_boot=n_boot, seed=seed)
        results.append(
            Rq3TaskContrast(
                name=name,
                condition_a="C1",
                condition_b="C0",
                focal_task="joint",
                reference_task=reference_task,
                interval=interval,
                reject=interval.ci_low > 0.0,
            )
        )
    return tuple(results)


def macro_f1_difference_interval(
    records: Sequence[ReleasedRecord],
    *,
    condition_a: Condition,
    condition_b: Condition,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> Interval:
    snippets, seeds, grids = _paired_grid(
        records,
        condition_a=condition_a,
        condition_b=condition_b,
        task="rule_identification",
        value=_rule_labels,
    )

    def statistic(sample_ids: list[str]) -> float:
        seed_differences: list[float] = []
        for observed_seed in seeds:
            pred_a, gold_a = zip(
                *(grids[condition_a][(snippet_id, observed_seed)] for snippet_id in sample_ids), strict=True
            )
            pred_b, gold_b = zip(
                *(grids[condition_b][(snippet_id, observed_seed)] for snippet_id in sample_ids), strict=True
            )
            seed_differences.append(macro_f1(pred_a, gold_a, CHECKS) - macro_f1(pred_b, gold_b, CHECKS))
        return float(np.mean(seed_differences))

    return _bootstrap_interval(snippets, statistic, n_boot=n_boot, seed=seed, alpha=alpha)


def task_contrast(
    records: Sequence[ReleasedRecord],
    *,
    name: str,
    condition_a: Condition,
    condition_b: Condition,
    task: Literal["rule_identification", "correction", "joint"],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> TaskContrast:
    if not isinstance(name, str) or not name:
        raise ValueError("contrast name must be a nonempty string")
    differences = seed_averaged_snippet_differences(
        records,
        condition_a=condition_a,
        condition_b=condition_b,
        task=task,
    )
    interval = cluster_bootstrap_interval(differences, n_boot=n_boot, seed=seed, alpha=alpha)
    return TaskContrast(
        name=name,
        condition_a=condition_a,
        condition_b=condition_b,
        task=task,
        interval=interval,
    )
