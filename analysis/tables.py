from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor, isfinite, log10
import statistics
from typing import Literal, TypeAlias, cast

from analysis.inputs import (
    CHECKS,
    TASK_TYPES,
    AnalysisMetadata,
    Condition,
    CorrectionRecord,
    ExperimentConfig,
    FineTunedCondition,
    JointRecord,
    ReleasedRecord,
    RuleIdentificationRecord,
    RunResults,
    SelectionPoint,
    SerializerAudit,
    StudyRow,
    expected_run_keys,
)
from analysis.metrics import (
    build_rq3_frame,
    cluster_bootstrap_interval,
    distribution_summary,
    exact_match_rate,
    fit_rq3_interaction,
    macro_f1,
    macro_f1_difference_interval,
    micro_f1,
    per_check_prf,
    rq3_task_contrasts,
    seed_averaged_snippet_differences,
    seed_mean_sd,
    task_contrast,
)

Scalar: TypeAlias = bool | int | float | str | None
OutputTargetKey: TypeAlias = tuple[str, str, str | None, str | None, str | None]


@dataclass(frozen=True)
class OutputCell:
    result_id: str | None
    value: Scalar
    display_digits: int | None = None


@dataclass(frozen=True)
class TableRow:
    key: str
    cells: dict[str, OutputCell | str]


@dataclass(frozen=True)
class TableData:
    filename: str
    columns: tuple[str, ...]
    rows: tuple[TableRow, ...]


@dataclass(frozen=True)
class GeneratedOutputs:
    results: dict[str, Scalar]
    tables: dict[str, TableData]


@dataclass(frozen=True)
class TrainingLengthSummary:
    p50: int
    p90: int
    p95: int
    p99: int
    max: int
    n: int


_TABLE_8_1_COLUMNS = (
    "condition",
    "rule_id_exact_match_mean",
    "rule_id_exact_match_sd",
    "rule_id_micro_f1_mean",
    "rule_id_micro_f1_sd",
    "rule_id_macro_f1_mean",
    "rule_id_macro_f1_sd",
    "correction_fix_rate_mean",
    "correction_fix_rate_sd",
    "joint_fix_rate_mean",
    "joint_fix_rate_sd",
)
_TABLE_8_1_ROWS: tuple[tuple[Condition, str, str], ...] = (
    ("zero-shot-raw", "zero_shot_raw", "Zero-shot raw"),
    ("zero-shot-syntax", "zero_shot_syntax", "Zero-shot syntax"),
    ("C0", "c0", "C0 (raw)"),
    ("C1", "c1", "C1 (syntax-input)"),
    ("C2", "c2", "C2 (syntax + aux)"),
    ("C2-control", "c2_control", "C2-control"),
)
_TABLE_8_1_METRICS = (
    "rule_id_exact_match",
    "rule_id_micro_f1",
    "rule_id_macro_f1",
    "correction_fix_rate",
    "joint_fix_rate",
)

_TABLE_8_2_COLUMNS = (
    "component",
    "c0_correction",
    "c1_correction",
    "c2_correction",
    "c2_control_correction",
    "c0_joint",
    "c1_joint",
    "c2_joint",
    "c2_control_joint",
)
_TABLE_8_2_ROWS = (
    ("extraction_success", "Extraction success"),
    ("go_fence", "via Go-tagged fence"),
    ("untagged_fence", "via untagged fence"),
    ("parse_probe", "via parse-probe recovery"),
    ("parse_validity", "Parse validity"),
    ("target_fixed", "Target check(s) fixed"),
    ("regression_adjusted_fix", "Regression-adjusted fix"),
)
_TABLE_8_2_CELLS = (
    ("C0", "correction", "c0_correction"),
    ("C1", "correction", "c1_correction"),
    ("C2", "correction", "c2_correction"),
    ("C2-control", "correction", "c2_control_correction"),
    ("C0", "joint", "c0_joint"),
    ("C1", "joint", "c1_joint"),
    ("C2", "joint", "c2_joint"),
    ("C2-control", "joint", "c2_control_joint"),
)

_TABLE_8_3_COLUMNS = (
    "task",
    "c1_minus_c0_point",
    "c1_minus_c0_low",
    "c1_minus_c0_high",
    "c2_minus_c0_point",
    "c2_minus_c0_low",
    "c2_minus_c0_high",
)
_TABLE_8_3_ROWS: tuple[tuple[Literal["rule_identification", "correction", "joint"], str, Literal[42, 43, 44]], ...] = (
    ("rule_identification", "Rule identification", 44),
    ("correction", "Correction", 43),
    ("joint", "Joint", 42),
)

_TABLE_8_4_COLUMNS = ("condition", "task", "a", "b", "c", "d", "invalid")
_TABLE_8_4_ROWS: tuple[tuple[Condition, str, str], ...] = (
    ("C0", "correction", "c0_correction"),
    ("C1", "correction", "c1_correction"),
    ("C2", "correction", "c2_correction"),
    ("C2-control", "correction", "c2_control_correction"),
    ("C0", "joint", "c0_joint"),
    ("C1", "joint", "c1_joint"),
    ("C2", "joint", "c2_joint"),
    ("C2-control", "joint", "c2_control_joint"),
)

_TABLE_8_5_COLUMNS = ("condition_task", "identity", "p50", "p90", "p95", "p99", "max", "n")
_TABLE_8_5_NUMERIC_ROWS = (
    ("c0_rule_identification", "C0 : rule identification", "C0", "rule_identification"),
    ("c0_explanation", "C0 : explanation", "C0", "explanation"),
    ("c0_correction", "C0 : correction", "C0", "correction"),
    ("c0_joint", "C0 : joint", "C0", "joint"),
    ("c1_rule_identification", "C1 : rule identification", "C1", "rule_identification"),
    ("c1_explanation", "C1 : explanation", "C1", "explanation"),
    ("c1_correction", "C1 : correction", "C1", "correction"),
    ("c1_joint", "C1 : joint", "C1", "joint"),
    ("c2_auxiliary", "C2 : syntax auxiliary", "C2", "syntax_summary"),
    (
        "c2_control_rule_identification",
        "C2-control : rule identification",
        "C2-control",
        "rule_identification",
    ),
    ("c2_control_explanation", "C2-control : explanation", "C2-control", "explanation"),
    ("c2_control_correction", "C2-control : correction", "C2-control", "correction"),
    ("c2_control_joint", "C2-control : joint", "C2-control", "joint"),
)

_TABLE_8_6_COLUMNS = ("quantity", "c0", "c1", "c2", "c2_control")
_TABLE_8_6_ROWS = (
    ("total_tokens_m", "Total tokens (millions)", "total_tokens", 1_000_000.0, 1),
    ("supervised_tokens_m", "Supervised tokens (millions)", "supervised_tokens", 1_000_000.0, 2),
    ("optimizer_steps", "Optimizer steps", "optimizer_steps", 1.0, 0),
    ("training_minutes", "Training minutes", "wall_clock_train_s", 60.0, 0),
    ("end_to_end_minutes", "End-to-end minutes", "wall_clock_total_s", 60.0, 0),
    (
        "peak_allocated_gpu_memory_gib",
        "Peak allocated GPU memory (GiB)",
        "peak_allocated_gpu_memory_gib",
        1.0,
        1,
    ),
)


def _add_registry_entry(
    registry: dict[str, OutputTargetKey],
    targets: set[OutputTargetKey],
    result_id: str,
    target: OutputTargetKey,
) -> None:
    if not result_id:
        raise ValueError("output cell requires a nonempty result ID")
    if result_id in registry:
        raise ValueError(f"duplicate result ID: {result_id}")
    if target in targets:
        raise ValueError(f"duplicate physical target coordinate: {target!r}")
    registry[result_id] = target
    targets.add(target)


def output_registry(outputs: GeneratedOutputs) -> dict[str, OutputTargetKey]:
    registry: dict[str, OutputTargetKey] = {}
    targets: set[OutputTargetKey] = set()
    for result_id, value in outputs.results.items():
        if not result_id:
            raise ValueError("JSON result requires a nonempty result ID")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"JSON result {result_id} must be finite")
        _add_registry_entry(
            registry,
            targets,
            result_id,
            ("json", "results.json", result_id, None, None),
        )

    seen_files: set[str] = set()
    for mapping_filename, table in outputs.tables.items():
        if mapping_filename != table.filename:
            raise ValueError(
                f"table mapping filename {mapping_filename!r} is inconsistent with "
                f"TableData filename {table.filename!r}"
            )
        duplicate_filename = table.filename in seen_files
        seen_files.add(table.filename)
        if not table.columns or len(table.columns) != len(set(table.columns)):
            raise ValueError(f"table {table.filename} requires nonempty unique columns")
        seen_rows: set[str] = set()
        for row in table.rows:
            if row.key in seen_rows:
                raise ValueError(f"duplicate row key in {table.filename}: {row.key}")
            seen_rows.add(row.key)
            if set(row.cells) != set(table.columns):
                raise ValueError(f"table {table.filename} row {row.key} cells do not match columns")
            for column, cell in row.cells.items():
                if isinstance(cell, str):
                    continue
                if not isinstance(cell, OutputCell):
                    raise ValueError(f"table {table.filename} row {row.key} contains an invalid cell")
                if isinstance(cell.value, float) and not isfinite(cell.value):
                    raise ValueError(f"table {table.filename} row {row.key} column {column} must be finite")
                if cell.display_digits is not None and (
                    type(cell.display_digits) is not int or cell.display_digits < 0
                ):
                    raise ValueError("display digits must be a nonnegative integer or null")
                _add_registry_entry(
                    registry,
                    targets,
                    cell.result_id or "",
                    ("csv", table.filename, None, row.key, column),
                )
        if duplicate_filename:
            raise ValueError(f"duplicate table filename: {table.filename}")
    return registry


def _condition_records(
    records: Sequence[ReleasedRecord],
    *,
    condition: Condition,
    task_type: str,
) -> list[ReleasedRecord]:
    selected = [record for record in records if record.condition == condition and record.task_type == task_type]
    if not selected:
        raise ValueError(f"{condition} {task_type} requires at least one record")
    return selected


def _seed_metric_values(records: Sequence[ReleasedRecord], condition: Condition, metric: str) -> list[float]:
    expected_seeds = (42,) if condition.startswith("zero-shot") else (42, 43, 44)
    values: list[float] = []
    if metric.startswith("rule_id_"):
        selected = _condition_records(records, condition=condition, task_type="rule_identification")
        for seed in expected_seeds:
            seed_records = [record for record in selected if record.seed == seed]
            if not seed_records:
                raise ValueError(f"{condition} rule_identification seed {seed} requires records")
            rule_records = cast(list[RuleIdentificationRecord], seed_records)
            preds = [record.pred for record in rule_records]
            golds = [record.gold for record in rule_records]
            if metric == "rule_id_exact_match":
                values.append(exact_match_rate(preds, golds))
            elif metric == "rule_id_micro_f1":
                values.append(micro_f1(preds, golds, CHECKS))
            else:
                values.append(macro_f1(preds, golds, CHECKS))
    else:
        task_type = "correction" if metric == "correction_fix_rate" else "joint"
        selected = _condition_records(records, condition=condition, task_type=task_type)
        for seed in expected_seeds:
            seed_records = [record for record in selected if record.seed == seed]
            if not seed_records:
                raise ValueError(f"{condition} {task_type} seed {seed} requires records")
            repair_records = cast(list[CorrectionRecord | JointRecord], seed_records)
            values.append(sum(record.outcome.overall_fixed for record in repair_records) / len(repair_records))
    if {record.seed for record in selected} != set(expected_seeds):
        raise ValueError(f"{condition} {metric} has an unexpected seed set")
    return values


def _build_table_8_1(scored_records: Sequence[ReleasedRecord]) -> TableData:
    rows: list[TableRow] = []
    for condition, row_key, label in _TABLE_8_1_ROWS:
        cells: dict[str, OutputCell | str] = {"condition": label}
        for metric in _TABLE_8_1_METRICS:
            mean, sd = seed_mean_sd(_seed_metric_values(scored_records, condition, metric))
            prefix = f"table_8_1.{row_key}.{metric}"
            suffix = "value" if condition.startswith("zero-shot") else "mean"
            cells[f"{metric}_mean"] = OutputCell(f"{prefix}.{suffix}", mean * 100.0, 2)
            if sd is None:
                cells[f"{metric}_sd"] = ""
            else:
                cells[f"{metric}_sd"] = OutputCell(f"{prefix}.sd", sd * 100.0, 2)
        rows.append(TableRow(key=row_key, cells=cells))
    return TableData(filename="table-8-1.csv", columns=_TABLE_8_1_COLUMNS, rows=tuple(rows))


def _repair_predicate(row_key: str, record: ReleasedRecord) -> bool:
    outcome = cast(CorrectionRecord | JointRecord, record).outcome
    if row_key == "extraction_success":
        return outcome.extracted
    if row_key == "go_fence":
        return outcome.extraction_status == "go_block"
    if row_key == "untagged_fence":
        return outcome.extraction_status == "fenced_block"
    if row_key == "parse_probe":
        return outcome.extraction_status == "largest_parseable"
    if row_key == "parse_validity":
        return outcome.parse_ok
    if row_key == "target_fixed":
        return outcome.target_fixed
    return outcome.overall_fixed


def _build_table_8_2(scored_records: Sequence[ReleasedRecord]) -> TableData:
    rows: list[TableRow] = []
    for row_key, label in _TABLE_8_2_ROWS:
        cells: dict[str, OutputCell | str] = {"component": label}
        for condition, task_type, column in _TABLE_8_2_CELLS:
            selected = _condition_records(scored_records, condition=condition, task_type=task_type)
            if {record.seed for record in selected} != {42, 43, 44}:
                raise ValueError(f"{condition} {task_type} has an unexpected seed set")
            value = sum(_repair_predicate(row_key, record) for record in selected) / len(selected) * 100.0
            cells[column] = OutputCell(f"table_8_2.{row_key}.{column}", value, 1)
        rows.append(TableRow(key=row_key, cells=cells))
    return TableData(filename="table-8-2.csv", columns=_TABLE_8_2_COLUMNS, rows=tuple(rows))


def _build_table_8_3(scored_records: Sequence[ReleasedRecord]) -> TableData:
    rows: list[TableRow] = []
    for task_type, label, c1_seed in _TABLE_8_3_ROWS:
        intervals = {
            "c1_minus_c0": task_contrast(
                scored_records,
                name=f"{task_type}.c1_minus_c0",
                condition_a="C1",
                condition_b="C0",
                task=task_type,
                n_boot=10_000,
                seed=c1_seed,
            ).interval,
            "c2_minus_c0": task_contrast(
                scored_records,
                name=f"{task_type}.c2_minus_c0",
                condition_a="C2",
                condition_b="C0",
                task=task_type,
                n_boot=10_000,
                seed=42,
            ).interval,
        }
        cells: dict[str, OutputCell | str] = {"task": label}
        for comparison, interval in intervals.items():
            for statistic, value in (
                ("point", interval.point),
                ("low", interval.ci_low),
                ("high", interval.ci_high),
            ):
                column = f"{comparison}_{statistic}"
                result_id = f"table_8_3.{task_type}.{comparison}.{statistic}"
                cells[column] = OutputCell(result_id, value * 100.0, 2)
        rows.append(TableRow(key=task_type, cells=cells))
    return TableData(filename="table-8-3.csv", columns=_TABLE_8_3_COLUMNS, rows=tuple(rows))


def _build_table_8_4(scored_records: Sequence[ReleasedRecord]) -> TableData:
    rows: list[TableRow] = []
    for condition, task_type, row_key in _TABLE_8_4_ROWS:
        selected = _condition_records(scored_records, condition=condition, task_type=task_type)
        cells: dict[str, OutputCell | str] = {
            "condition": condition,
            "task": task_type,
        }
        for category in ("A", "B", "C", "D", "INVALID"):
            column = category.lower()
            count = sum(
                cast(CorrectionRecord | JointRecord, record).outcome.category == category for record in selected
            )
            cells[column] = OutputCell(f"table_8_4.{row_key}.{column}", count, 0)
        rows.append(TableRow(key=row_key, cells=cells))
    return TableData(filename="table-8-4.csv", columns=_TABLE_8_4_COLUMNS, rows=tuple(rows))


def _training_length_summary(values: Sequence[int]) -> TrainingLengthSummary:
    if not values:
        raise ValueError("training length summary requires at least one value")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("training lengths must be nonnegative integers")
    ordered = sorted(values)

    def nearest_rank(percent: int) -> int:
        return ordered[max(1, ceil(percent / 100.0 * len(ordered))) - 1]

    return TrainingLengthSummary(
        p50=nearest_rank(50),
        p90=nearest_rank(90),
        p95=nearest_rank(95),
        p99=nearest_rank(99),
        max=ordered[-1],
        n=len(ordered),
    )


def _expanded_training_lengths(study_rows: Sequence[StudyRow]) -> dict[tuple[str, str], list[int]]:
    result: dict[tuple[str, str], list[int]] = {}
    for row in study_rows:
        for contribution in row.training_contributions:
            key = (contribution.condition, contribution.task_type)
            result.setdefault(key, []).extend([contribution.total_tokens] * contribution.multiplicity)
    return result


def _main_training_identity(
    study_rows: Sequence[StudyRow],
    condition: Literal["C1", "C2"],
) -> Counter[tuple[str, str, int, int, int, int]]:
    return Counter(
        (
            row.base_snippet_id,
            contribution.task_type,
            contribution.prompt_tokens,
            contribution.response_tokens,
            contribution.total_tokens,
            contribution.multiplicity,
        )
        for row in study_rows
        for contribution in row.training_contributions
        if contribution.condition == condition and contribution.pool == "main" and contribution.task_type in TASK_TYPES
    )


def _build_table_8_5(study_rows: Sequence[StudyRow]) -> TableData:
    lengths = _expanded_training_lengths(study_rows)
    if _main_training_identity(study_rows, "C2") != _main_training_identity(study_rows, "C1"):
        raise ValueError("C2 main rows must be identical to C1")

    rows: list[TableRow] = []
    for row_key, label, condition, task_type in _TABLE_8_5_NUMERIC_ROWS:
        summary = _training_length_summary(lengths.get((condition, task_type), ()))
        cells: dict[str, OutputCell | str] = {"condition_task": label, "identity": ""}
        for statistic in ("p50", "p90", "p95", "p99", "max", "n"):
            cells[statistic] = OutputCell(
                f"table_8_5.{row_key}.{statistic}",
                cast(int, getattr(summary, statistic)),
                0,
            )
        rows.append(TableRow(key=row_key, cells=cells))
        if row_key == "c1_joint":
            rows.append(
                TableRow(
                    key="c2_main",
                    cells={
                        "condition_task": "C2 : main tasks",
                        "identity": OutputCell(
                            "table_8_5.c2_main.identity",
                            "identical_to_c1_main_rows",
                        ),
                        "p50": "",
                        "p90": "",
                        "p95": "",
                        "p99": "",
                        "max": "",
                        "n": "",
                    },
                )
            )
    return TableData(filename="table-8-5.csv", columns=_TABLE_8_5_COLUMNS, rows=tuple(rows))


def _build_table_8_6(run_results: Sequence[RunResults], metadata: AnalysisMetadata) -> TableData:
    del metadata
    rows: list[TableRow] = []
    condition_columns = (("C0", "c0"), ("C1", "c1"), ("C2", "c2"), ("C2-control", "c2_control"))
    for row_key, label, field, divisor, display_digits in _TABLE_8_6_ROWS:
        cells: dict[str, OutputCell | str] = {"quantity": label}
        for condition, column in condition_columns:
            values = [getattr(result.metrics.compute, field) for result in run_results if result.condition == condition]
            if not values or any(value is None for value in values):
                raise ValueError(f"{condition} {field} requires complete run values")
            transformed = statistics.mean(cast(Sequence[int | float], values)) / divisor
            cells[column] = OutputCell(f"table_8_6.{row_key}.{column}", transformed, display_digits)
        rows.append(TableRow(key=row_key, cells=cells))
    return TableData(filename="table-8-6.csv", columns=_TABLE_8_6_COLUMNS, rows=tuple(rows))


_CHECK_RESULT_NAMES = {
    "assignOp": "assign_op",
    "builtinShadow": "builtin_shadow",
    "captLocal": "capt_local",
    "commentFormatting": "comment_formatting",
    "elseif": "elseif",
    "ifElseChain": "if_else_chain",
    "paramTypeCombine": "param_type_combine",
    "singleCaseSwitch": "single_case_switch",
}
_FINE_TUNED_CONDITIONS: tuple[FineTunedCondition, ...] = ("C0", "C1", "C2", "C2-control")
_CONDITION_RESULT_NAMES = {
    "C0": "c0",
    "C1": "c1",
    "C2": "c2",
    "C2-control": "c2_control",
    "zero-shot-raw": "zero_shot_raw",
    "zero-shot-syntax": "zero_shot_syntax",
}

_SECONDARY_SEEDS = (42, 43, 44)
_TABLE_8_7_COLUMNS = (
    "check",
    "n",
    "c0_rule_f1",
    "c1_rule_f1",
    "c1_minus_c0_rule_f1",
    "c0_joint_fix",
    "c1_joint_fix",
    "c1_minus_c0_joint_fix",
    "c2_rule_f1",
    "c2_control_rule_f1",
)
_TABLE_8_8_COLUMNS = (
    "stratum",
    "bin",
    "n",
    "rule_id_c0",
    "rule_id_c1_minus_c0_point",
    "rule_id_c1_minus_c0_low",
    "rule_id_c1_minus_c0_high",
    "rule_id_c2_minus_c0_point",
    "rule_id_c2_minus_c0_low",
    "rule_id_c2_minus_c0_high",
    "joint_c0",
    "joint_c1_minus_c0_point",
    "joint_c1_minus_c0_low",
    "joint_c1_minus_c0_high",
    "joint_c2_minus_c0_point",
    "joint_c2_minus_c0_low",
    "joint_c2_minus_c0_high",
)
_TABLE_8_8_ROWS = (
    ("violations_1", "Violations", "= 1"),
    ("violations_2_plus", "Violations", ">= 2"),
    ("length_lt_50", "Length", "< 50 lines"),
    ("length_50_199", "Length", "50-199 lines"),
    ("length_200_plus", "Length", ">= 200 lines"),
    ("depth_0_1", "Nesting depth", "<= 1"),
    ("depth_2_3", "Nesting depth", "2-3"),
    ("depth_gt_3", "Nesting depth", "> 3"),
)
_TABLE_8_9_COLUMNS = ("check", "n", "zero_shot_raw", "zero_shot_syntax", "fine_tuned_c0")
_FAMILIARITY_CONDITION_SEEDS: dict[Condition, tuple[int, ...]] = {
    "C0": _SECONDARY_SEEDS,
    "zero-shot-raw": (42,),
    "zero-shot-syntax": (42,),
}

_SecondaryKey: TypeAlias = tuple[FineTunedCondition, int, str]
_FamiliarityKey: TypeAlias = tuple[Condition, int, str]


def _secondary_evaluation_grid(
    scored_records: Sequence[ReleasedRecord],
    study_rows: Sequence[StudyRow],
) -> tuple[
    dict[str, StudyRow],
    dict[_SecondaryKey, RuleIdentificationRecord],
    dict[_SecondaryKey, JointRecord],
]:
    rows: dict[str, StudyRow] = {}
    for row in study_rows:
        if row.split != "test" or row.oracle.status == "excluded" or row.quarantined:
            continue
        if row.base_snippet_id in rows:
            raise ValueError(f"duplicate secondary study-row join ID: {row.base_snippet_id}")
        rows[row.base_snippet_id] = row
    if len(rows) != 410:
        raise ValueError(f"secondary evaluation join requires exactly 410 study rows, got {len(rows)}")
    if any(row.violation_count < 1 for row in rows.values()):
        raise ValueError("secondary violation strata require at least one violation per scored row")

    rule_grid: dict[_SecondaryKey, RuleIdentificationRecord] = {}
    joint_grid: dict[_SecondaryKey, JointRecord] = {}
    for released_record in scored_records:
        if released_record.condition not in _FINE_TUNED_CONDITIONS:
            continue
        if released_record.task_type not in {"rule_identification", "joint"}:
            continue
        record = cast(RuleIdentificationRecord | JointRecord, released_record)
        if record.base_snippet_id not in rows:
            raise ValueError(f"secondary record has no scored study-row join: {record.base_snippet_id}")
        expected_targets = rows[record.base_snippet_id].target_checks
        if record.target_checks != expected_targets:
            raise ValueError(f"secondary record target checks disagree with study row: {record.base_snippet_id}")
        key = (cast(FineTunedCondition, record.condition), record.seed, record.base_snippet_id)
        if record.task_type == "rule_identification":
            rule_record = cast(RuleIdentificationRecord, record)
            if rule_record.gold != expected_targets:
                raise ValueError(f"secondary rule gold disagrees with study-row targets: {record.base_snippet_id}")
            if key in rule_grid:
                raise ValueError(f"duplicate secondary rule grid cell: {key!r}")
            rule_grid[key] = rule_record
        else:
            joint_record = cast(JointRecord, record)
            if key in joint_grid:
                raise ValueError(f"duplicate secondary joint grid cell: {key!r}")
            joint_grid[key] = joint_record

    expected_keys = {
        (condition, seed, snippet_id)
        for condition in _FINE_TUNED_CONDITIONS
        for seed in _SECONDARY_SEEDS
        for snippet_id in rows
    }
    if set(rule_grid) != expected_keys:
        raise ValueError("secondary rule grid must contain the exact 4-condition by 3-seed by 410-ID cells")
    if set(joint_grid) != expected_keys:
        raise ValueError("secondary joint grid must contain the exact 4-condition by 3-seed by 410-ID cells")
    return rows, rule_grid, joint_grid


def _secondary_rule_f1(
    rule_grid: Mapping[_SecondaryKey, RuleIdentificationRecord],
    condition: FineTunedCondition,
) -> dict[str, float]:
    per_seed: list[dict[str, dict[str, float]]] = []
    for seed in _SECONDARY_SEEDS:
        records = sorted(
            (
                record
                for (cell_condition, cell_seed, _snippet_id), record in rule_grid.items()
                if cell_condition == condition and cell_seed == seed
            ),
            key=lambda record: record.base_snippet_id,
        )
        per_seed.append(
            per_check_prf(
                [record.pred for record in records],
                [record.gold for record in records],
                CHECKS,
            )
        )
    return {check: statistics.mean(seed_values[check]["f1"] for seed_values in per_seed) * 100.0 for check in CHECKS}


def _secondary_joint_fix(
    rows: Mapping[str, StudyRow],
    joint_grid: Mapping[_SecondaryKey, JointRecord],
    condition: FineTunedCondition,
    check: str,
) -> float:
    snippet_ids = sorted(snippet_id for snippet_id, row in rows.items() if check in row.target_checks)
    if not snippet_ids:
        raise ValueError(f"secondary check {check} requires positive support")
    return (
        statistics.mean(
            statistics.mean(
                joint_grid[(condition, seed, snippet_id)].outcome.overall_fixed for snippet_id in snippet_ids
            )
            for seed in _SECONDARY_SEEDS
        )
        * 100.0
    )


def _build_table_8_7(
    scored_records: Sequence[ReleasedRecord],
    study_rows: Sequence[StudyRow],
) -> TableData:
    rows_by_id, rule_grid, joint_grid = _secondary_evaluation_grid(scored_records, study_rows)
    rule_f1 = {condition: _secondary_rule_f1(rule_grid, condition) for condition in _FINE_TUNED_CONDITIONS}
    table_rows: list[TableRow] = []
    for check in CHECKS:
        row_key = _CHECK_RESULT_NAMES[check]
        c0_joint = _secondary_joint_fix(rows_by_id, joint_grid, "C0", check)
        c1_joint = _secondary_joint_fix(rows_by_id, joint_grid, "C1", check)
        values: dict[str, int | float] = {
            "n": sum(check in row.target_checks for row in rows_by_id.values()),
            "c0_rule_f1": rule_f1["C0"][check],
            "c1_rule_f1": rule_f1["C1"][check],
            "c1_minus_c0_rule_f1": rule_f1["C1"][check] - rule_f1["C0"][check],
            "c0_joint_fix": c0_joint,
            "c1_joint_fix": c1_joint,
            "c1_minus_c0_joint_fix": c1_joint - c0_joint,
            "c2_rule_f1": rule_f1["C2"][check],
            "c2_control_rule_f1": rule_f1["C2-control"][check],
        }
        cells: dict[str, OutputCell | str] = {"check": check}
        for column, value in values.items():
            cells[column] = OutputCell(
                f"table_8_7.{row_key}.{column}",
                value,
                0 if column == "n" else 1,
            )
        table_rows.append(TableRow(key=row_key, cells=cells))
    return TableData(filename="table-8-7.csv", columns=_TABLE_8_7_COLUMNS, rows=tuple(table_rows))


def _secondary_row_in_stratum(row_key: str, row: StudyRow) -> bool:
    if row_key == "violations_1":
        return row.violation_count == 1
    if row_key == "violations_2_plus":
        return row.violation_count >= 2
    if row_key == "length_lt_50":
        return row.source_line_count < 50
    if row_key == "length_50_199":
        return 50 <= row.source_line_count < 200
    if row_key == "length_200_plus":
        return row.source_line_count >= 200
    if row_key == "depth_0_1":
        return row.serializer.maximum_depth <= 1
    if row_key == "depth_2_3":
        return 2 <= row.serializer.maximum_depth <= 3
    if row_key == "depth_gt_3":
        return row.serializer.maximum_depth > 3
    raise ValueError(f"unknown secondary stratum row: {row_key}")


def _secondary_stratum_ids(row_key: str, rows: Mapping[str, StudyRow]) -> tuple[str, ...]:
    result = tuple(sorted(snippet_id for snippet_id, row in rows.items() if _secondary_row_in_stratum(row_key, row)))
    if not result:
        raise ValueError(f"secondary stratum {row_key} requires positive support")
    return result


def _secondary_stratum_baseline(
    values: Mapping[_SecondaryKey, float],
    snippet_ids: Sequence[str],
) -> float:
    return statistics.mean(
        statistics.mean(values[("C0", seed, snippet_id)] for snippet_id in snippet_ids) for seed in _SECONDARY_SEEDS
    )


def _secondary_stratum_differences(
    values: Mapping[_SecondaryKey, float],
    snippet_ids: Sequence[str],
    condition: Literal["C1", "C2"],
) -> dict[str, float]:
    return {
        snippet_id: statistics.mean(
            values[(condition, seed, snippet_id)] - values[("C0", seed, snippet_id)] for seed in _SECONDARY_SEEDS
        )
        for snippet_id in snippet_ids
    }


def _build_table_8_8(
    scored_records: Sequence[ReleasedRecord],
    study_rows: Sequence[StudyRow],
) -> TableData:
    rows_by_id, rule_grid, joint_grid = _secondary_evaluation_grid(scored_records, study_rows)
    endpoints: tuple[tuple[str, str, dict[_SecondaryKey, float]], ...] = (
        (
            "rule_id",
            "rule_id_c0",
            {key: float(record.exact_match) for key, record in rule_grid.items()},
        ),
        (
            "joint",
            "joint_c0",
            {key: float(record.outcome.overall_fixed) for key, record in joint_grid.items()},
        ),
    )
    table_rows: list[TableRow] = []
    for row_key, stratum, bin_label in _TABLE_8_8_ROWS:
        snippet_ids = _secondary_stratum_ids(row_key, rows_by_id)
        cells: dict[str, OutputCell | str] = {
            "stratum": stratum,
            "bin": bin_label,
            "n": OutputCell(f"table_8_8.{row_key}.n", len(snippet_ids), 0),
        }
        for endpoint, baseline_column, values in endpoints:
            baseline = _secondary_stratum_baseline(values, snippet_ids)
            cells[baseline_column] = OutputCell(
                f"table_8_8.{row_key}.{baseline_column}",
                baseline * 100.0,
                1,
            )
            for condition, comparison in (("C1", "c1_minus_c0"), ("C2", "c2_minus_c0")):
                differences = _secondary_stratum_differences(values, snippet_ids, condition)
                interval = cluster_bootstrap_interval(
                    differences,
                    n_boot=10_000,
                    seed=42,
                    alpha=0.05,
                )
                for statistic, value in (
                    ("point", interval.point),
                    ("low", interval.ci_low),
                    ("high", interval.ci_high),
                ):
                    column = f"{endpoint}_{comparison}_{statistic}"
                    cells[column] = OutputCell(
                        f"table_8_8.{row_key}.{column}",
                        value * 100.0,
                        1,
                    )
        table_rows.append(TableRow(key=row_key, cells=cells))
    return TableData(filename="table-8-8.csv", columns=_TABLE_8_8_COLUMNS, rows=tuple(table_rows))


def _familiarity_rule_grid(
    scored_records: Sequence[ReleasedRecord],
) -> tuple[tuple[str, ...], dict[_FamiliarityKey, RuleIdentificationRecord]]:
    grid: dict[_FamiliarityKey, RuleIdentificationRecord] = {}
    for released_record in scored_records:
        if released_record.condition not in _FAMILIARITY_CONDITION_SEEDS:
            continue
        if released_record.task_type != "rule_identification":
            continue
        record = cast(RuleIdentificationRecord, released_record)
        if record.seed not in _FAMILIARITY_CONDITION_SEEDS[record.condition]:
            raise ValueError(f"familiarity rule grid has an unexpected seed for {record.condition}")
        key = (record.condition, record.seed, record.base_snippet_id)
        if key in grid:
            raise ValueError(f"duplicate familiarity rule grid cell: {key!r}")
        grid[key] = record

    snippet_ids = tuple(
        sorted(
            record.base_snippet_id
            for (condition, seed, _snippet_id), record in grid.items()
            if condition == "C0" and seed == 42
        )
    )
    if len(snippet_ids) != 410 or len(set(snippet_ids)) != 410:
        raise ValueError("familiarity rule grid requires exactly 410 C0 reference IDs")
    expected_keys = {
        (condition, seed, snippet_id)
        for condition, seeds in _FAMILIARITY_CONDITION_SEEDS.items()
        for seed in seeds
        for snippet_id in snippet_ids
    }
    if set(grid) != expected_keys:
        raise ValueError("familiarity rule grid requires exact raw, syntax, and three-seed C0 ID cells")

    reference_gold = {snippet_id: grid[("C0", 42, snippet_id)].gold for snippet_id in snippet_ids}
    for (_condition, _seed, snippet_id), record in grid.items():
        if record.gold != reference_gold[snippet_id]:
            raise ValueError(f"familiarity rule gold disagrees across grids: {snippet_id}")
        if record.target_checks != reference_gold[snippet_id]:
            raise ValueError(f"familiarity target checks disagree with gold: {snippet_id}")
    return snippet_ids, grid


def _familiarity_cell_f1(
    snippet_ids: Sequence[str],
    grid: Mapping[_FamiliarityKey, RuleIdentificationRecord],
    condition: Condition,
    seed: int,
) -> dict[str, float]:
    records = [grid[(condition, seed, snippet_id)] for snippet_id in snippet_ids]
    metrics = per_check_prf(
        [record.pred for record in records],
        [record.gold for record in records],
        CHECKS,
    )
    return {check: metrics[check]["f1"] * 100.0 for check in CHECKS}


def _familiarity_values(
    scored_records: Sequence[ReleasedRecord],
) -> tuple[
    tuple[str, ...],
    dict[_FamiliarityKey, RuleIdentificationRecord],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    snippet_ids, grid = _familiarity_rule_grid(scored_records)
    raw = _familiarity_cell_f1(snippet_ids, grid, "zero-shot-raw", 42)
    syntax = _familiarity_cell_f1(snippet_ids, grid, "zero-shot-syntax", 42)
    c0_per_seed = [_familiarity_cell_f1(snippet_ids, grid, "C0", seed) for seed in _SECONDARY_SEEDS]
    c0 = {check: statistics.mean(seed_values[check] for seed_values in c0_per_seed) for check in CHECKS}
    return snippet_ids, grid, raw, syntax, c0


def _build_table_8_9(scored_records: Sequence[ReleasedRecord]) -> TableData:
    snippet_ids, grid, raw, syntax, c0 = _familiarity_values(scored_records)
    reference_gold = {snippet_id: grid[("C0", 42, snippet_id)].gold for snippet_id in snippet_ids}
    table_rows: list[TableRow] = []
    for check in CHECKS:
        row_key = _CHECK_RESULT_NAMES[check]
        values: dict[str, int | float] = {
            "n": sum(check in gold for gold in reference_gold.values()),
            "zero_shot_raw": raw[check],
            "zero_shot_syntax": syntax[check],
            "fine_tuned_c0": c0[check],
        }
        cells: dict[str, OutputCell | str] = {"check": check}
        for column, value in values.items():
            cells[column] = OutputCell(
                f"table_8_9.{row_key}.{column}",
                value,
                0 if column == "n" else 1,
            )
        table_rows.append(TableRow(key=row_key, cells=cells))
    return TableData(filename="table-8-9.csv", columns=_TABLE_8_9_COLUMNS, rows=tuple(table_rows))


def _build_zero_shot_transitions(scored_records: Sequence[ReleasedRecord]) -> dict[str, Scalar]:
    condition_maps: dict[Condition, dict[str, bool]] = {
        "zero-shot-raw": {},
        "zero-shot-syntax": {},
    }
    for released_record in scored_records:
        if released_record.condition not in condition_maps or released_record.task_type != "joint":
            continue
        record = cast(JointRecord, released_record)
        if record.seed != 42:
            raise ValueError(f"zero-shot transition grid has an unexpected seed for {record.condition}")
        condition_map = condition_maps[record.condition]
        if record.base_snippet_id in condition_map:
            raise ValueError(f"duplicate zero-shot transition ID: {record.base_snippet_id}")
        condition_map[record.base_snippet_id] = record.outcome.overall_fixed

    raw = condition_maps["zero-shot-raw"]
    syntax = condition_maps["zero-shot-syntax"]
    if set(raw) != set(syntax):
        raise ValueError("zero-shot transition ID sets must match")
    if len(raw) != 410:
        raise ValueError(f"zero-shot transition grid requires exactly 410 matched IDs, got {len(raw)}")
    favor_syntax = sum(not raw[snippet_id] and syntax[snippet_id] for snippet_id in raw)
    favor_raw = sum(raw[snippet_id] and not syntax[snippet_id] for snippet_id in raw)
    net_files = favor_syntax - favor_raw
    return {
        "section_8_10.zero_shot_transition.changed": favor_syntax + favor_raw,
        "section_8_10.zero_shot_transition.favor_syntax": favor_syntax,
        "section_8_10.zero_shot_transition.favor_raw": favor_raw,
        "section_8_10.zero_shot_transition.net_files": net_files,
        "section_8_10.zero_shot_transition.net_points": net_files / len(raw) * 100.0,
    }


def _build_familiarity_results(scored_records: Sequence[ReleasedRecord]) -> dict[str, Scalar]:
    snippet_ids, grid, raw, syntax, c0 = _familiarity_values(scored_records)
    raw_records = [grid[("zero-shot-raw", 42, snippet_id)] for snippet_id in snippet_ids]
    syntax_records = [grid[("zero-shot-syntax", 42, snippet_id)] for snippet_id in snippet_ids]
    raw_macro = macro_f1(
        [record.pred for record in raw_records],
        [record.gold for record in raw_records],
        CHECKS,
    )
    syntax_macro = macro_f1(
        [record.pred for record in syntax_records],
        [record.gold for record in syntax_records],
        CHECKS,
    )
    zero_shot_values = tuple(raw.values()) + tuple(syntax.values())
    return {
        "sensitivity.familiarity.zero_shot.minimum_percent": round(min(zero_shot_values)),
        "sensitivity.familiarity.zero_shot.maximum_percent": round(max(zero_shot_values)),
        "sensitivity.familiarity.c0.minimum_percent": round(min(c0.values())),
        "sensitivity.familiarity.c0.maximum_percent": round(max(c0.values())),
        "sensitivity.familiarity.zero_shot.overall_approx_percent": round(
            statistics.mean((raw_macro, syntax_macro)) * 100.0
        ),
    }


def _build_rq3_results(scored_records: Sequence[ReleasedRecord]) -> dict[str, Scalar]:
    fit = fit_rq3_interaction(build_rq3_frame(scored_records))
    contrasts = rq3_task_contrasts(scored_records, n_boot=10_000, seed=42)
    by_reference = {contrast.reference_task: contrast.interval for contrast in contrasts}
    if set(by_reference) != {"correction", "rule_identification"} or len(contrasts) != 2:
        raise ValueError("RQ3 requires exact joint-minus-correction and joint-minus-rule contrasts")
    correction = by_reference["correction"]
    rule = by_reference["rule_identification"]
    return {
        "rq3.vb.c1_joint_interaction.coefficient": fit.coefficient,
        "rq3.vb.c1_joint_interaction.low": fit.ci_low,
        "rq3.vb.c1_joint_interaction.high": fit.ci_high,
        "rq3.vb.c1_joint_interaction.sd": fit.sd,
        "rq3.risk.joint_minus_correction.point": correction.point * 100.0,
        "rq3.risk.joint_minus_correction.low": correction.ci_low * 100.0,
        "rq3.risk.joint_minus_correction.high": correction.ci_high * 100.0,
        "rq3.risk.joint_minus_rule.point": rule.point * 100.0,
        "rq3.risk.joint_minus_rule.low": rule.ci_low * 100.0,
        "rq3.risk.joint_minus_rule.high": rule.ci_high * 100.0,
        "rq3.h3_supported": correction.ci_low > 0.0 and rule.ci_low > 0.0,
    }


def _section_rule_grid(
    scored_records: Sequence[ReleasedRecord],
) -> tuple[tuple[str, ...], dict[_FamiliarityKey, RuleIdentificationRecord]]:
    grid: dict[_FamiliarityKey, RuleIdentificationRecord] = {}
    for released_record in scored_records:
        if released_record.condition not in _FINE_TUNED_CONDITIONS:
            continue
        if released_record.task_type != "rule_identification":
            continue
        record = cast(RuleIdentificationRecord, released_record)
        key = (record.condition, record.seed, record.base_snippet_id)
        if key in grid:
            raise ValueError(f"duplicate Section 8.10 rule grid cell: {key!r}")
        grid[key] = record

    snippet_ids = tuple(
        sorted(
            record.base_snippet_id
            for (condition, seed, _snippet_id), record in grid.items()
            if condition == "C0" and seed == 42
        )
    )
    if len(snippet_ids) != 410 or len(set(snippet_ids)) != 410:
        raise ValueError("Section 8.10 rule grid requires exactly 410 reference IDs")
    expected_keys = {
        (condition, seed, snippet_id)
        for condition in _FINE_TUNED_CONDITIONS
        for seed in _SECONDARY_SEEDS
        for snippet_id in snippet_ids
    }
    if set(grid) != expected_keys:
        raise ValueError("Section 8.10 rule grid requires the exact condition-by-seed-by-ID cells")
    reference_gold = {snippet_id: grid[("C0", 42, snippet_id)].gold for snippet_id in snippet_ids}
    for (_condition, _seed, snippet_id), record in grid.items():
        if record.gold != reference_gold[snippet_id]:
            raise ValueError(f"Section 8.10 rule gold disagrees across grids: {snippet_id}")
        if record.target_checks != reference_gold[snippet_id]:
            raise ValueError(f"Section 8.10 target checks disagree with gold: {snippet_id}")
    return snippet_ids, grid


def _section_c2_joint_grid(
    scored_records: Sequence[ReleasedRecord],
    snippet_ids: Sequence[str],
    reference_gold: Mapping[str, tuple[str, ...]],
) -> dict[tuple[int, str], JointRecord]:
    grid: dict[tuple[int, str], JointRecord] = {}
    for released_record in scored_records:
        if released_record.condition != "C2" or released_record.task_type != "joint":
            continue
        record = cast(JointRecord, released_record)
        key = (record.seed, record.base_snippet_id)
        if key in grid:
            raise ValueError(f"duplicate Section 8.10 C2 joint grid cell: {key!r}")
        if record.base_snippet_id not in reference_gold:
            raise ValueError(f"Section 8.10 C2 joint record has an unmatched ID: {record.base_snippet_id}")
        if record.target_checks != reference_gold[record.base_snippet_id]:
            raise ValueError(f"Section 8.10 C2 joint targets disagree with rule gold: {record.base_snippet_id}")
        grid[key] = record
    expected_keys = {(seed, snippet_id) for seed in _SECONDARY_SEEDS for snippet_id in snippet_ids}
    if set(grid) != expected_keys:
        raise ValueError("Section 8.10 C2 joint grid requires the exact three-seed by 410-ID cells")
    return grid


def _build_section_8_10_record_results(scored_records: Sequence[ReleasedRecord]) -> dict[str, Scalar]:
    snippet_ids, rule_grid = _section_rule_grid(scored_records)
    reference_gold = {snippet_id: rule_grid[("C0", 42, snippet_id)].gold for snippet_id in snippet_ids}
    joint_grid = _section_c2_joint_grid(scored_records, snippet_ids, reference_gold)
    c0_all_seed = {
        snippet_id
        for snippet_id in snippet_ids
        if all(rule_grid[("C0", seed, snippet_id)].exact_match for seed in _SECONDARY_SEEDS)
    }
    c2_only_seeds = {
        snippet_id: {
            seed
            for seed in _SECONDARY_SEEDS
            if rule_grid[("C2", seed, snippet_id)].exact_match
            and not rule_grid[("C0", seed, snippet_id)].exact_match
            and not rule_grid[("C1", seed, snippet_id)].exact_match
        }
        for snippet_id in snippet_ids
    }
    category_b = [record for record in joint_grid.values() if record.outcome.category == "B"]
    return {
        "section_8_10.c0_all_seed_correct.count": len(c0_all_seed),
        "section_8_10.c0_all_seed_correct.percent": len(c0_all_seed) / len(snippet_ids) * 100.0,
        "section_8_10.c2_only.any_seed.files": sum(bool(seeds) for seeds in c2_only_seeds.values()),
        "section_8_10.c2_only.at_least_two_seeds.files": sum(len(seeds) >= 2 for seeds in c2_only_seeds.values()),
        "section_8_10.category_b.files": len({record.base_snippet_id for record in category_b}),
        "section_8_10.category_b.runs": len(category_b),
    }


def _merge_results(*groups: Mapping[str, Scalar]) -> dict[str, Scalar]:
    merged: dict[str, Scalar] = {}
    for group in groups:
        duplicate_ids = set(merged).intersection(group)
        if duplicate_ids:
            raise ValueError(f"duplicate core result ID: {min(duplicate_ids)}")
        merged.update(group)
    return merged


def _uniform(values: Iterable[int], *, name: str) -> int:
    distinct = set(values)
    if len(distinct) != 1:
        raise ValueError(f"{name} must have one common value")
    return next(iter(distinct))


def _build_dataset_repository_results(study_rows: Sequence[StudyRow]) -> dict[str, Scalar]:
    split_names = ("train", "validation", "test")
    by_split = {split: [row for row in study_rows if row.split == split] for split in split_names}
    results: dict[str, Scalar] = {
        "dataset.base_snippets.total": len(study_rows),
        "dataset.studied_checks.count": len(CHECKS),
        "dataset.task_variants_per_snippet": len(TASK_TYPES),
        "dataset.clean_snippets.total": sum(row.violation_count == 0 for row in study_rows),
        "dataset.near_duplicate_quarantine.total": sum(row.quarantined for row in study_rows),
        "dataset.cross_split_exact_collisions.total": sum(row.dataset_qc.exact_collision for row in study_rows),
        "dataset.canonical_normalization.changed": sum(
            row.dataset_qc.canonical_normalization_changed for row in study_rows
        ),
    }
    for split, rows in by_split.items():
        check_counts = Counter(check for row in rows for check in row.target_checks)
        results[f"dataset.base_snippets.{split}"] = len(rows)
        results[f"dataset.per_check_positives.{split}"] = _uniform(
            (check_counts[check] for check in CHECKS),
            name=f"{split} per-check positives",
        )
        results[f"dataset.label_instances.{split}"] = sum(check_counts.values())
        results[f"dataset.pre_qc_task_examples.{split}"] = len(rows) * len(TASK_TYPES)
        task_counts = Counter(task for row in rows for task in row.task_types)
        for task_type in TASK_TYPES:
            results[f"dataset.post_reference_qc.{split}.{task_type}"] = task_counts[task_type]
        results[f"dataset.post_reference_qc.{split}.total"] = sum(task_counts.values())

    results["dataset.pre_qc_task_examples.total"] = sum(
        cast(int, results[f"dataset.pre_qc_task_examples.{split}"]) for split in split_names
    )
    for task_type in TASK_TYPES:
        results[f"dataset.post_reference_qc.total.{task_type}"] = sum(
            cast(int, results[f"dataset.post_reference_qc.{split}.{task_type}"]) for split in split_names
        )
    results["dataset.post_reference_qc.total.total"] = sum(
        cast(int, results[f"dataset.post_reference_qc.{split}.total"]) for split in split_names
    )

    train_rows = by_split["train"]
    excluded = [row for row in train_rows if row.length_excluded]
    retained = [row for row in train_rows if not row.length_excluded]
    results.update(
        {
            "dataset.length_exclusion.base_snippets": len(excluded),
            "dataset.post_length.base_snippets.total": len(retained),
            "dataset.length_exclusion.label_instances.total": sum(len(row.target_checks) for row in excluded),
            "dataset.length_exclusion.multilabel_snippets": sum(len(row.target_checks) > 1 for row in excluded),
            "dataset.length_exclusion.reference_qc_overlap": sum(
                row.reference_qc.correction_status != "accepted" for row in excluded
            ),
        }
    )
    retained_task_counts = Counter(task for row in retained for task in row.task_types)
    for task_type in TASK_TYPES:
        results[f"dataset.post_length.task_examples.{task_type}"] = retained_task_counts[task_type]
    results["dataset.post_length.task_examples.total"] = sum(retained_task_counts.values())
    excluded_check_counts = Counter(check for row in excluded for check in row.target_checks)
    for check, result_name in _CHECK_RESULT_NAMES.items():
        results[f"dataset.length_exclusion.check.{result_name}"] = excluded_check_counts[check]

    groups_by_split = {split: {row.repository_group_id for row in rows} for split, rows in by_split.items()}
    scored_test = [row for row in by_split["test"] if row.oracle.status != "excluded" and not row.quarantined]
    scored_group_counts = Counter(row.repository_group_id for row in scored_test)
    train_groups = groups_by_split["train"]
    overlapping_scored = [row for row in scored_test if row.repository_group_id in train_groups]
    duplicate_group_sizes = [count for count in scored_group_counts.values() if count > 1]
    results.update(
        {
            "repository.total": len({row.repository_group_id for row in study_rows}),
            "repository.train_test_overlap.full_split": len(train_groups & groups_by_split["test"]),
            "repository.scored_files_overlapping_train": len(overlapping_scored),
            "repository.scored_repositories_overlapping_train": len(
                {row.repository_group_id for row in overlapping_scored}
            ),
            "repository.scored_test.total": len(scored_group_counts),
            "repository.scored_groups_with_two_files": sum(count == 2 for count in duplicate_group_sizes),
            "repository.scored_files_per_duplicate_group": _uniform(
                duplicate_group_sizes,
                name="scored duplicate-group size",
            ),
        }
    )
    return results


def _build_reference_results(study_rows: Sequence[StudyRow]) -> dict[str, Scalar]:
    train_rows = [row for row in study_rows if row.split == "train"]
    accepted = [row for row in train_rows if row.reference_qc.correction_status == "accepted"]
    excluded = [row for row in train_rows if row.reference_qc.correction_status != "accepted"]
    both_category_a = [
        row
        for row in train_rows
        if row.reference_qc.primary_category_a is True and row.reference_qc.secondary_category_a is True
    ]
    normalized_equal = [row for row in both_category_a if row.reference_qc.normalized_fixes_equal is True]
    build_counts = Counter(row.reference_qc.build_status for row in accepted)
    deviations = [row for row in train_rows if row.reference_qc.skip_mechanism == "configured_but_not_applied"]
    marker = [row for row in deviations if row.reference_qc.generated_marker_retained]
    multiline = [row for row in deviations if not row.reference_qc.generated_marker_retained]
    multiline_current_checks = {
        row.base_snippet_id: (set(row.target_checks) - set(row.oracle.missing_checks)) | set(row.oracle.extra_checks)
        for row in multiline
    }
    deviation_check_counts = Counter(check for row in deviations for check in row.target_checks)
    results: dict[str, Scalar] = {
        "reference_qc.correction.accepted": len(accepted),
        "reference_qc.explanation.accepted": sum(row.reference_qc.explanation_present for row in train_rows),
        "reference_qc.correction.excluded": len(excluded),
        "reference_qc.correction.exclusion.parse_fail": sum(
            row.reference_qc.correction_status == "parse_fail" for row in excluded
        ),
        "reference_qc.correction.exclusion.target_not_fixed": sum(
            row.reference_qc.correction_status == "target_not_fixed" for row in excluded
        ),
        "reference_qc.correction.exclusion.generation_attempts_each": _uniform(
            (row.reference_qc.generation_attempts for row in excluded),
            name="excluded reference generation attempts",
        ),
        "reference_qc.cross_generator.both_category_a.count": len(both_category_a),
        "reference_qc.cross_generator.both_category_a.rate": len(both_category_a) / len(train_rows),
        "reference_qc.cross_generator.normalized_fix_equal.count": len(normalized_equal),
        "reference_qc.cross_generator.normalized_fix_equal.rate": len(normalized_equal) / len(both_category_a),
        "reference_qc.substitution.count": sum(
            row.reference_qc.selected_generator_role == "secondary" for row in train_rows
        ),
        "reference_qc.substitution.rate": sum(
            row.reference_qc.selected_generator_role == "secondary" for row in train_rows
        )
        / len(accepted),
        "reference_qc.training_targets.primary_generator": sum(
            row.reference_qc.selected_generator_role == "primary" for row in train_rows
        ),
        "reference_qc.training_targets.secondary_generator": sum(
            row.reference_qc.selected_generator_role == "secondary" for row in train_rows
        ),
        "reference_qc.deviation.skip_listed": len(deviations),
        "reference_qc.deviation.generated_marker": len(marker),
        "reference_qc.deviation.multiline_param": len(multiline),
        "reference_qc.deviation.multiline.no_studied_detection": sum(
            not multiline_current_checks[row.base_snippet_id] for row in multiline
        ),
        "reference_qc.deviation.multiline.with_studied_detection": sum(
            bool(multiline_current_checks[row.base_snippet_id]) for row in multiline
        ),
        "reference_qc.deviation.multiline.dual_target_verified": sum(
            len(row.target_checks) > 1
            and any(
                check != "paramTypeCombine" and check in multiline_current_checks[row.base_snippet_id]
                for check in row.target_checks
            )
            for row in multiline
        ),
        "reference_qc.deviation.marker.normalized_fix_equal": sum(
            row.reference_qc.normalized_fixes_equal is True for row in marker
        ),
        "reference_qc.deviation.overall.normalized_fix_equal": sum(
            row.reference_qc.normalized_fixes_equal is True for row in deviations
        ),
        "reference_qc.deviation.overall.normalized_fix_equal_rate": sum(
            row.reference_qc.normalized_fixes_equal is True for row in deviations
        )
        / len(deviations),
        "reference_qc.deviation.byte_identical_noop": sum(
            row.reference_qc.accepted_correction_byte_identical is True for row in deviations
        ),
        "reference_qc.deviation.both_generators_category_a": sum(
            row.reference_qc.primary_category_a is True and row.reference_qc.secondary_category_a is True
            for row in deviations
        ),
        "reference_qc.deviation.length_excluded": sum(row.length_excluded for row in deviations),
        "reference_qc.deviation.post_length_affected": sum(not row.length_excluded for row in deviations),
        "reference_qc.deviation.check.elseif_percent_of_training_positives": deviation_check_counts["elseif"]
        / sum("elseif" in row.target_checks for row in train_rows)
        * 100.0,
        "reference_qc.deviation.label_instances": sum(deviation_check_counts.values()),
    }
    for status, result_name in (("OK", "ok"), ("NA", "na"), ("FAIL", "fail")):
        count = build_counts[status]
        results[f"reference_qc.build.{result_name}.count"] = count
        results[f"reference_qc.build.{result_name}.percent"] = count / len(accepted) * 100.0
    for check, result_name in _CHECK_RESULT_NAMES.items():
        if deviation_check_counts[check]:
            results[f"reference_qc.deviation.check.{result_name}"] = deviation_check_counts[check]
    return results


def _build_oracle_serializer_results(
    study_rows: Sequence[StudyRow],
    scored_records: Sequence[ReleasedRecord],
) -> dict[str, Scalar]:
    evaluation_rows = [row for row in study_rows if row.split in {"validation", "test"}]
    excluded_evaluation = [row for row in evaluation_rows if row.oracle.status == "excluded"]
    reproduced_evaluation = [row for row in evaluation_rows if row.oracle.status == "reproduced"]
    mechanism_counts: Counter[str] = Counter()
    for row in excluded_evaluation:
        if not row.oracle.missing_checks and row.oracle.extra_checks:
            mechanism_counts["extra_detection"] += 1
        elif row.oracle.missing_checks == ("paramTypeCombine",) and not row.oracle.extra_checks:
            mechanism_counts["multiline_param"] += 1
        elif (
            row.oracle.missing_checks
            and not row.oracle.extra_checks
            and set(row.oracle.missing_checks) == set(row.target_checks)
        ):
            mechanism_counts["generated_suppression"] += 1
        else:
            raise ValueError(f"excluded row {row.base_snippet_id} has no valid exclusion mechanism")
    if sum(mechanism_counts.values()) != len(excluded_evaluation):
        raise ValueError("oracle exclusion mechanism categories must partition excluded evaluation rows")

    scored_evaluation = [row for row in evaluation_rows if row.oracle.status != "excluded" and not row.quarantined]
    scored_test_ids = {row.base_snippet_id for row in scored_evaluation if row.split == "test"}
    record_ids_by_cell: dict[tuple[Condition, int, str], list[str]] = {}
    for record in scored_records:
        key = (record.condition, record.seed, record.task_type)
        record_ids_by_cell.setdefault(key, []).append(record.base_snippet_id)
    expected_record_cells = {
        (condition, seed, task_type) for condition, seed in expected_run_keys() for task_type in TASK_TYPES
    }
    if set(record_ids_by_cell) != expected_record_cells:
        raise ValueError("scored record cell matrix must contain exactly the expected 56 cells")
    for cell, record_ids in record_ids_by_cell.items():
        if len(record_ids) != len(set(record_ids)) or set(record_ids) != scored_test_ids:
            raise ValueError(f"scored record cell {cell!r} differs from the study-derived scored test IDs")

    results: dict[str, Scalar] = {
        "oracle.scoring_gate.total": len(evaluation_rows),
        "oracle.scoring_gate.exact": len(reproduced_evaluation),
        "oracle.scoring_gate.exact.percent": len(reproduced_evaluation) / len(evaluation_rows) * 100.0,
        "oracle.scoring_gate.excluded.total": len(excluded_evaluation),
        "oracle.scoring_gate.excluded.validation": sum(row.split == "validation" for row in excluded_evaluation),
        "oracle.scoring_gate.excluded.test": sum(row.split == "test" for row in excluded_evaluation),
        "oracle.scoring_gate.fixture_fixes": sum(row.oracle.status == "fixture_fix" for row in evaluation_rows),
        "oracle.scored.validation": sum(row.split == "validation" for row in scored_evaluation),
        "oracle.scored.test": sum(row.split == "test" for row in scored_evaluation),
        "oracle.scored_set.shared_across_cells": True,
    }
    for mechanism in ("generated_suppression", "multiline_param", "extra_detection"):
        results[f"oracle.scoring_gate.excluded.mechanism.{mechanism}"] = mechanism_counts[mechanism]

    test_rows = [row for row in study_rows if row.split == "test"]
    excluded_test = [row for row in test_rows if row.oracle.status == "excluded"]
    results["oracle.test.excluded_label_incidences"] = sum(len(row.target_checks) for row in excluded_test)
    for check, result_name in _CHECK_RESULT_NAMES.items():
        original = sum(check in row.target_checks for row in test_rows)
        retained = sum(
            check in row.target_checks and row.oracle.status != "excluded" and not row.quarantined for row in test_rows
        )
        prefix = f"oracle.test.check.{result_name}"
        results[f"{prefix}.original_support"] = original
        results[f"{prefix}.retained_support"] = retained
        results[f"{prefix}.excluded_incidence"] = original - retained
        results[f"{prefix}.retained_percent"] = retained / original * 100.0

    train_rows = [row for row in study_rows if row.split == "train"]
    mismatches = [row for row in train_rows if row.oracle.status != "reproduced"]
    missing_only = [row for row in mismatches if row.oracle.missing_checks and not row.oracle.extra_checks]
    missing_and_extra = [row for row in mismatches if row.oracle.missing_checks and row.oracle.extra_checks]
    extra_only = [row for row in mismatches if not row.oracle.missing_checks and row.oracle.extra_checks]
    extra_bearing = [row for row in mismatches if row.oracle.extra_checks]
    length_extra_only = [row for row in extra_only if row.length_excluded]
    post_length_extra_only = [row for row in extra_only if not row.length_excluded]
    post_length_extra_bearing = [row for row in extra_bearing if not row.length_excluded]
    extra_check_counts = Counter(check for row in extra_bearing for check in row.oracle.extra_checks)
    results.update(
        {
            "oracle.training.exact": sum(row.oracle.status == "reproduced" for row in train_rows),
            "oracle.training.mismatch.total": len(mismatches),
            "oracle.training.missing_target.total": sum(bool(row.oracle.missing_checks) for row in mismatches),
            "oracle.training.missing_only": len(missing_only),
            "oracle.training.missing_and_extra": len(missing_and_extra),
            "oracle.training.extra_only": len(extra_only),
            "oracle.training.extra_bearing_snippets": len(extra_bearing),
            "oracle.training.extra_findings": sum(len(row.oracle.extra_checks) for row in extra_bearing),
            "oracle.training.length_excluded_extra_only": len(length_extra_only),
            "oracle.training.length_excluded_extra_findings": sum(
                len(row.oracle.extra_checks) for row in length_extra_only
            ),
            "oracle.training.post_length.extra_only": len(post_length_extra_only),
            "oracle.training.post_length.extra_bearing_snippets": len(post_length_extra_bearing),
            "oracle.training.post_length.affected_rows": sum(len(row.task_types) for row in post_length_extra_bearing),
            "oracle.training.post_length.omitted_findings": sum(
                len(row.oracle.extra_checks) for row in post_length_extra_bearing
            ),
            "oracle.training.extra_bearing.reference_qc_overlap": sum(
                row.reference_qc.correction_status != "accepted" for row in extra_bearing
            ),
        }
    )
    for check, result_name in _CHECK_RESULT_NAMES.items():
        if extra_check_counts[check]:
            results[f"oracle.training.extra_findings.check.{result_name}"] = extra_check_counts[check]

    parse_failures = sum(not row.serializer.parse_ok for row in study_rows)
    excluded_constructs = sum(row.serializer.excluded_construct_count > 0 for row in study_rows)
    results.update(
        {
            "serializer.corpus.parse_failures": parse_failures,
            "serializer.excluded_constructs.count": excluded_constructs,
            "serializer.excluded_constructs.percent": excluded_constructs / len(study_rows) * 100.0,
        }
    )
    skipped_total = 0
    for split in ("train", "validation", "test"):
        split_rows = [row for row in study_rows if row.split == split]
        skipped = sum(row.serializer.summary_status == "skipped" for row in split_rows)
        truncated = sum(row.serializer.summary_status == "present_truncated" for row in split_rows)
        skipped_total += skipped
        results[f"serializer.summary_skip.{split}.count"] = skipped
        results[f"serializer.summary_truncation.{split}.count"] = truncated
        results[f"serializer.summary_truncation.{split}.percent"] = truncated / len(split_rows) * 100.0
    results["serializer.summary_skip.total"] = skipped_total
    results["serializer.summary_skip.overall_percent"] = skipped_total / len(study_rows) * 100.0

    audited = [row for row in study_rows if row.serializer_audit is not None]
    audits = [cast(SerializerAudit, row.serializer_audit) for row in audited]
    closure_scored = [row for row in audited if cast(SerializerAudit, row.serializer_audit).closure_scored]
    results.update(
        {
            "serializer.audit.coverage_pool": sum(row.serializer_audit_stage1_pool_member for row in study_rows),
            "serializer.audit.sample": len(audited),
            "serializer.audit.relevant_omissions": sum(audit.relevant_omission is True for audit in audits),
            "serializer.audit.rows_with_notes": sum(audit.has_note for audit in audits),
            "serializer.audit.rows_with_known_lossiness": sum(bool(audit.known_loss_categories) for audit in audits),
            "serializer.audit.closure_violation.count": sum(row.violation_in_closure is True for row in closure_scored),
            "serializer.audit.closure_violation.denominator": len(closure_scored),
            "serializer.audit.closure_violation.percent": sum(
                row.violation_in_closure is True for row in closure_scored
            )
            / len(closure_scored)
            * 100.0,
            "serializer.audit.closure_violation.oracle_excluded": len(audited) - len(closure_scored),
        }
    )
    return results


def _build_metadata_results(metadata: AnalysisMetadata) -> dict[str, Scalar]:
    architecture = metadata.architecture
    reference = metadata.reference_comparison
    training = metadata.training_path
    return {
        "architecture.parameter_count": architecture.parameter_count,
        "architecture.hidden_size": architecture.embedding_dimension,
        "architecture.attention_heads": architecture.query_heads,
        "architecture.key_value_heads": architecture.key_value_heads,
        "architecture.query_heads_per_key_value_head": architecture.query_heads_per_key_value_head,
        "architecture.layers": architecture.layers,
        "architecture.feed_forward_size": architecture.feed_forward_dimension,
        "architecture.vocabulary_size": architecture.vocabulary_size,
        "architecture.context_length": architecture.context_length,
        "architecture.compute_dtype": (
            "BF16" if architecture.compute_dtype == "bfloat16" else architecture.compute_dtype
        ),
        "architecture.rmsnorm_dtype": (
            "FP32" if architecture.rmsnorm_dtype == "float32" else architecture.rmsnorm_dtype
        ),
        "architecture.weight_tied": architecture.weight_tied,
        "verification.prompt_count": reference.prompt_count,
        "verification.scored_positions": reference.scored_position_count,
        "verification.tokenizer_exact": reference.tokenizer_exact,
        "verification.chat_template_match": reference.chat_template_match,
        "verification.chat_template_first_divergence": reference.chat_template_first_divergence,
        "verification.generation_exact": reference.generation_exact,
        "verification.fp32_reference.next_token_agreement": reference.next_token_agreement_fp32,
        "verification.fp32_reference.raw_disagreements": reference.disagreements_fp32,
        "verification.fp32_reference.systematic_disagreements": reference.systematic_disagreements_fp32,
        "verification.maximum_opposing_margin_threshold": reference.margin_threshold,
        "verification.fp32_reference.maximum_absolute_logit_difference": (reference.maximum_absolute_logit_difference),
        "verification.fp32_reference.mean_absolute_logit_difference": reference.mean_absolute_logit_difference,
        "verification.forward_tolerance": reference.null_forward_tolerance,
        "verification.bf16_reference.next_token_agreement": reference.next_token_agreement_bf16,
        "verification.bf16_reference.raw_disagreements": reference.disagreements_bf16,
        "verification.bf16_reference.systematic_disagreements": reference.systematic_disagreements_bf16,
        "verification.cached_generation.status": reference.cached_generation_test.status,
        "verification.sdpa_manual_attention.status": reference.sdpa_manual_test.status,
        "verification.loss_masking.status": reference.loss_masking_test.status,
        "verification.hard_gates.passed": sum(
            (reference.tokenizer_exact, reference.systematic_disagreements_fp32 == 0)
        ),
        "verification.training_path.steps": training.steps,
        "verification.training_path.examples": training.examples,
        "verification.training_path.mean_relative_loss_divergence": training.mean_relative_loss_divergence,
        "verification.training_path.maximum_relative_loss_divergence": training.maximum_relative_loss_divergence,
        "verification.training_path.final_loss_scratch": training.final_loss_scratch,
        "verification.training_path.final_loss_reference": training.final_loss_reference,
        "verification.training_path.validation_macro_f1_scratch": training.validation_macro_f1_scratch,
        "verification.training_path.validation_macro_f1_reference": training.validation_macro_f1_reference,
    }


def _round_two_significant_digits(value: int) -> int:
    if value <= 0:
        raise ValueError("two-significant-digit rounding requires a positive integer")
    digits = 1 - floor(log10(value))
    return int(round(value, digits))


def _expanded_contribution_values(
    study_rows: Sequence[StudyRow],
    *,
    condition: FineTunedCondition,
    pool: str,
    field: Literal["prompt_tokens", "response_tokens", "total_tokens"],
    post_length_only: bool = False,
) -> list[int]:
    values: list[int] = []
    for row in study_rows:
        if post_length_only and row.length_excluded:
            continue
        for contribution in row.training_contributions:
            if contribution.condition == condition and contribution.pool == pool:
                values.extend([getattr(contribution, field)] * contribution.multiplicity)
    if not values:
        raise ValueError(f"{condition} {pool} {field} requires contributions")
    return values


def _build_configuration_training_contribution_results(
    experiment_config: ExperimentConfig,
    run_results: Sequence[RunResults],
    study_rows: Sequence[StudyRow],
) -> dict[str, Scalar]:
    profile = experiment_config.profiles["paper"]
    fine_configs = {
        condition: config for condition, config in experiment_config.conditions.items() if config.kind == "fine_tuned"
    }
    zero_shot_configs = {
        condition: config for condition, config in experiment_config.conditions.items() if config.kind == "zero_shot"
    }
    fine_seeds = sorted({seed for config in fine_configs.values() for seed in config.seeds})
    generative_caps = {
        profile.generation_max_new_tokens.explanation,
        profile.generation_max_new_tokens.correction,
        profile.generation_max_new_tokens.joint,
    }
    results: dict[str, Scalar] = {
        "training.profile.max_steps": profile.max_steps,
        "training.length.allowed_max_length": profile.allowed_max_length,
        "training.profile.micro_batch_size": profile.micro_batch_size,
        "training.profile.gradient_accumulation_steps": profile.grad_accum_steps,
        "training.profile.effective_batch_size": profile.effective_batch_size,
        "training.profile.learning_rate": profile.learning_rate,
        "training.profile.beta_1": profile.betas[0],
        "training.profile.beta_2": profile.betas[1],
        "training.profile.epsilon": profile.epsilon,
        "training.profile.weight_decay": profile.weight_decay,
        "training.profile.warmup_ratio": profile.warmup_ratio,
        "training.profile.minimum_learning_rate_ratio": profile.minimum_learning_rate_ratio,
        "training.profile.maximum_gradient_norm": profile.maximum_gradient_norm,
        "training.checkpoint_selection.cadence_steps": profile.checkpoint_every_steps,
        "training.profile.activation_checkpointing": profile.activation_checkpointing,
        "training.profile.generation_max_new_tokens.rule_identification": (
            profile.generation_max_new_tokens.rule_identification
        ),
        "training.profile.generation_max_new_tokens.generative_tasks": _uniform(
            generative_caps,
            name="generative response caps",
        ),
        "training.composite.metric_count": len(("rule_id_macro_f1", "correction_fix_rate", "joint_fix_rate")),
        "training.conditions.fine_tuned": len(fine_configs),
        "training.seeds.count": len(fine_seeds),
        "training.runs.fine_tuned": sum(result.condition in _FINE_TUNED_CONDITIONS for result in run_results),
        "training.training_set_sizes": len(
            {
                result.metrics.provenance.data_fraction
                for result in run_results
                if result.condition in _FINE_TUNED_CONDITIONS
            }
        ),
        "training.zero_shot.executions.raw": len(zero_shot_configs["zero-shot-raw"].seeds),
        "training.zero_shot.executions.syntax": len(zero_shot_configs["zero-shot-syntax"].seeds),
        "training.zero_shot.provenance_seed": _uniform(
            (seed for config in zero_shot_configs.values() for seed in config.seeds),
            name="zero-shot provenance seed",
        ),
    }
    for seed in fine_seeds:
        results[f"training.seed.{seed}"] = seed

    syntax_contributions = [
        contribution
        for row in study_rows
        for contribution in row.training_contributions
        if contribution.condition == "C2" and contribution.pool == "syntax_auxiliary"
    ]
    control_contributions = [
        contribution
        for row in study_rows
        for contribution in row.training_contributions
        if contribution.condition == "C2-control" and contribution.pool == "duplicated_main_control"
    ]
    control_rows = [
        row
        for row in study_rows
        if any(
            contribution.condition == "C2-control" and contribution.pool == "duplicated_main_control"
            for contribution in row.training_contributions
        )
    ]
    syntax_pre_rows = sum(contribution.multiplicity for contribution in syntax_contributions)
    training_base_rows = sum(row.split == "train" for row in study_rows)
    results.update(
        {
            "training.pool.syntax.pre_exclusion_rows": syntax_pre_rows,
            "training.pool.syntax.unusable_exclusions": training_base_rows - syntax_pre_rows,
            "training.pool.control.pre_exclusion_rows": sum(
                contribution.multiplicity for contribution in control_contributions
            ),
            "training.pool.control.distinct_examples": len(control_contributions),
            "training.pool.control.distinct_base_files": len(control_rows),
            "training.pool.syntax.post_length_rows": sum(
                contribution.multiplicity
                for row in study_rows
                if not row.length_excluded
                for contribution in row.training_contributions
                if contribution.condition == "C2" and contribution.pool == "syntax_auxiliary"
            ),
            "training.pool.control.post_length_rows": sum(
                contribution.multiplicity
                for row in study_rows
                if not row.length_excluded
                for contribution in row.training_contributions
                if contribution.condition == "C2-control" and contribution.pool == "duplicated_main_control"
            ),
            "training.pool.response_tokens_p50.main": _training_length_summary(
                _expanded_contribution_values(
                    study_rows,
                    condition="C1",
                    pool="main",
                    field="response_tokens",
                    post_length_only=True,
                )
            ).p50,
            "training.pool.response_tokens_p50.syntax_auxiliary": _training_length_summary(
                _expanded_contribution_values(
                    study_rows,
                    condition="C2",
                    pool="syntax_auxiliary",
                    field="response_tokens",
                    post_length_only=True,
                )
            ).p50,
            "training.length.retained_truncations": sum(
                cast(int, result.metrics.length.realized_truncation)
                for result in run_results
                if result.condition in _FINE_TUNED_CONDITIONS
            ),
        }
    )

    syntax_main_contributions = [
        contribution
        for row in study_rows
        for contribution in row.training_contributions
        if contribution.condition in {"C1", "C2", "C2-control"} and contribution.pool == "main"
    ]
    over_budget = [
        contribution
        for contribution in syntax_main_contributions
        if contribution.total_tokens > profile.allowed_max_length
    ]
    c1_main = [
        contribution
        for row in study_rows
        for contribution in row.training_contributions
        if contribution.condition == "C1" and contribution.pool == "main"
    ]
    c0_main = [
        contribution
        for row in study_rows
        for contribution in row.training_contributions
        if contribution.condition == "C0" and contribution.pool == "main"
    ]
    results.update(
        {
            "training.length.pre_exclusion.over_budget_rows": sum(
                contribution.multiplicity for contribution in over_budget
            ),
            "training.length.pre_exclusion.over_budget_correction_per_syntax_condition": _uniform(
                (
                    sum(
                        contribution.multiplicity
                        for contribution in syntax_main_contributions
                        if contribution.condition == condition
                        and contribution.task_type == "correction"
                        and contribution.total_tokens > profile.allowed_max_length
                    )
                    for condition in ("C1", "C2", "C2-control")
                ),
                name="over-budget correction rows per syntax condition",
            ),
            "training.length.pre_exclusion.over_budget_joint_per_syntax_condition": _uniform(
                (
                    sum(
                        contribution.multiplicity
                        for contribution in syntax_main_contributions
                        if contribution.condition == condition
                        and contribution.task_type == "joint"
                        and contribution.total_tokens > profile.allowed_max_length
                    )
                    for condition in ("C1", "C2", "C2-control")
                ),
                name="over-budget joint rows per syntax condition",
            ),
            "training.length.pre_exclusion.prompt_only_over_budget": sum(
                contribution.multiplicity
                for contribution in syntax_main_contributions
                if contribution.prompt_tokens > profile.allowed_max_length
            ),
            "training.length.pre_exclusion.syntax_max_approx": _round_two_significant_digits(
                max(contribution.total_tokens for contribution in c1_main)
            ),
            "training.length.pre_exclusion.c0_max_approx": _round_two_significant_digits(
                max(contribution.total_tokens for contribution in c0_main)
            ),
        }
    )
    return results


def _fine_run_compute(
    run_results: Sequence[RunResults],
) -> dict[tuple[FineTunedCondition, int], object]:
    result: dict[tuple[FineTunedCondition, int], object] = {}
    for run in run_results:
        if run.condition not in _FINE_TUNED_CONDITIONS:
            continue
        key = (run.condition, run.seed)
        if key in result:
            raise ValueError(f"duplicate fine-tuned run: {key!r}")
        result[key] = run.metrics.compute
    expected = {(condition, seed) for condition in _FINE_TUNED_CONDITIONS for seed in (42, 43, 44)}
    if set(result) != expected:
        raise ValueError("fine-tuned run matrix is incomplete")
    return result


def _build_run_accounting_results(
    run_results: Sequence[RunResults],
    selection_traces: Mapping[tuple[Condition, int], tuple[SelectionPoint, ...]],
    study_rows: Sequence[StudyRow],
    experiment_config: ExperimentConfig,
    metadata: AnalysisMetadata,
) -> dict[str, Scalar]:
    fine_runs = [run for run in run_results if run.condition in _FINE_TUNED_CONDITIONS]
    compute_by_key = _fine_run_compute(run_results)
    evaluations = {
        len(trace) for (condition, _seed), trace in selection_traces.items() if condition in _FINE_TUNED_CONDITIONS
    }
    results: dict[str, Scalar] = {
        "training.checkpoint_selection.evaluations_per_run": _uniform(
            evaluations,
            name="fine-tuned checkpoint-evaluation count",
        )
    }
    for condition in _FINE_TUNED_CONDITIONS:
        name = _CONDITION_RESULT_NAMES[condition]
        compute = compute_by_key[(condition, 42)]
        results[f"training.seed_42.{name}.total_tokens_m"] = cast(int, getattr(compute, "total_tokens")) / 1_000_000.0
        results[f"training.seed_42.{name}.supervised_tokens_m"] = (
            cast(int, getattr(compute, "supervised_tokens")) / 1_000_000.0
        )

    for field, result_id in (
        ("total_tokens", "training.seed_variation.total_tokens.maximum_percent"),
        ("supervised_tokens", "training.seed_variation.supervised_tokens.maximum_percent"),
    ):
        maximum = max(
            abs(
                cast(int, getattr(compute_by_key[(condition, seed)], field))
                - cast(int, getattr(compute_by_key[(condition, 42)], field))
            )
            / cast(int, getattr(compute_by_key[(condition, 42)], field))
            * 100.0
            for condition in _FINE_TUNED_CONDITIONS
            for seed in (43, 44)
        )
        results[result_id] = ceil(maximum * 10.0) / 10.0

    accounting = metadata.token_accounting
    total_slots = {
        (condition, seed): sum(row.slot_count for row in accounting if row.condition == condition and row.seed == seed)
        for condition in _FINE_TUNED_CONDITIONS
        for seed in (42, 43, 44)
    }
    c2_main_slots = _uniform(
        (row.slot_count for row in accounting if row.condition == "C2" and row.pool == "main"),
        name="C2 main slots",
    )
    c2_auxiliary_slots = _uniform(
        (row.slot_count for row in accounting if row.condition == "C2" and row.pool == "syntax_auxiliary"),
        name="C2 auxiliary slots",
    )
    slots_per_run = _uniform(total_slots.values(), name="total slots per run")
    results.update(
        {
            "training.slots.total_per_run": slots_per_run,
            "training.slots.c2_main": c2_main_slots,
            "training.slots.c2_auxiliary": c2_auxiliary_slots,
            "training.slots.main_ratio": c2_main_slots / slots_per_run,
            "training.slots.auxiliary_ratio": c2_auxiliary_slots / slots_per_run,
        }
    )

    pool_specs = (
        ("C2", "main", "training.supervised_decomposition.c2.main_tokens_m"),
        ("C2", "syntax_auxiliary", "training.supervised_decomposition.c2.auxiliary_tokens_m"),
        ("C2-control", "main", "training.supervised_decomposition.c2_control.main_tokens_m"),
        (
            "C2-control",
            "duplicated_main_control",
            "training.supervised_decomposition.c2_control.duplicated_tokens_m",
        ),
    )
    pool_means: dict[tuple[str, str], float] = {}
    for condition, pool, result_id in pool_specs:
        values = [row.supervised_tokens for row in accounting if row.condition == condition and row.pool == pool]
        if len(values) != 3:
            raise ValueError(f"{condition} {pool} requires three accounting rows")
        mean_tokens = statistics.mean(values)
        pool_means[(condition, pool)] = mean_tokens
        results[result_id] = mean_tokens / 1_000_000.0
    c2_supervised = pool_means[("C2", "main")] + pool_means[("C2", "syntax_auxiliary")]
    control_supervised = pool_means[("C2-control", "main")] + pool_means[("C2-control", "duplicated_main_control")]
    results["training.supervised_decomposition.c2.auxiliary_percent"] = (
        pool_means[("C2", "syntax_auxiliary")] / c2_supervised * 100.0
    )
    results["training.supervised_decomposition.c2_control.duplicated_percent"] = (
        pool_means[("C2-control", "duplicated_main_control")] / control_supervised * 100.0
    )

    def value(condition: FineTunedCondition, seed: int, field: str) -> float:
        return float(cast(int | float, getattr(compute_by_key[(condition, seed)], field)))

    def seed_percent_differences(
        condition_a: FineTunedCondition,
        condition_b: FineTunedCondition,
        field: str,
    ) -> list[float]:
        return [
            (value(condition_a, seed, field) / value(condition_b, seed, field) - 1.0) * 100.0 for seed in (42, 43, 44)
        ]

    c2_supervised_diffs = seed_percent_differences("C2", "C1", "supervised_tokens")
    c2_total_diffs = seed_percent_differences("C2", "C1", "total_tokens")
    control_supervised_diffs = seed_percent_differences("C2-control", "C1", "supervised_tokens")
    control_total_diffs = seed_percent_differences("C2-control", "C1", "total_tokens")
    results.update(
        {
            "training.budget_match.c2.supervised_max_deviation_percent": max(
                abs(difference) for difference in c2_supervised_diffs
            ),
            "training.budget_match.c2.total_deficit_min_percent": min(-difference for difference in c2_total_diffs),
            "training.budget_match.c2.total_deficit_max_percent": max(-difference for difference in c2_total_diffs),
            "training.budget_match.c2_control.supervised_max_deviation_percent": max(
                abs(difference) for difference in control_supervised_diffs
            ),
            "training.budget_match.c2_control.total_max_deviation_percent": max(
                abs(difference) for difference in control_total_diffs
            ),
            "training.budget_match.c2_control.total_over_c2_percent": (
                statistics.mean(value("C2-control", seed, "total_tokens") for seed in (42, 43, 44))
                / statistics.mean(value("C2", seed, "total_tokens") for seed in (42, 43, 44))
                - 1.0
            )
            * 100.0,
        }
    )

    for field, result_id in (
        ("total_tokens", "training.cost.c1_over_c0.total_tokens_percent"),
        ("wall_clock_train_s", "training.cost.c1_over_c0.training_time_percent"),
        ("wall_clock_total_s", "training.cost.c1_over_c0.end_to_end_time_percent"),
        ("peak_allocated_gpu_memory_gib", "training.cost.c1_over_c0.peak_memory_percent"),
    ):
        c1_mean = statistics.mean(value("C1", seed, field) for seed in (42, 43, 44))
        c0_mean = statistics.mean(value("C0", seed, field) for seed in (42, 43, 44))
        results[result_id] = (c1_mean / c0_mean - 1.0) * 100.0
    results["training.sweep.training_hours"] = (
        sum(cast(float, run.metrics.compute.wall_clock_train_s) for run in fine_runs) / 3_600.0
    )
    results["training.sweep.end_to_end_hours"] = (
        sum(cast(float, run.metrics.compute.wall_clock_total_s) for run in fine_runs) / 3_600.0
    )

    observations = {observation.id: observation.approximate_value for observation in metadata.operator_log_observations}
    results.update(
        {
            "training.operator.device_memory_c0_gb": observations["device_memory_c0_gb"],
            "training.operator.device_memory_syntax_gb": observations["device_memory_syntax_gb"],
            "training.operator.provider_credits": observations["provider_credits_total"],
        }
    )
    nontraining_seconds = {
        condition: [
            value(condition, seed, "wall_clock_total_s") - value(condition, seed, "wall_clock_train_s")
            for seed in (42, 43, 44)
        ]
        for condition in _FINE_TUNED_CONDITIONS
    }
    condition_nontraining_minutes = [
        statistics.mean(nontraining_seconds[condition]) / 60.0 for condition in _FINE_TUNED_CONDITIONS
    ]
    results["training.end_to_end_nontraining_minutes.minimum"] = min(condition_nontraining_minutes)
    results["training.end_to_end_nontraining_minutes.maximum"] = max(condition_nontraining_minutes)
    full_test_outputs = sum(row.split == "test" for row in study_rows) * len(TASK_TYPES)
    results["training.throughput.c0.seconds_per_output"] = (
        statistics.mean(nontraining_seconds["C0"]) / full_test_outputs
    )
    results["training.throughput.syntax.seconds_per_output"] = (
        statistics.mean(
            seconds for condition in ("C1", "C2", "C2-control") for seconds in nontraining_seconds[condition]
        )
        / full_test_outputs
    )

    selected_steps = Counter(cast(int, run.metrics.checkpoint_selection.selected_step) for run in fine_runs)
    final_step = experiment_config.profiles["paper"].max_steps
    alternate_steps = [step for step in selected_steps if step != final_step]
    alternate_step = _uniform(alternate_steps, name="alternate selected checkpoint step")
    results.update(
        {
            "training.checkpoint_selection.final_step": final_step,
            "training.checkpoint_selection.final_count": selected_steps[final_step],
            "training.checkpoint_selection.alternate_step": alternate_step,
            "training.checkpoint_selection.alternate_count": selected_steps[alternate_step],
        }
    )
    for condition in _FINE_TUNED_CONDITIONS:
        composites = [
            cast(float, run.metrics.checkpoint_selection.best_composite)
            for run in fine_runs
            if run.condition == condition
        ]
        results[f"training.checkpoint_selection.best_composite.{_CONDITION_RESULT_NAMES[condition]}"] = statistics.mean(
            composites
        )
    return results


def _build_run_record_results(scored_records: Sequence[ReleasedRecord]) -> dict[str, Scalar]:
    results: dict[str, Scalar] = {}
    for condition in (*_FINE_TUNED_CONDITIONS, "zero-shot-raw", "zero-shot-syntax"):
        records = cast(
            list[RuleIdentificationRecord],
            [
                record
                for record in scored_records
                if record.condition == condition and record.task_type == "rule_identification"
            ],
        )
        if not records:
            raise ValueError(f"{condition} requires rule-identification records")
        name = _CONDITION_RESULT_NAMES[condition]
        rejected = sum(record.rejected_label_count for record in records)
        emitted = sum(record.n_emitted for record in records)
        results[f"run.label_normalization.{name}.rejected"] = rejected
        results[f"run.label_normalization.{name}.emitted_members"] = emitted
        if condition in _FINE_TUNED_CONDITIONS:
            results[f"run.label_normalization.{name}.failure_percent"] = rejected / emitted * 100.0 if emitted else 0.0
        else:
            results[f"run.label_normalization.{name}.no_recognized_array"] = sum(
                record.normalization_status == "no_recognized_array" for record in records
            )

    repair_records = cast(
        list[CorrectionRecord | JointRecord],
        [record for record in scored_records if record.task_type in {"correction", "joint"}],
    )
    results["run.fix_outputs.load_degraded"] = sum(
        record.outcome.output_tool_status == "load_degraded" for record in repair_records
    )
    fine_cell_counts = {
        (condition, task): sum(record.condition == condition and record.task_type == task for record in scored_records)
        for condition in _FINE_TUNED_CONDITIONS
        for task in ("rule_identification", "explanation", "correction", "joint")
    }
    results["run.scored_outputs.per_fine_tuned_condition_task"] = _uniform(
        fine_cell_counts.values(),
        name="fine-tuned scored condition-task count",
    )
    for task_type in ("correction", "joint"):
        for status, result_name in (("NA", "na"), ("OK", "ok"), ("FAIL", "fail")):
            rates = []
            for condition in _FINE_TUNED_CONDITIONS:
                selected = [
                    record
                    for record in repair_records
                    if record.condition == condition and record.task_type == task_type
                ]
                rates.append(sum(record.outcome.build_status == status for record in selected) / len(selected) * 100.0)
            prefix = f"run.build.{task_type}.{result_name}"
            results[f"{prefix}.minimum_percent"] = min(rates)
            results[f"{prefix}.maximum_percent"] = max(rates)
    return results


def _seed_task_rates(
    scored_records: Sequence[ReleasedRecord],
    *,
    condition: FineTunedCondition,
    task_type: Literal["rule_identification", "correction", "joint"],
) -> dict[int, float]:
    rates: dict[int, float] = {}
    for seed in (42, 43, 44):
        selected = [
            record
            for record in scored_records
            if record.condition == condition and record.seed == seed and record.task_type == task_type
        ]
        if not selected:
            raise ValueError(f"{condition} {task_type} seed {seed} requires records")
        if task_type == "rule_identification":
            rates[seed] = sum(cast(RuleIdentificationRecord, record).exact_match for record in selected) / len(selected)
        else:
            rates[seed] = sum(
                cast(CorrectionRecord | JointRecord, record).outcome.overall_fixed for record in selected
            ) / len(selected)
    return rates


def _seed_rate_differences(
    scored_records: Sequence[ReleasedRecord],
    *,
    condition_a: FineTunedCondition,
    condition_b: FineTunedCondition,
    task_type: Literal["rule_identification", "correction", "joint"],
) -> dict[int, float]:
    rates_a = _seed_task_rates(scored_records, condition=condition_a, task_type=task_type)
    rates_b = _seed_task_rates(scored_records, condition=condition_b, task_type=task_type)
    return {seed: rates_a[seed] - rates_b[seed] for seed in (42, 43, 44)}


def _build_rq1_rq2_results(scored_records: Sequence[ReleasedRecord]) -> dict[str, Scalar]:
    results: dict[str, Scalar] = {}
    rq1_joint = task_contrast(
        scored_records,
        name="rq1.joint.c1_minus_c0",
        condition_a="C1",
        condition_b="C0",
        task="joint",
        n_boot=10_000,
        seed=42,
    ).interval
    c0_joint_rates = _seed_task_rates(scored_records, condition="C0", task_type="joint")
    c1_joint_rates = _seed_task_rates(scored_records, condition="C1", task_type="joint")
    rq1_seed_differences = {seed: c1_joint_rates[seed] - c0_joint_rates[seed] for seed in (42, 43, 44)}
    rq1_macro = macro_f1_difference_interval(
        scored_records,
        condition_a="C1",
        condition_b="C0",
        n_boot=10_000,
        seed=45,
    )
    results["rq1.joint.relative_to_c0.percent"] = rq1_joint.point / statistics.mean(c0_joint_rates.values()) * 100.0
    for statistic, value in (
        ("point", rq1_macro.point),
        ("low", rq1_macro.ci_low),
        ("high", rq1_macro.ci_high),
    ):
        results[f"rq1.rule_id_macro_f1.c1_minus_c0.{statistic}"] = value * 100.0
    for seed in (42, 43, 44):
        results[f"rq1.joint.c1_minus_c0.seed_{seed}"] = rq1_seed_differences[seed] * 100.0
        results[f"rq1.joint.c0.seed_{seed}.percent"] = c0_joint_rates[seed] * 100.0
        results[f"rq1.joint.c1.seed_{seed}.percent"] = c1_joint_rates[seed] * 100.0
    results["rq1.joint.c1_minus_c0.seed_sd"] = statistics.stdev(rq1_seed_differences.values()) * 100.0
    c0_exact_rates = _seed_task_rates(scored_records, condition="C0", task_type="rule_identification")
    for seed in (42, 43, 44):
        results[f"rq1.rule_id_exact_match.c0.seed_{seed}.percent"] = c0_exact_rates[seed] * 100.0
    results["rq1.rule_id_exact_match.c0.seed_range_points"] = (
        max(c0_exact_rates.values()) - min(c0_exact_rates.values())
    ) * 100.0

    joint_by_key = {
        (record.condition, record.seed, record.base_snippet_id): record.outcome.overall_fixed
        for record in cast(Sequence[JointRecord], scored_records)
        if record.task_type == "joint" and record.condition in {"C0", "C1"}
    }
    discordance_rates = []
    for seed in (42, 43, 44):
        snippet_ids = {
            record.base_snippet_id
            for record in scored_records
            if record.condition == "C0" and record.seed == seed and record.task_type == "joint"
        }
        discordance_rates.append(
            sum(
                joint_by_key[("C0", seed, snippet_id)] != joint_by_key[("C1", seed, snippet_id)]
                for snippet_id in snippet_ids
            )
            / len(snippet_ids)
        )
    results["rq1.discordance.minimum_rate"] = min(discordance_rates)
    results["rq1.discordance.maximum_rate"] = max(discordance_rates)
    rq1_differences = seed_averaged_snippet_differences(
        scored_records,
        condition_a="C1",
        condition_b="C0",
        task="joint",
    )
    results["rq1.nonzero_seed_averaged_contrast.files"] = sum(
        difference != 0.0 for difference in rq1_differences.values()
    )

    joint_comparisons = (
        ("c2_control_minus_c1", "C2-control", "C1"),
        ("c2_minus_c1", "C2", "C1"),
        ("c2_minus_c2_control", "C2", "C2-control"),
    )
    for name, condition_a, condition_b in joint_comparisons:
        interval = task_contrast(
            scored_records,
            name=f"rq2.joint.{name}",
            condition_a=condition_a,
            condition_b=condition_b,
            task="joint",
            n_boot=10_000,
            seed=42,
        ).interval
        for statistic, value in (
            ("point", interval.point),
            ("low", interval.ci_low),
            ("high", interval.ci_high),
        ):
            results[f"rq2.joint.{name}.{statistic}"] = value * 100.0

    control_differences = _seed_rate_differences(
        scored_records,
        condition_a="C2-control",
        condition_b="C1",
        task_type="joint",
    )
    for seed, difference in control_differences.items():
        results[f"rq2.joint.c2_control_minus_c1.seed_{seed}"] = difference * 100.0

    for name, condition_b in (("c2_minus_c1", "C1"), ("c2_minus_c2_control", "C2-control")):
        exact_interval = task_contrast(
            scored_records,
            name=f"rq2.rule_id_exact_match.{name}",
            condition_a="C2",
            condition_b=condition_b,
            task="rule_identification",
            n_boot=10_000,
            seed=44,
        ).interval
        macro_interval = macro_f1_difference_interval(
            scored_records,
            condition_a="C2",
            condition_b=condition_b,
            n_boot=10_000,
            seed=45,
        )
        for metric, interval in (
            ("rule_id_exact_match", exact_interval),
            ("rule_id_macro_f1", macro_interval),
        ):
            for statistic, value in (
                ("point", interval.point),
                ("low", interval.ci_low),
                ("high", interval.ci_high),
            ):
                results[f"rq2.{metric}.{name}.{statistic}"] = value * 100.0

        correction = task_contrast(
            scored_records,
            name=f"rq2.correction.{name}",
            condition_a="C2",
            condition_b=condition_b,
            task="correction",
            n_boot=10_000,
            seed=43,
        ).interval
        results[f"rq2.correction.{name}.point"] = correction.point * 100.0

        joint_seed_differences = _seed_rate_differences(
            scored_records,
            condition_a="C2",
            condition_b=condition_b,
            task_type="joint",
        )
        for seed, difference in joint_seed_differences.items():
            results[f"rq2.joint.{name}.seed_{seed}"] = difference * 100.0
        results[f"rq2.joint.{name}.seed_sd"] = statistics.stdev(joint_seed_differences.values()) * 100.0
    return results


def _build_core_sensitivity_results(
    scored_records: Sequence[ReleasedRecord],
    experiment_config: ExperimentConfig,
) -> dict[str, Scalar]:
    fine_repair_records = cast(
        list[CorrectionRecord | JointRecord],
        [
            record
            for record in scored_records
            if record.condition in _FINE_TUNED_CONDITIONS and record.task_type in {"correction", "joint"}
        ],
    )
    studied_counts = []
    enabled_counts = []
    for condition in _FINE_TUNED_CONDITIONS:
        for task_type in ("correction", "joint"):
            selected = [
                record
                for record in fine_repair_records
                if record.condition == condition and record.task_type == task_type
            ]
            studied_counts.append(sum(record.outcome.studied_regression is True for record in selected))
            enabled_counts.append(sum(record.outcome.enabled_regression is True for record in selected))
    scored_per_cell = _uniform(
        (
            sum(record.condition == condition and record.task_type == task_type for record in fine_repair_records)
            for condition in _FINE_TUNED_CONDITIONS
            for task_type in ("correction", "joint")
        ),
        name="repair records per condition-task cell",
    )
    results: dict[str, Scalar] = {
        "sensitivity.regression.studied.maximum_count": max(studied_counts),
        "sensitivity.regression.studied.maximum_percent": max(studied_counts) / scored_per_cell * 100.0,
        "sensitivity.regression.enabled.maximum_count": max(enabled_counts),
    }

    syntax_prompt_maps: dict[FineTunedCondition, dict[tuple[int, str, str], int]] = {}
    for condition in ("C1", "C2", "C2-control"):
        selected = [record for record in scored_records if record.condition == condition]
        prompt_map = {
            (record.seed, record.base_snippet_id, record.task_type): record.prompt_tokens for record in selected
        }
        if len(prompt_map) != len(selected):
            raise ValueError(f"{condition} syntax prompt records must have unique seed/base/task keys")
        syntax_prompt_maps[condition] = prompt_map
    if any(syntax_prompt_maps[condition] != syntax_prompt_maps["C1"] for condition in ("C2", "C2-control")):
        raise ValueError("syntax prompt token maps must match across C1, C2, and C2-control")

    prompt_summaries: dict[tuple[str, str], object] = {}
    for condition in ("C0", "C1"):
        for task_type in ("rule_identification", "explanation", "correction", "joint"):
            prompt_summaries[(condition, task_type)] = distribution_summary(
                [
                    record.prompt_tokens
                    for record in scored_records
                    if record.condition == condition and record.task_type == task_type
                ]
            )
    c0_joint = prompt_summaries[("C0", "joint")]
    syntax_joint = prompt_summaries[("C1", "joint")]
    c0_p50s = [
        cast(float, getattr(prompt_summaries[("C0", task)], "p50"))
        for task in (
            "rule_identification",
            "explanation",
            "correction",
            "joint",
        )
    ]
    syntax_p50s = [
        cast(float, getattr(prompt_summaries[("C1", task)], "p50"))
        for task in (
            "rule_identification",
            "explanation",
            "correction",
            "joint",
        )
    ]
    c0_joint_p50 = cast(float, getattr(c0_joint, "p50"))
    syntax_joint_p50 = cast(float, getattr(syntax_joint, "p50"))
    results.update(
        {
            "sensitivity.prompt.joint.c0.p50": c0_joint_p50,
            "sensitivity.prompt.joint.c0.max": cast(float, getattr(c0_joint, "max")),
            "sensitivity.prompt.joint.syntax.p50": syntax_joint_p50,
            "sensitivity.prompt.joint.syntax.max": cast(float, getattr(syntax_joint, "max")),
            "sensitivity.prompt.joint.syntax_median_overhead_percent": (syntax_joint_p50 / c0_joint_p50 - 1.0) * 100.0,
            "sensitivity.prompt.task_p50.c0.minimum": min(c0_p50s),
            "sensitivity.prompt.task_p50.c0.maximum": max(c0_p50s),
            "sensitivity.prompt.task_p50.syntax.minimum": min(syntax_p50s),
            "sensitivity.prompt.task_p50.syntax.maximum": max(syntax_p50s),
        }
    )

    c1_joint_records = cast(
        list[JointRecord],
        [record for record in scored_records if record.condition == "C1" and record.task_type == "joint"],
    )
    c0_joint_by_key = {
        (record.base_snippet_id, record.seed): cast(JointRecord, record)
        for record in scored_records
        if record.condition == "C0" and record.task_type == "joint"
    }
    for status, name in (("present", "present"), ("present_truncated", "present_truncated")):
        selected_c1 = [record for record in c1_joint_records if record.summary_status == status]
        selected_c0 = [c0_joint_by_key[(record.base_snippet_id, record.seed)] for record in selected_c1]
        c1_rate = sum(record.outcome.overall_fixed for record in selected_c1) / len(selected_c1) * 100.0
        c0_rate = sum(record.outcome.overall_fixed for record in selected_c0) / len(selected_c0) * 100.0
        results[f"sensitivity.summary_stratum.c1.{name}.fix_percent"] = c1_rate
        results[f"sensitivity.summary_stratum.{name}.outputs"] = len(selected_c1)
        results[f"sensitivity.summary_stratum.c0.{name}.fix_percent"] = c0_rate
        results[f"sensitivity.summary_stratum.{name}.c1_minus_c0_points"] = c1_rate - c0_rate
    skipped = [record for record in c1_joint_records if record.summary_status == "skipped"]
    results["sensitivity.summary_stratum.skipped.files"] = len({record.base_snippet_id for record in skipped})
    results["sensitivity.summary_stratum.skipped.outputs"] = len(skipped)

    generation_caps = experiment_config.profiles["paper"].generation_max_new_tokens
    response_caps = {
        "correction": generation_caps.correction,
        "joint": generation_caps.joint,
    }
    cap_rate_ranges = []
    for task_type, cap in response_caps.items():
        counts = []
        rates = []
        for condition in _FINE_TUNED_CONDITIONS:
            selected = [
                record for record in scored_records if record.condition == condition and record.task_type == task_type
            ]
            count = sum(record.retokenized_response_token_proxy >= cap for record in selected)
            counts.append(count)
            rates.append(count / len(selected) * 100.0)
        results[f"sensitivity.response_cap_proxy.{task_type}.minimum_percent"] = min(rates)
        results[f"sensitivity.response_cap_proxy.{task_type}.maximum_percent"] = max(rates)
        results[f"sensitivity.response_cap_proxy.{task_type}.minimum_count"] = min(counts)
        results[f"sensitivity.response_cap_proxy.{task_type}.maximum_count"] = max(counts)
        cap_rate_ranges.append(max(rates) - min(rates))
    for task_type in ("rule_identification", "explanation"):
        summary = distribution_summary(
            [
                record.retokenized_response_token_proxy
                for record in scored_records
                if record.condition in _FINE_TUNED_CONDITIONS and record.task_type == task_type
            ]
        )
        results[f"sensitivity.response_cap_proxy.{task_type}.p50"] = summary.p50
    results["sensitivity.response_cap_proxy.per_condition_rate_range_max_points"] = int(round(max(cap_rate_ranges)))
    return results


def _build_remaining_sensitivity_results(scored_records: Sequence[ReleasedRecord]) -> dict[str, Scalar]:
    repair_records = [
        cast(CorrectionRecord | JointRecord, record)
        for record in scored_records
        if record.condition in _FINE_TUNED_CONDITIONS and record.task_type in {"correction", "joint"}
    ]
    successes = [record for record in repair_records if record.outcome.overall_fixed]
    for record in successes:
        if (record.extracted_similarity is None) != (record.sensitivity_class is None):
            raise ValueError("successful extracted-code indicators must be both present or both null")
    retained = [record for record in successes if record.sensitivity_class is not None]
    condition_counts = Counter(record.condition for record in retained)
    joint_retained = Counter(record.condition for record in retained if record.task_type == "joint")
    joint_denominators = Counter(record.condition for record in repair_records if record.task_type == "joint")
    if set(joint_denominators) != set(_FINE_TUNED_CONDITIONS) or any(
        joint_denominators[condition] != 1230 for condition in _FINE_TUNED_CONDITIONS
    ):
        raise ValueError("extracted-code sensitivity requires exactly 1,230 joint records per condition")

    c1_task_rates: dict[str, float] = {}
    for task_type in ("correction", "joint"):
        selected = [record for record in repair_records if record.condition == "C1" and record.task_type == task_type]
        if len(selected) != 1230:
            raise ValueError(f"C1 {task_type} task-rate sensitivity requires exactly 1,230 records")
        c1_task_rates[task_type] = sum(record.outcome.overall_fixed for record in selected) / len(selected)

    multilabel = [
        cast(RuleIdentificationRecord, record)
        for record in scored_records
        if record.task_type == "rule_identification" and len(cast(RuleIdentificationRecord, record).gold) > 1
    ]
    if len(multilabel) != 350:
        raise ValueError(f"multilabel sensitivity requires exactly 350 released rule records, got {len(multilabel)}")

    results: dict[str, Scalar] = {
        "sensitivity.extracted_code.scored_fix_successes": len(successes),
        "sensitivity.extracted_code.low_similarity_outputs": len(retained),
        "sensitivity.extracted_code.low_similarity_shared_function": sum(
            record.sensitivity_class == "same_file_truncated" for record in retained
        ),
        "sensitivity.extracted_code.low_similarity_package_only": sum(
            record.sensitivity_class == "same_pkg_no_shared_func" for record in retained
        ),
        "sensitivity.extracted_code.unique_snippets": len({record.base_snippet_id for record in retained}),
        "sensitivity.extracted_code.condition.c0": condition_counts["C0"],
        "sensitivity.extracted_code.condition.c1": condition_counts["C1"],
        "sensitivity.extracted_code.condition.c2": condition_counts["C2"],
        "sensitivity.extracted_code.condition.c2_control": condition_counts["C2-control"],
        "sensitivity.extracted_code.adjustment.c1_minus_c0.maximum_points": (
            joint_retained["C1"] / joint_denominators["C1"] - joint_retained["C0"] / joint_denominators["C0"]
        )
        * 100.0,
        "sensitivity.extracted_code.adjustment.c2_minus_c1.maximum_points": (
            joint_retained["C2"] / joint_denominators["C2"] - joint_retained["C1"] / joint_denominators["C1"]
        )
        * 100.0,
        "sensitivity.extracted_code.adjustment.c2_minus_c2_control.maximum_points": (
            joint_retained["C2"] / joint_denominators["C2"]
            - joint_retained["C2-control"] / joint_denominators["C2-control"]
        )
        * 100.0,
        "sensitivity.task_rate.c1.correction_minus_joint.approx_points": round(
            (c1_task_rates["correction"] - c1_task_rates["joint"]) * 100.0
        ),
        "sensitivity.multilabel.rule_exact.all_conditions_percent": sum(record.exact_match for record in multilabel)
        / len(multilabel)
        * 100.0,
    }
    return results


def _build_core_results(
    experiment_config: ExperimentConfig,
    scored_records: Sequence[ReleasedRecord],
    run_results: Sequence[RunResults],
    selection_traces: Mapping[tuple[Condition, int], tuple[SelectionPoint, ...]],
    study_rows: Sequence[StudyRow],
    metadata: AnalysisMetadata,
) -> dict[str, Scalar]:
    results = _merge_results(
        _build_dataset_repository_results(study_rows),
        _build_reference_results(study_rows),
        _build_oracle_serializer_results(study_rows, scored_records),
        _build_metadata_results(metadata),
        _build_configuration_training_contribution_results(
            experiment_config,
            run_results,
            study_rows,
        ),
        _build_run_accounting_results(
            run_results,
            selection_traces,
            study_rows,
            experiment_config,
            metadata,
        ),
        _build_run_record_results(scored_records),
        _build_rq1_rq2_results(scored_records),
        _build_core_sensitivity_results(scored_records, experiment_config),
    )
    if len(results) != 450:
        raise ValueError(f"core result registry requires exactly 450 JSON leaves, got {len(results)}")
    return results


def build_outputs(
    *,
    experiment_config: ExperimentConfig,
    scored_records: Sequence[ReleasedRecord],
    run_results: Sequence[RunResults],
    selection_traces: Mapping[tuple[Condition, int], tuple[SelectionPoint, ...]],
    study_rows: Sequence[StudyRow],
    metadata: AnalysisMetadata,
) -> GeneratedOutputs:
    core_tables = (
        _build_table_8_1(scored_records),
        _build_table_8_2(scored_records),
        _build_table_8_3(scored_records),
        _build_table_8_4(scored_records),
        _build_table_8_5(study_rows),
        _build_table_8_6(run_results, metadata),
        _build_table_8_7(scored_records, study_rows),
        _build_table_8_8(scored_records, study_rows),
        _build_table_8_9(scored_records),
    )
    secondary_results = _merge_results(
        _build_rq3_results(scored_records),
        _build_remaining_sensitivity_results(scored_records),
        _build_familiarity_results(scored_records),
        _build_section_8_10_record_results(scored_records),
        _build_zero_shot_transitions(scored_records),
    )
    return GeneratedOutputs(
        results=_merge_results(
            _build_core_results(
                experiment_config,
                scored_records,
                run_results,
                selection_traces,
                study_rows,
                metadata,
            ),
            secondary_results,
        ),
        tables={table.filename: table for table in core_tables},
    )
