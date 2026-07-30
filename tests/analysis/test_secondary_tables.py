from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from typing import cast

import pytest

import analysis.tables as tables
from analysis.inputs import (
    CHECKS,
    Condition,
    CorrectionRecord,
    JointRecord,
    ReleasedRecord,
    RuleIdentificationRecord,
    StudyRow,
)
from analysis.metrics import (
    Interval,
    Rq3Result,
    Rq3TaskContrast,
    build_rq3_frame,
    cluster_bootstrap_interval,
    fit_rq3_interaction,
    macro_f1,
    per_check_prf,
    rq3_task_contrasts,
)
from analysis.tables import GeneratedOutputs, OutputCell, TableData, build_outputs

CHECK_ROWS = (
    ("assignOp", "assign_op"),
    ("builtinShadow", "builtin_shadow"),
    ("captLocal", "capt_local"),
    ("commentFormatting", "comment_formatting"),
    ("elseif", "elseif"),
    ("ifElseChain", "if_else_chain"),
    ("paramTypeCombine", "param_type_combine"),
    ("singleCaseSwitch", "single_case_switch"),
)
CHECK_ROW_KEYS = tuple(row_key for _check, row_key in CHECK_ROWS)
CHECK_BY_ROW = {row_key: check for check, row_key in CHECK_ROWS}
FINE_TUNED_CONDITIONS: tuple[Condition, ...] = ("C0", "C1", "C2", "C2-control")

TABLE_8_7_COLUMNS = (
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
TABLE_8_8_COLUMNS = (
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
TABLE_8_9_COLUMNS = ("check", "n", "zero_shot_raw", "zero_shot_syntax", "fine_tuned_c0")

STRATUM_SPECS: tuple[tuple[str, str, str, Callable[[StudyRow], bool]], ...] = (
    ("violations_1", "Violations", "= 1", lambda row: row.violation_count == 1),
    ("violations_2_plus", "Violations", ">= 2", lambda row: row.violation_count >= 2),
    ("length_lt_50", "Length", "< 50 lines", lambda row: row.source_line_count < 50),
    (
        "length_50_199",
        "Length",
        "50-199 lines",
        lambda row: 50 <= row.source_line_count < 200,
    ),
    ("length_200_plus", "Length", ">= 200 lines", lambda row: row.source_line_count >= 200),
    ("depth_0_1", "Nesting depth", "<= 1", lambda row: row.serializer.maximum_depth <= 1),
    (
        "depth_2_3",
        "Nesting depth",
        "2-3",
        lambda row: 2 <= row.serializer.maximum_depth <= 3,
    ),
    ("depth_gt_3", "Nesting depth", "> 3", lambda row: row.serializer.maximum_depth > 3),
)
STRATUM_ROW_KEYS = tuple(row_key for row_key, _stratum, _bin, _predicate in STRATUM_SPECS)

SECONDARY_JSON_IDS = frozenset(
    {
        "rq3.vb.c1_joint_interaction.coefficient",
        "rq3.vb.c1_joint_interaction.low",
        "rq3.vb.c1_joint_interaction.high",
        "rq3.vb.c1_joint_interaction.sd",
        "rq3.risk.joint_minus_correction.point",
        "rq3.risk.joint_minus_correction.low",
        "rq3.risk.joint_minus_correction.high",
        "rq3.risk.joint_minus_rule.point",
        "rq3.risk.joint_minus_rule.low",
        "rq3.risk.joint_minus_rule.high",
        "rq3.h3_supported",
        "sensitivity.extracted_code.scored_fix_successes",
        "sensitivity.extracted_code.low_similarity_outputs",
        "sensitivity.extracted_code.low_similarity_shared_function",
        "sensitivity.extracted_code.low_similarity_package_only",
        "sensitivity.extracted_code.unique_snippets",
        "sensitivity.extracted_code.condition.c0",
        "sensitivity.extracted_code.condition.c1",
        "sensitivity.extracted_code.condition.c2",
        "sensitivity.extracted_code.condition.c2_control",
        "sensitivity.extracted_code.adjustment.c1_minus_c0.maximum_points",
        "sensitivity.extracted_code.adjustment.c2_minus_c1.maximum_points",
        "sensitivity.extracted_code.adjustment.c2_minus_c2_control.maximum_points",
        "sensitivity.familiarity.zero_shot.minimum_percent",
        "sensitivity.familiarity.zero_shot.maximum_percent",
        "sensitivity.familiarity.c0.minimum_percent",
        "sensitivity.familiarity.c0.maximum_percent",
        "sensitivity.familiarity.zero_shot.overall_approx_percent",
        "sensitivity.task_rate.c1.correction_minus_joint.approx_points",
        "sensitivity.multilabel.rule_exact.all_conditions_percent",
        "section_8_10.c0_all_seed_correct.count",
        "section_8_10.c0_all_seed_correct.percent",
        "section_8_10.c2_only.any_seed.files",
        "section_8_10.c2_only.at_least_two_seeds.files",
        "section_8_10.category_b.files",
        "section_8_10.category_b.runs",
        "section_8_10.zero_shot_transition.changed",
        "section_8_10.zero_shot_transition.favor_syntax",
        "section_8_10.zero_shot_transition.favor_raw",
        "section_8_10.zero_shot_transition.net_files",
        "section_8_10.zero_shot_transition.net_points",
    }
)


def _build_from(release_inputs) -> GeneratedOutputs:
    return build_outputs(
        experiment_config=release_inputs.config,
        scored_records=release_inputs.scored_records,
        run_results=release_inputs.results,
        selection_traces=release_inputs.selection_traces,
        study_rows=release_inputs.study_rows,
        metadata=release_inputs.metadata,
    )


@pytest.fixture(scope="module")
def secondary_outputs(release_inputs) -> GeneratedOutputs:
    return _build_from(release_inputs)


def _cell(table: TableData, row_key: str, column: str) -> OutputCell | str:
    row = next(row for row in table.rows if row.key == row_key)
    return row.cells[column]


def _output_cell(table: TableData, row_key: str, column: str) -> OutputCell:
    cell = _cell(table, row_key, column)
    assert isinstance(cell, OutputCell)
    return cell


def _value(table: TableData, row_key: str, column: str) -> bool | int | float | str | None:
    return _output_cell(table, row_key, column).value


def _scored_study_rows(release_inputs) -> tuple[StudyRow, ...]:
    rows = tuple(
        row
        for row in release_inputs.study_rows
        if row.split == "test" and row.oracle.status != "excluded" and not row.quarantined
    )
    assert len(rows) == 410
    assert len({row.base_snippet_id for row in rows}) == 410
    return rows


def _rule_records(
    records: Sequence[ReleasedRecord],
    condition: Condition,
    seed: int,
) -> tuple[RuleIdentificationRecord, ...]:
    selected = tuple(
        cast(RuleIdentificationRecord, record)
        for record in records
        if record.condition == condition and record.seed == seed and record.task_type == "rule_identification"
    )
    assert len(selected) == 410
    assert len({record.base_snippet_id for record in selected}) == 410
    return selected


def _joint_records(
    records: Sequence[ReleasedRecord],
    condition: Condition,
    seed: int,
) -> tuple[JointRecord, ...]:
    selected = tuple(
        cast(JointRecord, record)
        for record in records
        if record.condition == condition and record.seed == seed and record.task_type == "joint"
    )
    assert len(selected) == 410
    assert len({record.base_snippet_id for record in selected}) == 410
    return selected


def _per_check_f1(records: Sequence[RuleIdentificationRecord]) -> dict[str, float]:
    values = per_check_prf(
        [record.pred for record in records],
        [record.gold for record in records],
        CHECKS,
    )
    return {check: values[check]["f1"] * 100.0 for check in CHECKS}


def _seed_mean_per_check_f1(
    records: Sequence[ReleasedRecord],
    condition: Condition,
) -> dict[str, float]:
    per_seed = [_per_check_f1(_rule_records(records, condition, seed)) for seed in (42, 43, 44)]
    return {check: statistics.mean(values[check] for values in per_seed) for check in CHECKS}


def _per_check_joint_fix(
    records: Sequence[ReleasedRecord],
    condition: Condition,
    check: str,
) -> float:
    rates = []
    for seed in (42, 43, 44):
        selected = [record for record in _joint_records(records, condition, seed) if check in record.target_checks]
        assert selected
        rates.append(sum(record.outcome.overall_fixed for record in selected) / len(selected) * 100.0)
    return statistics.mean(rates)


def _scored_ids_by_stratum(release_inputs) -> dict[str, frozenset[str]]:
    rows = _scored_study_rows(release_inputs)
    result = {
        row_key: frozenset(row.base_snippet_id for row in rows if predicate(row))
        for row_key, _stratum, _bin, predicate in STRATUM_SPECS
    }
    assert tuple(len(result[row_key]) for row_key in STRATUM_ROW_KEYS) == (385, 25, 105, 220, 85, 56, 191, 163)
    return result


def _success_grid(
    records: Sequence[ReleasedRecord],
    *,
    task: str,
) -> dict[tuple[Condition, int, str], float]:
    result: dict[tuple[Condition, int, str], float] = {}
    for record in records:
        if record.condition not in FINE_TUNED_CONDITIONS or record.task_type != task:
            continue
        key = (record.condition, record.seed, record.base_snippet_id)
        assert key not in result
        if task == "rule_identification":
            result[key] = float(cast(RuleIdentificationRecord, record).exact_match)
        else:
            result[key] = float(cast(JointRecord, record).outcome.overall_fixed)
    return result


def _stratum_baseline(
    grid: dict[tuple[Condition, int, str], float],
    ids: frozenset[str],
) -> float:
    return statistics.mean(
        statistics.mean(grid[("C0", seed, snippet_id)] for snippet_id in ids) for seed in (42, 43, 44)
    )


def _stratum_differences(
    grid: dict[tuple[Condition, int, str], float],
    ids: frozenset[str],
    condition: Condition,
) -> dict[str, float]:
    return {
        snippet_id: statistics.mean(
            grid[(condition, seed, snippet_id)] - grid[("C0", seed, snippet_id)] for seed in (42, 43, 44)
        )
        for snippet_id in ids
    }


def test_secondary_table_names_columns_rows_and_populated_cell_counts(secondary_outputs) -> None:
    assert list(secondary_outputs.tables)[6:] == ["table-8-7.csv", "table-8-8.csv", "table-8-9.csv"]

    table_8_7 = secondary_outputs.tables["table-8-7.csv"]
    table_8_8 = secondary_outputs.tables["table-8-8.csv"]
    table_8_9 = secondary_outputs.tables["table-8-9.csv"]
    assert table_8_7.columns == TABLE_8_7_COLUMNS
    assert table_8_8.columns == TABLE_8_8_COLUMNS
    assert table_8_9.columns == TABLE_8_9_COLUMNS
    assert tuple(row.key for row in table_8_7.rows) == CHECK_ROW_KEYS
    assert tuple(row.key for row in table_8_8.rows) == STRATUM_ROW_KEYS
    assert tuple(row.key for row in table_8_9.rows) == CHECK_ROW_KEYS

    populated = {
        table.filename: sum(isinstance(cell, OutputCell) for row in table.rows for cell in row.cells.values())
        for table in (table_8_7, table_8_8, table_8_9)
    }
    assert populated == {"table-8-7.csv": 72, "table-8-8.csv": 120, "table-8-9.csv": 32}


def test_table_8_7_per_check_denominators_and_formulas(secondary_outputs, release_inputs) -> None:
    table = secondary_outputs.tables["table-8-7.csv"]
    records = release_inputs.scored_records
    supports = Counter(check for record in _rule_records(records, "C0", 42) for check in record.gold)
    expected_rule = {condition: _seed_mean_per_check_f1(records, condition) for condition in FINE_TUNED_CONDITIONS}

    for check, row_key in CHECK_ROWS:
        assert _cell(table, row_key, "check") == check
        expected = {
            "n": supports[check],
            "c0_rule_f1": expected_rule["C0"][check],
            "c1_rule_f1": expected_rule["C1"][check],
            "c1_minus_c0_rule_f1": expected_rule["C1"][check] - expected_rule["C0"][check],
            "c0_joint_fix": _per_check_joint_fix(records, "C0", check),
            "c1_joint_fix": _per_check_joint_fix(records, "C1", check),
            "c2_rule_f1": expected_rule["C2"][check],
            "c2_control_rule_f1": expected_rule["C2-control"][check],
        }
        expected["c1_minus_c0_joint_fix"] = expected["c1_joint_fix"] - expected["c0_joint_fix"]
        for column, value in expected.items():
            cell = _output_cell(table, row_key, column)
            assert cell.result_id == f"table_8_7.{row_key}.{column}"
            assert cell.value == pytest.approx(value)
            assert cell.display_digits == (0 if column == "n" else 1)

    assert tuple(supports[check] for check, _row_key in CHECK_ROWS) == (58, 60, 56, 59, 38, 58, 55, 56)


@pytest.mark.parametrize("mutation", ["missing_record", "duplicate_record", "gold_mismatch", "study_target_mismatch"])
def test_table_8_7_rejects_broken_scored_joins(release_inputs, mutation: str) -> None:
    records = list(release_inputs.scored_records)
    study_rows = list(release_inputs.study_rows)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.condition == "C1" and record.seed == 43 and record.task_type == "rule_identification"
    )
    if mutation == "missing_record":
        records.pop(record_index)
    elif mutation == "duplicate_record":
        records.append(records[record_index])
    elif mutation == "gold_mismatch":
        record = cast(RuleIdentificationRecord, records[record_index])
        replacement = next(check for check in CHECKS if check not in record.gold)
        records[record_index] = record.model_copy(update={"gold": (*record.gold, replacement)})
    else:
        snippet_id = records[record_index].base_snippet_id
        row_index = next(index for index, row in enumerate(study_rows) if row.base_snippet_id == snippet_id)
        row = study_rows[row_index]
        replacement = next(check for check in CHECKS if check not in row.target_checks)
        study_rows[row_index] = row.model_copy(update={"target_checks": (*row.target_checks, replacement)})

    with pytest.raises(ValueError, match="grid|duplicate|gold|target|join"):
        tables._build_table_8_7(tuple(records), tuple(study_rows))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "target_mismatch"])
def test_table_8_7_rejects_broken_joint_grids(release_inputs, mutation: str) -> None:
    records = list(release_inputs.scored_records)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.condition == "C2" and record.seed == 44 and record.task_type == "joint"
    )
    if mutation == "missing":
        records.pop(record_index)
    elif mutation == "duplicate":
        records.append(records[record_index])
    else:
        record = cast(JointRecord, records[record_index])
        replacement = next(check for check in CHECKS if check not in record.target_checks)
        records[record_index] = record.model_copy(update={"target_checks": (*record.target_checks, replacement)})

    with pytest.raises(ValueError, match="grid|duplicate|target|join"):
        tables._build_table_8_7(tuple(records), release_inputs.study_rows)


def test_table_8_7_is_input_order_invariant(release_inputs) -> None:
    expected = tables._build_table_8_7(release_inputs.scored_records, release_inputs.study_rows)
    actual = tables._build_table_8_7(
        tuple(reversed(release_inputs.scored_records)),
        tuple(reversed(release_inputs.study_rows)),
    )
    assert actual == expected


def test_table_8_8_fixed_bins_baselines_and_bootstrap_intervals(secondary_outputs, release_inputs) -> None:
    table = secondary_outputs.tables["table-8-8.csv"]
    ids_by_stratum = _scored_ids_by_stratum(release_inputs)
    grids = {
        "rule_id": _success_grid(release_inputs.scored_records, task="rule_identification"),
        "joint": _success_grid(release_inputs.scored_records, task="joint"),
    }
    labels = {row_key: (stratum, bin_label) for row_key, stratum, bin_label, _predicate in STRATUM_SPECS}

    for row_key in STRATUM_ROW_KEYS:
        ids = ids_by_stratum[row_key]
        assert (_cell(table, row_key, "stratum"), _cell(table, row_key, "bin")) == labels[row_key]
        support = _output_cell(table, row_key, "n")
        assert support.value == len(ids)
        assert support.display_digits == 0
        assert support.result_id == f"table_8_8.{row_key}.n"

        for task_name, baseline_column in (("rule_id", "rule_id_c0"), ("joint", "joint_c0")):
            grid = grids[task_name]
            baseline = _stratum_baseline(grid, ids) * 100.0
            cell = _output_cell(table, row_key, baseline_column)
            assert cell.value == pytest.approx(baseline)
            assert cell.display_digits == 1
            for condition_name, comparison_name in (("C1", "c1_minus_c0"), ("C2", "c2_minus_c0")):
                differences = _stratum_differences(grid, ids, cast(Condition, condition_name))
                interval = cluster_bootstrap_interval(differences, n_boot=10_000, seed=42, alpha=0.05)
                for statistic, value in (
                    ("point", interval.point),
                    ("low", interval.ci_low),
                    ("high", interval.ci_high),
                ):
                    column = f"{task_name}_{comparison_name}_{statistic}"
                    comparison_cell = _output_cell(table, row_key, column)
                    assert comparison_cell.value == pytest.approx(value * 100.0)
                    assert comparison_cell.display_digits == 1
                    assert comparison_cell.result_id == f"table_8_8.{row_key}.{column}"


def test_table_8_8_bootstraps_exactly_four_restricted_grids_per_bin(monkeypatch, release_inputs) -> None:
    calls: list[tuple[frozenset[str], int, int, float]] = []

    def fake_interval(values, *, n_boot: int, seed: int, alpha: float = 0.05) -> Interval:
        calls.append((frozenset(values), n_boot, seed, alpha))
        point = statistics.mean(values.values())
        return Interval(point, point, point, 1.0, n_boot, seed, alpha, len(values))

    monkeypatch.setattr(tables, "cluster_bootstrap_interval", fake_interval, raising=False)
    tables._build_table_8_8(release_inputs.scored_records, release_inputs.study_rows)

    assert len(calls) == 32
    expected_sets = Counter(_scored_ids_by_stratum(release_inputs).values())
    actual_sets = Counter(call[0] for call in calls)
    assert actual_sets == Counter({ids: multiplicity * 4 for ids, multiplicity in expected_sets.items()})
    assert {(n_boot, seed, alpha) for _ids, n_boot, seed, alpha in calls} == {(10_000, 42, 0.05)}


def test_table_8_8_boundaries_and_degenerate_bins(secondary_outputs, release_inputs) -> None:
    table = secondary_outputs.tables["table-8-8.csv"]
    assert [_value(table, row_key, "n") for row_key in STRATUM_ROW_KEYS] == [385, 25, 105, 220, 85, 56, 191, 163]
    for column in (
        "rule_id_c0",
        "rule_id_c1_minus_c0_point",
        "rule_id_c1_minus_c0_low",
        "rule_id_c1_minus_c0_high",
        "rule_id_c2_minus_c0_point",
        "rule_id_c2_minus_c0_low",
        "rule_id_c2_minus_c0_high",
    ):
        assert _value(table, "violations_2_plus", column) == pytest.approx(0.0)
    assert _value(table, "length_200_plus", "rule_id_c1_minus_c0_point") == pytest.approx(0.0, abs=1e-14)

    scored_rows = _scored_study_rows(release_inputs)
    assert sum(row.source_line_count == 50 for row in scored_rows) > 0
    assert sum(row.source_line_count == 199 for row in scored_rows) > 0
    assert sum(row.serializer.maximum_depth == 1 for row in scored_rows) > 0
    assert sum(row.serializer.maximum_depth == 2 for row in scored_rows) > 0
    assert sum(row.serializer.maximum_depth == 3 for row in scored_rows) > 0
    for row_key, _stratum, _bin, predicate in STRATUM_SPECS:
        assert _value(table, row_key, "n") == sum(predicate(row) for row in scored_rows)


@pytest.mark.parametrize(
    ("field", "value", "source_row", "destination_row"),
    [
        ("source_line_count", 50, "length_lt_50", "length_50_199"),
        ("source_line_count", 200, "length_50_199", "length_200_plus"),
        ("maximum_depth", 2, "depth_0_1", "depth_2_3"),
        ("maximum_depth", 4, "depth_2_3", "depth_gt_3"),
    ],
)
def test_table_8_8_boundary_mutations_move_one_snippet(
    release_inputs,
    field: str,
    value: int,
    source_row: str,
    destination_row: str,
) -> None:
    rows = list(release_inputs.study_rows)
    scored_ids = _scored_ids_by_stratum(release_inputs)
    candidate_id = next(iter(scored_ids[source_row] - scored_ids[destination_row]))
    row_index = next(index for index, row in enumerate(rows) if row.base_snippet_id == candidate_id)
    row = rows[row_index]
    if field == "source_line_count":
        rows[row_index] = row.model_copy(update={field: value})
    else:
        rows[row_index] = row.model_copy(
            update={"serializer": row.serializer.model_copy(update={"maximum_depth": value})}
        )

    table = tables._build_table_8_8(release_inputs.scored_records, tuple(rows))
    assert _value(table, source_row, "n") == len(scored_ids[source_row]) - 1
    assert _value(table, destination_row, "n") == len(scored_ids[destination_row]) + 1


def test_table_8_8_rejects_zero_violation_and_broken_record_grids(release_inputs) -> None:
    rows = list(release_inputs.study_rows)
    scored_id = _scored_study_rows(release_inputs)[0].base_snippet_id
    row_index = next(index for index, row in enumerate(rows) if row.base_snippet_id == scored_id)
    rows[row_index] = rows[row_index].model_copy(update={"violation_count": 0})
    with pytest.raises(ValueError, match="violation|stratum|bin"):
        tables._build_table_8_8(release_inputs.scored_records, tuple(rows))

    records = list(release_inputs.scored_records)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.condition == "C2" and record.seed == 44 and record.task_type == "joint"
    )
    duplicate = (*records, records[record_index])
    with pytest.raises(ValueError, match="duplicate|grid"):
        tables._build_table_8_8(duplicate, release_inputs.study_rows)
    records.pop(record_index)
    with pytest.raises(ValueError, match="grid|missing|join"):
        tables._build_table_8_8(tuple(records), release_inputs.study_rows)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_table_8_8_requires_the_exact_410_scored_study_ids(release_inputs, mutation: str) -> None:
    rows = list(release_inputs.study_rows)
    scored_id = _scored_study_rows(release_inputs)[0].base_snippet_id
    row_index = next(index for index, row in enumerate(rows) if row.base_snippet_id == scored_id)
    if mutation == "missing":
        rows.pop(row_index)
    else:
        rows.append(rows[row_index])

    with pytest.raises(ValueError, match="410|duplicate|grid|join"):
        tables._build_table_8_8(release_inputs.scored_records, tuple(rows))


def test_table_8_8_is_input_order_invariant(release_inputs) -> None:
    expected = tables._build_table_8_8(release_inputs.scored_records, release_inputs.study_rows)
    actual = tables._build_table_8_8(
        tuple(reversed(release_inputs.scored_records)),
        tuple(reversed(release_inputs.study_rows)),
    )
    assert actual == expected


def test_table_8_9_familiarity_uses_per_check_f1_and_c0_seed_means(secondary_outputs, release_inputs) -> None:
    table = secondary_outputs.tables["table-8-9.csv"]
    records = release_inputs.scored_records
    supports = Counter(check for record in _rule_records(records, "C0", 42) for check in record.gold)
    raw = _per_check_f1(_rule_records(records, "zero-shot-raw", 42))
    syntax = _per_check_f1(_rule_records(records, "zero-shot-syntax", 42))
    c0 = _seed_mean_per_check_f1(records, "C0")

    for check, row_key in CHECK_ROWS:
        assert _cell(table, row_key, "check") == check
        expected = {
            "n": supports[check],
            "zero_shot_raw": raw[check],
            "zero_shot_syntax": syntax[check],
            "fine_tuned_c0": c0[check],
        }
        for column, value in expected.items():
            cell = _output_cell(table, row_key, column)
            assert cell.value == pytest.approx(value)
            assert cell.result_id == f"table_8_9.{row_key}.{column}"
            assert cell.display_digits == (0 if column == "n" else 1)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "gold_mismatch"])
def test_table_8_9_rejects_broken_condition_grids(release_inputs, mutation: str) -> None:
    records = list(release_inputs.scored_records)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.condition == "zero-shot-syntax" and record.task_type == "rule_identification"
    )
    if mutation == "missing":
        records.pop(record_index)
    elif mutation == "duplicate":
        records.append(records[record_index])
    else:
        record = cast(RuleIdentificationRecord, records[record_index])
        replacement = next(check for check in CHECKS if check not in record.gold)
        records[record_index] = record.model_copy(update={"gold": (*record.gold, replacement)})

    with pytest.raises(ValueError, match="grid|duplicate|gold|join"):
        tables._build_table_8_9(tuple(records))


def test_table_8_9_is_input_order_invariant(release_inputs) -> None:
    expected = tables._build_table_8_9(release_inputs.scored_records)
    actual = tables._build_table_8_9(tuple(reversed(release_inputs.scored_records)))
    assert actual == expected


def _zero_shot_transition_expected(records: Sequence[ReleasedRecord]) -> dict[str, int | float]:
    maps: dict[Condition, dict[str, bool]] = {"zero-shot-raw": {}, "zero-shot-syntax": {}}
    for record in records:
        if record.condition not in maps or record.seed != 42 or record.task_type != "joint":
            continue
        selected = cast(JointRecord, record)
        assert selected.base_snippet_id not in maps[selected.condition]
        maps[selected.condition][selected.base_snippet_id] = selected.outcome.overall_fixed
    assert set(maps["zero-shot-raw"]) == set(maps["zero-shot-syntax"])
    assert len(maps["zero-shot-raw"]) == 410
    pairs = [
        (maps["zero-shot-raw"][snippet_id], maps["zero-shot-syntax"][snippet_id])
        for snippet_id in sorted(maps["zero-shot-raw"])
    ]
    favor_syntax = sum(not raw and syntax for raw, syntax in pairs)
    favor_raw = sum(raw and not syntax for raw, syntax in pairs)
    net_files = favor_syntax - favor_raw
    return {
        "section_8_10.zero_shot_transition.changed": favor_syntax + favor_raw,
        "section_8_10.zero_shot_transition.favor_syntax": favor_syntax,
        "section_8_10.zero_shot_transition.favor_raw": favor_raw,
        "section_8_10.zero_shot_transition.net_files": net_files,
        "section_8_10.zero_shot_transition.net_points": net_files / len(pairs) * 100.0,
    }


def test_zero_shot_transitions_use_matched_joint_overall_fix(release_inputs) -> None:
    expected = _zero_shot_transition_expected(release_inputs.scored_records)
    actual = tables._build_zero_shot_transitions(release_inputs.scored_records)
    assert set(actual) == set(expected)
    for result_id, value in expected.items():
        assert actual[result_id] == pytest.approx(value)
    assert expected == {
        "section_8_10.zero_shot_transition.changed": 143,
        "section_8_10.zero_shot_transition.favor_syntax": 59,
        "section_8_10.zero_shot_transition.favor_raw": 84,
        "section_8_10.zero_shot_transition.net_files": -25,
        "section_8_10.zero_shot_transition.net_points": pytest.approx(-25 / 410 * 100.0),
    }


def test_zero_shot_transition_direction_flip_updates_all_derived_counts(release_inputs) -> None:
    records = list(release_inputs.scored_records)
    joint_by_key = {
        (record.condition, record.base_snippet_id): index
        for index, record in enumerate(records)
        if record.condition in {"zero-shot-raw", "zero-shot-syntax"}
        and record.seed == 42
        and record.task_type == "joint"
    }
    snippet_id = next(
        snippet_id
        for condition, snippet_id in joint_by_key
        if condition == "zero-shot-raw"
        and not cast(JointRecord, records[joint_by_key[("zero-shot-raw", snippet_id)]]).outcome.overall_fixed
        and cast(JointRecord, records[joint_by_key[("zero-shot-syntax", snippet_id)]]).outcome.overall_fixed
    )
    for condition, fixed in (("zero-shot-raw", True), ("zero-shot-syntax", False)):
        index = joint_by_key[(condition, snippet_id)]
        record = cast(JointRecord, records[index])
        records[index] = record.model_copy(
            update={"outcome": record.outcome.model_copy(update={"overall_fixed": fixed})}
        )

    expected = _zero_shot_transition_expected(tuple(records))
    actual = tables._build_zero_shot_transitions(tuple(records))
    assert actual == pytest.approx(expected)
    assert actual["section_8_10.zero_shot_transition.changed"] == (
        actual["section_8_10.zero_shot_transition.favor_syntax"] + actual["section_8_10.zero_shot_transition.favor_raw"]
    )
    assert actual["section_8_10.zero_shot_transition.net_files"] == (
        actual["section_8_10.zero_shot_transition.favor_syntax"] - actual["section_8_10.zero_shot_transition.favor_raw"]
    )
    assert actual["section_8_10.zero_shot_transition.net_points"] == pytest.approx(
        actual["section_8_10.zero_shot_transition.net_files"] / 410 * 100.0
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_zero_shot_transitions_reject_unmatched_or_duplicate_ids(release_inputs, mutation: str) -> None:
    records = list(release_inputs.scored_records)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.condition == "zero-shot-raw" and record.seed == 42 and record.task_type == "joint"
    )
    if mutation == "missing":
        records.pop(record_index)
    else:
        records.append(records[record_index])
    with pytest.raises(ValueError, match="duplicate|ID|identifier|join|set"):
        tables._build_zero_shot_transitions(tuple(records))


def test_rq3_registry_values_use_existing_fit_and_risk_contrasts(release_inputs) -> None:
    actual = tables._build_rq3_results(release_inputs.scored_records)
    frame = build_rq3_frame(release_inputs.scored_records)
    fit = fit_rq3_interaction(frame)
    contrasts = rq3_task_contrasts(release_inputs.scored_records, n_boot=10_000, seed=42)
    by_reference = {contrast.reference_task: contrast.interval for contrast in contrasts}
    expected = {
        "rq3.vb.c1_joint_interaction.coefficient": fit.coefficient,
        "rq3.vb.c1_joint_interaction.low": fit.ci_low,
        "rq3.vb.c1_joint_interaction.high": fit.ci_high,
        "rq3.vb.c1_joint_interaction.sd": fit.sd,
        "rq3.risk.joint_minus_correction.point": by_reference["correction"].point * 100.0,
        "rq3.risk.joint_minus_correction.low": by_reference["correction"].ci_low * 100.0,
        "rq3.risk.joint_minus_correction.high": by_reference["correction"].ci_high * 100.0,
        "rq3.risk.joint_minus_rule.point": by_reference["rule_identification"].point * 100.0,
        "rq3.risk.joint_minus_rule.low": by_reference["rule_identification"].ci_low * 100.0,
        "rq3.risk.joint_minus_rule.high": by_reference["rule_identification"].ci_high * 100.0,
        "rq3.h3_supported": all(interval.ci_low > 0.0 for interval in by_reference.values()),
    }
    assert set(actual) == set(expected)
    for result_id, value in expected.items():
        if isinstance(value, bool):
            assert actual[result_id] is value
        else:
            assert actual[result_id] == pytest.approx(value)
    assert actual["rq3.h3_supported"] is False


@pytest.mark.parametrize(
    ("correction_low", "rule_low", "supported"),
    [(0.1, -0.1, False), (-0.1, 0.1, False), (0.1, 0.1, True)],
)
def test_rq3_uses_supplied_records_fixed_contrasts_and_both_risk_lower_bounds(
    monkeypatch,
    release_inputs,
    correction_low: float,
    rule_low: float,
    supported: bool,
) -> None:
    supplied = release_inputs.scored_records
    frame_marker = object()
    calls: list[tuple[str, int, int] | tuple[str]] = []

    def fake_frame(records):
        assert records is supplied
        calls.append(("frame",))
        return frame_marker

    def fake_fit(frame):
        assert frame is frame_marker
        calls.append(("fit",))
        return Rq3Result(7.0, 0.5, 6.0, 8.0, True, True, "synthetic-term")

    def fake_contrasts(records, *, n_boot: int, seed: int):
        assert records is supplied
        calls.append(("contrasts", n_boot, seed))
        return (
            Rq3TaskContrast(
                "joint_minus_correction",
                "C1",
                "C0",
                "joint",
                "correction",
                Interval(0.2, correction_low, 0.3, 0.0, n_boot, seed, 0.05, 410),
                correction_low > 0.0,
            ),
            Rq3TaskContrast(
                "joint_minus_rule_identification",
                "C1",
                "C0",
                "joint",
                "rule_identification",
                Interval(0.2, rule_low, 0.3, 0.0, n_boot, seed, 0.05, 410),
                rule_low > 0.0,
            ),
        )

    monkeypatch.setattr(tables, "build_rq3_frame", fake_frame, raising=False)
    monkeypatch.setattr(tables, "fit_rq3_interaction", fake_fit, raising=False)
    monkeypatch.setattr(tables, "rq3_task_contrasts", fake_contrasts, raising=False)

    actual = tables._build_rq3_results(supplied)
    assert calls == [("frame",), ("fit",), ("contrasts", 10_000, 42)]
    assert actual["rq3.vb.c1_joint_interaction.coefficient"] == 7.0
    assert actual["rq3.h3_supported"] is supported


def _section_8_10_expected(records: Sequence[ReleasedRecord]) -> dict[str, int | float]:
    grid = _success_grid(records, task="rule_identification")
    snippet_ids = sorted({snippet_id for condition, _seed, snippet_id in grid if condition == "C0"})
    c0_all_seed = {
        snippet_id for snippet_id in snippet_ids if all(grid[("C0", seed, snippet_id)] == 1.0 for seed in (42, 43, 44))
    }
    c2_only_seeds = {
        snippet_id: {
            seed
            for seed in (42, 43, 44)
            if grid[("C2", seed, snippet_id)] == 1.0
            and grid[("C0", seed, snippet_id)] == 0.0
            and grid[("C1", seed, snippet_id)] == 0.0
        }
        for snippet_id in snippet_ids
    }
    category_b = [
        cast(JointRecord, record)
        for record in records
        if record.condition == "C2"
        and record.task_type == "joint"
        and cast(JointRecord, record).outcome.category == "B"
    ]
    return {
        "section_8_10.c0_all_seed_correct.count": len(c0_all_seed),
        "section_8_10.c0_all_seed_correct.percent": len(c0_all_seed) / len(snippet_ids) * 100.0,
        "section_8_10.c2_only.any_seed.files": sum(bool(seeds) for seeds in c2_only_seeds.values()),
        "section_8_10.c2_only.at_least_two_seeds.files": sum(len(seeds) >= 2 for seeds in c2_only_seeds.values()),
        "section_8_10.category_b.files": len({record.base_snippet_id for record in category_b}),
        "section_8_10.category_b.runs": len(category_b),
    }


def test_section_8_10_record_counts_use_exact_predicates(release_inputs) -> None:
    expected = _section_8_10_expected(release_inputs.scored_records)
    actual = tables._build_section_8_10_record_results(release_inputs.scored_records)
    assert actual == pytest.approx(expected)
    assert expected == {
        "section_8_10.c0_all_seed_correct.count": 206,
        "section_8_10.c0_all_seed_correct.percent": pytest.approx(206 / 410 * 100.0),
        "section_8_10.c2_only.any_seed.files": 20,
        "section_8_10.c2_only.at_least_two_seeds.files": 1,
        "section_8_10.category_b.files": 1,
        "section_8_10.category_b.runs": 2,
    }


def test_section_8_10_c0_all_seed_count_is_an_intersection(release_inputs) -> None:
    records = list(release_inputs.scored_records)
    grid = _success_grid(records, task="rule_identification")
    snippet_id = next(
        snippet_id
        for condition, _seed, snippet_id in grid
        if condition == "C0" and all(grid[("C0", seed, snippet_id)] == 1.0 for seed in (42, 43, 44))
    )
    baseline = tables._build_section_8_10_record_results(tuple(records))
    record_index = _rule_record_index(records, "C0", 42, snippet_id)
    records[record_index] = _flip_rule_exactness(records[record_index])
    actual = tables._build_section_8_10_record_results(tuple(records))

    assert actual["section_8_10.c0_all_seed_correct.count"] == baseline["section_8_10.c0_all_seed_correct.count"] - 1
    assert actual["section_8_10.c0_all_seed_correct.percent"] == pytest.approx(
        baseline["section_8_10.c0_all_seed_correct.percent"] - 100.0 / 410
    )


def test_section_8_10_c2_only_at_least_two_threshold_keeps_any_seed(release_inputs) -> None:
    records = list(release_inputs.scored_records)
    grid = _success_grid(records, task="rule_identification")
    snippet_ids = {snippet_id for condition, _seed, snippet_id in grid if condition == "C2"}
    qualifying = {
        snippet_id: tuple(
            seed
            for seed in (42, 43, 44)
            if grid[("C2", seed, snippet_id)] == 1.0
            and grid[("C0", seed, snippet_id)] == 0.0
            and grid[("C1", seed, snippet_id)] == 0.0
        )
        for snippet_id in snippet_ids
    }
    snippet_id, seeds = next((snippet_id, seeds) for snippet_id, seeds in qualifying.items() if len(seeds) == 2)
    baseline = tables._build_section_8_10_record_results(tuple(records))
    record_index = _rule_record_index(records, "C2", seeds[0], snippet_id)
    records[record_index] = _flip_rule_exactness(records[record_index])
    actual = tables._build_section_8_10_record_results(tuple(records))

    assert actual["section_8_10.c2_only.any_seed.files"] == baseline["section_8_10.c2_only.any_seed.files"]
    assert (
        actual["section_8_10.c2_only.at_least_two_seeds.files"]
        == baseline["section_8_10.c2_only.at_least_two_seeds.files"] - 1
    )

    three_seed_records = list(release_inputs.scored_records)
    third_seed = next(seed for seed in (42, 43, 44) if seed not in seeds)
    c2_index = _rule_record_index(three_seed_records, "C2", third_seed, snippet_id)
    c0_index = _rule_record_index(three_seed_records, "C0", third_seed, snippet_id)
    c1_index = _rule_record_index(three_seed_records, "C1", third_seed, snippet_id)
    three_seed_records[c2_index] = _make_rule_exact(three_seed_records[c2_index])
    three_seed_records[c0_index] = _make_rule_non_exact(three_seed_records[c0_index])
    three_seed_records[c1_index] = _make_rule_non_exact(three_seed_records[c1_index])
    three_seed = tables._build_section_8_10_record_results(tuple(three_seed_records))
    assert three_seed["section_8_10.c2_only.any_seed.files"] == baseline["section_8_10.c2_only.any_seed.files"]
    assert (
        three_seed["section_8_10.c2_only.at_least_two_seeds.files"]
        == baseline["section_8_10.c2_only.at_least_two_seeds.files"]
    )


def _rule_record_index(
    records: Sequence[ReleasedRecord],
    condition: Condition,
    seed: int,
    snippet_id: str,
) -> int:
    return next(
        index
        for index, record in enumerate(records)
        if record.condition == condition
        and record.seed == seed
        and record.base_snippet_id == snippet_id
        and record.task_type == "rule_identification"
    )


def _make_rule_exact(record: ReleasedRecord) -> RuleIdentificationRecord:
    rule = cast(RuleIdentificationRecord, record)
    return rule.model_copy(update={"pred": rule.gold, "exact_match": True})


def _make_rule_non_exact(record: ReleasedRecord) -> RuleIdentificationRecord:
    rule = cast(RuleIdentificationRecord, record)
    return rule.model_copy(update={"pred": (), "exact_match": False})


def _flip_rule_exactness(record: ReleasedRecord) -> RuleIdentificationRecord:
    rule = cast(RuleIdentificationRecord, record)
    if rule.exact_match:
        return _make_rule_non_exact(rule)
    return _make_rule_exact(rule)


@pytest.mark.parametrize("blocking_condition", ["C0", "C1"])
def test_section_8_10_c2_only_ignores_control_but_requires_c0_and_c1_failures(
    release_inputs,
    blocking_condition: Condition,
) -> None:
    records = list(release_inputs.scored_records)
    grid = _success_grid(records, task="rule_identification")
    snippet_id, seed = next(
        (snippet_id, seed)
        for condition, seed, snippet_id in grid
        if condition == "C2"
        and grid[("C2", seed, snippet_id)] == 1.0
        and grid[("C0", seed, snippet_id)] == 0.0
        and grid[("C1", seed, snippet_id)] == 0.0
        and sum(
            grid[("C2", candidate_seed, snippet_id)] == 1.0
            and grid[("C0", candidate_seed, snippet_id)] == 0.0
            and grid[("C1", candidate_seed, snippet_id)] == 0.0
            for candidate_seed in (42, 43, 44)
        )
        == 1
    )
    baseline = tables._build_section_8_10_record_results(tuple(records))

    control_index = _rule_record_index(records, "C2-control", seed, snippet_id)
    records[control_index] = _flip_rule_exactness(records[control_index])
    assert tables._build_section_8_10_record_results(tuple(records)) == baseline

    blocking_index = _rule_record_index(records, blocking_condition, seed, snippet_id)
    records[blocking_index] = _make_rule_exact(records[blocking_index])
    mutated = tables._build_section_8_10_record_results(tuple(records))
    assert mutated["section_8_10.c2_only.any_seed.files"] == baseline["section_8_10.c2_only.any_seed.files"] - 1


@pytest.mark.parametrize(("condition", "task_type"), [("C1", "joint"), ("C2", "correction")])
def test_section_8_10_category_b_counts_only_c2_joint_records(
    release_inputs,
    condition: Condition,
    task_type: str,
) -> None:
    records = list(release_inputs.scored_records)
    baseline = tables._build_section_8_10_record_results(tuple(records))
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.condition == condition
        and record.task_type == task_type
        and cast(CorrectionRecord | JointRecord, record).outcome.category != "B"
    )
    record = cast(CorrectionRecord | JointRecord, records[record_index])
    records[record_index] = record.model_copy(update={"outcome": record.outcome.model_copy(update={"category": "B"})})
    actual = tables._build_section_8_10_record_results(tuple(records))
    assert actual["section_8_10.category_b.files"] == baseline["section_8_10.category_b.files"]
    assert actual["section_8_10.category_b.runs"] == baseline["section_8_10.category_b.runs"]


def _similarity_sensitivity_expected(records: Sequence[ReleasedRecord]) -> dict[str, int | float]:
    successes = [
        cast(CorrectionRecord | JointRecord, record)
        for record in records
        if record.condition in FINE_TUNED_CONDITIONS
        and record.task_type in {"correction", "joint"}
        and cast(CorrectionRecord | JointRecord, record).outcome.overall_fixed
    ]
    retained = [record for record in successes if record.sensitivity_class is not None]
    assert all(record.extracted_similarity is not None for record in retained)
    assert all((record.extracted_similarity is None) == (record.sensitivity_class is None) for record in successes)
    condition_counts = Counter(record.condition for record in retained)
    joint_retained = Counter(record.condition for record in retained if record.task_type == "joint")
    joint_denominators = Counter(
        record.condition
        for record in records
        if record.condition in FINE_TUNED_CONDITIONS and record.task_type == "joint"
    )
    return {
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
    }


def test_similarity_sensitivity_uses_only_successful_released_tail_indicators(
    secondary_outputs,
    release_inputs,
) -> None:
    expected = _similarity_sensitivity_expected(release_inputs.scored_records)
    actual = {result_id: secondary_outputs.results[result_id] for result_id in expected}
    for result_id, value in expected.items():
        assert actual[result_id] == pytest.approx(value)
    assert expected["sensitivity.extracted_code.scored_fix_successes"] == 4_782
    assert expected["sensitivity.extracted_code.low_similarity_outputs"] == 139
    assert expected["sensitivity.extracted_code.low_similarity_shared_function"] == 117
    assert expected["sensitivity.extracted_code.low_similarity_package_only"] == 22
    assert expected["sensitivity.extracted_code.unique_snippets"] == 18
    assert [expected[f"sensitivity.extracted_code.condition.{name}"] for name in ("c0", "c1", "c2", "c2_control")] == [
        30,
        38,
        39,
        32,
    ]
    assert expected["sensitivity.extracted_code.adjustment.c1_minus_c0.maximum_points"] == pytest.approx(
        6 / 1230 * 100.0
    )
    assert expected["sensitivity.extracted_code.adjustment.c2_minus_c1.maximum_points"] == pytest.approx(
        1 / 1230 * 100.0
    )
    assert expected["sensitivity.extracted_code.adjustment.c2_minus_c2_control.maximum_points"] == pytest.approx(
        7 / 1230 * 100.0
    )
    assert "sensitivity.extracted_code.cross_package_substitutions" not in secondary_outputs.results


def test_similarity_sensitivity_ignores_failed_indicator_records(release_inputs) -> None:
    records = list(release_inputs.scored_records)
    baseline = tables._build_remaining_sensitivity_results(records)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.condition in FINE_TUNED_CONDITIONS
        and record.task_type == "joint"
        and not cast(JointRecord, record).outcome.overall_fixed
        and cast(JointRecord, record).sensitivity_class is None
    )
    record = cast(JointRecord, records[record_index])
    records[record_index] = record.model_copy(
        update={"extracted_similarity": 0.1, "sensitivity_class": "same_file_truncated"}
    )
    actual = tables._build_remaining_sensitivity_results(tuple(records))
    for result_id in _similarity_sensitivity_expected(release_inputs.scored_records):
        assert actual[result_id] == baseline[result_id]


def test_similarity_adjustments_use_joint_records_only(release_inputs) -> None:
    records = list(release_inputs.scored_records)
    baseline = _similarity_sensitivity_expected(records)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.condition in FINE_TUNED_CONDITIONS
        and record.task_type == "correction"
        and cast(CorrectionRecord, record).outcome.overall_fixed
        and cast(CorrectionRecord, record).sensitivity_class is not None
    )
    record = cast(CorrectionRecord, records[record_index])
    records[record_index] = record.model_copy(update={"extracted_similarity": None, "sensitivity_class": None})
    actual = tables._build_remaining_sensitivity_results(tuple(records))
    expected = _similarity_sensitivity_expected(records)
    for result_id, value in expected.items():
        assert actual[result_id] == pytest.approx(value)
    adjustment_ids = {
        result_id for result_id in SECONDARY_JSON_IDS if result_id.startswith("sensitivity.extracted_code.adjustment.")
    }
    assert {result_id: actual[result_id] for result_id in adjustment_ids} == {
        result_id: baseline[result_id] for result_id in adjustment_ids
    }


def test_similarity_c1_joint_retained_hit_controls_count_and_adjustment(release_inputs) -> None:
    records = list(release_inputs.scored_records)
    baseline = _similarity_sensitivity_expected(records)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.condition == "C1"
        and record.task_type == "joint"
        and cast(JointRecord, record).outcome.overall_fixed
        and cast(JointRecord, record).sensitivity_class is not None
    )
    record = cast(JointRecord, records[record_index])
    records[record_index] = record.model_copy(update={"extracted_similarity": None, "sensitivity_class": None})
    actual = tables._build_remaining_sensitivity_results(tuple(records))
    expected = _similarity_sensitivity_expected(records)
    for result_id, value in expected.items():
        assert actual[result_id] == pytest.approx(value)
    assert (
        expected["sensitivity.extracted_code.condition.c1"] == baseline["sensitivity.extracted_code.condition.c1"] - 1
    )
    assert expected["sensitivity.extracted_code.adjustment.c1_minus_c0.maximum_points"] == pytest.approx(
        baseline["sensitivity.extracted_code.adjustment.c1_minus_c0.maximum_points"] - 100.0 / 1230
    )


def test_familiarity_results_use_per_check_ranges_and_unweighted_macro_mean(
    secondary_outputs,
    release_inputs,
) -> None:
    records = release_inputs.scored_records
    raw_records = _rule_records(records, "zero-shot-raw", 42)
    syntax_records = _rule_records(records, "zero-shot-syntax", 42)
    raw_per_check = _per_check_f1(raw_records)
    syntax_per_check = _per_check_f1(syntax_records)
    zero_shot_values = tuple(raw_per_check.values()) + tuple(syntax_per_check.values())
    c0_per_check = _seed_mean_per_check_f1(records, "C0")
    raw_macro = macro_f1([record.pred for record in raw_records], [record.gold for record in raw_records], CHECKS)
    syntax_macro = macro_f1(
        [record.pred for record in syntax_records],
        [record.gold for record in syntax_records],
        CHECKS,
    )
    expected = {
        "sensitivity.familiarity.zero_shot.minimum_percent": round(min(zero_shot_values)),
        "sensitivity.familiarity.zero_shot.maximum_percent": round(max(zero_shot_values)),
        "sensitivity.familiarity.c0.minimum_percent": round(min(c0_per_check.values())),
        "sensitivity.familiarity.c0.maximum_percent": round(max(c0_per_check.values())),
        "sensitivity.familiarity.zero_shot.overall_approx_percent": round(
            statistics.mean((raw_macro, syntax_macro)) * 100.0
        ),
    }
    for result_id, value in expected.items():
        assert secondary_outputs.results[result_id] == value
    assert tuple(expected.values()) == (14, 26, 39, 92, 23)


def test_task_gap_and_multilabel_formulas(secondary_outputs, release_inputs) -> None:
    records = release_inputs.scored_records
    c1_rates = {}
    for task_type in ("correction", "joint"):
        selected = [
            cast(CorrectionRecord | JointRecord, record)
            for record in records
            if record.condition == "C1" and record.task_type == task_type
        ]
        c1_rates[task_type] = sum(record.outcome.overall_fixed for record in selected) / len(selected)
    c1_successes = [
        sum(
            cast(CorrectionRecord | JointRecord, record).outcome.overall_fixed
            for record in records
            if record.condition == "C1" and record.task_type == task
        )
        for task in ("correction", "joint")
    ]
    assert c1_successes == [690, 513]
    assert secondary_outputs.results["sensitivity.task_rate.c1.correction_minus_joint.approx_points"] == round(
        (c1_rates["correction"] - c1_rates["joint"]) * 100.0
    )

    multilabel = [
        cast(RuleIdentificationRecord, record)
        for record in records
        if record.task_type == "rule_identification" and len(cast(RuleIdentificationRecord, record).gold) > 1
    ]
    assert len(multilabel) == 350
    assert {record.condition for record in multilabel} == {
        "C0",
        "C1",
        "C2",
        "C2-control",
        "zero-shot-raw",
        "zero-shot-syntax",
    }
    expected_multilabel = sum(record.exact_match for record in multilabel) / len(multilabel) * 100.0
    assert expected_multilabel == 0.0
    assert secondary_outputs.results["sensitivity.multilabel.rule_exact.all_conditions_percent"] == expected_multilabel


def test_c1_task_gap_recomputes_across_rounding_boundary(release_inputs) -> None:
    records = list(release_inputs.scored_records)
    success_indices = [
        index
        for index, record in enumerate(records)
        if record.condition == "C1"
        and record.task_type == "correction"
        and cast(CorrectionRecord, record).outcome.overall_fixed
    ][:12]
    assert len(success_indices) == 12
    for index in success_indices:
        record = cast(CorrectionRecord, records[index])
        records[index] = record.model_copy(
            update={"outcome": record.outcome.model_copy(update={"overall_fixed": False})}
        )

    actual = tables._build_remaining_sensitivity_results(tuple(records))
    assert actual["sensitivity.task_rate.c1.correction_minus_joint.approx_points"] == 13


def test_multilabel_rule_exact_recomputes_from_all_six_conditions(release_inputs) -> None:
    records = list(release_inputs.scored_records)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.task_type == "rule_identification"
        and len(cast(RuleIdentificationRecord, record).gold) > 1
        and not cast(RuleIdentificationRecord, record).exact_match
    )
    record = cast(RuleIdentificationRecord, records[record_index])
    records[record_index] = record.model_copy(update={"pred": record.gold, "exact_match": True})

    actual = tables._build_remaining_sensitivity_results(tuple(records))
    assert actual["sensitivity.multilabel.rule_exact.all_conditions_percent"] == pytest.approx(100.0 / 350)
