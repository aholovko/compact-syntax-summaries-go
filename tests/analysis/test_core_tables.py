from __future__ import annotations

import statistics
from collections import Counter
from types import SimpleNamespace
from typing import Literal

import pytest

import analysis.tables as tables
from analysis.inputs import (
    CHECKS,
    TASK_TYPES,
    Condition,
    CorrectionRecord,
    JointRecord,
    RepairOutcome,
    RuleIdentificationRecord,
    TrainingContribution,
)
from analysis.metrics import task_contrast as compute_task_contrast
from analysis.tables import GeneratedOutputs, OutputCell, TableData, build_outputs

TestSeed = Literal[42, 43, 44]


def _snippet(index: int) -> str:
    return f"sha256:{index:064x}"


def _outcome(
    *,
    extracted: bool = True,
    extraction_status: Literal["go_block", "fenced_block", "largest_parseable", "failed"] = "go_block",
    parse_ok: bool = True,
    target_fixed: bool = True,
    overall_fixed: bool = True,
    category: Literal["A", "B", "C", "D", "INVALID"] = "A",
) -> RepairOutcome:
    return RepairOutcome(
        target_fixed=target_fixed,
        overall_fixed=overall_fixed,
        studied_regression=False,
        enabled_regression=False,
        extracted=extracted,
        extraction_status=extraction_status,
        parse_ok=parse_ok,
        lint_ok=parse_ok,
        original_tool_status="ok",
        output_tool_status="ok" if parse_ok else "load_failed",
        build_status="OK" if parse_ok else "NA",
        category=category,
        introduced_checks=(),
        residual_findings=(),
    )


def _rule_record(
    index: int,
    condition: Condition,
    seed: TestSeed,
    *,
    pred: tuple[str, ...],
    gold: tuple[str, ...],
) -> RuleIdentificationRecord:
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
        gold=gold,
        pred=pred,
        rejected_label_count=0,
        exact_match=set(pred) == set(gold),
        n_emitted=len(pred),
        normalization_status="recognized_array",
    )


def _repair_record(
    index: int,
    condition: Condition,
    seed: TestSeed,
    task: Literal["correction", "joint"],
    *,
    outcome: RepairOutcome,
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
        "outcome": outcome,
        "extracted_similarity": 1.0 if outcome.extracted else None,
        "sensitivity_class": None,
    }
    if task == "correction":
        return CorrectionRecord(task_type=task, **common)
    return JointRecord(task_type=task, **common)


def _table_8_1_records() -> tuple[RuleIdentificationRecord | CorrectionRecord | JointRecord, ...]:
    records: list[RuleIdentificationRecord | CorrectionRecord | JointRecord] = []
    for condition in ("C0", "C1", "C2", "C2-control", "zero-shot-raw", "zero-shot-syntax"):
        seeds = (42,) if condition.startswith("zero-shot") else (42, 43, 44)
        for seed in seeds:
            rule_hits = 1 if condition.startswith("zero-shot") else seed - 42
            for index, label in ((1, "assignOp"), (2, "builtinShadow")):
                pred = (label,) if index <= rule_hits else ()
                records.append(_rule_record(index, condition, seed, pred=pred, gold=(label,)))
            correction_hits = 1 if condition.startswith("zero-shot") else seed - 42
            joint_hits = 2
            for index in (1, 2):
                records.append(
                    _repair_record(
                        index,
                        condition,
                        seed,
                        "correction",
                        outcome=_outcome(
                            target_fixed=index <= correction_hits,
                            overall_fixed=index <= correction_hits,
                        ),
                    )
                )
                records.append(
                    _repair_record(
                        index,
                        condition,
                        seed,
                        "joint",
                        outcome=_outcome(target_fixed=index <= joint_hits, overall_fixed=index <= joint_hits),
                    )
                )
    return tuple(records)


def _table_8_2_records() -> tuple[CorrectionRecord | JointRecord, ...]:
    outcomes = (
        _outcome(),
        _outcome(extraction_status="fenced_block", target_fixed=False, overall_fixed=False),
        _outcome(extraction_status="largest_parseable", overall_fixed=False),
        _outcome(
            extracted=False,
            extraction_status="failed",
            parse_ok=False,
            target_fixed=False,
            overall_fixed=False,
            category="INVALID",
        ),
    )
    return tuple(
        _repair_record(index, condition, seed, task, outcome=outcome)
        for condition in ("C0", "C1", "C2", "C2-control")
        for seed in (42, 43, 44)
        for task in ("correction", "joint")
        for index, outcome in enumerate(outcomes, start=1)
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
def generated_outputs(release_inputs) -> GeneratedOutputs:
    return _build_from(release_inputs)


def _cell(table: TableData, row_key: str, column: str) -> OutputCell | str:
    row = next(row for row in table.rows if row.key == row_key)
    return row.cells[column]


def _value(table: TableData, row_key: str, column: str) -> bool | int | float | str | None:
    cell = _cell(table, row_key, column)
    assert isinstance(cell, OutputCell)
    return cell.value


def test_table_8_1_hand_calculated() -> None:
    table = tables._build_table_8_1(_table_8_1_records())

    assert table.filename == "table-8-1.csv"
    assert tuple(row.key for row in table.rows) == (
        "zero_shot_raw",
        "zero_shot_syntax",
        "c0",
        "c1",
        "c2",
        "c2_control",
    )
    assert _value(table, "zero_shot_raw", "rule_id_exact_match_mean") == pytest.approx(50.0)
    assert _value(table, "zero_shot_raw", "rule_id_micro_f1_mean") == pytest.approx(200.0 / 3.0)
    assert _value(table, "zero_shot_raw", "rule_id_macro_f1_mean") == pytest.approx(12.5)
    assert _value(table, "zero_shot_raw", "correction_fix_rate_mean") == pytest.approx(50.0)
    assert _value(table, "zero_shot_raw", "joint_fix_rate_mean") == pytest.approx(100.0)
    assert _cell(table, "zero_shot_raw", "rule_id_exact_match_sd") == ""
    for row_key in ("c0", "c1", "c2", "c2_control"):
        assert _value(table, row_key, "rule_id_exact_match_mean") == pytest.approx(50.0)
        assert _value(table, row_key, "rule_id_exact_match_sd") == pytest.approx(50.0)
        assert _value(table, row_key, "rule_id_micro_f1_mean") == pytest.approx(500.0 / 9.0)
        assert _value(table, row_key, "rule_id_micro_f1_sd") == pytest.approx(100.0 * (7.0 / 27.0) ** 0.5)
        assert _value(table, row_key, "rule_id_macro_f1_mean") == pytest.approx(12.5)
        assert _value(table, row_key, "rule_id_macro_f1_sd") == pytest.approx(12.5)
        assert _value(table, row_key, "correction_fix_rate_mean") == pytest.approx(50.0)
        assert _value(table, row_key, "correction_fix_rate_sd") == pytest.approx(50.0)
        assert _value(table, row_key, "joint_fix_rate_mean") == pytest.approx(100.0)
        assert _value(table, row_key, "joint_fix_rate_sd") == pytest.approx(0.0)
    assert all(
        cell.display_digits == 2 for row in table.rows for cell in row.cells.values() if isinstance(cell, OutputCell)
    )


def test_table_8_1_full_inputs_are_recomputed_without_display_rounding(generated_outputs, release_inputs) -> None:
    table = generated_outputs.tables["table-8-1.csv"]
    records = [
        record
        for record in release_inputs.scored_records
        if record.condition == "C0" and record.task_type == "rule_identification"
    ]
    per_seed = [
        sum(record.exact_match for record in records if record.seed == seed)
        / sum(record.seed == seed for record in records)
        * 100.0
        for seed in (42, 43, 44)
    ]
    mean_cell = _cell(table, "c0", "rule_id_exact_match_mean")
    sd_cell = _cell(table, "c0", "rule_id_exact_match_sd")
    assert isinstance(mean_cell, OutputCell) and isinstance(sd_cell, OutputCell)
    assert mean_cell.value == statistics.mean(per_seed)
    assert sd_cell.value == pytest.approx(statistics.stdev(per_seed))
    assert mean_cell.display_digits == sd_cell.display_digits == 2
    assert mean_cell.value != round(float(mean_cell.value), 2)


def test_table_8_2_extraction_categories() -> None:
    table = tables._build_table_8_2(_table_8_2_records())

    assert table.filename == "table-8-2.csv"
    assert tuple(row.key for row in table.rows) == (
        "extraction_success",
        "go_fence",
        "untagged_fence",
        "parse_probe",
        "parse_validity",
        "target_fixed",
        "regression_adjusted_fix",
    )
    expected = {
        "extraction_success": 75.0,
        "go_fence": 25.0,
        "untagged_fence": 25.0,
        "parse_probe": 25.0,
        "parse_validity": 75.0,
        "target_fixed": 50.0,
        "regression_adjusted_fix": 25.0,
    }
    for row_key, value in expected.items():
        for column in table.columns[1:]:
            cell = _cell(table, row_key, column)
            assert isinstance(cell, OutputCell)
            assert cell.value == pytest.approx(value)
            assert cell.display_digits == 1
            assert cell.result_id == f"table_8_2.{row_key}.{column}"


def test_table_8_2_full_inputs_pool_exactly_three_seeds(generated_outputs, release_inputs) -> None:
    table = generated_outputs.tables["table-8-2.csv"]
    records = [
        record for record in release_inputs.scored_records if record.condition == "C1" and record.task_type == "joint"
    ]
    assert len(records) == 1_230
    expected = sum(record.outcome.extracted for record in records) / len(records) * 100.0
    cell = _cell(table, "extraction_success", "c1_joint")
    assert isinstance(cell, OutputCell)
    assert cell.value == expected
    assert cell.display_digits == 1
    assert cell.value != round(float(cell.value), 1)


def test_table_8_2_rejects_an_incomplete_three_seed_pool() -> None:
    records = tuple(record for record in _table_8_2_records() if record.seed != 44)

    with pytest.raises(ValueError, match="seed set"):
        tables._build_table_8_2(records)


def test_explanation_is_coverage_only_not_a_scored_endpoint(generated_outputs) -> None:
    assert all("explanation_score" not in key for key in generated_outputs.results)
    assert all("explanation_fix" not in key for key in generated_outputs.results)


def test_core_table_names_columns_and_row_keys(generated_outputs) -> None:
    assert list(generated_outputs.tables)[:6] == [f"table-8-{number}.csv" for number in range(1, 7)]
    expected_columns = {
        "table-8-1.csv": (
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
        ),
        "table-8-2.csv": (
            "component",
            "c0_correction",
            "c1_correction",
            "c2_correction",
            "c2_control_correction",
            "c0_joint",
            "c1_joint",
            "c2_joint",
            "c2_control_joint",
        ),
        "table-8-3.csv": (
            "task",
            "c1_minus_c0_point",
            "c1_minus_c0_low",
            "c1_minus_c0_high",
            "c2_minus_c0_point",
            "c2_minus_c0_low",
            "c2_minus_c0_high",
        ),
        "table-8-4.csv": ("condition", "task", "a", "b", "c", "d", "invalid"),
        "table-8-5.csv": ("condition_task", "identity", "p50", "p90", "p95", "p99", "max", "n"),
        "table-8-6.csv": ("quantity", "c0", "c1", "c2", "c2_control"),
    }
    assert {name: generated_outputs.tables[name].columns for name in expected_columns} == expected_columns
    expected_row_keys = {
        "table-8-1.csv": ("zero_shot_raw", "zero_shot_syntax", "c0", "c1", "c2", "c2_control"),
        "table-8-2.csv": (
            "extraction_success",
            "go_fence",
            "untagged_fence",
            "parse_probe",
            "parse_validity",
            "target_fixed",
            "regression_adjusted_fix",
        ),
        "table-8-3.csv": ("rule_identification", "correction", "joint"),
        "table-8-4.csv": (
            "c0_correction",
            "c1_correction",
            "c2_correction",
            "c2_control_correction",
            "c0_joint",
            "c1_joint",
            "c2_joint",
            "c2_control_joint",
        ),
        "table-8-5.csv": (
            "c0_rule_identification",
            "c0_explanation",
            "c0_correction",
            "c0_joint",
            "c1_rule_identification",
            "c1_explanation",
            "c1_correction",
            "c1_joint",
            "c2_main",
            "c2_auxiliary",
            "c2_control_rule_identification",
            "c2_control_explanation",
            "c2_control_correction",
            "c2_control_joint",
        ),
        "table-8-6.csv": (
            "total_tokens_m",
            "supervised_tokens_m",
            "optimizer_steps",
            "training_minutes",
            "end_to_end_minutes",
            "peak_allocated_gpu_memory_gib",
        ),
    }
    assert {
        filename: tuple(row.key for row in generated_outputs.tables[filename].rows) for filename in expected_row_keys
    } == expected_row_keys


def test_table_8_3_paired_contrasts_use_fixed_endpoint_seeds(monkeypatch) -> None:
    calls: list[tuple[str, str, int, int]] = []

    def fake_contrast(
        records,
        *,
        name: str,
        condition_a: str,
        condition_b: str,
        task: str,
        n_boot: int,
        seed: int,
        alpha: float = 0.05,
    ):
        del records, name, alpha
        calls.append((condition_a, task, n_boot, seed))
        offset = 1.0 if condition_a == "C1" else 2.0
        return SimpleNamespace(interval=SimpleNamespace(point=offset, ci_low=offset - 0.5, ci_high=offset + 0.5))

    monkeypatch.setattr(tables, "task_contrast", fake_contrast)
    table = tables._build_table_8_3(())

    assert {(condition, task): (n_boot, seed) for condition, task, n_boot, seed in calls} == {
        ("C1", "rule_identification"): (10_000, 44),
        ("C1", "correction"): (10_000, 43),
        ("C1", "joint"): (10_000, 42),
        ("C2", "rule_identification"): (10_000, 42),
        ("C2", "correction"): (10_000, 42),
        ("C2", "joint"): (10_000, 42),
    }
    assert _value(table, "joint", "c1_minus_c0_point") == 100.0
    assert _value(table, "joint", "c2_minus_c0_high") == 250.0


def test_table_8_3_full_input_joint_interval_is_recomputed_and_unrounded(generated_outputs, release_inputs) -> None:
    expected = compute_task_contrast(
        release_inputs.scored_records,
        name="joint.c1_minus_c0",
        condition_a="C1",
        condition_b="C0",
        task="joint",
        n_boot=10_000,
        seed=42,
    ).interval
    table = generated_outputs.tables["table-8-3.csv"]
    expected_values = {
        "c1_minus_c0_point": expected.point * 100.0,
        "c1_minus_c0_low": expected.ci_low * 100.0,
        "c1_minus_c0_high": expected.ci_high * 100.0,
    }

    for column, expected_value in expected_values.items():
        cell = _cell(table, "joint", column)
        assert isinstance(cell, OutputCell)
        assert cell.value == expected_value
        assert cell.display_digits == 2
    low = _cell(table, "joint", "c1_minus_c0_low")
    assert isinstance(low, OutputCell) and isinstance(low.value, float)
    assert low.value != round(low.value, low.display_digits or 0)


def test_table_8_4_four_way_counts() -> None:
    categories = ("A", "A", "B", "C", "C", "C", "D", "D", "D", "D", *("INVALID",) * 5)
    records = tuple(
        _repair_record(
            index,
            condition,
            42,
            task,
            outcome=_outcome(category=category, parse_ok=category != "INVALID"),
        )
        for condition in ("C0", "C1", "C2", "C2-control")
        for task in ("correction", "joint")
        for index, category in enumerate(categories, start=1)
    )
    table = tables._build_table_8_4(records)

    assert table.columns == ("condition", "task", "a", "b", "c", "d", "invalid")
    for row in table.rows:
        assert [_value(table, row.key, column) for column in ("a", "b", "c", "d", "invalid")] == [2, 1, 3, 4, 5]


def _contribution(
    condition: Literal["C0", "C1", "C2", "C2-control"],
    pool: Literal["main", "syntax_auxiliary", "duplicated_main_control"],
    task: Literal["rule_identification", "correction", "joint", "explanation", "syntax_summary"],
    total_tokens: int,
    multiplicity: int = 1,
) -> TrainingContribution:
    return TrainingContribution(
        condition=condition,
        pool=pool,
        task_type=task,
        prompt_tokens=total_tokens - 1,
        response_tokens=1,
        total_tokens=total_tokens,
        multiplicity=multiplicity,
    )


def _length_rows(*, unequal_c2_main: bool = False) -> tuple[SimpleNamespace, ...]:
    rows: list[SimpleNamespace] = []
    row_index = 1
    for task in ("rule_identification", "explanation", "correction", "joint"):
        rows.extend(
            (
                SimpleNamespace(
                    base_snippet_id=_snippet(row_index),
                    training_contributions=(_contribution("C0", "main", task, 10, 2),),
                ),
                SimpleNamespace(
                    base_snippet_id=_snippet(row_index + 1),
                    training_contributions=(_contribution("C0", "main", task, 40, 2),),
                ),
                SimpleNamespace(
                    base_snippet_id=_snippet(row_index + 2),
                    training_contributions=(
                        _contribution("C1", "main", task, 20),
                        _contribution("C2", "main", task, 20),
                    ),
                ),
                SimpleNamespace(
                    base_snippet_id=_snippet(row_index + 3),
                    training_contributions=(
                        _contribution("C1", "main", task, 30),
                        _contribution("C2", "main", task, 31 if unequal_c2_main else 30),
                    ),
                ),
                SimpleNamespace(
                    base_snippet_id=_snippet(row_index + 4),
                    training_contributions=(
                        _contribution("C2-control", "main", task, 15),
                        _contribution("C2-control", "duplicated_main_control", task, 35),
                    ),
                ),
            )
        )
        row_index += 5
    rows.extend(
        (
            SimpleNamespace(
                base_snippet_id=_snippet(row_index),
                training_contributions=(_contribution("C2", "syntax_auxiliary", "syntax_summary", 12, 2),),
            ),
            SimpleNamespace(
                base_snippet_id=_snippet(row_index + 1),
                training_contributions=(_contribution("C2", "syntax_auxiliary", "syntax_summary", 50),),
            ),
        )
    )
    return tuple(rows)


def test_table_8_5_multiplicity_and_training_nearest_rank() -> None:
    summary = tables._training_length_summary([10, 20, 30, 40])
    assert (summary.p50, summary.p90, summary.p95, summary.p99, summary.max, summary.n) == (20, 40, 40, 40, 40, 4)
    table = tables._build_table_8_5(_length_rows())

    assert [_value(table, "c0_rule_identification", column) for column in ("p50", "p90", "max", "n")] == [
        10,
        40,
        40,
        4,
    ]
    identity = _cell(table, "c2_main", "identity")
    assert isinstance(identity, OutputCell)
    assert identity.value == "identical_to_c1_main_rows"
    assert all(_cell(table, "c2_main", column) == "" for column in ("p50", "p90", "p95", "p99", "max", "n"))
    with pytest.raises(ValueError, match="C2 main"):
        tables._build_table_8_5(_length_rows(unequal_c2_main=True))


def test_table_8_5_rejects_same_total_with_different_prompt_response_identity(release_inputs) -> None:
    rows = list(release_inputs.study_rows)
    owner_index = next(
        index
        for index, row in enumerate(rows)
        if any(
            contribution.condition == "C2" and contribution.pool == "main" and contribution.response_tokens > 0
            for contribution in row.training_contributions
        )
    )
    owner = rows[owner_index]
    contributions = list(owner.training_contributions)
    contribution_index = next(
        index
        for index, contribution in enumerate(contributions)
        if contribution.condition == "C2" and contribution.pool == "main" and contribution.response_tokens > 0
    )
    original = contributions[contribution_index]
    contributions[contribution_index] = original.model_copy(
        update={
            "prompt_tokens": original.prompt_tokens + 1,
            "response_tokens": original.response_tokens - 1,
        }
    )
    rows[owner_index] = owner.model_copy(update={"training_contributions": tuple(contributions)})

    with pytest.raises(ValueError, match="C2 main"):
        tables._build_table_8_5(tuple(rows))


def test_table_8_6_seed_means() -> None:
    run_results = []
    for condition, offset in (("C0", 0), ("C1", 10), ("C2", 20), ("C2-control", 30)):
        for seed, multiplier in ((42, 1), (43, 2)):
            compute = SimpleNamespace(
                total_tokens=(offset + multiplier) * 1_000_000,
                supervised_tokens=(offset + multiplier) * 100_000,
                optimizer_steps=offset + multiplier,
                wall_clock_train_s=(offset + multiplier) * 60,
                wall_clock_total_s=(offset + multiplier) * 120,
                peak_allocated_gpu_memory_gib=float(offset + multiplier),
            )
            run_results.append(
                SimpleNamespace(condition=condition, seed=seed, metrics=SimpleNamespace(compute=compute))
            )
    table = tables._build_table_8_6(tuple(run_results), SimpleNamespace())

    assert _value(table, "total_tokens_m", "c0") == pytest.approx(1.5)
    assert _value(table, "supervised_tokens_m", "c1") == pytest.approx(1.15)
    assert _value(table, "optimizer_steps", "c2") == pytest.approx(21.5)
    assert _value(table, "training_minutes", "c2_control") == pytest.approx(31.5)
    assert _value(table, "end_to_end_minutes", "c0") == pytest.approx(3.0)
    assert _value(table, "peak_allocated_gpu_memory_gib", "c1") == pytest.approx(11.5)


def test_core_registry_uses_config_and_actual_checkpoint_trace_lengths(generated_outputs, release_inputs) -> None:
    assert generated_outputs.results["training.profile.max_steps"] == release_inputs.config.profiles["paper"].max_steps
    assert generated_outputs.results["training.profile.generation_max_new_tokens.rule_identification"] == 64
    assert generated_outputs.results["training.profile.generation_max_new_tokens.generative_tasks"] == 512
    assert generated_outputs.results["training.checkpoint_selection.evaluations_per_run"] == 5
    assert {
        len(trace)
        for (condition, _seed), trace in release_inputs.selection_traces.items()
        if not condition.startswith("zero-shot")
    } == {5}


def test_core_configuration_pool_exclusions_stay_within_the_training_population(
    generated_outputs,
) -> None:
    results = generated_outputs.results
    assert results["training.pool.syntax.unusable_exclusions"] == (
        results["dataset.base_snippets.train"] - results["training.pool.syntax.pre_exclusion_rows"]
    )


def test_core_hard_gates_use_tokenizer_and_fp32_systematic_agreement(release_inputs) -> None:
    reference = release_inputs.metadata.reference_comparison.model_copy(
        update={"tokenizer_exact": False, "generation_exact": True, "systematic_disagreements_fp32": 0}
    )
    metadata = release_inputs.metadata.model_copy(update={"reference_comparison": reference})
    results = tables._build_metadata_results(metadata)
    assert results["verification.hard_gates.passed"] == 1

    reference = reference.model_copy(update={"generation_exact": False, "systematic_disagreements_fp32": 1})
    metadata = metadata.model_copy(update={"reference_comparison": reference})
    results = tables._build_metadata_results(metadata)
    assert results["verification.hard_gates.passed"] == 0


def test_core_response_cap_proxy_uses_the_experiment_configuration(release_inputs) -> None:
    profile = release_inputs.config.profiles["paper"]
    generation_caps = profile.generation_max_new_tokens.model_copy(update={"correction": 500, "joint": 500})
    altered_profile = profile.model_copy(update={"generation_max_new_tokens": generation_caps})
    config = release_inputs.config.model_copy(update={"profiles": {"paper": altered_profile}})

    results = tables._build_core_sensitivity_results(release_inputs.scored_records, config)
    for task_type in ("correction", "joint"):
        counts = [
            sum(
                record.retokenized_response_token_proxy >= getattr(generation_caps, task_type)
                for record in release_inputs.scored_records
                if record.condition == condition and record.task_type == task_type
            )
            for condition in ("C0", "C1", "C2", "C2-control")
        ]
        assert results[f"sensitivity.response_cap_proxy.{task_type}.minimum_count"] == min(counts)
        assert results[f"sensitivity.response_cap_proxy.{task_type}.maximum_count"] == max(counts)


def test_rq1_relative_gain_is_the_ratio_of_seed_means(generated_outputs, release_inputs) -> None:
    rates = {}
    for condition in ("C0", "C1"):
        for seed in (42, 43, 44):
            records = [
                record
                for record in release_inputs.scored_records
                if record.condition == condition and record.seed == seed and record.task_type == "joint"
            ]
            rates[(condition, seed)] = sum(record.outcome.overall_fixed for record in records) / len(records)
    c0_mean = statistics.mean(rates[("C0", seed)] for seed in (42, 43, 44))
    c1_mean = statistics.mean(rates[("C1", seed)] for seed in (42, 43, 44))
    expected = (c1_mean - c0_mean) / c0_mean * 100.0
    mean_of_seedwise_ratios = statistics.mean(
        (rates[("C1", seed)] - rates[("C0", seed)]) / rates[("C0", seed)] * 100.0 for seed in (42, 43, 44)
    )

    assert generated_outputs.results["rq1.joint.relative_to_c0.percent"] == pytest.approx(expected)
    assert expected != pytest.approx(mean_of_seedwise_ratios)


def test_core_reference_overlap_means_generation_exclusion(generated_outputs, release_inputs) -> None:
    results = generated_outputs.results
    length_excluded = [row for row in release_inputs.study_rows if row.split == "train" and row.length_excluded]
    extra_bearing = [row for row in release_inputs.study_rows if row.split == "train" and row.oracle.extra_checks]
    assert results["dataset.length_exclusion.reference_qc_overlap"] == sum(
        row.reference_qc.correction_status != "accepted" for row in length_excluded
    )
    assert results["oracle.training.extra_bearing.reference_qc_overlap"] == sum(
        row.reference_qc.correction_status != "accepted" for row in extra_bearing
    )


def _multiline_detection_counts(study_rows) -> tuple[int, int, int]:
    multiline = [
        row
        for row in study_rows
        if row.split == "train"
        and row.reference_qc.skip_mechanism == "configured_but_not_applied"
        and not row.reference_qc.generated_marker_retained
    ]
    current_checks = {
        row.base_snippet_id: (set(row.target_checks) - set(row.oracle.missing_checks)) | set(row.oracle.extra_checks)
        for row in multiline
    }
    no_detection = sum(not current_checks[row.base_snippet_id] for row in multiline)
    with_detection = len(multiline) - no_detection
    dual_target = sum(
        len(row.target_checks) > 1
        and any(
            check != "paramTypeCombine" and check in row.target_checks and check in current_checks[row.base_snippet_id]
            for check in row.target_checks
        )
        for row in multiline
    )
    return no_detection, with_detection, dual_target


def test_reference_multiline_detection_does_not_use_normalized_fix_equality(release_inputs) -> None:
    rows = list(release_inputs.study_rows)
    row_index = next(
        index
        for index, row in enumerate(rows)
        if row.split == "train"
        and row.reference_qc.skip_mechanism == "configured_but_not_applied"
        and not row.reference_qc.generated_marker_retained
        and row.reference_qc.normalized_fixes_equal is True
    )
    row = rows[row_index]
    reference_qc = row.reference_qc.model_copy(update={"normalized_fixes_equal": False})
    rows[row_index] = row.model_copy(update={"reference_qc": reference_qc})

    expected_no, expected_with, _expected_dual = _multiline_detection_counts(rows)
    results = tables._build_reference_results(tuple(rows))
    assert results["reference_qc.deviation.multiline.no_studied_detection"] == expected_no
    assert results["reference_qc.deviation.multiline.with_studied_detection"] == expected_with


def test_reference_multiline_dual_target_requires_a_detected_non_ptc_target(release_inputs) -> None:
    rows = list(release_inputs.study_rows)
    row_index = next(
        index
        for index, row in enumerate(rows)
        if row.split == "train"
        and row.reference_qc.skip_mechanism == "configured_but_not_applied"
        and not row.reference_qc.generated_marker_retained
        and len(row.target_checks) > 1
    )
    row = rows[row_index]
    oracle = row.oracle.model_copy(update={"missing_checks": row.target_checks, "extra_checks": ()})
    rows[row_index] = row.model_copy(update={"oracle": oracle})

    expected_no, expected_with, expected_dual = _multiline_detection_counts(rows)
    results = tables._build_reference_results(tuple(rows))
    assert results["reference_qc.deviation.multiline.no_studied_detection"] == expected_no
    assert results["reference_qc.deviation.multiline.with_studied_detection"] == expected_with
    assert results["reference_qc.deviation.multiline.dual_target_verified"] == expected_dual


def _expected_exclusion_mechanisms(study_rows) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in study_rows:
        if row.split not in {"validation", "test"} or row.oracle.status != "excluded":
            continue
        if not row.oracle.missing_checks and row.oracle.extra_checks:
            counts["extra_detection"] += 1
        elif row.oracle.missing_checks == ("paramTypeCombine",) and not row.oracle.extra_checks:
            counts["multiline_param"] += 1
        elif (
            row.oracle.missing_checks
            and not row.oracle.extra_checks
            and set(row.oracle.missing_checks) == set(row.target_checks)
        ):
            counts["generated_suppression"] += 1
        else:
            raise AssertionError("test fixture contains an unaccounted exclusion pattern")
    return counts


def test_oracle_multitarget_single_missing_ptc_is_a_multiline_exclusion(release_inputs) -> None:
    rows = list(release_inputs.study_rows)
    row_index = next(
        index
        for index, row in enumerate(rows)
        if row.split in {"validation", "test"}
        and row.oracle.status == "excluded"
        and row.oracle.missing_checks
        and not row.oracle.extra_checks
        and row.oracle.missing_checks != ("paramTypeCombine",)
    )
    row = rows[row_index]
    oracle = row.oracle.model_copy(update={"missing_checks": ("paramTypeCombine",), "extra_checks": ()})
    rows[row_index] = row.model_copy(update={"target_checks": ("assignOp", "paramTypeCombine"), "oracle": oracle})
    expected = _expected_exclusion_mechanisms(rows)

    results = tables._build_oracle_serializer_results(tuple(rows), release_inputs.scored_records)
    for mechanism, count in expected.items():
        assert results[f"oracle.scoring_gate.excluded.mechanism.{mechanism}"] == count


@pytest.mark.parametrize("pattern", ["mixed", "empty"])
def test_oracle_exclusion_mechanisms_reject_unaccounted_patterns(release_inputs, pattern: str) -> None:
    rows = list(release_inputs.study_rows)
    row_index = next(
        index
        for index, row in enumerate(rows)
        if row.split in {"validation", "test"} and row.oracle.status == "excluded"
    )
    row = rows[row_index]
    if pattern == "mixed":
        oracle = row.oracle.model_copy(update={"missing_checks": ("assignOp",), "extra_checks": ("elseif",)})
    else:
        oracle = row.oracle.model_copy(update={"missing_checks": (), "extra_checks": ()})
    rows[row_index] = row.model_copy(update={"oracle": oracle})

    with pytest.raises(ValueError, match="exclusion mechanism"):
        tables._build_oracle_serializer_results(tuple(rows), release_inputs.scored_records)


def test_oracle_exclusion_mechanisms_reject_partial_missing_target_sets(release_inputs) -> None:
    rows = list(release_inputs.study_rows)
    row_index = next(
        index
        for index, row in enumerate(rows)
        if row.split in {"validation", "test"}
        and row.oracle.status == "excluded"
        and len(row.oracle.missing_checks) == 1
        and row.oracle.missing_checks != ("paramTypeCombine",)
        and not row.oracle.extra_checks
        and set(row.oracle.missing_checks) == set(row.target_checks)
    )
    row = rows[row_index]
    additional_check = next(check for check in CHECKS if check not in row.target_checks)
    rows[row_index] = row.model_copy(update={"target_checks": (*row.target_checks, additional_check)})

    with pytest.raises(ValueError, match="exclusion mechanism"):
        tables._build_oracle_serializer_results(tuple(rows), release_inputs.scored_records)


def test_oracle_scored_population_rejects_a_missing_record_cell(release_inputs) -> None:
    with pytest.raises(ValueError, match="scored.*cell"):
        tables._build_oracle_serializer_results(
            release_inputs.study_rows,
            release_inputs.scored_records[1:],
        )


def test_core_zero_shot_normalization_uses_the_canonical_public_fields(
    generated_outputs,
    release_inputs,
) -> None:
    for condition, result_name in (
        ("zero-shot-raw", "zero_shot_raw"),
        ("zero-shot-syntax", "zero_shot_syntax"),
    ):
        records = [
            record
            for record in release_inputs.scored_records
            if record.condition == condition and record.task_type == "rule_identification"
        ]
        assert generated_outputs.results[f"run.label_normalization.{result_name}.rejected"] == sum(
            record.rejected_label_count for record in records
        )
        assert generated_outputs.results[f"run.label_normalization.{result_name}.emitted_members"] == sum(
            record.n_emitted for record in records
        )
        assert generated_outputs.results[f"run.label_normalization.{result_name}.no_recognized_array"] == sum(
            record.normalization_status == "no_recognized_array" for record in records
        )


def test_core_registry_hand_calculated_checkpoint_timing_and_token_accounting(release_inputs) -> None:
    condition_offsets = {"C0": 0, "C1": 10, "C2": 20, "C2-control": 30}
    synthetic_runs = []
    for run in release_inputs.results:
        if run.condition.startswith("zero-shot"):
            synthetic_runs.append(run)
            continue
        value = condition_offsets[run.condition] + run.seed - 41
        compute = run.metrics.compute.model_copy(
            update={
                "total_tokens": value * 1_000_000,
                "supervised_tokens": value * 100_000,
                "optimizer_steps": value,
                "wall_clock_train_s": value * 60.0,
                "wall_clock_total_s": value * 120.0,
                "peak_allocated_gpu_memory_gib": float(value),
            }
        )
        synthetic_runs.append(run.model_copy(update={"metrics": run.metrics.model_copy(update={"compute": compute})}))

    synthetic_rows = []
    supervised = {
        ("C2", "main"): {42: 4_000_000, 43: 6_000_000, 44: 8_000_000},
        ("C2", "syntax_auxiliary"): {42: 1_000_000, 43: 2_000_000, 44: 3_000_000},
        ("C2-control", "main"): {42: 3_000_000, 43: 5_000_000, 44: 7_000_000},
        ("C2-control", "duplicated_main_control"): {42: 1_000_000, 43: 3_000_000, 44: 5_000_000},
    }
    for row in release_inputs.metadata.token_accounting:
        if row.condition == "C2":
            slot_count = 80 if row.pool == "main" else 20
        elif row.condition == "C2-control":
            slot_count = 80 if row.pool == "main" else 20
        else:
            slot_count = 100
        synthetic_rows.append(
            row.model_copy(
                update={
                    "slot_count": slot_count,
                    "forwarded_tokens": slot_count * 1_000,
                    "supervised_tokens": supervised.get(
                        (row.condition, row.pool),
                        {row.seed: slot_count * 100},
                    )[row.seed],
                }
            )
        )
    metadata = release_inputs.metadata.model_copy(update={"token_accounting": tuple(synthetic_rows)})
    traces = {
        key: trace[:2] if not key[0].startswith("zero-shot") else trace
        for key, trace in release_inputs.selection_traces.items()
    }

    results = tables._build_run_accounting_results(
        tuple(synthetic_runs),
        traces,
        release_inputs.study_rows,
        release_inputs.config,
        metadata,
    )

    assert results["training.checkpoint_selection.evaluations_per_run"] == 2
    assert results["training.seed_42.c0.total_tokens_m"] == 1.0
    assert results["training.seed_42.c0.supervised_tokens_m"] == 0.1
    assert results["training.slots.total_per_run"] == 100
    assert results["training.slots.c2_main"] == 80
    assert results["training.slots.c2_auxiliary"] == 20
    assert results["training.slots.main_ratio"] == 0.8
    assert results["training.slots.auxiliary_ratio"] == 0.2
    assert results["training.supervised_decomposition.c2.main_tokens_m"] == 6.0
    assert results["training.supervised_decomposition.c2.auxiliary_tokens_m"] == 2.0
    assert results["training.supervised_decomposition.c2_control.main_tokens_m"] == 5.0
    assert results["training.supervised_decomposition.c2_control.duplicated_tokens_m"] == 3.0
    assert results["training.supervised_decomposition.c2.auxiliary_percent"] == 25.0
    assert results["training.supervised_decomposition.c2_control.duplicated_percent"] == 37.5
    assert results["training.budget_match.c2_control.total_over_c2_percent"] == pytest.approx(
        (32.0 / 22.0 - 1.0) * 100.0
    )
    assert results["training.end_to_end_nontraining_minutes.minimum"] == 2.0
    assert results["training.end_to_end_nontraining_minutes.maximum"] == 32.0
    full_test_outputs = sum(row.split == "test" for row in release_inputs.study_rows) * len(TASK_TYPES)
    assert results["training.throughput.c0.seconds_per_output"] == pytest.approx(120.0 / full_test_outputs)
    assert results["training.throughput.syntax.seconds_per_output"] == pytest.approx(1_320.0 / full_test_outputs)
    expected_training_hours = (
        sum(
            run.metrics.compute.wall_clock_train_s
            for run in synthetic_runs
            if not run.condition.startswith("zero-shot")
        )
        / 3_600.0
    )
    expected_end_to_end_hours = (
        sum(
            run.metrics.compute.wall_clock_total_s
            for run in synthetic_runs
            if not run.condition.startswith("zero-shot")
        )
        / 3_600.0
    )
    assert results["training.sweep.training_hours"] == expected_training_hours
    assert results["training.sweep.end_to_end_hours"] == expected_end_to_end_hours


def test_run_accounting_throughput_uses_the_study_derived_full_test_output_count(release_inputs) -> None:
    removed = next(row for row in release_inputs.study_rows if row.split == "test")
    study_rows = tuple(row for row in release_inputs.study_rows if row.base_snippet_id != removed.base_snippet_id)
    results = tables._build_run_accounting_results(
        release_inputs.results,
        release_inputs.selection_traces,
        study_rows,
        release_inputs.config,
        release_inputs.metadata,
    )
    denominator = sum(row.split == "test" for row in study_rows) * len(TASK_TYPES)
    c0_nontraining = [
        run.metrics.compute.wall_clock_total_s - run.metrics.compute.wall_clock_train_s
        for run in release_inputs.results
        if run.condition == "C0"
    ]
    syntax_nontraining = [
        run.metrics.compute.wall_clock_total_s - run.metrics.compute.wall_clock_train_s
        for run in release_inputs.results
        if run.condition in {"C1", "C2", "C2-control"}
    ]
    assert results["training.throughput.c0.seconds_per_output"] == pytest.approx(
        statistics.mean(c0_nontraining) / denominator
    )
    assert results["training.throughput.syntax.seconds_per_output"] == pytest.approx(
        statistics.mean(syntax_nontraining) / denominator
    )


def test_run_accounting_final_checkpoint_is_the_configured_final_step(release_inputs) -> None:
    runs = []
    for run in release_inputs.results:
        if run.condition.startswith("zero-shot"):
            runs.append(run)
            continue
        selection = run.metrics.checkpoint_selection.model_copy(update={"selected_step": 480})
        metrics = run.metrics.model_copy(update={"checkpoint_selection": selection})
        runs.append(run.model_copy(update={"metrics": metrics}))

    results = tables._build_run_accounting_results(
        tuple(runs),
        release_inputs.selection_traces,
        release_inputs.study_rows,
        release_inputs.config,
        release_inputs.metadata,
    )
    assert results["training.checkpoint_selection.final_step"] == 600
    assert results["training.checkpoint_selection.final_count"] == 0
    assert results["training.checkpoint_selection.alternate_step"] == 480
    assert results["training.checkpoint_selection.alternate_count"] == 12


def test_syntax_prompt_summaries_reject_cross_condition_prompt_map_divergence(release_inputs) -> None:
    records = list(release_inputs.scored_records)
    record_index = next(
        index for index, record in enumerate(records) if record.condition == "C2" and record.task_type == "joint"
    )
    record = records[record_index]
    records[record_index] = record.model_copy(update={"prompt_tokens": record.prompt_tokens + 1_000})

    with pytest.raises(ValueError, match="syntax prompt"):
        tables._build_core_sensitivity_results(tuple(records), release_inputs.config)
