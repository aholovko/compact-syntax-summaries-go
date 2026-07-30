from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from analysis.inputs import (
    CorrectionRecord,
    InputValidationError,
    RuleIdentificationRecord,
    expected_run_keys,
    load_experiment_config,
    load_inventory,
    load_metadata,
    load_records_file,
    load_run,
    load_runs,
    load_study_rows,
    scored_test_ids,
    select_scored_records,
)

TASK_TYPES = ("rule_identification", "correction", "joint", "explanation")


def _rule_record() -> dict[str, object]:
    return {
        "base_snippet_id": "sha256:" + "a" * 64,
        "condition": "C0",
        "seed": 42,
        "task_type": "rule_identification",
        "target_checks": ("assignOp",),
        "summary_status": "not_applicable",
        "prompt_tokens": 10,
        "retokenized_response_token_proxy": 3,
        "latency_ms": 1.0,
        "gold": ("assignOp",),
        "pred": ("assignOp",),
        "rejected_label_count": 0,
        "exact_match": True,
        "n_emitted": 1,
        "normalization_status": "recognized_array",
    }


def _correction_record() -> dict[str, object]:
    return {
        "base_snippet_id": "sha256:" + "b" * 64,
        "condition": "C0",
        "seed": 42,
        "task_type": "correction",
        "target_checks": ("assignOp",),
        "summary_status": "failed",
        "prompt_tokens": 10,
        "retokenized_response_token_proxy": 3,
        "latency_ms": None,
        "outcome": {
            "target_fixed": False,
            "overall_fixed": False,
            "studied_regression": None,
            "enabled_regression": None,
            "extracted": False,
            "extraction_status": "failed",
            "parse_ok": False,
            "lint_ok": False,
            "original_tool_status": "load_degraded",
            "output_tool_status": "load_failed",
            "build_status": "FAIL",
            "category": "INVALID",
            "introduced_checks": (),
            "residual_findings": (),
        },
        "extracted_similarity": None,
        "sensitivity_class": None,
    }


def _write_jsonl(path: Path, row: dict[str, object]) -> None:
    path.write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")


def _study_row(index: int, *, excluded: bool = False, length_excluded: bool = False) -> dict[str, object]:
    return {
        "base_snippet_id": f"sha256:{index:064x}",
        "split": "test",
        "task_types": list(TASK_TYPES),
        "target_checks": ["assignOp"],
        "violation_count": 1,
        "repository_group_id": f"repository-group-{index:04d}",
        "length_excluded": length_excluded,
        "quarantined": False,
        "oracle": {
            "status": "excluded" if excluded else "reproduced",
            "missing_checks": [],
            "extra_checks": [],
        },
        "source_line_count": 10,
        "serializer": {
            "parse_ok": True,
            "parse_strategy": "file",
            "excluded_construct_count": 0,
            "maximum_depth": 2,
            "summary_status": "present",
        },
        "dataset_qc": {
            "exact_collision": False,
            "canonical_normalization_sampled": False,
            "canonical_normalization_changed": False,
            "pair_indicators": [],
        },
        "reference_qc": {
            "correction_status": "accepted",
            "explanation_present": True,
            "primary_category_a": True,
            "secondary_category_a": None,
            "normalized_fixes_equal": None,
            "selected_generator_role": "primary",
            "generation_attempts": 1,
            "skip_mechanism": "none",
            "generated_marker_retained": False,
            "accepted_correction_byte_identical": False,
            "build_status": "OK",
        },
        "training_contributions": [],
        "serializer_audit_stage1_pool_member": False,
        "serializer_audit": None,
        "violation_in_closure": None,
        "license_class": "permissive",
    }


def _metadata() -> dict[str, object]:
    test_status = {"status": "passed", "evidence_class": "recovered_current"}
    return {
        "architecture": {
            "vocabulary_size": 128_256,
            "context_length": 131_072,
            "embedding_dimension": 2_048,
            "query_heads": 32,
            "key_value_heads": 8,
            "query_heads_per_key_value_head": 4,
            "layers": 16,
            "feed_forward_dimension": 8_192,
            "rope_base": 500_000.0,
            "parameter_count": 1_235_814_400,
            "weight_tied": True,
            "compute_dtype": "bfloat16",
            "rmsnorm_dtype": "float32",
        },
        "reference_comparison": {
            "prompt_count": 215,
            "scored_position_count": 5666,
            "tokenizer_exact": True,
            "chat_template_match": True,
            "chat_template_first_divergence": None,
            "generation_exact": True,
            "generation_first_divergence": None,
            "next_token_agreement_fp32": 1.0,
            "disagreements_fp32": 0,
            "systematic_disagreements_fp32": 0,
            "next_token_agreement_bf16": 1.0,
            "disagreements_bf16": 0,
            "systematic_disagreements_bf16": 0,
            "margin_threshold": 0.1,
            "near_tie_epsilon": 0.01,
            "maximum_absolute_logit_difference": 0.0,
            "mean_absolute_logit_difference": 0.0,
            "null_forward_tolerance": None,
            "cached_generation_test": test_status,
            "sdpa_manual_test": test_status,
            "loss_masking_test": test_status,
        },
        "training_path": {
            "steps": 1,
            "examples": 1,
            "validation_examples": 1,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "learning_rate": 2e-5,
            "mean_relative_loss_divergence": 0.0,
            "maximum_relative_loss_divergence": 0.0,
            "final_loss_scratch": 1.0,
            "final_loss_reference": 1.0,
            "validation_macro_f1_scratch": 1.0,
            "validation_macro_f1_reference": 1.0,
        },
        "token_accounting": [],
        "operator_log_observations": [],
    }


def test_record_rejects_generated_text_and_nonfinite_latency() -> None:
    base = _rule_record()
    RuleIdentificationRecord.model_validate(base)
    with pytest.raises(ValidationError):
        RuleIdentificationRecord.model_validate(base | {"raw_output": "model text"})
    with pytest.raises(ValidationError):
        RuleIdentificationRecord.model_validate(base | {"latency_ms": float("nan")})


def test_no_recognized_array_has_zero_counts_and_empty_prediction() -> None:
    invalid = _rule_record() | {
        "condition": "zero-shot-raw",
        "pred": ("assignOp",),
        "exact_match": False,
        "n_emitted": 0,
        "normalization_status": "no_recognized_array",
    }
    with pytest.raises(ValidationError, match="empty pred"):
        RuleIdentificationRecord.model_validate(invalid)


def test_correction_preserves_independent_tool_and_failed_summary_statuses() -> None:
    record = CorrectionRecord.model_validate(_correction_record())
    assert record.summary_status == "failed"
    assert record.outcome.original_tool_status == "load_degraded"
    assert record.outcome.output_tool_status == "load_failed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary_status", "error"),
        ("outcome.original_tool_status", "degraded"),
        ("outcome.output_tool_status", "failed"),
    ],
)
def test_unknown_status_strings_are_rejected(field: str, value: str) -> None:
    record = _correction_record()
    if field.startswith("outcome."):
        nested = field.removeprefix("outcome.")
        record["outcome"] = record["outcome"] | {nested: value}  # type: ignore[operator]
    else:
        record[field] = value
    with pytest.raises(ValidationError):
        CorrectionRecord.model_validate(record)


def test_expected_run_matrix_is_exact() -> None:
    assert expected_run_keys() == (
        ("C0", 42),
        ("C0", 43),
        ("C0", 44),
        ("C1", 42),
        ("C1", 43),
        ("C1", 44),
        ("C2", 42),
        ("C2", 43),
        ("C2", 44),
        ("C2-control", 42),
        ("C2-control", 43),
        ("C2-control", 44),
        ("zero-shot-raw", 42),
        ("zero-shot-syntax", 42),
    )


def test_loader_rejects_missing_or_extra_run(run_release_fixture: Path) -> None:
    missing = run_release_fixture / "data/runs/c0/seed-42"
    moved = run_release_fixture / "missing-run"
    missing.rename(moved)
    try:
        with pytest.raises(ValueError, match="run matrix"):
            load_runs(run_release_fixture)
    finally:
        moved.rename(missing)

    extra = run_release_fixture / "data/runs/c0/seed-99"
    extra.mkdir()
    try:
        with pytest.raises(ValueError, match="run matrix"):
            load_runs(run_release_fixture)
    finally:
        extra.rmdir()


@pytest.mark.parametrize(
    ("field", "value"),
    [("seed", "42"), ("exact_match", "false"), ("latency_ms", "1.0")],
)
def test_json_scalar_coercions_are_rejected(tmp_path: Path, field: str, value: str) -> None:
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, _rule_record() | {field: value})
    with pytest.raises(InputValidationError, match=field):
        load_records_file(path)


def test_nested_json_duplicate_reports_path_record_and_field(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    payload = json.dumps(_correction_record(), default=list, separators=(",", ":"))
    payload = payload.replace('"build_status":"FAIL"', '"build_status":"FAIL","build_status":"OK"')
    path.write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(InputValidationError) as caught:
        load_records_file(path)
    message = str(caught.value)
    assert str(path) in message
    assert "sha256:" + "b" * 64 in message
    assert "duplicate key" in message and "build_status" in message


def test_yaml_loader_rejects_nested_duplicate_and_nonfinite_number(tmp_path: Path) -> None:
    source = (Path(__file__).resolve().parents[2] / "config/experiments.yaml").read_text(encoding="utf-8")
    duplicate = source.replace("    max_steps: 600", "    max_steps: 600\n    max_steps: 600")
    duplicate_path = tmp_path / "duplicate.yaml"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(InputValidationError, match="duplicate key.*max_steps"):
        load_experiment_config(duplicate_path)

    nonfinite_path = tmp_path / "nonfinite.yaml"
    nonfinite_path.write_text(source.replace("2.0e-5", ".nan"), encoding="utf-8")
    with pytest.raises(InputValidationError, match="finite"):
        load_experiment_config(nonfinite_path)


def test_experiment_loader_rejects_condition_path_mismatch(tmp_path: Path) -> None:
    source = (Path(__file__).resolve().parents[2] / "config/experiments.yaml").read_text(encoding="utf-8")
    path = tmp_path / "experiments.yaml"
    path.write_text(source.replace("    path: c0", "    path: c1", 1), encoding="utf-8")
    with pytest.raises(InputValidationError, match="condition path"):
        load_experiment_config(path)


def test_experiment_loader_requires_exact_paper_profile(tmp_path: Path) -> None:
    source = (Path(__file__).resolve().parents[2] / "config/experiments.yaml").read_text(encoding="utf-8")
    profiles_start = source.index("profiles:\n")
    conditions_start = source.index("conditions:\n")
    path = tmp_path / "experiments.yaml"
    path.write_text(source[:profiles_start] + "profiles: {}\n" + source[conditions_start:], encoding="utf-8")
    with pytest.raises(InputValidationError, match="profile.*paper"):
        load_experiment_config(path)


def test_experiment_loader_requires_exact_generation_caps(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "config/experiments.yaml"
    source = config_path.read_text(encoding="utf-8")
    config = load_experiment_config(config_path)
    assert config.profiles["paper"].generation_max_new_tokens.model_dump(mode="json") == {
        "rule_identification": 64,
        "explanation": 512,
        "correction": 512,
        "joint": 512,
    }

    mutations = (
        source.replace("      joint: 512\n", "", 1),
        source.replace("      joint: 512\n", "      joint: 512\n      unknown_task: 512\n", 1),
        source.replace("      rule_identification: 64\n", "      rule_identification: 65\n", 1),
        source.replace("      rule_identification: 64\n", "      rule_identification: 64.0\n", 1),
    )
    assert all(mutated != source for mutated in mutations)
    for index, mutated in enumerate(mutations):
        path = tmp_path / f"invalid-generation-caps-{index}.yaml"
        path.write_text(mutated, encoding="utf-8")
        with pytest.raises(InputValidationError, match=r"profiles\.paper\.generation_max_new_tokens"):
            load_experiment_config(path)


def test_run_loader_rejects_missing_file_and_nested_result_error(run_release_fixture: Path) -> None:
    result_path = run_release_fixture / "data/runs/c0/seed-42/results.yaml"
    moved = result_path.with_suffix(".missing")
    result_path.rename(moved)
    try:
        with pytest.raises(InputValidationError, match="required input is missing"):
            load_run(run_release_fixture, "C0", 42)
    finally:
        moved.rename(result_path)

    original = result_path.read_text(encoding="utf-8")
    result_path.write_text(original.replace("profile: paper", "profile: exploratory"), encoding="utf-8")
    try:
        with pytest.raises(InputValidationError, match="profile"):
            load_run(run_release_fixture, "C0", 42)
    finally:
        result_path.write_text(original, encoding="utf-8")


def test_run_loader_rejects_trace_and_provenance_mismatches(run_release_fixture: Path) -> None:
    run_dir = run_release_fixture / "data/runs/c0/seed-42"
    trace_path = run_dir / "selection_trace.json"
    original_trace = trace_path.read_text(encoding="utf-8")
    trace_path.write_text(original_trace.replace('"step":240', '"step":121'), encoding="utf-8")
    try:
        with pytest.raises(InputValidationError, match="selection trace"):
            load_run(run_release_fixture, "C0", 42)
    finally:
        trace_path.write_text(original_trace, encoding="utf-8")

    manifest_path = run_dir / "manifest.yaml"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        original_manifest.replace("16aadb26296b291538de481265a149dcb6db8876", "f" * 40),
        encoding="utf-8",
    )
    try:
        with pytest.raises(InputValidationError, match="provenance"):
            load_run(run_release_fixture, "C0", 42)
    finally:
        manifest_path.write_text(original_manifest, encoding="utf-8")


@pytest.mark.parametrize(
    "changes",
    [
        (
            ("results.yaml", "condition: C0", "condition: C1"),
            ("manifest.yaml", "condition: C0", "condition: C1"),
        ),
        (
            ("results.yaml", "seed: 42", "seed: 43"),
            ("manifest.yaml", "seed: 42", "seed: 43"),
        ),
        (("manifest.yaml", "historical_run_id: paper-c0-42", "historical_run_id: paper-c0-other"),),
        (("manifest.yaml", "configuration_id: paper-c0-v1", "configuration_id: paper-c1-v1"),),
    ],
    ids=("condition", "seed", "historical-run-id", "configuration-id"),
)
def test_run_loader_rejects_cross_file_identity_mismatches(
    run_release_fixture: Path,
    changes: tuple[tuple[str, str, str], ...],
) -> None:
    run_dir = run_release_fixture / "data/runs/c0/seed-42"
    originals: dict[Path, str] = {}
    try:
        for filename, old, new in changes:
            path = run_dir / filename
            original = originals.setdefault(path, path.read_text(encoding="utf-8"))
            assert old in original
            path.write_text(original.replace(old, new, 1), encoding="utf-8")
        with pytest.raises(InputValidationError, match="identities must agree"):
            load_run(run_release_fixture, "C0", 42)
    finally:
        for path, original in originals.items():
            path.write_text(original, encoding="utf-8")


def test_run_loader_rejects_selected_trace_metric_mismatch(run_release_fixture: Path) -> None:
    path = run_release_fixture / "data/runs/c0/seed-42/selection_trace.json"
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace('"composite_score":0.7', '"composite_score":0.6', 1), encoding="utf-8")
    try:
        with pytest.raises(InputValidationError, match="selected point disagrees"):
            load_run(run_release_fixture, "C0", 42)
    finally:
        path.write_text(original, encoding="utf-8")


def test_run_loader_rejects_nonempty_zero_shot_trace(run_release_fixture: Path) -> None:
    path = run_release_fixture / "data/runs/zero-shot-raw/seed-42/selection_trace.json"
    original = path.read_text(encoding="utf-8")
    trace = [
        {
            "step": 120,
            "validation_loss": 1.0,
            "composite_score": 0.5,
            "rule_id_macro_f1": 0.5,
            "correction_fix_rate": 0.5,
            "joint_fix_rate": 0.5,
        }
    ]
    path.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    try:
        with pytest.raises(InputValidationError, match="zero-shot selection trace must be empty"):
            load_run(run_release_fixture, "zero-shot-raw", 42)
    finally:
        path.write_text(original, encoding="utf-8")


def test_run_loader_rejects_unexpected_extra_file(run_release_fixture: Path) -> None:
    path = run_release_fixture / "data/runs/c0/seed-42/unexpected.txt"
    path.write_text("unexpected\n", encoding="utf-8")
    try:
        with pytest.raises(InputValidationError, match="exactly the four released inputs"):
            load_run(run_release_fixture, "C0", 42)
    finally:
        path.unlink()


def test_run_loader_rejects_duplicate_cell_substitution_with_same_row_count(run_release_fixture: Path) -> None:
    path = run_release_fixture / "data/runs/c0/seed-42/records.jsonl"
    original = path.read_text(encoding="utf-8")
    rows = original.splitlines()
    rows[-1] = rows[0]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    try:
        with pytest.raises(InputValidationError, match="duplicate record matrix cell"):
            load_run(run_release_fixture, "C0", 42)
    finally:
        path.write_text(original, encoding="utf-8")


def test_run_loader_rejects_incomplete_record_cells(run_release_fixture: Path) -> None:
    path = run_release_fixture / "data/runs/c0/seed-42/records.jsonl"
    original = path.read_text(encoding="utf-8")
    path.write_text("\n".join(original.splitlines()[:-1]) + "\n", encoding="utf-8")
    try:
        with pytest.raises(InputValidationError, match="record matrix"):
            load_run(run_release_fixture, "C0", 42)
    finally:
        path.write_text(original, encoding="utf-8")


def test_study_loader_and_single_scored_population_policy(
    tmp_path: Path,
    run_release_fixture: Path,
) -> None:
    path = tmp_path / "analysis_inputs.jsonl"
    rows = [_study_row(index, excluded=index < 38, length_excluded=index == 100) for index in range(448)]
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    study_rows = load_study_rows(path)
    scored_ids = scored_test_ids(study_rows)
    assert len(scored_ids) == 410
    assert f"sha256:{100:064x}" in scored_ids
    selected = select_scored_records(load_runs(run_release_fixture).records, study_rows)
    assert len(selected) == 22_960
    assert len({record.base_snippet_id for record in selected}) == 410


def test_scored_population_rejects_quarantine_and_substituted_run_id(
    tmp_path: Path,
    run_release_fixture: Path,
) -> None:
    rows = [_study_row(index, excluded=index < 38) for index in range(448)]
    rows[100]["quarantined"] = True
    path = tmp_path / "quarantine.jsonl"
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(InputValidationError, match="quarantines"):
        scored_test_ids(load_study_rows(path))

    valid_path = tmp_path / "valid.jsonl"
    rows[100]["quarantined"] = False
    valid_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    records = list(load_runs(run_release_fixture).records)
    first = records[0].model_copy(update={"base_snippet_id": "sha256:" + "f" * 64})
    records[0] = first
    with pytest.raises(InputValidationError, match="test ID set"):
        select_scored_records(tuple(records), load_study_rows(valid_path))


def test_metadata_and_inventory_loaders_are_strict_and_tuple_preserving(tmp_path: Path) -> None:
    metadata_path = tmp_path / "analysis_metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(_metadata(), sort_keys=False), encoding="utf-8")
    metadata = load_metadata(metadata_path)
    assert metadata.token_accounting == ()

    inventory_path = tmp_path / "manuscript_results.yaml"
    inventory_payload = {
        "schema_version": 1,
        "results": [
            {
                "id": "table8.metric",
                "manuscript_locations": ["table:8.1#metric"],
                "target": {
                    "kind": "csv",
                    "file": "table-8-1.csv",
                    "identifier": None,
                    "row": "c0",
                    "column": "value",
                },
            }
        ],
    }
    inventory_path.write_text(yaml.safe_dump(inventory_payload, sort_keys=False), encoding="utf-8")
    inventory = load_inventory(inventory_path)
    assert inventory.results[0].manuscript_locations == ("table:8.1#metric",)

    inventory_payload["results"][0]["note"] = "prose is forbidden"  # type: ignore[index]
    inventory_path.write_text(yaml.safe_dump(inventory_payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(InputValidationError, match="note"):
        load_inventory(inventory_path)
