from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import go_ast_assistant.paper4.results as results_module
from analysis.inputs import ExplanationRecord, RuleIdentificationRecord, load_experiment_config
from go_ast_assistant.paper4.eval.evaluator import EvaluationAggregateMetrics, EvaluationResult
from go_ast_assistant.paper4.gocheck.toolchain import ToolchainInfo
from go_ast_assistant.paper4.preflight import ValidatedRequest
from go_ast_assistant.paper4.results import (
    AttemptComputeMetrics,
    CompletedTrainingRun,
    TrainingAttemptManifest,
    TrainingAttemptMetrics,
    TrainingAttemptResults,
    write_training_run,
)
from go_ast_assistant.paper4.training.driver import (
    TrainingComputeMetrics,
    TrainingLengthMetrics,
    TrainingRunResult,
    TrainingSelectionMetrics,
    TrainingSelectionPoint,
)


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
DATASET_REVISION = "7b951fd57d19286153b46ba219aa2cb87fcc4d2b"
MODEL_IDENTIFIER = "meta-llama/Llama-3.2-1B-Instruct"
CONFIGURATION_IDS = {
    "C0": "paper-c0-v1",
    "C1": "paper-c1-v1",
    "C2": "paper-c2-v1",
    "C2-control": "paper-c2-control-v1",
}
IDENTITY_FIELDS = (
    "schema_version",
    "run_kind",
    "provenance_status",
    "condition",
    "seed",
    "profile",
    "declared_dataset_revision",
    "configured_model_identifier",
    "configuration_id",
    "go_version",
    "go_critic_version",
    "release_version",
)
RUN_FILES = {
    "manifest.yaml",
    "records.jsonl",
    "results.yaml",
    "selection_trace.json",
}
PREEXISTING_STAGE_NAME = ".attempt.staging"
PREEXISTING_STAGE_SENTINEL = b"PRE-EXISTING STAGING DATA MUST SURVIVE\n"
FORBIDDEN_OUTPUT_FIELDS = {
    "best_checkpoint",
    "checkpoint_path",
    "dataset_path",
    "git_commit",
    "go_binary",
    "go_critic_binary",
    "gofmt_binary",
    "historical_run_id",
    "historical_source_commit",
    "input_path",
    "model_dir",
    "os_snapshot",
    "output_dir",
    "platform",
    "provider_state",
    "raw_output",
    "raw_prompt",
    "raw_response",
    "source_code",
    "study_data_dir",
    "timestamp",
    "tool_output",
}


def _point(step: int, score: float) -> TrainingSelectionPoint:
    return TrainingSelectionPoint(
        step=step,  # type: ignore[arg-type]
        validation_loss=1.0 - score,
        composite_score=score,
        rule_id_macro_f1=score,
        correction_fix_rate=score,
        joint_fix_rate=score,
    )


def _selection(step: int, score: float) -> TrainingSelectionMetrics:
    return TrainingSelectionMetrics(
        selected_step=step,  # type: ignore[arg-type]
        best_composite=score,
        rule_id_macro_f1=score,
        correction_fix_rate=score,
        joint_fix_rate=score,
    )


def _training_result(root: Path) -> TrainingRunResult:
    checkpoint = root / "private-checkpoint" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"PRIVATE MODEL WEIGHTS MUST NOT BE READ OR PUBLISHED")
    scores = (0.2, 0.4, 0.7, 0.6, 0.5)
    trace = tuple(_point(step, score) for step, score in zip((120, 240, 360, 480, 600), scores, strict=True))
    return TrainingRunResult(
        checkpoint_selection=_selection(360, 0.7),
        compute=TrainingComputeMetrics(
            optimizer_steps=600,
            examples_seen=19_200,
            total_tokens=1_234_567,
            supervised_tokens=456_789,
            peak_allocated_gpu_memory_gib=12.5,
            wall_clock_train_s=7_200.25,
        ),
        length=TrainingLengthMetrics(allowed_max_length=9_305, realized_truncation=0),
        selection_trace=trace,
        best_checkpoint=checkpoint,
    )


def _toolchain(root: Path) -> ToolchainInfo:
    tool_root = root / "tools"
    tool_root.mkdir()
    paths = {name: tool_root / name for name in ("go", "gofmt", "go-critic")}
    for path in paths.values():
        path.write_text("local executable\n", encoding="utf-8")
        path.chmod(0o755)
    return ToolchainInfo(
        go_binary=paths["go"],
        gofmt_binary=paths["gofmt"],
        go_critic_binary=paths["go-critic"],
        go_version="go1.26.99",
        go_critic_version="v0.14.4",
    )


def _rule_record(condition: str = "C2", seed: int = 43) -> RuleIdentificationRecord:
    return RuleIdentificationRecord(
        base_snippet_id="sha256:" + "a" * 64,
        condition=condition,  # type: ignore[arg-type]
        seed=seed,  # type: ignore[arg-type]
        task_type="rule_identification",
        target_checks=("assignOp",),
        summary_status="not_applicable" if condition == "C0" else "present",
        prompt_tokens=17,
        retokenized_response_token_proxy=5,
        latency_ms=None,
        gold=("assignOp",),
        pred=("assignOp",),
        rejected_label_count=0,
        exact_match=True,
        n_emitted=1,
        normalization_status="recognized_array",
    )


def _explanation_record(condition: str = "C2", seed: int = 43) -> ExplanationRecord:
    return ExplanationRecord(
        base_snippet_id="sha256:" + "b" * 64,
        condition=condition,  # type: ignore[arg-type]
        seed=seed,  # type: ignore[arg-type]
        task_type="explanation",
        target_checks=("captLocal",),
        summary_status="not_applicable" if condition == "C0" else "skipped",
        prompt_tokens=29,
        retokenized_response_token_proxy=11,
        latency_ms=None,
    )


def _aggregate() -> EvaluationAggregateMetrics:
    return EvaluationAggregateMetrics(
        rule_id_macro_f1=0.71,
        rule_id_micro_f1=0.72,
        rule_id_exact_match=0.55,
        correction_fix_rate=0.61,
        joint_fix_rate=0.51,
    )


def _validated_request(root: Path, condition: str = "C2", seed: int = 43) -> ValidatedRequest:
    return ValidatedRequest(
        config=load_experiment_config(BUNDLE_ROOT / "config" / "experiments.yaml"),
        condition=condition,  # type: ignore[arg-type]
        seed=seed,  # type: ignore[arg-type]
        profile="paper",
        study_data_dir=root / "private-study",
        model_dir=root / "private-model",
        output_dir=root / "request-output",
        device="cpu",
    )


def _completed_run(
    root: Path,
    *,
    request_condition: str = "C2",
    request_seed: int = 43,
    record_condition: str | None = None,
    record_seed: int | None = None,
) -> CompletedTrainingRun:
    condition = request_condition if record_condition is None else record_condition
    seed = request_seed if record_seed is None else record_seed
    records = (_rule_record(condition, seed), _explanation_record(condition, seed))
    evaluation = EvaluationResult(
        records=records,
        aggregate_metrics=_aggregate(),
        toolchain=_toolchain(root),
        n_examples=len(records),
        n_excluded=3,
    )
    return CompletedTrainingRun(
        request=_validated_request(root, request_condition, request_seed),
        training=_training_result(root),
        evaluation=evaluation,
        wall_clock_total_s=10_800.5,
    )


def _identity_payload(condition: str = "C2", seed: int = 43) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_kind": "retraining_attempt",
        "provenance_status": "user_supplied_unverified",
        "condition": condition,
        "seed": seed,
        "profile": "paper",
        "declared_dataset_revision": DATASET_REVISION,
        "configured_model_identifier": MODEL_IDENTIFIER,
        "configuration_id": CONFIGURATION_IDS[condition],
        "go_version": "go1.26.99",
        "go_critic_version": "v0.14.4",
        "release_version": "1.0.0",
    }


def _attempt_compute_payload() -> dict[str, object]:
    return {
        "optimizer_steps": 600,
        "examples_seen": 19_200,
        "total_tokens": 1_234_567,
        "supervised_tokens": 456_789,
        "peak_allocated_gpu_memory_gib": 12.5,
        "wall_clock_train_s": 7_200.25,
        "wall_clock_total_s": 10_800.5,
    }


def _attempt_metrics() -> TrainingAttemptMetrics:
    return TrainingAttemptMetrics(
        aggregate_metrics=_aggregate(),
        checkpoint_selection=_selection(360, 0.7),
        compute=AttemptComputeMetrics.model_validate(_attempt_compute_payload()),
        length=TrainingLengthMetrics(allowed_max_length=9_305, realized_truncation=0),
    )


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _walk_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _walk_keys(item)}
    return set()


def _walk_strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {item for nested in value.values() for item in _walk_strings(nested)}
    if isinstance(value, list):
        return {item for nested in value for item in _walk_strings(nested)}
    return set()


def _loaded_outputs(output_dir: Path) -> dict[str, object]:
    records = [json.loads(line) for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    return {
        "records": records,
        "results": yaml.safe_load((output_dir / "results.yaml").read_text(encoding="utf-8")),
        "selection_trace": json.loads((output_dir / "selection_trace.json").read_text(encoding="utf-8")),
        "manifest": yaml.safe_load((output_dir / "manifest.yaml").read_text(encoding="utf-8")),
    }


def _publication_root(root: Path) -> tuple[Path, Path]:
    parent = root / "publication"
    parent.mkdir()
    (parent / "keep.txt").write_text("untouched\n", encoding="utf-8")
    preexisting_stage = parent / PREEXISTING_STAGE_NAME
    preexisting_stage.mkdir()
    (preexisting_stage / "sentinel.bin").write_bytes(PREEXISTING_STAGE_SENTINEL)
    return parent, parent / "attempt"


def _assert_preexisting_stage_is_preserved(parent: Path) -> None:
    preexisting_stage = parent / PREEXISTING_STAGE_NAME
    assert preexisting_stage.is_dir()
    assert {entry.name for entry in preexisting_stage.iterdir()} == {"sentinel.bin"}
    assert (preexisting_stage / "sentinel.bin").read_bytes() == PREEXISTING_STAGE_SENTINEL


def _assert_failed_publication_is_clean(parent: Path, output_dir: Path) -> None:
    assert not output_dir.exists()
    assert not output_dir.is_symlink()
    assert {entry.name for entry in parent.iterdir()} == {PREEXISTING_STAGE_NAME, "keep.txt"}
    _assert_preexisting_stage_is_preserved(parent)


def test_attempt_schemas_have_exact_fields_and_truthful_fixed_identity() -> None:
    manifest = TrainingAttemptManifest.model_validate(_identity_payload())
    results = TrainingAttemptResults.model_validate(_identity_payload() | {"metrics": _attempt_metrics()})

    assert tuple(TrainingAttemptManifest.model_fields) == IDENTITY_FIELDS
    assert tuple(AttemptComputeMetrics.model_fields) == (
        "optimizer_steps",
        "examples_seen",
        "total_tokens",
        "supervised_tokens",
        "peak_allocated_gpu_memory_gib",
        "wall_clock_train_s",
        "wall_clock_total_s",
    )
    assert tuple(TrainingAttemptMetrics.model_fields) == (
        "aggregate_metrics",
        "checkpoint_selection",
        "compute",
        "length",
    )
    assert tuple(TrainingAttemptResults.model_fields) == IDENTITY_FIELDS + ("metrics",)
    assert manifest.provenance_status == "user_supplied_unverified"
    assert results.provenance_status == "user_supplied_unverified"
    assert results.declared_dataset_revision == DATASET_REVISION
    assert results.configured_model_identifier == MODEL_IDENTIFIER


@pytest.mark.parametrize(
    ("updates", "match"),
    (
        ({"schema_version": 2}, "schema_version"),
        ({"run_kind": "historical_reconstruction"}, "run_kind"),
        ({"provenance_status": "verified"}, "provenance_status"),
        ({"condition": "zero-shot-raw"}, "condition"),
        ({"seed": 41}, "seed"),
        ({"profile": "smoke"}, "profile"),
        ({"declared_dataset_revision": "0" * 40}, "declared_dataset_revision"),
        ({"configured_model_identifier": "some/local/model"}, "configured_model_identifier"),
        ({"configuration_id": "../paper-c2-v1"}, "configuration_id"),
        ({"go_version": "go version go1.26.99 linux/amd64"}, "go_version"),
        ({"go_critic_version": "v0.14.3"}, "go_critic_version"),
        ({"release_version": "1.0.1"}, "release_version"),
        ({"historical_source_commit": "f" * 40}, "historical_source_commit"),
        ({"model_dir": "/private/model"}, "model_dir"),
    ),
)
def test_attempt_identity_rejects_untruthful_or_extra_fields(updates: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        TrainingAttemptManifest.model_validate(_identity_payload() | updates)


@pytest.mark.parametrize(
    ("updates", "match"),
    (
        ({"optimizer_steps": 599}, "optimizer_steps"),
        ({"optimizer_steps": True}, "optimizer_steps"),
        ({"examples_seen": -1}, "examples_seen"),
        ({"total_tokens": -1}, "total_tokens"),
        ({"supervised_tokens": -1}, "supervised_tokens"),
        ({"peak_allocated_gpu_memory_gib": -0.1}, "peak_allocated_gpu_memory_gib"),
        ({"peak_allocated_gpu_memory_gib": float("nan")}, "peak_allocated_gpu_memory_gib"),
        ({"wall_clock_train_s": float("inf")}, "wall_clock_train_s"),
        ({"wall_clock_total_s": -0.1}, "wall_clock_total_s"),
        ({"local_path": "/private/run"}, "local_path"),
    ),
)
def test_attempt_compute_rejects_invalid_counts_times_and_extra_fields(
    updates: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        AttemptComputeMetrics.model_validate(_attempt_compute_payload() | updates)


@pytest.mark.parametrize(
    "wall_clock_total_s",
    (-0.1, float("nan"), float("inf")),
)
def test_completed_run_rejects_invalid_total_time(tmp_path: Path, wall_clock_total_s: float) -> None:
    run = _completed_run(tmp_path)

    with pytest.raises(ValueError, match="total.*time|finite|nonnegative"):
        replace(run, wall_clock_total_s=wall_clock_total_s)


@pytest.mark.parametrize(
    ("record_condition", "record_seed"),
    (("C1", None), (None, 42)),
)
def test_writer_rejects_cross_component_identity_before_staging(
    tmp_path: Path,
    record_condition: str | None,
    record_seed: int | None,
) -> None:
    run = _completed_run(tmp_path, record_condition=record_condition, record_seed=record_seed)
    parent, output_dir = _publication_root(tmp_path)

    with pytest.raises(ValueError, match="condition|seed"):
        write_training_run(run, output_dir)

    _assert_failed_publication_is_clean(parent, output_dir)


@pytest.mark.parametrize("target_kind", ("directory", "file", "symlink"))
def test_writer_requires_an_absent_target_without_modifying_it(tmp_path: Path, target_kind: str) -> None:
    run = _completed_run(tmp_path)
    parent, output_dir = _publication_root(tmp_path)
    if target_kind == "directory":
        output_dir.mkdir()
        (output_dir / "existing.txt").write_text("keep\n", encoding="utf-8")
    elif target_kind == "file":
        output_dir.write_text("keep\n", encoding="utf-8")
    else:
        output_dir.symlink_to(parent / "missing-target")

    before = tuple(sorted((path.name, path.is_symlink(), path.is_file()) for path in parent.iterdir()))
    with pytest.raises(ValueError, match="absent|exists|target"):
        write_training_run(run, output_dir)

    after = tuple(sorted((path.name, path.is_symlink(), path.is_file()) for path in parent.iterdir()))
    assert after == before
    _assert_preexisting_stage_is_preserved(parent)


@pytest.mark.parametrize(
    ("condition", "seed", "configuration_id"),
    (
        ("C0", 42, "paper-c0-v1"),
        ("C1", 43, "paper-c1-v1"),
        ("C2", 44, "paper-c2-v1"),
        ("C2-control", 42, "paper-c2-control-v1"),
    ),
)
def test_writer_publishes_exact_canonical_source_free_four_file_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    seed: int,
    configuration_id: str,
) -> None:
    run = _completed_run(tmp_path, request_condition=condition, request_seed=seed)
    parent, output_dir = _publication_root(tmp_path)
    real_path_open = Path.open

    def guard_checkpoint_open(path: Path, *args: Any, **kwargs: Any):
        if path == run.training.best_checkpoint:
            raise AssertionError("the writer read the private checkpoint")
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_checkpoint_open)

    write_training_run(run, output_dir)

    assert {path.name for path in output_dir.iterdir()} == RUN_FILES
    assert {entry.name for entry in parent.iterdir()} == {PREEXISTING_STAGE_NAME, "attempt", "keep.txt"}
    _assert_preexisting_stage_is_preserved(parent)

    expected_records = "".join(
        json.dumps(record.model_dump(mode="json"), allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        for record in run.evaluation.records
    )
    expected_trace = (
        json.dumps(
            [asdict(point) for point in run.training.selection_trace],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    expected_identity = _identity_payload(condition, seed)
    expected_manifest = TrainingAttemptManifest.model_validate(expected_identity)
    expected_results = TrainingAttemptResults.model_validate(expected_identity | {"metrics": _attempt_metrics()})
    assert (output_dir / "records.jsonl").read_text(encoding="utf-8") == expected_records
    assert (output_dir / "selection_trace.json").read_text(encoding="utf-8") == expected_trace
    assert (output_dir / "manifest.yaml").read_text(encoding="utf-8") == yaml.safe_dump(
        expected_manifest.model_dump(mode="json"),
        sort_keys=False,
    )
    assert (output_dir / "results.yaml").read_text(encoding="utf-8") == yaml.safe_dump(
        expected_results.model_dump(mode="json"),
        sort_keys=False,
    )

    loaded = _loaded_outputs(output_dir)
    assert loaded["manifest"]["condition"] == condition  # type: ignore[index]
    assert loaded["manifest"]["seed"] == seed  # type: ignore[index]
    assert loaded["manifest"]["configuration_id"] == configuration_id  # type: ignore[index]
    assert loaded["results"]["condition"] == condition  # type: ignore[index]
    assert loaded["results"]["seed"] == seed  # type: ignore[index]
    assert loaded["results"]["configuration_id"] == configuration_id  # type: ignore[index]
    assert not (_walk_keys(loaded) & FORBIDDEN_OUTPUT_FIELDS)
    serialized_strings = _walk_strings(loaded)
    private_values = {
        str(tmp_path),
        str(run.request.study_data_dir),
        str(run.request.model_dir),
        str(run.request.output_dir),
        str(run.training.best_checkpoint),
        str(run.evaluation.toolchain.go_binary),
        str(run.evaluation.toolchain.gofmt_binary),
        str(run.evaluation.toolchain.go_critic_binary),
    }
    assert not (serialized_strings & private_values)
    assert not any(Path(value).is_absolute() for value in serialized_strings)
    assert b"PRIVATE MODEL WEIGHTS" not in b"".join(path.read_bytes() for path in output_dir.iterdir())


def test_serialization_failure_removes_sibling_staging_and_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(tmp_path)
    parent, output_dir = _publication_root(tmp_path)

    def fail_serialization(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(results_module.yaml, "safe_dump", fail_serialization)
    with pytest.raises(RuntimeError, match="injected serialization failure"):
        write_training_run(run, output_dir)

    _assert_failed_publication_is_clean(parent, output_dir)
