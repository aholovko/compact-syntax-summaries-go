"""Validate and atomically publish one results-only retraining attempt."""

from __future__ import annotations

import ctypes
import errno
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, FiniteFloat

from analysis.inputs import BoundedIdentifier, FineTunedCondition, StrictModel
from go_ast_assistant.paper4.eval.evaluator import EvaluationAggregateMetrics, EvaluationResult
from go_ast_assistant.paper4.preflight import ValidatedRequest
from go_ast_assistant.paper4.training.driver import (
    TrainingLengthMetrics,
    TrainingRunResult,
    TrainingSelectionMetrics,
)


_RELEASE_VERSION = "1.0.0"


class AttemptComputeMetrics(StrictModel):
    optimizer_steps: Literal[600]
    examples_seen: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    supervised_tokens: int = Field(ge=0)
    peak_allocated_gpu_memory_gib: FiniteFloat | None = Field(default=None, ge=0)
    wall_clock_train_s: FiniteFloat = Field(ge=0)
    wall_clock_total_s: FiniteFloat = Field(ge=0)


class _AttemptIdentity(StrictModel):
    schema_version: Literal[1]
    run_kind: Literal["retraining_attempt"]
    provenance_status: Literal["user_supplied_unverified"]
    condition: FineTunedCondition
    seed: Literal[42, 43, 44]
    profile: Literal["paper"]
    declared_dataset_revision: Literal["7b951fd57d19286153b46ba219aa2cb87fcc4d2b"]
    configured_model_identifier: Literal["meta-llama/Llama-3.2-1B-Instruct"]
    configuration_id: BoundedIdentifier
    go_version: BoundedIdentifier
    go_critic_version: Literal["v0.14.4"]
    release_version: Literal["1.0.0"]


class TrainingAttemptManifest(_AttemptIdentity):
    pass


class TrainingAttemptMetrics(StrictModel):
    aggregate_metrics: EvaluationAggregateMetrics
    checkpoint_selection: TrainingSelectionMetrics
    compute: AttemptComputeMetrics
    length: TrainingLengthMetrics


class TrainingAttemptResults(_AttemptIdentity):
    metrics: TrainingAttemptMetrics


@dataclass(frozen=True)
class CompletedTrainingRun:
    request: ValidatedRequest
    training: TrainingRunResult
    evaluation: EvaluationResult
    wall_clock_total_s: float

    def __post_init__(self) -> None:
        value = self.wall_clock_total_s
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("total run time must be finite and nonnegative")
        if value < 0:
            raise ValueError("total run time must be finite and nonnegative")


def _require_exact_nonnegative_int(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _validate_shared_identity(run: CompletedTrainingRun) -> None:
    condition = run.request.condition
    seed = run.request.seed
    configuration = run.request.config.conditions.get(condition)
    if configuration is None or configuration.kind != "fine_tuned":
        raise ValueError(f"condition is not a configured fine-tuned condition: {condition!r}")
    if seed not in configuration.seeds:
        raise ValueError(f"seed is not configured for condition {condition}: {seed!r}")

    n_examples = _require_exact_nonnegative_int("evaluation n_examples", run.evaluation.n_examples)
    _require_exact_nonnegative_int("evaluation n_excluded", run.evaluation.n_excluded)
    if n_examples != len(run.evaluation.records):
        raise ValueError("evaluation n_examples must equal the number of records")
    for index, record in enumerate(run.evaluation.records):
        if record.condition != condition:
            raise ValueError(
                f"evaluation record {index} condition {record.condition!r} "
                f"does not match request condition {condition!r}"
            )
        if record.seed != seed:
            raise ValueError(f"evaluation record {index} seed {record.seed!r} does not match request seed {seed!r}")

    compute = run.training.compute
    for label in ("optimizer_steps", "examples_seen", "total_tokens", "supervised_tokens"):
        _require_exact_nonnegative_int(f"training {label}", getattr(compute, label))
    if compute.optimizer_steps != 600:
        raise ValueError("training optimizer_steps must be 600")


def _identity(run: CompletedTrainingRun) -> dict[str, object]:
    request = run.request
    configuration = request.config.conditions[request.condition]
    return {
        "schema_version": 1,
        "run_kind": "retraining_attempt",
        "provenance_status": "user_supplied_unverified",
        "condition": request.condition,
        "seed": request.seed,
        "profile": request.profile,
        "declared_dataset_revision": request.config.dataset.revision,
        "configured_model_identifier": request.config.model.identifier,
        "configuration_id": configuration.configuration_id,
        "go_version": run.evaluation.toolchain.go_version,
        "go_critic_version": run.evaluation.toolchain.go_critic_version,
        "release_version": _RELEASE_VERSION,
    }


def _validated_outputs(
    run: CompletedTrainingRun,
) -> tuple[TrainingAttemptManifest, TrainingAttemptResults]:
    _validate_shared_identity(run)
    identity = _identity(run)
    manifest = TrainingAttemptManifest.model_validate(identity)
    compute = run.training.compute
    attempt_compute = AttemptComputeMetrics(
        optimizer_steps=compute.optimizer_steps,
        examples_seen=compute.examples_seen,
        total_tokens=compute.total_tokens,
        supervised_tokens=compute.supervised_tokens,
        peak_allocated_gpu_memory_gib=compute.peak_allocated_gpu_memory_gib,
        wall_clock_train_s=compute.wall_clock_train_s,
        wall_clock_total_s=run.wall_clock_total_s,
    )
    metrics = TrainingAttemptMetrics(
        aggregate_metrics=run.evaluation.aggregate_metrics,
        checkpoint_selection=run.training.checkpoint_selection,
        compute=attempt_compute,
        length=run.training.length,
    )
    results = TrainingAttemptResults.model_validate(identity | {"metrics": metrics})
    return manifest, results


def _compact_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"


def _serialized_files(
    run: CompletedTrainingRun,
    manifest: TrainingAttemptManifest,
    results: TrainingAttemptResults,
) -> tuple[tuple[str, str], ...]:
    records = "".join(_compact_json(record.model_dump(mode="json")) for record in run.evaluation.records)
    selection_trace = _compact_json([asdict(point) for point in run.training.selection_trace])
    manifest_yaml = yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False)
    results_yaml = yaml.safe_dump(results.model_dump(mode="json"), sort_keys=False)
    return (
        ("records.jsonl", records),
        ("results.yaml", results_yaml),
        ("selection_trace.json", selection_trace),
        ("manifest.yaml", manifest_yaml),
    )


def _write_and_flush(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _flush_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _rename_no_replace(source: Path, target: Path) -> None:
    """Atomically publish one directory while refusing every existing target."""
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        rename_exclusive = ctypes.CDLL(None, use_errno=True).renamex_np
        rename_exclusive.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename_exclusive = libc.renameat2
        except AttributeError as error:  # pragma: no cover - modern supported glibc exposes renameat2
            raise OSError(errno.ENOTSUP, "exclusive directory publication is unavailable", target) from error
        rename_exclusive.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(-100, source_bytes, -100, target_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif os.name == "nt":  # pragma: no cover - Windows os.rename already refuses an existing destination
        os.rename(source, target)
        return
    else:  # pragma: no cover - fail closed on an unsupported publication primitive
        raise OSError(errno.ENOTSUP, "exclusive directory publication is unavailable", target)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), target)


def write_training_run(run: CompletedTrainingRun, output_dir: Path) -> None:
    """Publish four source-free result files without retaining the checkpoint."""

    target = Path(output_dir)
    if _path_present(target):
        raise ValueError(f"output target must be absent: {target}")
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"output parent must be an existing non-symlink directory: {parent}")

    # Validate the complete projection before creating any writer-owned staging state.
    manifest, results = _validated_outputs(run)

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
    try:
        for name, content in _serialized_files(run, manifest, results):
            _write_and_flush(staging / name, content)
        _flush_directory(staging)
        if _path_present(target):
            raise ValueError(f"output target must remain absent until publication: {target}")
        _rename_no_replace(staging, target)
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


__all__ = [
    "AttemptComputeMetrics",
    "CompletedTrainingRun",
    "TrainingAttemptManifest",
    "TrainingAttemptMetrics",
    "TrainingAttemptResults",
    "write_training_run",
]
