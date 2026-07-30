from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch.utils.data import DataLoader

from go_ast_assistant.paper4.prepared_study import PreparedStudy
from go_ast_assistant.paper4.records import TaskExample
from go_ast_assistant.paper4.runtime.collate import PAD_ID, instruction_collate_fn
from go_ast_assistant.paper4.runtime.generate import generate_bucketed, strip_assistant_header
from go_ast_assistant.paper4.runtime.loss import calc_loss_loader
from go_ast_assistant.paper4.training.budget import BudgetMeter
from go_ast_assistant.paper4.training.checkpoint import CheckpointManager
from go_ast_assistant.paper4.training.composite import ValidationComposite
from go_ast_assistant.paper4.training.conditions import (
    AUX_ENCODING,
    CONDITIONS,
    Condition,
    NullSummaryStore,
    SummaryStore,
)
from go_ast_assistant.paper4.training.config import TrainingConfig
from go_ast_assistant.paper4.training.instruction_dataset import EOT_ID, InstructionDataset, encode_example
from go_ast_assistant.paper4.training.loop import train_loop
from go_ast_assistant.paper4.training.sampling import build_mixture_stream, length_stratified_aux_sample
from go_ast_assistant.paper4.training.seeding import seed_everything
from go_ast_assistant.paper4.training.summary_store import SerializedSummaryStore


_CHECKPOINT_STEPS = (120, 240, 360, 480, 600)


def _require_exact_int(label: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact integer")


def _require_finite(label: str, *values: float) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain only finite values")


def _require_rate(label: str, *values: float) -> None:
    _require_finite(label, *values)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{label} must be within [0, 1]")


@dataclass(frozen=True)
class TrainingSelectionPoint:
    step: Literal[120, 240, 360, 480, 600]
    validation_loss: float
    composite_score: float
    rule_id_macro_f1: float
    correction_fix_rate: float
    joint_fix_rate: float

    def __post_init__(self) -> None:
        _require_exact_int("selection step", self.step)
        if self.step not in _CHECKPOINT_STEPS:
            raise ValueError("selection step must use the fixed 120-step cadence")
        _require_finite("validation loss", self.validation_loss)
        if self.validation_loss < 0.0:
            raise ValueError("validation loss must be nonnegative")
        _require_rate(
            "selection scores",
            self.composite_score,
            self.rule_id_macro_f1,
            self.correction_fix_rate,
            self.joint_fix_rate,
        )


@dataclass(frozen=True)
class TrainingSelectionMetrics:
    selected_step: Literal[120, 240, 360, 480, 600]
    best_composite: float
    rule_id_macro_f1: float
    correction_fix_rate: float
    joint_fix_rate: float

    def __post_init__(self) -> None:
        _require_exact_int("selected step", self.selected_step)
        if self.selected_step not in _CHECKPOINT_STEPS:
            raise ValueError("selected step must use the fixed 120-step cadence")
        _require_rate(
            "selection metrics",
            self.best_composite,
            self.rule_id_macro_f1,
            self.correction_fix_rate,
            self.joint_fix_rate,
        )


@dataclass(frozen=True)
class TrainingComputeMetrics:
    optimizer_steps: Literal[600]
    examples_seen: int
    total_tokens: int
    supervised_tokens: int
    peak_allocated_gpu_memory_gib: float | None
    wall_clock_train_s: float

    def __post_init__(self) -> None:
        count_fields = {
            "optimizer_steps": self.optimizer_steps,
            "examples_seen": self.examples_seen,
            "total_tokens": self.total_tokens,
            "supervised_tokens": self.supervised_tokens,
        }
        for label, value in count_fields.items():
            _require_exact_int(label, value)
        if self.optimizer_steps != 600:
            raise ValueError("optimizer_steps must be 600")
        if any(value < 0 for value in count_fields.values()):
            raise ValueError("compute counts must be nonnegative")
        _require_finite("training time", self.wall_clock_train_s)
        if self.wall_clock_train_s < 0:
            raise ValueError("training time must be nonnegative")
        if self.peak_allocated_gpu_memory_gib is not None:
            _require_finite("peak memory", self.peak_allocated_gpu_memory_gib)
            if self.peak_allocated_gpu_memory_gib < 0:
                raise ValueError("peak memory must be nonnegative")


@dataclass(frozen=True)
class TrainingLengthMetrics:
    allowed_max_length: Literal[9305]
    realized_truncation: Literal[0]

    def __post_init__(self) -> None:
        _require_exact_int("allowed_max_length", self.allowed_max_length)
        _require_exact_int("realized_truncation", self.realized_truncation)
        if self.allowed_max_length != 9305:
            raise ValueError("allowed_max_length must be 9305")
        if self.realized_truncation != 0:
            raise ValueError("paper-profile training must not truncate")


@dataclass(frozen=True)
class TrainingRunResult:
    checkpoint_selection: TrainingSelectionMetrics
    compute: TrainingComputeMetrics
    length: TrainingLengthMetrics
    selection_trace: tuple[TrainingSelectionPoint, ...]
    best_checkpoint: Path

    def __post_init__(self) -> None:
        steps = tuple(point.step for point in self.selection_trace)
        if steps != _CHECKPOINT_STEPS:
            raise ValueError("selection trace must contain exactly steps 120, 240, 360, 480, and 600")
        selected = [point for point in self.selection_trace if point.step == self.checkpoint_selection.selected_step]
        if len(selected) != 1:
            raise ValueError("selected step must occur exactly once in selection trace")
        point = selected[0]
        expected = (
            point.composite_score,
            point.rule_id_macro_f1,
            point.correction_fix_rate,
            point.joint_fix_rate,
        )
        actual = (
            self.checkpoint_selection.best_composite,
            self.checkpoint_selection.rule_id_macro_f1,
            self.checkpoint_selection.correction_fix_rate,
            self.checkpoint_selection.joint_fix_rate,
        )
        if actual != expected:
            raise ValueError("selected metrics must equal the selected trace point")
        first_best = max(self.selection_trace, key=lambda item: item.composite_score)
        if point.step != first_best.step:
            raise ValueError("selected step must be the first maximum composite score")
        if self.best_checkpoint.is_symlink() or not self.best_checkpoint.is_file():
            raise ValueError(f"checkpoint must be a regular local file: {self.best_checkpoint}")


def filter_examples(
    examples: Iterable[TaskExample],
    drop_ids: frozenset[str],
) -> tuple[TaskExample, ...]:
    return tuple(example for example in examples if example.id not in drop_ids)


def val_scoring_exclusions(study: PreparedStudy) -> frozenset[str]:
    validation_ids = frozenset(example.id for example in study.tasks_by_split["validation"])
    quarantined = study.quarantine_ids & validation_ids
    adjudicated = frozenset(
        adjudication.id
        for adjudication in study.adjudications.values()
        if adjudication.split == "validation" and adjudication.resolution == "exclude"
    )
    return quarantined | adjudicated


def val_loss_subset(val: Iterable[TaskExample]) -> tuple[TaskExample, ...]:
    subset = tuple(example for example in val if example.task_type == "rule_identification")
    if not subset:
        raise ValueError("no rule_identification rows for validation loss")
    return subset


def select_composite_examples(
    val: Iterable[TaskExample],
    ids: frozenset[str],
) -> tuple[TaskExample, ...]:
    if not ids:
        raise ValueError("empty composite validation ID set")
    examples = tuple(val)
    missing = ids - frozenset(example.id for example in examples)
    if missing:
        raise ValueError(f"composite validation IDs are missing from the filtered validation set: {sorted(missing)}")
    return tuple(example for example in examples if example.id in ids)


def aux_encoding_for(condition: Condition) -> Condition:
    return AUX_ENCODING if condition.aux == "syntax" else condition


def build_condition_stream(
    main_resp_lengths: Sequence[int],
    aux_resp_lengths: Sequence[int],
    condition: Condition,
    max_steps: int,
    eff_batch: int,
    seed: int,
    aux_ratio: float | None = None,
) -> list[tuple[str, int]] | tuple[tuple[str, int], ...]:
    """Build the fixed stream, stratifying C2 by supervised response length."""
    total = max_steps * eff_batch
    ratio = condition.aux_ratio if aux_ratio is None else aux_ratio
    pattern = build_mixture_stream(
        len(main_resp_lengths),
        len(aux_resp_lengths),
        ratio,
        total,
        seed,
    )
    if not aux_resp_lengths or condition.aux != "syntax":
        return pattern

    auxiliary_count = sum(pool == "aux" for pool, _index in pattern)
    selected_auxiliary = iter(
        length_stratified_aux_sample(
            tuple(aux_resp_lengths),
            tuple(main_resp_lengths),
            auxiliary_count,
            seed,
        )
    )
    return [("aux", next(selected_auxiliary)) if pool == "aux" else ("main", index) for pool, index in pattern]


def iter_micro_batches(
    stream: Iterable[tuple[str, int]],
    main_ds: Sequence[Mapping[str, object]],
    aux_ds: Sequence[Mapping[str, object]],
    micro_batch_size: int,
    allowed_max_length: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict[str, int]]]:
    """Collate fixed-pool draws on CPU and retain exact exposure counts."""
    batch: list[Mapping[str, object]] = []
    n_main = 0
    n_auxiliary = 0
    for pool, index in stream:
        if pool == "aux":
            batch.append(aux_ds[index])
            n_auxiliary += 1
        elif pool == "main":
            batch.append(main_ds[index])
            n_main += 1
        else:
            raise ValueError(f"unknown training pool: {pool!r}")

        if len(batch) != micro_batch_size:
            continue
        inputs, targets = instruction_collate_fn(batch, allowed_max_length=allowed_max_length)
        if inputs.device.type != "cpu" or targets.device.type != "cpu":
            raise ValueError("training micro-batches must be collated on CPU")
        yield (
            inputs,
            targets,
            {
                "n_main": n_main,
                "n_aux": n_auxiliary,
                "n_real_tokens": int((inputs != PAD_ID).sum().item()),
                "n_supervised_tokens": int((targets != -100).sum().item()),
            },
        )
        batch = []
        n_main = 0
        n_auxiliary = 0

    if batch:
        raise ValueError("training stream must contain complete micro-batches")


def _require_checkpoint_dir(checkpoint_dir: Path) -> Path:
    directory = Path(checkpoint_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"checkpoint directory must be an existing non-symlink directory: {directory}")
    try:
        nonempty = next(directory.iterdir(), None) is not None
    except OSError as error:
        raise ValueError(f"checkpoint directory cannot be inspected: {directory}") from error
    if nonempty:
        raise ValueError(f"checkpoint directory must be empty: {directory}")
    return directory


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    if actual.type != expected.type:
        return False
    return expected.index is None or actual.index == expected.index


def _require_model_device(model: torch.nn.Module, device: torch.device) -> None:
    for kind, named_tensors in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in named_tensors:
            if not _device_matches(tensor.device, device):
                raise ValueError(f"model {kind} {name!r} is on {tensor.device}; expected device {device}")


def _summary_store(study: PreparedStudy, tokenizer: Any, condition: Condition) -> SummaryStore:
    if not condition.use_summary:
        return NullSummaryStore()
    if study.summaries is None:
        raise ValueError(f"condition {condition.name} requires injected prepared summaries")
    return SerializedSummaryStore(
        study.summaries,
        lambda text: len(tokenizer.tok.encode(text)),
    )


def _dataset_items(dataset: Any) -> tuple[Mapping[str, object], ...]:
    return tuple(cast(Mapping[str, object], dataset[index]) for index in range(len(dataset)))


def _response_lengths(items: Sequence[Mapping[str, object]]) -> tuple[int, ...]:
    return tuple(len(cast(Sequence[object], item["input_ids"])) - cast(int, item["prompt_len"]) for item in items)


def _realized_truncation(
    stream: Sequence[tuple[str, int]],
    main_items: Sequence[Mapping[str, object]],
    auxiliary_items: Sequence[Mapping[str, object]],
    allowed_max_length: int,
) -> int:
    count = 0
    for pool, index in stream:
        item = auxiliary_items[index] if pool == "aux" else main_items[index]
        count += len(cast(Sequence[object], item["input_ids"])) > allowed_max_length
    return count


def _generation_function(
    model: torch.nn.Module,
    tokenizer: Any,
    store: SummaryStore,
    condition: Condition,
):
    def generate_fn(
        examples: tuple[TaskExample, ...],
        max_new_tokens: int,
    ) -> tuple[str, ...]:
        prompts = []
        for example in examples:
            item = encode_example(example, tokenizer, condition, store)
            prompts.append(item["input_ids"][: item["prompt_len"]])
        try:
            context_size = model.cfg["context_length"]  # type: ignore[attr-defined]
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError("model must expose its fixed context length") from error
        if type(context_size) is not int or context_size <= 0:
            raise ValueError("model context length must be a positive integer")
        outputs = generate_bucketed(
            model,
            prompts,
            max_new_tokens,
            context_size,
            eos_id=EOT_ID,
        )
        return tuple(strip_assistant_header(tokenizer.decode(output)) for output in outputs)

    return generate_fn


def _selection_trace(checkpoint: CheckpointManager) -> tuple[TrainingSelectionPoint, ...]:
    step_type = Literal[120, 240, 360, 480, 600]
    return tuple(
        TrainingSelectionPoint(
            step=cast(step_type, point.step),
            validation_loss=point.validation_loss,
            composite_score=point.composite_score,
            rule_id_macro_f1=point.rule_id_macro_f1,
            correction_fix_rate=point.correction_fix_rate,
            joint_fix_rate=point.joint_fix_rate,
        )
        for point in checkpoint.trace
    )


def run_training(
    config: TrainingConfig,
    study: PreparedStudy,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    composite: ValidationComposite,
    *,
    checkpoint_dir: Path,
) -> TrainingRunResult:
    checkpoint_path = _require_checkpoint_dir(checkpoint_dir)
    _require_model_device(model, device)
    if config.activation_checkpointing is not False:
        raise ValueError("activation checkpointing must remain disabled")

    seed_everything(config.seed)
    condition = CONDITIONS[config.condition]
    store = _summary_store(study, tokenizer, condition)
    excluded = study.length_exclusion_ids
    train_examples = filter_examples(study.tasks_by_split["train"], excluded)
    auxiliary_examples = filter_examples(study.auxiliary_examples, excluded)
    if not train_examples:
        raise ValueError(f"condition {config.condition} has no main examples after global exclusions")
    if condition.aux is not None and not auxiliary_examples:
        raise ValueError(f"condition {config.condition} has no auxiliary examples after global exclusions")

    validation_examples = filter_examples(
        study.tasks_by_split["validation"],
        val_scoring_exclusions(study),
    )
    if not validation_examples:
        raise ValueError("empty validation set after quarantine and adjudication exclusions")
    composite_examples = select_composite_examples(validation_examples, study.composite_validation_ids)

    main_dataset = InstructionDataset(train_examples, tokenizer, condition, store)
    auxiliary_dataset = InstructionDataset(
        auxiliary_examples,
        tokenizer,
        aux_encoding_for(condition),
        store,
    )
    main_items = _dataset_items(main_dataset)
    auxiliary_items = _dataset_items(auxiliary_dataset)
    stream = tuple(
        build_condition_stream(
            _response_lengths(main_items),
            _response_lengths(auxiliary_items),
            condition,
            config.max_steps,
            config.effective_batch_size,
            config.seed,
            aux_ratio=config.auxiliary_ratio,
        )
    )
    realized_truncation = _realized_truncation(
        stream,
        main_items,
        auxiliary_items,
        config.allowed_max_length,
    )
    if realized_truncation:
        raise ValueError(
            f"paper-profile training would truncate {realized_truncation} stream examples at "
            f"allowed_max_length={config.allowed_max_length}"
        )
    micro_batches = iter_micro_batches(
        stream,
        main_dataset,
        auxiliary_dataset,
        config.micro_batch_size,
        config.allowed_max_length,
    )

    validation_dataset = InstructionDataset(
        val_loss_subset(validation_examples),
        tokenizer,
        condition,
        store,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        collate_fn=lambda batch: instruction_collate_fn(
            batch,
            allowed_max_length=config.allowed_max_length,
        ),
    )

    def validation_loss() -> float:
        return calc_loss_loader(validation_loader, model, device)

    checkpoint = CheckpointManager(checkpoint_path)
    meter = BudgetMeter(device)
    train_loop(
        model=model,
        micro_batches=micro_batches,
        val_loss_fn=validation_loss,
        val_examples=composite_examples,
        composite=composite,
        ckpt=checkpoint,
        meter=meter,
        cfg=config,
        device=device,
        generate_fn=_generation_function(model, tokenizer, store, condition),
    )

    trace = _selection_trace(checkpoint)
    if tuple(point.step for point in trace) != _CHECKPOINT_STEPS:
        raise ValueError("selection trace must contain exactly steps 120, 240, 360, 480, and 600")
    best = max(trace, key=lambda point: point.composite_score)
    selection = TrainingSelectionMetrics(
        selected_step=best.step,
        best_composite=best.composite_score,
        rule_id_macro_f1=best.rule_id_macro_f1,
        correction_fix_rate=best.correction_fix_rate,
        joint_fix_rate=best.joint_fix_rate,
    )
    budget = meter.report()
    main_count = cast(int, budget["main_examples"])
    auxiliary_count = cast(int, budget["aux_examples"])
    compute = TrainingComputeMetrics(
        optimizer_steps=cast(Literal[600], budget["optimizer_steps"]),
        examples_seen=main_count + auxiliary_count,
        total_tokens=cast(int, budget["tokens_processed"]),
        supervised_tokens=cast(int, budget["supervised_tokens"]),
        peak_allocated_gpu_memory_gib=cast(float | None, budget["peak_gpu_mem_gib"]),
        wall_clock_train_s=cast(float, budget["wall_clock_s"]),
    )
    return TrainingRunResult(
        checkpoint_selection=selection,
        compute=compute,
        length=TrainingLengthMetrics(config.allowed_max_length, 0),
        selection_trace=trace,
        best_checkpoint=checkpoint.best_path,
    )
