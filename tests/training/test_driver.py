from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import go_ast_assistant.paper4.training.driver as driver_module
from analysis.inputs import load_experiment_config
from go_ast_assistant.paper4.adjudication import Adjudication
from go_ast_assistant.paper4.preflight import ValidatedRequest
from go_ast_assistant.paper4.prepared_study import PreparedStudy, PreparedSummaryLine, PreparedSummaryRecord
from go_ast_assistant.paper4.records import TaskExample
from go_ast_assistant.paper4.training.composite import CompositeResult
from go_ast_assistant.paper4.training.conditions import AUX_ENCODING, CONDITIONS
from go_ast_assistant.paper4.training.config import training_config_for
from go_ast_assistant.paper4.training.driver import (
    TrainingComputeMetrics,
    TrainingLengthMetrics,
    TrainingRunResult,
    TrainingSelectionMetrics,
    TrainingSelectionPoint,
    aux_encoding_for,
    build_condition_stream,
    filter_examples,
    iter_micro_batches,
    run_training,
    select_composite_examples,
    val_loss_subset,
    val_scoring_exclusions,
)


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_STEPS = (120, 240, 360, 480, 600)


def _example(
    label: str,
    task_type: str = "rule_identification",
    *,
    split: str = "validation",
) -> TaskExample:
    return TaskExample(
        id=f"sha256:{hashlib.sha256(label.encode()).hexdigest()}",
        split=split,  # type: ignore[arg-type]
        task_type=task_type,  # type: ignore[arg-type]
        target_checks=("assignOp",),
        code=f"code:{label}",
        target="target",
        meta={},
    )


def _summary(example: TaskExample) -> PreparedSummaryRecord:
    return PreparedSummaryRecord(
        id=example.id,
        ok=True,
        parse_strategy="file",
        type_facts_available=False,
        lines=(PreparedSummaryLine(tier=0, depth=0, text="func f()", segments=()),),
        excluded_constructs=(),
        parse_error=None,
    )


def _request(condition: str = "C2-control", seed: int = 42) -> ValidatedRequest:
    return ValidatedRequest(
        config=load_experiment_config(BUNDLE_ROOT / "config" / "experiments.yaml"),
        condition=condition,  # type: ignore[arg-type]
        seed=seed,  # type: ignore[arg-type]
        profile="paper",
        study_data_dir=Path("unused-study"),
        model_dir=Path("unused-model"),
        output_dir=Path("unused-output"),
        device="cpu",
    )


def _point(step: int, score: float) -> TrainingSelectionPoint:
    return TrainingSelectionPoint(
        step=step,  # type: ignore[arg-type]
        validation_loss=1.0,
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


def _valid_result(tmp_path: Path) -> TrainingRunResult:
    best = tmp_path / "best.pt"
    best.write_bytes(b"private checkpoint")
    trace = tuple(_point(step, score) for step, score in zip(CHECKPOINT_STEPS, (0.2, 0.4, 0.7, 0.6, 0.5)))
    return TrainingRunResult(
        checkpoint_selection=_selection(360, 0.7),
        compute=TrainingComputeMetrics(
            optimizer_steps=600,
            examples_seen=19_200,
            total_tokens=100,
            supervised_tokens=50,
            peak_allocated_gpu_memory_gib=None,
            wall_clock_train_s=1.0,
        ),
        length=TrainingLengthMetrics(allowed_max_length=9_305, realized_truncation=0),
        selection_trace=trace,
        best_checkpoint=best,
    )


class _AlreadyPlacedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.transfer_calls = 0

    def to(self, *_args: object, **_kwargs: object):
        self.transfer_calls += 1
        raise AssertionError("run_training transferred the injected model")


class _DriverTokenizer:
    class _Raw:
        def encode(self, _text: str, allowed_special: object = None) -> list[int]:
            del allowed_special
            return [1]

    def __init__(self) -> None:
        self.tok = self._Raw()

    def encode(
        self,
        _user_message: str,
        _system_message: str | None = None,
        _allowed_special: object = None,
    ) -> list[int]:
        return [1, 2]

    def decode(self, _tokens: list[int]) -> str:
        return "decoded"


class _DriverMeter:
    instances: list[_DriverMeter] = []

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.main_examples = 0
        self.aux_examples = 0
        self.tokens = 0
        self.supervised = 0
        self.optimizer_steps = 0
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def add_micro_batch(
        self,
        n_main: int,
        n_aux: int,
        n_real_tokens: int,
        n_supervised_tokens: int,
    ) -> None:
        self.main_examples += n_main
        self.aux_examples += n_aux
        self.tokens += n_real_tokens
        self.supervised += n_supervised_tokens

    def add_step(self) -> None:
        self.optimizer_steps += 1

    def stop(self) -> None:
        self.stopped = True

    def report(self) -> dict[str, int | float | None]:
        return {
            "main_examples": self.main_examples,
            "aux_examples": self.aux_examples,
            "tokens_processed": self.tokens,
            "supervised_tokens": self.supervised,
            "optimizer_steps": self.optimizer_steps,
            "wall_clock_s": 2.5,
            "peak_gpu_mem_gib": None,
        }


def test_training_result_rejects_invalid_boundary_values(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    invalid_points = (
        lambda: _point(121, 0.5),
        lambda: replace(result.selection_trace[0], validation_loss=-1.0),
        lambda: replace(result.selection_trace[0], composite_score=float("nan")),
        lambda: replace(result.selection_trace[0], rule_id_macro_f1=1.1),
        lambda: replace(result.checkpoint_selection, best_composite=float("nan")),
        lambda: replace(result.checkpoint_selection, correction_fix_rate=-0.1),
        lambda: replace(result.compute, optimizer_steps=599),
        lambda: replace(result.compute, examples_seen=-1),
        lambda: replace(result.compute, total_tokens=-1),
        lambda: replace(result.compute, supervised_tokens=-1),
        lambda: replace(result.compute, peak_allocated_gpu_memory_gib=-1.0),
        lambda: replace(result.compute, wall_clock_train_s=-0.1),
        lambda: replace(result.compute, wall_clock_train_s=float("inf")),
        lambda: replace(result.length, allowed_max_length=9_304),
        lambda: replace(result.length, realized_truncation=1),
        lambda: _selection(121, 0.5),
    )
    for construct in invalid_points:
        with pytest.raises(ValueError):
            construct()

    with pytest.raises(ValueError, match="120.*240.*360.*480.*600"):
        replace(result, selection_trace=result.selection_trace[:-1] + (result.selection_trace[-2],))

    with pytest.raises(ValueError, match="maximum"):
        replace(result, checkpoint_selection=_selection(240, 0.4))

    tied_trace = tuple(_point(step, score) for step, score in zip(CHECKPOINT_STEPS, (0.2, 0.7, 0.7, 0.6, 0.5)))
    with pytest.raises(ValueError, match="first maximum"):
        replace(result, checkpoint_selection=_selection(360, 0.7), selection_trace=tied_trace)

    with pytest.raises(ValueError, match="selected metrics"):
        replace(result, checkpoint_selection=replace(result.checkpoint_selection, joint_fix_rate=0.6))

    result.best_checkpoint.unlink()
    with pytest.raises(ValueError, match="regular"):
        replace(result)


def test_training_result_rejects_checkpoint_symlink(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    target = result.best_checkpoint
    target.rename(tmp_path / "target.pt")
    target.symlink_to(tmp_path / "target.pt")

    with pytest.raises(ValueError, match="regular"):
        replace(result)


@pytest.mark.parametrize("invalid_kind", ("bool", "equal-float"))
@pytest.mark.parametrize(
    "field_name",
    (
        "step",
        "selected_step",
        "optimizer_steps",
        "examples_seen",
        "total_tokens",
        "supervised_tokens",
        "allowed_max_length",
        "realized_truncation",
    ),
)
def test_training_boundaries_reject_non_integer_exact_fields(field_name: str, invalid_kind: str) -> None:
    valid_values = {
        "step": 120,
        "selected_step": 120,
        "optimizer_steps": 600,
        "examples_seen": 1,
        "total_tokens": 1,
        "supervised_tokens": 1,
        "allowed_max_length": 9_305,
        "realized_truncation": 0,
    }
    invalid: object
    if invalid_kind == "bool":
        invalid = False if field_name == "realized_truncation" else True
    else:
        invalid = float(valid_values[field_name])

    with pytest.raises(ValueError, match="integer"):
        if field_name == "step":
            replace(_point(120, 0.5), step=invalid)  # type: ignore[arg-type]
        elif field_name == "selected_step":
            replace(_selection(120, 0.5), selected_step=invalid)  # type: ignore[arg-type]
        elif field_name in {
            "optimizer_steps",
            "examples_seen",
            "total_tokens",
            "supervised_tokens",
        }:
            replace(
                TrainingComputeMetrics(600, 1, 1, 1, None, 1.0),
                **{field_name: invalid},
            )
        elif field_name == "allowed_max_length":
            replace(TrainingLengthMetrics(9_305, 0), allowed_max_length=invalid)  # type: ignore[arg-type]
        else:
            replace(TrainingLengthMetrics(9_305, 0), realized_truncation=invalid)  # type: ignore[arg-type]


def test_filter_and_selection_helpers_preserve_prepared_tuple_order_and_repetitions() -> None:
    first = _example("a")
    second = _example("b", "correction")
    repeated = first
    rows = (first, second, repeated)

    assert filter_examples(rows, frozenset()) == rows
    assert filter_examples(rows, frozenset({second.id})) == (first, repeated)
    assert filter_examples(rows, frozenset({first.id, second.id})) == ()
    assert val_loss_subset(rows) == (first, repeated)
    assert select_composite_examples(rows, frozenset({first.id, second.id})) == rows
    assert rows == (first, second, repeated)

    with pytest.raises(ValueError, match="rule_identification"):
        val_loss_subset((second,))
    with pytest.raises(ValueError, match="empty|composite"):
        select_composite_examples(rows, frozenset())
    with pytest.raises(ValueError, match="missing|filtered"):
        select_composite_examples(rows, frozenset({_example("missing").id}))


def test_validation_scoring_exclusions_use_only_validation_quarantine_and_excludes() -> None:
    quarantined_validation = _example("quarantined-validation")
    quarantined_test = _example("quarantined-test", split="test")
    excluded_validation = _example("excluded-validation")
    fixture_fixed_validation = _example("fixture-fixed-validation")
    excluded_test = _example("excluded-test", split="test")
    adjudications = {
        excluded_validation.id: Adjudication(
            id=excluded_validation.id,
            split="validation",
            resolution="exclude",
            reason="synthetic validation exclusion",
        ),
        fixture_fixed_validation.id: Adjudication(
            id=fixture_fixed_validation.id,
            split="validation",
            resolution="fixture_fix",
            reason="synthetic fixture correction",
        ),
        excluded_test.id: Adjudication(
            id=excluded_test.id,
            split="test",
            resolution="exclude",
            reason="synthetic test exclusion",
        ),
    }
    study = PreparedStudy(
        root=Path("must-not-be-read"),
        tasks_by_split={
            "train": (),
            "validation": (quarantined_validation, excluded_validation, fixture_fixed_validation),
            "test": (quarantined_test, excluded_test),
        },
        length_budget=object(),  # type: ignore[arg-type]
        length_exclusion_ids=frozenset(),
        composite_validation_ids=frozenset(),
        quarantine_ids=frozenset({quarantined_validation.id, quarantined_test.id}),
        adjudications=adjudications,
        summaries=None,
        auxiliary_examples=(),
    )

    assert val_scoring_exclusions(study) == frozenset({quarantined_validation.id, excluded_validation.id})


def test_auxiliary_encoding_uses_raw_syntax_targets_only_for_c2() -> None:
    assert aux_encoding_for(CONDITIONS["C2"]) is AUX_ENCODING
    for condition_name in ("C0", "C1", "C2-control"):
        condition = CONDITIONS[condition_name]
        assert aux_encoding_for(condition) is condition


def test_condition_stream_stratifies_c2_only_and_preserves_control_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_lengths = (2, 5, 8)
    auxiliary_lengths = (1, 3, 7)
    base_stream = (("main", 2), ("aux", 0), ("main", 1), ("aux", 1))
    mixture_calls: list[tuple[int, int, float, int, int]] = []
    stratified_calls: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []

    def fake_mixture(n_main: int, n_aux: int, ratio: float, total: int, seed: int):
        mixture_calls.append((n_main, n_aux, ratio, total, seed))
        return base_stream

    def fake_stratified(
        aux_lengths: tuple[int, ...],
        reference_lengths: tuple[int, ...],
        k: int,
        seed: int,
    ):
        stratified_calls.append((aux_lengths, reference_lengths, k, seed))
        return (2, 0)

    monkeypatch.setattr(driver_module, "build_mixture_stream", fake_mixture)
    monkeypatch.setattr(driver_module, "length_stratified_aux_sample", fake_stratified)

    c2_stream = build_condition_stream(
        main_lengths,
        auxiliary_lengths,
        CONDITIONS["C2"],
        600,
        32,
        42,
    )
    control_stream = build_condition_stream(
        main_lengths,
        auxiliary_lengths,
        CONDITIONS["C2-control"],
        600,
        32,
        43,
        aux_ratio=0.2,
    )

    assert tuple(c2_stream) == (("main", 2), ("aux", 2), ("main", 1), ("aux", 0))
    assert tuple(control_stream) == base_stream
    assert mixture_calls == [
        (3, 3, 0.2, 19_200, 42),
        (3, 3, 0.2, 19_200, 43),
    ]
    assert stratified_calls == [((1, 3, 7), (2, 5, 8), 2, 42)]


def test_micro_batch_iterator_preserves_pool_indices_repetitions_and_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collated: list[tuple[str, ...]] = []

    def fake_collate(batch: list[dict[str, str]], *, allowed_max_length: int):
        assert allowed_max_length == 9_305
        collated.append(tuple(item["name"] for item in batch))
        return (
            torch.tensor(((1, 128004), (2, 3)), dtype=torch.long),
            torch.tensor(((-100, -100), (2, -100)), dtype=torch.long),
        )

    monkeypatch.setattr(driver_module, "instruction_collate_fn", fake_collate)
    main = ({"name": "main-a"}, {"name": "main-b"})
    auxiliary = ({"name": "aux-a"},)
    stream = (("main", 1), ("aux", 0), ("main", 0), ("main", 1))

    batches = tuple(iter_micro_batches(stream, main, auxiliary, 2, 9_305))

    assert collated == [("main-b", "aux-a"), ("main-a", "main-b")]
    assert [batch[2] for batch in batches] == [
        {"n_main": 1, "n_aux": 1, "n_real_tokens": 3, "n_supervised_tokens": 1},
        {"n_main": 2, "n_aux": 0, "n_real_tokens": 3, "n_supervised_tokens": 1},
    ]


def test_run_training_returns_complete_injected_result_without_reordering_pools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_second = _example("train-second", split="train")
    train_first = _example("train-first", split="train")
    train = (train_second, train_first)
    control = (train_first, train_second, train_first)
    validation = (_example("validation"),)
    summary_examples = (*train, *validation)
    study = PreparedStudy(
        root=Path("must-not-be-read"),
        tasks_by_split={"train": train, "validation": validation, "test": ()},
        length_budget=object(),  # type: ignore[arg-type]
        length_exclusion_ids=frozenset(),
        composite_validation_ids=frozenset({validation[0].id}),
        quarantine_ids=frozenset(),
        adjudications={},
        summaries={example.id: _summary(example) for example in summary_examples},
        auxiliary_examples=control,
    )
    config = training_config_for(_request())
    model = _AlreadyPlacedModel()
    tokenizer = _DriverTokenizer()
    device = torch.device("cpu")
    composite = object()
    checkpoint_dir = tmp_path / "private-checkpoints"
    checkpoint_dir.mkdir()
    dataset_calls: list[tuple[TaskExample, ...]] = []

    class RecordingDataset:
        def __init__(self, examples, _tokenizer, _condition, _store) -> None:
            self.examples = tuple(examples)
            dataset_calls.append(self.examples)
            self.items = [{"input_ids": [1, 2, 128009], "prompt_len": 1} for _example_item in self.examples]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):
            return self.items[index]

    def fake_train_loop(
        *,
        model: _AlreadyPlacedModel,
        micro_batches,
        val_loss_fn,
        val_examples,
        composite: object,
        ckpt,
        meter: _DriverMeter,
        cfg,
        device: torch.device,
        generate_fn,
    ) -> dict[str, list[object]]:
        del micro_batches
        assert callable(val_loss_fn)
        assert callable(generate_fn)
        assert tuple(val_examples) == validation
        assert composite is injected_composite
        assert cfg is config
        assert device == injected_device
        meter.start()
        meter.add_micro_batch(
            n_main=15_360,
            n_aux=3_840,
            n_real_tokens=57_600,
            n_supervised_tokens=38_400,
        )
        for _ in range(600):
            meter.add_step()
        for index, (step, score) in enumerate(
            zip(CHECKPOINT_STEPS, (0.2, 0.4, 0.7, 0.6, 0.5)),
            start=1,
        ):
            with torch.no_grad():
                model.weight.fill_(float(index))
            ckpt.consider(
                step=step,
                result=CompositeResult(
                    composite=score,
                    components={
                        "rule_id_macro_f1": score,
                        "correction_fix_rate": score,
                        "joint_fix_rate": score,
                    },
                ),
                val_loss=1.0 / index,
                model=model,
            )
        meter.stop()
        return {"train_loss": [], "val_loss": []}

    injected_composite = composite
    injected_device = device
    _DriverMeter.instances.clear()
    monkeypatch.setattr(driver_module, "InstructionDataset", RecordingDataset)
    monkeypatch.setattr(driver_module, "BudgetMeter", _DriverMeter)
    monkeypatch.setattr(driver_module, "train_loop", fake_train_loop)

    result = run_training(
        config,
        study,
        model,
        tokenizer,
        device,
        composite,  # type: ignore[arg-type]
        checkpoint_dir=checkpoint_dir,
    )

    assert train in dataset_calls
    assert control in dataset_calls
    assert study.tasks_by_split["train"] == train
    assert study.auxiliary_examples == control
    assert model.transfer_calls == 0
    assert len(_DriverMeter.instances) == 1
    assert _DriverMeter.instances[0].device == device
    assert _DriverMeter.instances[0].started and _DriverMeter.instances[0].stopped
    assert result.checkpoint_selection == _selection(360, 0.7)
    assert result.compute == TrainingComputeMetrics(600, 19_200, 57_600, 38_400, None, 2.5)
    assert result.length == TrainingLengthMetrics(9_305, 0)
    assert tuple(point.step for point in result.selection_trace) == CHECKPOINT_STEPS
    assert tuple(point.composite_score for point in result.selection_trace) == (0.2, 0.4, 0.7, 0.6, 0.5)
    assert result.best_checkpoint == checkpoint_dir / "best.pt"
    assert {path.name for path in checkpoint_dir.iterdir()} == {"best.pt"}
    state = torch.load(result.best_checkpoint, map_location="cpu", weights_only=True)
    assert torch.equal(state["weight"], torch.tensor(3.0))


def test_run_training_rejects_model_on_a_different_device_without_transferring_it(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "private-checkpoints"
    checkpoint_dir.mkdir()
    model = _AlreadyPlacedModel()
    study = PreparedStudy(
        root=Path("must-not-be-read"),
        tasks_by_split={"train": (), "validation": (), "test": ()},
        length_budget=object(),  # type: ignore[arg-type]
        length_exclusion_ids=frozenset(),
        composite_validation_ids=frozenset(),
        quarantine_ids=frozenset(),
        adjudications={},
        summaries=None,
        auxiliary_examples=(),
    )

    with pytest.raises(ValueError, match="device"):
        run_training(
            training_config_for(_request("C0")),
            study,
            model,
            _DriverTokenizer(),
            torch.device("meta"),
            object(),  # type: ignore[arg-type]
            checkpoint_dir=checkpoint_dir,
        )

    assert model.transfer_calls == 0
    assert not any(checkpoint_dir.iterdir())


@pytest.mark.parametrize("case", ["missing", "nonempty", "symlink"])
def test_run_training_rejects_unsafe_checkpoint_directories_first(tmp_path: Path, case: str) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    if case == "nonempty":
        checkpoint_dir.mkdir()
        (checkpoint_dir / "user-file").write_text("keep", encoding="utf-8")
    elif case == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        checkpoint_dir.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="checkpoint"):
        run_training(
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            checkpoint_dir=checkpoint_dir,
        )

    if case == "nonempty":
        assert (checkpoint_dir / "user-file").read_text(encoding="utf-8") == "keep"
