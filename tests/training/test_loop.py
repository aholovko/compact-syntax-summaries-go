from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import go_ast_assistant.paper4.training.loop as loop_module
from analysis.inputs import load_experiment_config
from go_ast_assistant.paper4.preflight import ValidatedRequest
from go_ast_assistant.paper4.training.composite import CompositeResult
from go_ast_assistant.paper4.training.config import training_config_for
from go_ast_assistant.paper4.training.loop import train_loop


BUNDLE_ROOT = Path(__file__).resolve().parents[2]


def _config():
    request = ValidatedRequest(
        config=load_experiment_config(BUNDLE_ROOT / "config" / "experiments.yaml"),
        condition="C0",
        seed=42,
        profile="paper",
        study_data_dir=Path("unused-study"),
        model_dir=Path("unused-model"),
        output_dir=Path("unused-output"),
        device="cpu",
    )
    return training_config_for(request)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.transfer_calls = 0

    def to(self, *_args: object, **_kwargs: object):
        self.transfer_calls += 1
        raise AssertionError("train_loop transferred an already placed model")


class _Meter:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.optimizer_steps = 0
        self.examples_seen = 0
        self.tokens = 0
        self.supervised = 0

    def start(self) -> None:
        self.started = True

    def add_micro_batch(
        self,
        n_main: int,
        n_aux: int,
        n_real_tokens: int,
        n_supervised_tokens: int,
    ) -> None:
        self.examples_seen += n_main + n_aux
        self.tokens += n_real_tokens
        self.supervised += n_supervised_tokens

    def add_step(self) -> None:
        self.optimizer_steps += 1

    def stop(self) -> None:
        self.stopped = True


class _Composite:
    def __init__(self) -> None:
        self.steps: list[int] = []
        self.eval_states: list[tuple[bool, bool]] = []

    def evaluate(self, model: _TinyModel, examples: tuple[object, ...], generate_fn):
        del examples, generate_fn
        self.eval_states.append((model.training, torch.is_grad_enabled()))
        step = len(self.steps) + 1
        self.steps.append(step)
        score = step / 10
        return CompositeResult(
            composite=score,
            components={
                "rule_id_macro_f1": score,
                "correction_fix_rate": score,
                "joint_fix_rate": score,
            },
        )


class _Checkpoint:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float, float]] = []

    def consider(
        self,
        step: int,
        result: CompositeResult,
        val_loss: float,
        model: _TinyModel,
    ) -> None:
        del model
        self.calls.append((step, result.composite, val_loss))


def _micro_batches():
    inputs = torch.ones((2, 2), dtype=torch.long)
    targets = torch.ones((2, 2), dtype=torch.long)
    counts = {"n_main": 2, "n_aux": 0, "n_real_tokens": 4, "n_supervised_tokens": 2}
    for _ in range(600 * 16):
        yield inputs, targets, counts


def test_loop_runs_exact_paper_steps_and_only_five_full_composites(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _TinyModel()
    initial_weight = model.weight.detach().clone()
    meter = _Meter()
    composite = _Composite()
    checkpoint = _Checkpoint()
    validation_calls = 0
    validation_states: list[tuple[bool, bool]] = []

    def fake_loss(_inputs, _targets, current_model, _device):
        return current_model.weight.square() + 1.0

    def validation_loss() -> float:
        nonlocal validation_calls
        validation_calls += 1
        validation_states.append((model.training, torch.is_grad_enabled()))
        return 1.0

    monkeypatch.setattr(loop_module, "calc_loss_batch", fake_loss)

    curves = train_loop(
        model=model,
        micro_batches=_micro_batches(),
        val_loss_fn=validation_loss,
        val_examples=(),
        composite=composite,
        ckpt=checkpoint,
        meter=meter,
        cfg=_config(),
        device=torch.device("cpu"),
        generate_fn=lambda _examples, _cap: (),
    )

    assert meter.started and meter.stopped
    assert meter.optimizer_steps == 600
    assert meter.examples_seen == 19_200
    assert meter.tokens == 38_400
    assert meter.supervised == 19_200
    assert validation_calls == 5
    assert validation_states == [(False, False)] * 5
    assert composite.eval_states == [(False, False)] * 5
    assert [step for step, _, _ in checkpoint.calls] == [120, 240, 360, 480, 600]
    assert len(curves["train_loss"]) == 600
    assert [step for step, _ in curves["val_loss"]] == [120, 240, 360, 480, 600]
    assert model.training is True
    assert model.transfer_calls == 0
    assert not torch.equal(model.weight.detach(), initial_weight)
