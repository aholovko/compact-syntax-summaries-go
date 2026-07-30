from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import go_ast_assistant.paper4.training.checkpoint as checkpoint_module
from go_ast_assistant.paper4.training.checkpoint import CheckpointManager, load_best_checkpoint
from go_ast_assistant.paper4.training.composite import CompositeResult


COMPONENT_NAMES = ("rule_id_macro_f1", "correction_fix_rate", "joint_fix_rate")


def _result(score: float) -> CompositeResult:
    return CompositeResult(composite=score, components=dict.fromkeys(COMPONENT_NAMES, score))


def test_checkpoint_manager_writes_only_atomic_best_and_preserves_first_tie(tmp_path) -> None:
    model = nn.Linear(1, 1, bias=False)
    manager = CheckpointManager(tmp_path)
    points = (
        (120, 0.4, 1.0),
        (240, 0.7, 2.0),
        (360, 0.7, 3.0),
        (480, 0.6, 4.0),
        (600, 0.5, 5.0),
    )

    for step, score, weight in points:
        with torch.no_grad():
            model.weight.fill_(weight)
        manager.consider(step=step, result=_result(score), val_loss=1.0, model=model)

    assert {path.name for path in tmp_path.iterdir()} == {"best.pt"}
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / "last.pt").exists()
    assert not (tmp_path / "selection_trace.json").exists()
    assert tuple(point.step for point in manager.trace) == (120, 240, 360, 480, 600)
    state = torch.load(manager.best_path, map_location="cpu", weights_only=True)
    assert torch.equal(state["weight"], torch.tensor([[2.0]]))


def test_failed_improvement_preserves_atomic_best_and_selection_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1, bias=False)
    manager = CheckpointManager(tmp_path)
    with torch.no_grad():
        model.weight.fill_(1.0)
    manager.consider(step=120, result=_result(0.4), val_loss=1.0, model=model)
    original = manager.best_path.read_bytes()
    real_save = checkpoint_module.torch.save

    def fail_after_partial_write(state, path, *args, **kwargs) -> None:
        del state, args, kwargs
        Path(path).write_bytes(b"partial checkpoint")
        raise RuntimeError("simulated checkpoint write failure")

    with torch.no_grad():
        model.weight.fill_(2.0)
    monkeypatch.setattr(checkpoint_module.torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated checkpoint"):
        manager.consider(step=240, result=_result(0.9), val_loss=0.9, model=model)

    assert manager.best_path.read_bytes() == original
    assert {path.name for path in tmp_path.iterdir()} == {"best.pt"}
    assert tuple(point.step for point in manager.trace) == (120,)

    monkeypatch.setattr(checkpoint_module.torch, "save", real_save)
    with torch.no_grad():
        model.weight.fill_(3.0)
    manager.consider(step=360, result=_result(0.8), val_loss=0.8, model=model)

    state = torch.load(manager.best_path, map_location="cpu", weights_only=True)
    assert torch.equal(state["weight"], torch.tensor([[3.0]]))
    assert {path.name for path in tmp_path.iterdir()} == {"best.pt"}
    assert tuple(point.step for point in manager.trace) == (120, 360)


class _LoadSpy:
    def __init__(self, *, tied: bool = True, break_tie_on_load: bool = False) -> None:
        shared = object()
        self.tok_emb = SimpleNamespace(weight=shared)
        self.out_head = SimpleNamespace(weight=shared if tied else object())
        self.break_tie_on_load = break_tie_on_load
        self.loaded: tuple[dict[str, torch.Tensor], bool] | None = None

    def load_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool) -> None:
        self.loaded = (state, strict)
        if self.break_tie_on_load:
            self.out_head.weight = object()

    def to(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("checkpoint reload transferred the model")


def test_safe_best_checkpoint_load_is_cpu_strict_weights_only_and_keeps_tie(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "best.pt"
    path.write_bytes(b"local checkpoint placeholder")
    state = {"tok_emb.weight": torch.ones(1)}
    calls: list[tuple[object, object, object]] = []

    def fake_load(source, *, map_location, weights_only):
        calls.append((source, map_location, weights_only))
        return state

    monkeypatch.setattr(checkpoint_module.torch, "load", fake_load)
    model = _LoadSpy()

    load_best_checkpoint(model, path)

    assert calls == [(path, "cpu", True)]
    assert model.loaded == (state, True)
    assert model.out_head.weight is model.tok_emb.weight


def test_safe_best_checkpoint_rejects_untied_model_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "best.pt"
    path.write_bytes(b"local checkpoint placeholder")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("an untied model must be rejected before torch.load")

    monkeypatch.setattr(checkpoint_module.torch, "load", unexpected_load)

    with pytest.raises(ValueError, match="tied"):
        load_best_checkpoint(_LoadSpy(tied=False), path)


def test_safe_best_checkpoint_rejects_load_that_breaks_weight_tie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "best.pt"
    path.write_bytes(b"local checkpoint placeholder")
    state = {"tok_emb.weight": torch.ones(1)}
    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *_args, **_kwargs: state)

    with pytest.raises(ValueError, match="tied"):
        load_best_checkpoint(_LoadSpy(break_tie_on_load=True), path)


@pytest.mark.parametrize("payload", [{1: torch.ones(1)}, {"weight": 1.0}, [torch.ones(1)]])
def test_safe_best_checkpoint_rejects_non_string_tensor_mapping(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    path = tmp_path / "best.pt"
    path.write_bytes(b"local checkpoint placeholder")
    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *_args, **_kwargs: payload)

    with pytest.raises(ValueError, match="state"):
        load_best_checkpoint(_LoadSpy(), path)


def test_safe_best_checkpoint_requires_one_regular_non_symlink_best_file(tmp_path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="regular"):
        load_best_checkpoint(_LoadSpy(), missing)

    target = tmp_path / "target.pt"
    target.write_bytes(b"checkpoint")
    link = tmp_path / "best.pt"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular"):
        load_best_checkpoint(_LoadSpy(), link)
