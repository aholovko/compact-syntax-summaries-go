from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from go_ast_assistant.paper4.training.composite import CompositeResult


class _StateModel(Protocol):
    def state_dict(self) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True)
class _SelectionPoint:
    step: int
    validation_loss: float
    composite_score: float
    rule_id_macro_f1: float
    correction_fix_rate: float
    joint_fix_rate: float


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


class CheckpointManager:
    """Keep the first strict composite maximum in one atomic best checkpoint."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.best_path = self.out_dir / "best.pt"
        self.trace: list[_SelectionPoint] = []
        self._best = float("-inf")

    def consider(
        self,
        step: int,
        result: CompositeResult,
        val_loss: float,
        model: _StateModel,
    ) -> None:
        point = _SelectionPoint(
            step=step,
            validation_loss=val_loss,
            composite_score=result.composite,
            rule_id_macro_f1=result.components["rule_id_macro_f1"],
            correction_fix_rate=result.components["correction_fix_rate"],
            joint_fix_rate=result.components["joint_fix_rate"],
        )
        if result.composite > self._best:
            state = model.state_dict()
            _atomic_write(self.best_path, lambda path: torch.save(state, path))
            self._best = result.composite
        self.trace.append(point)


def _require_tied_weights(model: object) -> None:
    try:
        tied = model.out_head.weight is model.tok_emb.weight  # type: ignore[attr-defined]
    except AttributeError as error:
        raise ValueError("model must expose tied tok_emb and out_head weights") from error
    if not tied:
        raise ValueError("model token embedding and output-head weights must remain tied")


def load_best_checkpoint(model: object, path: Path) -> None:
    checkpoint = Path(path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError(f"checkpoint must be one regular non-symlink file: {checkpoint}")
    _require_tied_weights(model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or any(
        type(name) is not str or not isinstance(tensor, torch.Tensor) for name, tensor in state.items()
    ):
        raise ValueError("checkpoint state must be a string-to-tensor mapping")
    model.load_state_dict(state, strict=True)  # type: ignore[attr-defined]
    _require_tied_weights(model)
