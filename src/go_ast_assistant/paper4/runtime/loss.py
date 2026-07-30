from __future__ import annotations

from collections.abc import Sized
from typing import Any

import torch
import torch.nn.functional as functional


def calc_loss_batch(
    input_batch: Any,
    target_batch: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    inputs = input_batch.to(device)
    targets = target_batch.to(device)
    logits = model(inputs)
    return functional.cross_entropy(logits.flatten(0, 1), targets.flatten())


def calc_loss_loader(
    data_loader: Sized,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    available = len(data_loader)
    count = available if num_batches is None else min(num_batches, available)
    if count <= 0:
        return float("nan")

    total = 0.0
    for batch_index, (inputs, targets) in enumerate(data_loader):  # type: ignore[attr-defined]
        if batch_index >= count:
            break
        total += calc_loss_batch(inputs, targets, model, device).item()
    return total / count
