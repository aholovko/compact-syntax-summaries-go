from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Any

import torch

from go_ast_assistant.paper4.runtime.loss import calc_loss_batch
from go_ast_assistant.paper4.training.config import TrainingConfig
from go_ast_assistant.paper4.training.schedule import cosine_with_warmup_lambda


def train_loop(
    *,
    model: torch.nn.Module,
    micro_batches: Iterable[tuple[torch.Tensor, torch.Tensor, dict[str, int]]],
    val_loss_fn: Callable[[], float],
    val_examples: tuple[Any, ...],
    composite: Any,
    ckpt: Any,
    meter: Any,
    cfg: TrainingConfig,
    device: torch.device,
    generate_fn: Callable[..., tuple[str, ...]],
) -> dict[str, list[Any]]:
    """Run the fixed 600-step paper loop and select on five full composites."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        eps=cfg.epsilon,
        weight_decay=cfg.weight_decay,
    )
    warmup_steps = round(cfg.warmup_ratio * cfg.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        cosine_with_warmup_lambda(
            cfg.max_steps,
            warmup_steps,
            cfg.minimum_learning_rate_ratio,
        ),
    )
    curves: dict[str, list[Any]] = {"train_loss": [], "val_loss": []}
    meter.start()
    model.train()
    optimizer.zero_grad()
    optimizer_steps = 0
    accumulated = 0
    step_loss = torch.zeros((), device=device)

    for inputs, targets, counts in micro_batches:
        if counts["n_supervised_tokens"] == 0:
            warnings.warn(
                "all-ignore micro-batch skipped after global length exclusion",
                stacklevel=2,
            )
            continue
        loss = calc_loss_batch(inputs, targets, model, device) / cfg.grad_accum_steps
        loss.backward()
        step_loss = step_loss + loss.detach() * cfg.grad_accum_steps
        meter.add_micro_batch(
            counts["n_main"],
            counts["n_aux"],
            counts["n_real_tokens"],
            counts["n_supervised_tokens"],
        )
        accumulated += 1
        if accumulated != cfg.grad_accum_steps:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.maximum_gradient_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        optimizer_steps += 1
        meter.add_step()
        curves["train_loss"].append((optimizer_steps, (step_loss / accumulated).item()))
        accumulated = 0
        step_loss = torch.zeros((), device=device)

        if optimizer_steps % cfg.checkpoint_every_steps == 0:
            model.eval()
            with torch.no_grad():
                validation_loss = val_loss_fn()
                result = composite.evaluate(model, val_examples, generate_fn)
                curves["val_loss"].append((optimizer_steps, validation_loss))
                ckpt.consider(
                    step=optimizer_steps,
                    result=result,
                    val_loss=validation_loss,
                    model=model,
                )
            model.train()
        if optimizer_steps == cfg.max_steps:
            break

    close = getattr(micro_batches, "close", None)
    if callable(close):
        close()
    if optimizer_steps != cfg.max_steps:
        raise ValueError(f"training stream ended after {optimizer_steps} optimizer steps; expected {cfg.max_steps}")
    meter.stop()
    return curves
