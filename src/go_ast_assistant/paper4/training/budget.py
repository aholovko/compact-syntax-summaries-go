from __future__ import annotations

import time

import torch


class BudgetMeter:
    """Track compute spent by one already-resolved training device."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self.main_examples = 0
        self.aux_examples = 0
        self.tokens_processed = 0
        self.supervised_tokens = 0
        self.optimizer_steps = 0
        self._started_at: float | None = None
        self.wall_clock_s = 0.0

    def start(self) -> None:
        self._started_at = time.perf_counter()
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

    def add_micro_batch(
        self,
        n_main: int,
        n_aux: int,
        n_real_tokens: int,
        n_supervised_tokens: int = 0,
    ) -> None:
        self.main_examples += n_main
        self.aux_examples += n_aux
        self.tokens_processed += n_real_tokens
        self.supervised_tokens += n_supervised_tokens

    def add_step(self) -> None:
        self.optimizer_steps += 1

    def stop(self) -> None:
        if self._started_at is not None:
            self.wall_clock_s = time.perf_counter() - self._started_at

    def report(self) -> dict[str, int | float | None]:
        peak = None
        if self._device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self._device) / 1024**3
        return {
            "main_examples": self.main_examples,
            "aux_examples": self.aux_examples,
            "tokens_processed": self.tokens_processed,
            "supervised_tokens": self.supervised_tokens,
            "optimizer_steps": self.optimizer_steps,
            "wall_clock_s": self.wall_clock_s,
            "peak_gpu_mem_gib": peak,
        }


def token_budget_guard(
    tokens_by_condition: dict[str, int],
    reference: str = "C1",
    tol: float = 0.05,
) -> dict[str, dict[str, float | bool]]:
    """Compare the two RQ2 conditions with the C1 token reference."""
    guarded_present = {condition for condition in tokens_by_condition if condition in {"C2", "C2-control"}}
    if guarded_present and reference not in tokens_by_condition:
        raise ValueError(
            f"token_budget_guard reference {reference!r} absent but guarded conditions present: {guarded_present}"
        )

    reference_tokens = tokens_by_condition.get(reference, 0)
    if guarded_present and reference_tokens <= 0:
        raise ValueError(
            f"token_budget_guard reference {reference!r} must be positive when guarded conditions are present"
        )
    result: dict[str, dict[str, float | bool]] = {}
    for condition, tokens in tokens_by_condition.items():
        delta = (tokens - reference_tokens) / reference_tokens if reference_tokens else 0.0
        guarded = condition in {"C2", "C2-control"}
        result[condition] = {
            "delta": delta,
            "guarded": guarded,
            "exceeds": guarded and abs(delta) > tol,
        }
    return result
