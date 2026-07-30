from __future__ import annotations

import math
from collections.abc import Callable


def cosine_with_warmup_lambda(
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    """Return the fixed linear-warmup and cosine-decay multiplier."""

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return multiplier
