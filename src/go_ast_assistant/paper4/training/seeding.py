from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed the training RNGs and request deterministic Torch algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def seeded_generator(seed: int) -> torch.Generator:
    """Return a local CPU generator without changing global Torch RNG state."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator
