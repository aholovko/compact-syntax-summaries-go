from __future__ import annotations

from typing import Literal

import torch


def resolve_device(kind: Literal["cuda", "mps", "cpu"]) -> torch.device:
    if kind == "cpu":
        return torch.device("cpu")
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if kind == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    raise ValueError(f"unsupported device kind: {kind!r}")
