from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .silu import SiLU


class FeedForward(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=cfg["dtype"])
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=cfg["dtype"])
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"])
        self.silu = SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.silu(self.fc1(values)) * self.fc2(values))
