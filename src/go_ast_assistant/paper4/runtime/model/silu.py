from __future__ import annotations

import torch
from torch import nn


class SiLU(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * torch.sigmoid(values)
