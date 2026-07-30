from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim, dtype=torch.float32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        input_dtype = values.dtype
        normalized = values.to(torch.float32)
        normalized = normalized * torch.rsqrt(normalized.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * normalized.to(input_dtype)).to(input_dtype)
