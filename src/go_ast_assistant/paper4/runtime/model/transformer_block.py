from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .feed_forward import FeedForward
from .grouped_query_attention import GroupedQueryAttention
from .rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        embedding_dimension = cfg["emb_dim"]
        self.att = GroupedQueryAttention(
            d_in=embedding_dimension,
            d_out=embedding_dimension,
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(embedding_dimension)
        self.norm2 = RMSNorm(embedding_dimension)
        self.compute_dtype = cfg["dtype"]

    def forward(
        self,
        values: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        *,
        start_pos: int = 0,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attn_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        assert values.dim() == 3, "values must be (batch, seq_len, emb_dim)"
        residual = values
        attended, next_cache = self.att(
            self.norm1(values).to(self.compute_dtype),
            cos,
            sin,
            start_pos=start_pos,
            cache=cache,
            use_cache=use_cache,
            attn_mask=attn_mask,
            position_ids=position_ids,
        )
        values = residual + attended
        return values + self.ff(self.norm2(values).to(self.compute_dtype)), next_cache
