from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .rope import compute_rope


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        num_kv_groups: int,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_out // num_heads
        self.group_size = num_heads // num_kv_groups
        kv_dim = num_kv_groups * self.head_dim
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, kv_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, kv_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

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
        batch_size, token_count, _ = values.shape
        queries = self.W_query(values).view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        new_keys = self.W_key(values).view(batch_size, token_count, self.num_kv_groups, self.head_dim).transpose(1, 2)
        new_values = (
            self.W_value(values).view(batch_size, token_count, self.num_kv_groups, self.head_dim).transpose(1, 2)
        )

        if cos is None and sin is None:
            pass
        elif cos is None or sin is None:
            raise ValueError("RoPE requires both cos and sin or neither")
        else:
            queries = compute_rope(queries, cos, sin, offset=start_pos, position_ids=position_ids)
            new_keys = compute_rope(new_keys, cos, sin, offset=start_pos, position_ids=position_ids)

        if cache is None:
            keys, cached_values = new_keys, new_values
        else:
            keys = torch.cat((cache[0], new_keys), dim=2)
            cached_values = torch.cat((cache[1], new_values), dim=2)
        next_cache = (keys, cached_values) if use_cache else None

        expanded_keys = keys.repeat_interleave(self.group_size, dim=1)
        expanded_values = cached_values.repeat_interleave(self.group_size, dim=1)
        query_length = queries.shape[2]
        key_length = expanded_keys.shape[2]
        if attn_mask is not None:
            context = functional.scaled_dot_product_attention(
                queries,
                expanded_keys,
                expanded_values,
                attn_mask=attn_mask,
            )
        elif query_length == key_length:
            context = functional.scaled_dot_product_attention(
                queries,
                expanded_keys,
                expanded_values,
                is_causal=True,
            )
        else:
            assert query_length == 1, "chunked prefill is unsupported"
            context = functional.scaled_dot_product_attention(queries, expanded_keys, expanded_values)

        merged = context.transpose(1, 2).reshape(batch_size, token_count, self.d_out)
        return self.out_proj(merged), next_cache
