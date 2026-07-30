from __future__ import annotations

import torch

from go_ast_assistant.paper4.runtime.model.grouped_query_attention import GroupedQueryAttention
from go_ast_assistant.paper4.runtime.model.rope import compute_rope, precompute_rope_params


def _attention() -> GroupedQueryAttention:
    torch.manual_seed(0)
    attention = GroupedQueryAttention(
        d_in=32,
        d_out=32,
        num_heads=4,
        num_kv_groups=2,
        dtype=torch.float32,
    )
    attention.eval()
    return attention


def _manual_attention_reference(
    attention: GroupedQueryAttention,
    values: torch.Tensor,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, tokens, _ = values.shape
    queries = attention.W_query(values).view(batch, tokens, attention.num_heads, attention.head_dim).transpose(1, 2)
    keys = attention.W_key(values).view(batch, tokens, attention.num_kv_groups, attention.head_dim).transpose(1, 2)
    projected_values = (
        attention.W_value(values).view(batch, tokens, attention.num_kv_groups, attention.head_dim).transpose(1, 2)
    )
    if cos is not None and sin is not None:
        queries = compute_rope(queries, cos, sin)
        keys = compute_rope(keys, cos, sin)
    keys = keys.repeat_interleave(attention.group_size, dim=1)
    projected_values = projected_values.repeat_interleave(attention.group_size, dim=1)
    scores = queries @ keys.transpose(2, 3)
    future = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1)
    weights = torch.softmax(scores.masked_fill(future, -torch.inf) / attention.head_dim**0.5, dim=-1)
    context = (weights @ projected_values).transpose(1, 2).reshape(batch, tokens, attention.d_out)
    return attention.out_proj(context)


def test_sdpa_matches_manual_attention_reference() -> None:
    attention = _attention()
    values = torch.randn(2, 10, 32)
    cos, sin = precompute_rope_params(head_dim=8, theta_base=10000, context_length=64)

    actual, cache = attention(values, cos, sin)
    expected = _manual_attention_reference(attention, values, cos, sin)

    assert cache is None
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    actual_without_rope, cache_without_rope = attention(values)
    expected_without_rope = _manual_attention_reference(attention, values)

    assert cache_without_rope is None
    torch.testing.assert_close(actual_without_rope, expected_without_rope, rtol=1e-5, atol=1e-5)


def test_left_padded_attention_matches_unpadded_real_tokens() -> None:
    attention = _attention()
    cos, sin = precompute_rope_params(head_dim=8, theta_base=10000, context_length=16)
    real = torch.randn(1, 4, 32)
    padded = torch.cat([torch.randn(1, 2, 32), real], dim=1)
    total_length = padded.shape[1]
    causal = torch.tril(torch.ones(total_length, total_length, dtype=torch.bool))
    key_is_real = torch.tensor([[False, False, True, True, True, True]])
    attention_mask = (causal.unsqueeze(0) & key_is_real.unsqueeze(1)).unsqueeze(1)
    position_ids = torch.tensor([[0, 0, 0, 1, 2, 3]])

    padded_output, _ = attention(
        padded,
        cos,
        sin,
        attn_mask=attention_mask,
        position_ids=position_ids,
    )
    real_output, _ = attention(real, cos, sin)

    torch.testing.assert_close(padded_output[:, 2:, :], real_output, rtol=0, atol=1e-6)
