from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .kv_cache import KVCache
from .rms_norm import RMSNorm
from .rope import precompute_rope_params
from .transformer_block import TransformerBlock


class Llama3Model(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg["n_layers"]))
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        self.out_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        cache: KVCache | None = None,
        start_pos: int = 0,
        attn_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert start_pos + token_ids.shape[1] <= self.cfg["context_length"], "seq_len exceeds context_length"
        use_cache = cache is not None
        assert not (use_cache and self.training), "KV cache is generation-only"
        values = self.tok_emb(token_ids)

        for layer_index, block in enumerate(self.trf_blocks):
            values, next_cache = block(
                values,
                self.cos,
                self.sin,
                start_pos=start_pos,
                cache=cache.get(layer_index) if cache is not None else None,
                use_cache=use_cache,
                attn_mask=attn_mask,
                position_ids=position_ids,
            )
            if cache is not None:
                assert next_cache is not None
                cache.update(layer_index, next_cache)

        return self.out_head(self.final_norm(values).to(self.cfg["dtype"]))


def build_llama32_1b_instruct() -> Llama3Model:
    config = {
        "vocab_size": 128256,
        "context_length": 131072,
        "emb_dim": 2048,
        "n_heads": 32,
        "n_layers": 16,
        "hidden_dim": 8192,
        "n_kv_groups": 8,
        "rope_base": 500000.0,
        "rope_freq": {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        },
        "dtype": torch.bfloat16,
    }
    model = Llama3Model(config)
    model.out_head.weight = model.tok_emb.weight
    return model
