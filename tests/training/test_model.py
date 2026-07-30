from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from go_ast_assistant.paper4.runtime.model import llama3_model as model_module
from go_ast_assistant.paper4.runtime.model.llama3_model import Llama3Model, build_llama32_1b_instruct


_FIXED_CONFIG = {
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

_TINY_CONFIG = {
    "vocab_size": 64,
    "context_length": 64,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 64,
    "n_kv_groups": 2,
    "rope_base": 10000.0,
    "rope_freq": None,
    "dtype": torch.float32,
}


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_tiny_model_backward_preserves_the_transformer_residual(dtype: torch.dtype) -> None:
    config = {**_TINY_CONFIG, "dtype": dtype, "n_layers": 1}
    model = Llama3Model(config)
    model.train()
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_residual(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        values = inputs[0]
        captured.append((values, values.detach().clone()))

    handle = model.trf_blocks[0].norm1.register_forward_pre_hook(capture_residual)
    try:
        try:
            logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
        except RuntimeError as error:
            unsupported = "not implemented" in str(error).lower() or "unsupported" in str(error).lower()
            if dtype == torch.bfloat16 and unsupported:
                pytest.skip(f"local torch build does not support bfloat16 model execution: {error}")
            raise
    finally:
        handle.remove()

    logits.float().square().mean().backward()

    assert model.tok_emb.weight.grad is not None
    assert len(captured) == 1
    residual, snapshot = captured[0]
    torch.testing.assert_close(residual.detach(), snapshot)


def _unique_parameter_count(config: dict[str, Any]) -> int:
    embedding = config["vocab_size"] * config["emb_dim"]
    head_dim = config["emb_dim"] // config["n_heads"]
    kv_dim = config["n_kv_groups"] * head_dim
    attention = 2 * config["emb_dim"] ** 2 + 2 * config["emb_dim"] * kv_dim
    feed_forward = 3 * config["emb_dim"] * config["hidden_dim"]
    norms = 2 * config["emb_dim"]
    return embedding + config["n_layers"] * (attention + feed_forward + norms) + config["emb_dim"]


def test_fixed_llama32_1b_factory_uses_exact_config_count_and_tied_weights(monkeypatch) -> None:
    constructed: list[dict[str, Any]] = []

    class FakeModel:
        def __init__(self, config: dict[str, Any]) -> None:
            constructed.append(config)
            self.cfg = config
            self.tok_emb = SimpleNamespace(weight=object())
            self.out_head = SimpleNamespace(weight=object())

        def to(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("the fixed factory must construct on CPU without transferring")

    monkeypatch.setattr(model_module, "Llama3Model", FakeModel)

    model = build_llama32_1b_instruct()

    assert constructed == [_FIXED_CONFIG]
    assert model.cfg == _FIXED_CONFIG
    assert model.out_head.weight is model.tok_emb.weight
    assert _unique_parameter_count(model.cfg) == 1_235_814_400


def test_model_left_padded_real_token_logits_match_unpadded() -> None:
    torch.manual_seed(0)
    model = Llama3Model(dict(_TINY_CONFIG))
    model.eval()
    real = torch.randint(1, _TINY_CONFIG["vocab_size"], (1, 5))
    padded = torch.cat([torch.zeros(1, 2, dtype=torch.long), real], dim=1)
    total_length = padded.shape[1]
    causal = torch.tril(torch.ones(total_length, total_length, dtype=torch.bool))
    key_is_real = torch.tensor([[False, False, True, True, True, True, True]])
    attention_mask = (causal.unsqueeze(0) & key_is_real.unsqueeze(1)).unsqueeze(1)
    position_ids = torch.tensor([[0, 0, 0, 1, 2, 3, 4]])

    with torch.no_grad():
        padded_logits = model(padded, attn_mask=attention_mask, position_ids=position_ids)
        unpadded_logits = model(real)

    torch.testing.assert_close(padded_logits[:, 2:, :], unpadded_logits, rtol=0, atol=1e-5)


def test_model_state_dict_uses_the_exact_weight_loader_names() -> None:
    model = Llama3Model(dict(_TINY_CONFIG))
    expected = {
        "tok_emb.weight",
        "final_norm.weight",
        "out_head.weight",
        *{
            f"trf_blocks.{layer}.{leaf}"
            for layer in range(_TINY_CONFIG["n_layers"])
            for leaf in (
                "att.W_query.weight",
                "att.W_key.weight",
                "att.W_value.weight",
                "att.out_proj.weight",
                "ff.fc1.weight",
                "ff.fc2.weight",
                "ff.fc3.weight",
                "norm1.weight",
                "norm2.weight",
            )
        },
    }

    assert set(model.state_dict()) == expected
