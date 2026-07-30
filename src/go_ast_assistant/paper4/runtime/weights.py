from __future__ import annotations

from typing import Any

import torch
from safetensors.torch import load_file

from go_ast_assistant.paper4.preflight import LocalModelLayout
from go_ast_assistant.paper4.records import reject_duplicate_json_keys

from .model.llama3_model import Llama3Model, build_llama32_1b_instruct


def _expected_tensor_names() -> tuple[str, ...]:
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for layer_index in range(16):
        prefix = f"model.layers.{layer_index}"
        names.extend(
            (
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            )
        )
    return tuple(names)


_EXPECTED_TENSORS = frozenset(_expected_tensor_names())


def _assign(left: torch.Tensor, right: torch.Tensor, tensor_name: str = "unknown") -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch for {tensor_name}: expected {tuple(left.shape)}, got {tuple(right.shape)}")
    with torch.no_grad():
        left.copy_(right.to(device=left.device, dtype=left.dtype))
    return left


def _read_weight_index(layout: LocalModelLayout) -> dict[str, str]:
    assert layout.weight_index_path is not None
    try:
        parsed = reject_duplicate_json_keys(layout.weight_index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read model weight index: {layout.weight_index_path}") from error
    if not isinstance(parsed, dict):
        raise ValueError("model weight index must be an object")
    weight_map = parsed.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model weight index requires a weight_map")
    if any(type(name) is not str or type(owner) is not str for name, owner in weight_map.items()):
        raise ValueError("model weight index must map strings to strings")
    return weight_map


def _load_payload(layout: LocalModelLayout) -> dict[str, torch.Tensor]:
    if layout.weight_index_path is None:
        if len(layout.weight_paths) != 1:
            raise ValueError("monolithic model layout requires exactly one weight file")
        payload = load_file(str(layout.weight_paths[0]), device="cpu")
        if set(payload) != _EXPECTED_TENSORS:
            raise ValueError("monolithic checkpoint does not contain the exact official tensor set")
        return payload

    weight_map = _read_weight_index(layout)
    if set(weight_map) != _EXPECTED_TENSORS:
        raise ValueError("weight index does not contain the exact official tensor set")
    paths_by_name = {path.name: path for path in layout.weight_paths}
    if set(paths_by_name) != set(weight_map.values()):
        raise ValueError("weight paths do not match the indexed shards")

    combined: dict[str, torch.Tensor] = {}
    for shard_name in sorted(paths_by_name):
        shard = load_file(str(paths_by_name[shard_name]), device="cpu")
        declared = {name for name, owner in weight_map.items() if owner == shard_name}
        if set(shard) != declared:
            raise ValueError(f"shard contents do not match index ownership: {shard_name}")
        if combined.keys() & shard.keys():
            raise ValueError(f"tensor appears in more than one shard: {shard_name}")
        combined.update(shard)
    if set(combined) != _EXPECTED_TENSORS:
        raise ValueError("indexed checkpoint does not contain the exact official tensor set")
    return combined


def _targets(model: Any) -> dict[str, torch.Tensor]:
    targets = {
        "model.embed_tokens.weight": model.tok_emb.weight,
        "model.norm.weight": model.final_norm.weight,
    }
    for layer_index in range(16):
        block = model.trf_blocks[layer_index]
        prefix = f"model.layers.{layer_index}"
        targets.update(
            {
                f"{prefix}.self_attn.q_proj.weight": block.att.W_query.weight,
                f"{prefix}.self_attn.k_proj.weight": block.att.W_key.weight,
                f"{prefix}.self_attn.v_proj.weight": block.att.W_value.weight,
                f"{prefix}.self_attn.o_proj.weight": block.att.out_proj.weight,
                f"{prefix}.mlp.gate_proj.weight": block.ff.fc1.weight,
                f"{prefix}.mlp.up_proj.weight": block.ff.fc2.weight,
                f"{prefix}.mlp.down_proj.weight": block.ff.fc3.weight,
                f"{prefix}.input_layernorm.weight": block.norm1.weight,
                f"{prefix}.post_attention_layernorm.weight": block.norm2.weight,
            }
        )
    return targets


def load_local_weights(model: Llama3Model, layout: LocalModelLayout) -> None:
    payload = _load_payload(layout)
    targets = _targets(model)
    if set(targets) != _EXPECTED_TENSORS:
        raise ValueError("model does not expose the exact official tensor targets")
    for name in _expected_tensor_names():
        _assign(targets[name], payload[name], name)
    model.out_head.weight = model.tok_emb.weight


def load_fixed_local_model(layout: LocalModelLayout) -> Llama3Model:
    model = build_llama32_1b_instruct()
    load_local_weights(model, layout)
    return model
