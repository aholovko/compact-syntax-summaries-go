from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from go_ast_assistant.paper4.preflight import LocalModelLayout
from go_ast_assistant.paper4.runtime import device as device_module
from go_ast_assistant.paper4.runtime import weights as weights_module
from go_ast_assistant.paper4.runtime.weights import _assign, load_fixed_local_model, load_local_weights


def _expected_tensor_names() -> tuple[str, ...]:
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for layer in range(16):
        prefix = f"model.layers.{layer}"
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
    result = tuple(names)
    assert len(result) == 146
    assert len(set(result)) == 146
    return result


EXPECTED_TENSORS = _expected_tensor_names()


def _slot() -> SimpleNamespace:
    return SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(1)))


class _FakeModel:
    def __init__(self) -> None:
        self.cfg = {"n_layers": 16}
        self.tok_emb = _slot()
        self.final_norm = _slot()
        self.out_head = SimpleNamespace(weight=self.tok_emb.weight)
        self.trf_blocks = []
        for _ in range(16):
            attention = SimpleNamespace(
                W_query=_slot(),
                W_key=_slot(),
                W_value=_slot(),
                out_proj=_slot(),
            )
            feed_forward = SimpleNamespace(fc1=_slot(), fc2=_slot(), fc3=_slot())
            self.trf_blocks.append(SimpleNamespace(att=attention, ff=feed_forward, norm1=_slot(), norm2=_slot()))

    def to(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("local weight loading must not transfer the model")


def _target_parameters(model: _FakeModel) -> dict[str, torch.Tensor]:
    targets = {
        "model.embed_tokens.weight": model.tok_emb.weight,
        "model.norm.weight": model.final_norm.weight,
    }
    for index, block in enumerate(model.trf_blocks):
        prefix = f"model.layers.{index}"
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


def _payload() -> dict[str, torch.Tensor]:
    return {name: torch.tensor([float(index + 1)]) for index, name in enumerate(EXPECTED_TENSORS)}


def _assert_loaded_payload(model: _FakeModel, payload: dict[str, torch.Tensor]) -> None:
    actual = _target_parameters(model)
    assert set(actual) == set(EXPECTED_TENSORS)
    for name, expected in payload.items():
        torch.testing.assert_close(actual[name], expected)
    assert model.out_head.weight is model.tok_emb.weight


def test_assign_preserves_parameter_identity_and_rejects_shape_mismatch() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))

    assigned = _assign(parameter, torch.tensor([1.0, 2.0]), "fixture.weight")

    assert assigned is parameter
    torch.testing.assert_close(parameter, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="fixture.weight"):
        _assign(parameter, torch.ones(3), "fixture.weight")


def _monolithic_layout(tmp_path: Path, payload: dict[str, torch.Tensor]) -> LocalModelLayout:
    tokenizer_path = tmp_path / "original" / "tokenizer.model"
    tokenizer_path.parent.mkdir()
    tokenizer_path.write_bytes(b"local tokenizer")
    weight_path = tmp_path / "model.safetensors"
    save_file(payload, weight_path)
    return LocalModelLayout(tmp_path, tokenizer_path, (weight_path,), None)


def _indexed_layout(
    tmp_path: Path,
    payloads: dict[str, dict[str, torch.Tensor]],
    weight_map: dict[str, str],
) -> LocalModelLayout:
    tokenizer_path = tmp_path / "original" / "tokenizer.model"
    tokenizer_path.parent.mkdir()
    tokenizer_path.write_bytes(b"local tokenizer")
    paths: list[Path] = []
    for filename, payload in payloads.items():
        path = tmp_path / filename
        save_file(payload, path)
        paths.append(path)
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": weight_map}) + "\n", encoding="utf-8")
    return LocalModelLayout(tmp_path, tokenizer_path, tuple(sorted(paths)), index_path)


def test_monolithic_loader_consumes_all_146_tensors(tmp_path: Path) -> None:
    payload = _payload()
    model = _FakeModel()

    load_local_weights(model, _monolithic_layout(tmp_path, payload))

    _assert_loaded_payload(model, payload)


def test_indexed_loader_consumes_all_146_tensors_from_their_declared_shards_once(tmp_path: Path) -> None:
    payload = _payload()
    weight_map = {
        name: "model-00001-of-00002.safetensors" if index % 2 == 0 else "model-00002-of-00002.safetensors"
        for index, name in enumerate(EXPECTED_TENSORS)
    }
    shard_payloads = {
        filename: {name: payload[name] for name, owner in weight_map.items() if owner == filename}
        for filename in set(weight_map.values())
    }
    layout = _indexed_layout(tmp_path, shard_payloads, weight_map)
    model = _FakeModel()

    load_local_weights(model, layout)

    _assert_loaded_payload(model, payload)


@pytest.mark.parametrize("case", ["missing", "extra", "lm-head"])
def test_loader_rejects_non_exact_official_tensor_sets(tmp_path: Path, case: str) -> None:
    payload = _payload()
    if case == "missing":
        payload.pop(EXPECTED_TENSORS[-1])
    elif case == "extra":
        payload["model.unexpected.weight"] = torch.ones(1)
    else:
        payload["lm_head.weight"] = torch.ones(1)
    layout = _monolithic_layout(tmp_path, payload)

    with pytest.raises(ValueError):
        load_local_weights(_FakeModel(), layout)


@pytest.mark.parametrize("case", ["missing", "extra", "lm-head", "unindexed-extra"])
def test_indexed_loader_rejects_non_exact_official_tensor_sets(tmp_path: Path, case: str) -> None:
    payload = _payload()
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    weight_map = {name: first_shard if index % 2 == 0 else second_shard for index, name in enumerate(EXPECTED_TENSORS)}
    shard_payloads = {
        first_shard: {name: payload[name] for name, owner in weight_map.items() if owner == first_shard},
        second_shard: {name: payload[name] for name, owner in weight_map.items() if owner == second_shard},
    }
    if case == "missing":
        missing = EXPECTED_TENSORS[-1]
        owner = weight_map.pop(missing)
        shard_payloads[owner].pop(missing)
    elif case in {"extra", "lm-head"}:
        extra = "model.unexpected.weight" if case == "extra" else "lm_head.weight"
        weight_map[extra] = first_shard
        shard_payloads[first_shard][extra] = torch.ones(1)
    else:
        shard_payloads[first_shard]["model.unindexed.weight"] = torch.ones(1)
    layout = _indexed_layout(tmp_path, shard_payloads, weight_map)

    with pytest.raises(ValueError):
        load_local_weights(_FakeModel(), layout)


@pytest.mark.parametrize("case", ["duplicate", "wrong-shard"])
def test_indexed_loader_rejects_duplicate_or_wrong_shard_ownership(tmp_path: Path, case: str) -> None:
    payload = _payload()
    first_name = EXPECTED_TENSORS[0]
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    weight_map = {name: first_shard if index % 2 == 0 else second_shard for index, name in enumerate(EXPECTED_TENSORS)}
    shard_payloads = {
        first_shard: {name: payload[name] for name, owner in weight_map.items() if owner == first_shard},
        second_shard: {name: payload[name] for name, owner in weight_map.items() if owner == second_shard},
    }
    if case == "duplicate":
        shard_payloads[second_shard][first_name] = payload[first_name]
    else:
        shard_payloads[first_shard].pop(first_name)
        shard_payloads[second_shard][first_name] = payload[first_name]
    layout = _indexed_layout(tmp_path, shard_payloads, weight_map)

    with pytest.raises(ValueError):
        load_local_weights(_FakeModel(), layout)


def test_load_fixed_local_model_builds_and_loads_on_cpu_without_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _monolithic_layout(tmp_path, _payload())
    model = _FakeModel()
    monkeypatch.setattr(weights_module, "build_llama32_1b_instruct", lambda: model)

    def unexpected_probe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fixed local loading must not probe or resolve a device")

    with monkeypatch.context() as probes:
        probes.setattr(torch.cuda, "is_available", unexpected_probe)
        probes.setattr(torch.cuda, "device_count", unexpected_probe)
        probes.setattr(torch.cuda, "current_device", unexpected_probe)
        probes.setattr(torch.backends.mps, "is_available", unexpected_probe)
        probes.setattr(torch.backends.mps, "is_built", unexpected_probe)
        probes.setattr(device_module, "resolve_device", unexpected_probe)
        probes.setattr(weights_module, "resolve_device", unexpected_probe, raising=False)
        loaded = load_fixed_local_model(layout)

    assert loaded is model
    assert all(parameter.device.type == "cpu" for parameter in _target_parameters(model).values())
    _assert_loaded_payload(model, _payload())
