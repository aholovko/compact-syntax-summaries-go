from __future__ import annotations

import json
from pathlib import Path

import pytest

from go_ast_assistant.paper4.preflight import (
    LocalModelLayout,
    validate_model_layout,
)


BUNDLE_ROOT = Path(__file__).resolve().parents[2]


def _tokenizer(root: Path) -> Path:
    path = root / "original" / "tokenizer.model"
    path.parent.mkdir(parents=True)
    path.write_text("synthetic tokenizer", encoding="utf-8")
    return path


def _monolithic_model(root: Path) -> None:
    _tokenizer(root)
    (root / "model.safetensors").write_bytes(b"not opened")


def _sharded_model(root: Path, weight_map: dict[str, str]) -> None:
    _tokenizer(root)
    shards = sorted(set(weight_map.values()))
    for name in shards:
        (root / name).write_bytes(b"not opened")
    payload = {"metadata": {"total_size": 2}, "weight_map": weight_map}
    (root / "model.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")


def test_model_layout_accepts_the_monolithic_shape(tmp_path: Path) -> None:
    _monolithic_model(tmp_path)

    layout = validate_model_layout(tmp_path)

    assert layout == LocalModelLayout(
        root=tmp_path,
        tokenizer_path=tmp_path / "original" / "tokenizer.model",
        weight_paths=(tmp_path / "model.safetensors",),
        weight_index_path=None,
    )


def test_model_layout_accepts_unique_shards_in_lexical_order(tmp_path: Path) -> None:
    _sharded_model(
        tmp_path,
        {
            "model.z": "model-00002-of-00002.safetensors",
            "model.a": "model-00001-of-00002.safetensors",
            "model.a_alias": "model-00001-of-00002.safetensors",
        },
    )

    layout = validate_model_layout(tmp_path)

    assert layout.weight_paths == (
        tmp_path / "model-00001-of-00002.safetensors",
        tmp_path / "model-00002-of-00002.safetensors",
    )
    assert layout.weight_index_path == tmp_path / "model.safetensors.index.json"


@pytest.mark.parametrize(
    "case",
    ["missing-tokenizer", "tokenizer-directory", "missing-weight", "weight-directory"],
)
def test_model_layout_requires_regular_tokenizer_and_weight_files(tmp_path: Path, case: str) -> None:
    _monolithic_model(tmp_path)
    tokenizer = tmp_path / "original" / "tokenizer.model"
    weight = tmp_path / "model.safetensors"
    target = tokenizer if case.startswith("tokenizer") else weight
    target.unlink()
    if case.endswith("directory"):
        target.mkdir()

    with pytest.raises(ValueError):
        validate_model_layout(tmp_path)


def test_monolithic_and_indexed_layouts_are_mutually_exclusive(tmp_path: Path) -> None:
    _sharded_model(tmp_path, {"model.a": "model-00001-of-00001.safetensors"})
    (tmp_path / "model.safetensors").write_bytes(b"ambiguous")

    with pytest.raises(ValueError):
        validate_model_layout(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"weight_map": {}},
        {"weight_map": {"": "model-00001-of-00001.safetensors"}},
        {"weight_map": {"model.a": 1}},
        {"weight_map": {"model.a": "model.bin"}},
        {"weight_map": {"model.a": "model.safetensors"}, "unknown": True},
    ],
)
def test_model_layout_rejects_malformed_index_shapes(tmp_path: Path, payload: dict[str, object]) -> None:
    _tokenizer(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_model_layout(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "/absolute.safetensors",
        "../parent.safetensors",
        "./dot.safetensors",
        "sub/shard.safetensors",
        r"sub\shard.safetensors",
        ".safetensors",
        "..safetensors",
    ],
)
def test_model_layout_rejects_non_simple_shard_names(tmp_path: Path, name: str) -> None:
    _tokenizer(tmp_path)
    index = {"weight_map": {"model.a": name}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_model_layout(tmp_path)


def test_model_layout_rejects_invalid_and_duplicate_key_index_json(tmp_path: Path) -> None:
    _tokenizer(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_model_layout(tmp_path)

    index_path.write_text(
        '{"weight_map":{"model.a":"a.safetensors","model.a":"b.safetensors"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        validate_model_layout(tmp_path)


@pytest.mark.parametrize("case", ["missing", "extra", "directory"])
def test_model_layout_requires_the_exact_regular_indexed_shard_set(tmp_path: Path, case: str) -> None:
    _sharded_model(tmp_path, {"model.a": "model-00001-of-00001.safetensors"})
    shard = tmp_path / "model-00001-of-00001.safetensors"
    if case == "missing":
        shard.unlink()
    elif case == "directory":
        shard.unlink()
        shard.mkdir()
    else:
        (tmp_path / "extra.safetensors").write_bytes(b"extra")

    with pytest.raises(ValueError):
        validate_model_layout(tmp_path)


@pytest.mark.parametrize("layout_kind", ["monolithic", "sharded"])
def test_model_layout_never_opens_tensor_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout_kind: str,
) -> None:
    if layout_kind == "monolithic":
        _monolithic_model(tmp_path)
    else:
        _sharded_model(tmp_path, {"model.a": "model-00001-of-00001.safetensors"})
    real_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name.endswith(".safetensors"):
            raise AssertionError(f"tensor content opened: {path}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    validate_model_layout(tmp_path)
