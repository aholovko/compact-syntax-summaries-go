from __future__ import annotations

from pathlib import Path

import pytest

from go_ast_assistant.paper4.runtime import tokenizer as tokenizer_module
from go_ast_assistant.paper4.runtime.tokenizer import ChatFormat, Llama3Tokenizer, load_local_tokenizer


_OFFICIAL_SPECIAL_TOKEN_NAMES = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|finetune_right_pad_id|>",
    "<|reserved_special_token_2|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",
    "<|eot_id|>",
    "<|python_tag|>",
    *(f"<|reserved_special_token_{index}|>" for index in range(3, 248)),
)
_OFFICIAL_SPECIAL_IDS = {name: token_id for token_id, name in enumerate(_OFFICIAL_SPECIAL_TOKEN_NAMES, start=128000)}


def _patch_tiny_local_bpe(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    loaded: list[Path] = []

    def fake_load(path: str) -> dict[bytes, int]:
        loaded.append(Path(path))
        return {b"a": 0}

    monkeypatch.setattr(tokenizer_module, "load_tiktoken_bpe", fake_load)
    return loaded


def _patch_byte_local_bpe(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    loaded: list[Path] = []

    def fake_load(path: str) -> dict[bytes, int]:
        loaded.append(Path(path))
        return {bytes((value,)): value for value in range(256)}

    monkeypatch.setattr(tokenizer_module, "load_tiktoken_bpe", fake_load)
    return loaded


def test_tokenizer_uses_exact_official_special_id_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"local fixture")
    loaded = _patch_tiny_local_bpe(monkeypatch)

    tokenizer = Llama3Tokenizer(str(path))

    assert loaded == [path]
    assert tokenizer.special == _OFFICIAL_SPECIAL_IDS
    assert tokenizer.model.n_vocab == 128256
    assert tokenizer.bos_token_id == 128000
    assert tokenizer.eos_token_id == 128001


def test_load_local_tokenizer_uses_only_the_supplied_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"local fixture")
    loaded = _patch_tiny_local_bpe(monkeypatch)

    chat = load_local_tokenizer(path)

    assert isinstance(chat, ChatFormat)
    assert loaded == [path]
    assert chat.tok.model.n_vocab == 128256


def test_local_tokenizer_encodes_and_decodes_with_explicit_bos_and_eos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"local fixture")
    loaded = _patch_byte_local_bpe(monkeypatch)
    tokenizer = Llama3Tokenizer(str(path))

    encoded = tokenizer.encode("abc", bos=True, eos=True)

    assert loaded == [path]
    assert encoded == [128000, ord("a"), ord("b"), ord("c"), 128001]
    assert tokenizer.decode(encoded[1:-1]) == "abc"


def test_chat_format_emits_the_canonical_system_user_and_assistant_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"local fixture")
    _patch_byte_local_bpe(monkeypatch)
    tokenizer = Llama3Tokenizer(str(path))
    chat = ChatFormat(tokenizer)

    encoded = chat.encode("question", system_message="policy")

    start_header = tokenizer.special["<|start_header_id|>"]
    end_header = tokenizer.special["<|end_header_id|>"]
    end_turn = tokenizer.special["<|eot_id|>"]
    line_break = tokenizer.encode("\n\n")
    expected = [
        tokenizer.bos_token_id,
        start_header,
        *tokenizer.encode("system"),
        end_header,
        *line_break,
        *tokenizer.encode("policy"),
        end_turn,
        start_header,
        *tokenizer.encode("user"),
        end_header,
        *line_break,
        *tokenizer.encode("question"),
        end_turn,
        start_header,
        *tokenizer.encode("assistant"),
        end_header,
        *line_break,
    ]
    assert encoded == expected
    assert chat.decode(tokenizer.encode("answer")) == "answer"


def test_load_local_tokenizer_rejects_a_missing_path_without_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _patch_tiny_local_bpe(monkeypatch)

    with pytest.raises(FileNotFoundError):
        load_local_tokenizer(tmp_path / "missing.model")

    assert loaded == []
