from __future__ import annotations

from collections.abc import Collection, Set
from pathlib import Path
from typing import Literal

import tiktoken
from tiktoken.load import load_tiktoken_bpe


_SPECIAL_TOKEN_NAMES = (
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
_SPECIAL_TOKENS = {name: token_id for token_id, name in enumerate(_SPECIAL_TOKEN_NAMES, start=128000)}
_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class Llama3Tokenizer:
    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"tokenizer file is missing or non-regular: {path}")

        self.special = dict(_SPECIAL_TOKENS)
        self.model = tiktoken.Encoding(
            name=path.name,
            pat_str=_PATTERN,
            mergeable_ranks=load_tiktoken_bpe(str(path)),
            special_tokens=self.special,
        )
        self.bos_token_id = self.special["<|begin_of_text|>"]
        self.eos_token_id = self.special["<|end_of_text|>"]

    def encode(
        self,
        text: str,
        bos: bool = False,
        eos: bool = False,
        allowed_special: Literal["all"] | Set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = (),
    ) -> list[int]:
        allowed = set() if allowed_special is None else allowed_special
        token_ids = self.model.encode(
            text,
            allowed_special=allowed,
            disallowed_special=disallowed_special,
        )
        if bos:
            token_ids.insert(0, self.bos_token_id)
        if eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.model.decode(token_ids)


class ChatFormat:
    def __init__(
        self,
        tokenizer: Llama3Tokenizer,
        *,
        default_system: str = "You are a helpful assistant.",
    ) -> None:
        self.tok = tokenizer
        self.default_system = default_system

    def _encode_header(self, role: str) -> list[int]:
        return [
            self.tok.special["<|start_header_id|>"],
            *self.tok.encode(role),
            self.tok.special["<|end_header_id|>"],
            *self.tok.encode("\n\n"),
        ]

    def encode(
        self,
        user_message: str,
        system_message: str | None = None,
        allowed_special: Literal["all"] | Set[str] | None = None,
    ) -> list[int]:
        system = self.default_system if system_message is None else system_message
        return [
            self.tok.special["<|begin_of_text|>"],
            *self._encode_header("system"),
            *self.tok.encode(system, allowed_special=allowed_special),
            self.tok.special["<|eot_id|>"],
            *self._encode_header("user"),
            *self.tok.encode(user_message),
            self.tok.special["<|eot_id|>"],
            *self._encode_header("assistant"),
        ]

    def decode(self, token_ids: list[int]) -> str:
        return self.tok.decode(token_ids)


def load_local_tokenizer(path: Path) -> ChatFormat:
    local_path = Path(path)
    if not local_path.is_file():
        raise FileNotFoundError(f"tokenizer file is missing or non-regular: {local_path}")
    return ChatFormat(Llama3Tokenizer(local_path))
