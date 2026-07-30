from __future__ import annotations

from typing import Protocol, TypedDict

from torch.utils.data import Dataset

from go_ast_assistant.paper4.prompts.registry import SYSTEM_PROMPT
from go_ast_assistant.paper4.records import TaskExample
from go_ast_assistant.paper4.training.conditions import (
    Condition,
    SummaryStore,
    assemble_main_user_content,
)


EOT_ID = 128009


class _Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class _ChatFormat(Protocol):
    tok: _Tokenizer

    def encode(self, user_message: str, system_message: str | None = None) -> list[int]: ...


class _EncodedExample(TypedDict):
    input_ids: list[int]
    prompt_len: int


def encode_example(
    example: TaskExample,
    chat: _ChatFormat,
    condition: Condition,
    store: SummaryStore,
) -> _EncodedExample:
    code_tokens = len(chat.tok.encode(example.code))
    summary: str | None = None
    if condition.use_summary:
        rendered = store.render_for_main(example.id, code_tokens)
        if rendered.attached in {"present", "present_truncated"}:
            summary = rendered.text

    user_content = assemble_main_user_content(
        example.task_type,
        example.code,
        summary,
        example.target_checks,
    )
    prompt_ids = chat.encode(user_content, system_message=SYSTEM_PROMPT)
    response_ids = [*chat.tok.encode(example.target), EOT_ID]
    return {
        "input_ids": [*prompt_ids, *response_ids],
        "prompt_len": len(prompt_ids),
    }


class InstructionDataset(Dataset[_EncodedExample]):
    def __init__(
        self,
        examples: tuple[TaskExample, ...],
        chat: _ChatFormat,
        condition: Condition,
        store: SummaryStore,
    ) -> None:
        self._items = [encode_example(example, chat, condition, store) for example in examples]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> _EncodedExample:
        return self._items[index]
