from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from go_ast_assistant.paper4.prompts.registry import SYSTEM_PROMPT
from go_ast_assistant.paper4.records import TaskExample
from go_ast_assistant.paper4.training.conditions import (
    AUX_ENCODING,
    CONDITIONS,
    Attached,
    NullSummaryStore,
    SummaryRender,
)
from go_ast_assistant.paper4.training.instruction_dataset import (
    EOT_ID,
    InstructionDataset,
    encode_example,
)


class _FakeTokenizer:
    def encode(self, text: str, allowed_special: object = None) -> list[int]:
        return [ord(character) for character in text]


class _FakeChat:
    def __init__(self) -> None:
        self.tok = _FakeTokenizer()
        self.user_messages: list[str] = []
        self.system_messages: list[str | None] = []

    def encode(
        self,
        user_message: str,
        system_message: str | None = None,
        allowed_special: object = None,
    ) -> list[int]:
        self.user_messages.append(user_message)
        self.system_messages.append(system_message)
        return [700, *self.tok.encode(user_message), 999]


@dataclass
class _RecordingStore:
    text: str = "func f()"
    attached: Attached = "present"
    calls: list[tuple[str, int]] = field(default_factory=list)

    def render_for_main(self, snippet_id: str, code_tokens: int) -> SummaryRender:
        self.calls.append((snippet_id, code_tokens))
        return SummaryRender(self.text, self.attached)


def _example(
    character: str = "a",
    *,
    task_type: str = "rule_identification",
    target_checks: tuple[str, ...] = ("elseif",),
) -> TaskExample:
    return TaskExample(
        id=f"sha256:{character * 64}",
        split="train",
        task_type=task_type,  # type: ignore[arg-type]
        target_checks=target_checks,
        code=f"package {character}\n",
        target='["elseif"]',
        meta={},
    )


def test_encode_example_records_the_response_mask_boundary_and_appends_eot() -> None:
    example = _example()
    chat = _FakeChat()

    item = encode_example(example, chat, CONDITIONS["C0"], NullSummaryStore())

    assert item["input_ids"][: item["prompt_len"]] == [700, *chat.tok.encode(chat.user_messages[0]), 999]
    assert item["input_ids"][item["prompt_len"] :] == chat.tok.encode(example.target) + [EOT_ID]
    assert EOT_ID == 128009
    assert chat.system_messages == [SYSTEM_PROMPT]


@pytest.mark.parametrize(
    ("task_type", "checks", "expected"),
    [
        ("correction", ("captLocal", "elseif"), "checks: captLocal, elseif."),
        ("explanation", ("elseif",), "checks: elseif."),
    ],
)
def test_encode_example_threads_target_checks_to_labeled_renderers(
    task_type: str,
    checks: tuple[str, ...],
    expected: str,
) -> None:
    chat = _FakeChat()

    encode_example(
        _example(task_type=task_type, target_checks=checks),
        chat,
        CONDITIONS["C0"],
        NullSummaryStore(),
    )

    assert expected in chat.user_messages[0]


@pytest.mark.parametrize("condition_name", ["C1", "C2", "C2-control"])
def test_each_summary_condition_attaches_the_injected_prepared_summary(condition_name: str) -> None:
    example = _example()
    store = _RecordingStore()
    chat = _FakeChat()

    encode_example(example, chat, CONDITIONS[condition_name], store)

    assert store.calls == [(example.id, len(chat.tok.encode(example.code)))]
    assert chat.user_messages[0].rindex("```") < chat.user_messages[0].index("SYNTAX SUMMARY:\nfunc f()")


def test_auxiliary_encoding_never_requests_or_attaches_its_target_summary() -> None:
    store = _RecordingStore()

    chat = _FakeChat()
    encode_example(_example(task_type="syntax_summary"), chat, AUX_ENCODING, store)

    assert store.calls == []
    assert "SYNTAX SUMMARY" not in chat.user_messages[0]


def test_instruction_dataset_preserves_injected_tuple_order_and_repetitions() -> None:
    first = _example("a")
    second = _example("b")
    examples = (second, second, first)
    store = _RecordingStore()
    chat = _FakeChat()

    dataset = InstructionDataset(examples, chat, CONDITIONS["C1"], store)

    assert len(dataset) == 3
    assert [snippet_id for snippet_id, _ in store.calls] == [second.id, second.id, first.id]
    encoded_prompts = [dataset[index]["input_ids"][: dataset[index]["prompt_len"]] for index in range(len(dataset))]
    expected_prompts = [[700, *chat.tok.encode(message), 999] for message in chat.user_messages]
    assert encoded_prompts == expected_prompts
    assert encoded_prompts[0] == encoded_prompts[1]
    assert encoded_prompts[0] != encoded_prompts[2]
    assert examples == (second, second, first)
    assert set(dataset[0]) == {
        "input_ids",
        "prompt_len",
    }
