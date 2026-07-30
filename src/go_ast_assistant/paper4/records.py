from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import Field, TypeAdapter

from go_ast_assistant.paper4.config import StrictModel, TrainingTaskType


class TaskExample(StrictModel):
    id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split: Literal["train", "validation", "test"]
    task_type: TrainingTaskType
    target_checks: tuple[str, ...]
    code: str
    target: str
    meta: dict[str, Any]


_ModelT = TypeVar("_ModelT", bound=StrictModel)


def reject_duplicate_json_keys(text: str) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error}") from error


def load_jsonl(path: Path, model: type[_ModelT]) -> tuple[_ModelT, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read required JSONL file: {path}") from error
    if not text:
        return ()
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"JSONL contains a blank row: {path}")
    adapter = TypeAdapter(model)
    rows: list[_ModelT] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            reject_duplicate_json_keys(line)
            rows.append(adapter.validate_json(line))
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid row {line_number} in {path}: {error}") from error
    return tuple(rows)


def load_task_examples(path: Path) -> tuple[TaskExample, ...]:
    return load_jsonl(path, TaskExample)
