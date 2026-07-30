from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from go_ast_assistant.paper4.config import StrictModel
from go_ast_assistant.paper4.records import load_jsonl


class Adjudication(StrictModel):
    id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split: Literal["train", "validation", "test"]
    resolution: Literal["exclude", "fixture_fix"]
    reason: str

    @field_validator("reason")
    @classmethod
    def require_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("adjudication reason must be nonempty")
        return value


def load_adjudications(path: Path) -> dict[str, Adjudication]:
    result: dict[str, Adjudication] = {}
    for adjudication in load_jsonl(path, Adjudication):
        if adjudication.id in result:
            raise ValueError(f"duplicate adjudication ID: {adjudication.id}")
        result[adjudication.id] = adjudication
    return result
