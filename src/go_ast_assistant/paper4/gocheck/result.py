from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Finding:
    check: str
    line: int
    col: int


@dataclass(frozen=True)
class GoCheckResult:
    parse_ok: bool
    build_status: Literal["OK", "FAIL", "NA"]
    findings: tuple[Finding, ...]
    tool_status: Literal["ok", "load_degraded", "load_failed"]
