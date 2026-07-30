from __future__ import annotations

from go_ast_assistant.paper4.gocheck.fixture import Fixture, synthesize
from go_ast_assistant.paper4.gocheck.go_critic import parse_go_critic_output
from go_ast_assistant.paper4.gocheck.result import Finding, GoCheckResult
from go_ast_assistant.paper4.gocheck.runner import GoCheck, check, run
from go_ast_assistant.paper4.gocheck.toolchain import (
    ENABLED_CHECKS,
    GO_CRITIC_VERSION,
    STUDIED_CHECKS,
    ToolchainError,
    ToolchainInfo,
    resolve_toolchain,
)


__all__ = [
    "ENABLED_CHECKS",
    "GO_CRITIC_VERSION",
    "STUDIED_CHECKS",
    "Finding",
    "Fixture",
    "GoCheck",
    "GoCheckResult",
    "ToolchainError",
    "ToolchainInfo",
    "check",
    "parse_go_critic_output",
    "resolve_toolchain",
    "run",
    "synthesize",
]
