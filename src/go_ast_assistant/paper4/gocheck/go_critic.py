from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Literal

from go_ast_assistant.paper4.gocheck.result import Finding
from go_ast_assistant.paper4.gocheck.toolchain import ToolchainInfo, _local_environment


_FINDING_LINE = re.compile(r"^.+?:(\d+):(\d+):\s+(\w+):", re.MULTILINE)
_DEGRADED = re.compile(r"could not (load|import)|type-check|no required module", re.IGNORECASE)


def parse_go_critic_output(text: str, enable: tuple[str, ...]) -> tuple[Finding, ...]:
    keep_all = any(check.startswith("#") for check in enable)
    allowed = set(enable)
    findings = []
    for match in _FINDING_LINE.finditer(text):
        check = match.group(3)
        if keep_all or check in allowed:
            findings.append(Finding(check=check, line=int(match.group(1)), col=int(match.group(2))))
    return tuple(findings)


def lint(
    path: Path,
    enable: tuple[str, ...],
    toolchain: ToolchainInfo,
) -> tuple[tuple[Finding, ...], Literal["ok", "load_degraded", "load_failed"]]:
    try:
        completed = subprocess.run(
            [
                toolchain.go_critic_binary,
                "check",
                f"-enable={','.join(enable)}",
                os.fspath(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=_local_environment(
                toolchain.go_binary,
                workspace=path.parent,
                disable_proxy=True,
            ),
        )
    except subprocess.TimeoutExpired:
        return (), "load_failed"

    output = completed.stdout + completed.stderr
    findings = parse_go_critic_output(output, enable)
    degraded = _DEGRADED.search(output) is not None
    if completed.returncode != 0 and not findings and not degraded:
        return (), "load_failed"
    return findings, "load_degraded" if degraded else "ok"
