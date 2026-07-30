from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from go_ast_assistant.paper4.gocheck.toolchain import ToolchainInfo, _local_environment


_MISSING_DEPENDENCY = (
    "cannot find package",
    "no required module",
    "cannot find module",
    "missing go.sum",
)


def gofmt_parse_ok(source: str, toolchain: ToolchainInfo) -> bool:
    try:
        completed = subprocess.run(
            [toolchain.gofmt_binary, "-e"],
            input=source,
            capture_output=True,
            text=True,
            timeout=60,
            env=_local_environment(toolchain.go_binary),
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0


def go_build_status(pkg_dir: Path, toolchain: ToolchainInfo) -> Literal["OK", "FAIL", "NA"]:
    try:
        completed = subprocess.run(
            [toolchain.go_binary, "build", "./..."],
            cwd=pkg_dir,
            capture_output=True,
            text=True,
            timeout=120,
            env=_local_environment(
                toolchain.go_binary,
                workspace=pkg_dir,
                disable_proxy=True,
            ),
        )
    except subprocess.TimeoutExpired:
        return "FAIL"
    if completed.returncode == 0:
        return "OK"
    output = (completed.stderr + completed.stdout).lower()
    if any(fragment in output for fragment in _MISSING_DEPENDENCY):
        return "NA"
    return "FAIL"
