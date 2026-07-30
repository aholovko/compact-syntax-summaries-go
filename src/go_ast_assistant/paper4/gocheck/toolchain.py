from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


GO_CRITIC_VERSION = "v0.14.4"
GO_CRITIC_MODULE = "github.com/go-critic/go-critic"
STUDIED_CHECKS: tuple[str, ...] = (
    "assignOp",
    "builtinShadow",
    "captLocal",
    "commentFormatting",
    "elseif",
    "ifElseChain",
    "paramTypeCombine",
    "singleCaseSwitch",
)
ENABLED_CHECKS: tuple[str, ...] = ("#style",)

_GO_VERSION = re.compile(r"(?<![A-Za-z0-9_])(go(\d+)\.(\d+)\.(\d+))(?![A-Za-z0-9_.])")


@dataclass(frozen=True)
class ToolchainInfo:
    go_binary: Path
    gofmt_binary: Path
    go_critic_binary: Path
    go_version: str
    go_critic_version: Literal["v0.14.4"]


class ToolchainError(RuntimeError):
    pass


def _local_environment(
    go_binary: Path,
    *,
    workspace: Path | None = None,
    disable_proxy: bool = False,
) -> dict[str, str]:
    environment = {
        "GOENV": "off",
        "GOTOOLCHAIN": "local",
        "GOWORK": "off",
        "PATH": os.fspath(go_binary.parent),
    }
    if workspace is not None:
        environment["GOCACHE"] = os.fspath(workspace / ".gocache")
        environment["GOPATH"] = os.fspath(workspace / ".gopath")
    if disable_proxy:
        environment["GOPROXY"] = "off"
    return environment


def _resolved_executable(name: str, located: str | os.PathLike[str] | None) -> Path:
    if located is None:
        raise ToolchainError(f"`{name}` not found on PATH")
    try:
        path = Path(located).resolve(strict=True)
    except OSError as error:
        raise ToolchainError(f"`{name}` does not resolve to an existing file") from error
    if not path.is_file():
        raise ToolchainError(f"`{name}` is not a regular file")
    if not os.access(path, os.X_OK):
        raise ToolchainError(f"`{name}` is not executable")
    return path


def parse_go_version(raw: str) -> str:
    match = _GO_VERSION.search(raw)
    if match is None:
        raise ToolchainError("cannot parse a normalized Go patch version")
    return match.group(1)


def _run_version_command(go_binary: Path, *arguments: str | Path) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            [go_binary, *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            env=_local_environment(go_binary),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ToolchainError("toolchain version command failed") from error
    if completed.returncode != 0:
        raise ToolchainError("toolchain version command returned a failure")
    return completed


def _require_supported_go(raw: str) -> str:
    token = parse_go_version(raw)
    match = _GO_VERSION.fullmatch(token)
    if match is None:
        raise ToolchainError("Go version token was not normalized")
    version = tuple(int(match.group(index)) for index in (2, 3, 4))
    if version < (1, 26, 4) or version >= (1, 27, 0):
        raise ToolchainError("Go must be >=1.26.4 and <1.27")
    return token


def _require_go_critic_module(raw: str) -> None:
    versions = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "mod" and fields[1] == GO_CRITIC_MODULE:
            versions.append(fields[2])
    if versions != [GO_CRITIC_VERSION]:
        raise ToolchainError(f"go-critic must be module {GO_CRITIC_MODULE} at {GO_CRITIC_VERSION}")


def resolve_toolchain() -> ToolchainInfo:
    go = _resolved_executable("go", shutil.which("go"))
    go_critic = _resolved_executable("go-critic", shutil.which("go-critic"))
    gofmt = _resolved_executable("gofmt", go.parent / "gofmt")

    go_version_result = _run_version_command(go, "version")
    go_version = _require_supported_go(go_version_result.stdout + go_version_result.stderr)

    module_result = _run_version_command(go, "version", "-m", go_critic)
    _require_go_critic_module(module_result.stdout + module_result.stderr)

    return ToolchainInfo(
        go_binary=go,
        gofmt_binary=gofmt,
        go_critic_binary=go_critic,
        go_version=go_version,
        go_critic_version=GO_CRITIC_VERSION,
    )
