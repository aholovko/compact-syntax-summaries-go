from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from go_ast_assistant.paper4.gocheck.fixture import Fixture, synthesize
from go_ast_assistant.paper4.gocheck.go_critic import lint
from go_ast_assistant.paper4.gocheck.parse import go_build_status, gofmt_parse_ok
from go_ast_assistant.paper4.gocheck.result import GoCheckResult
from go_ast_assistant.paper4.gocheck.toolchain import ToolchainInfo


def run(fixture: Fixture, enable: tuple[str, ...], toolchain: ToolchainInfo) -> GoCheckResult:
    parse_ok = gofmt_parse_ok(fixture.source, toolchain)
    with tempfile.TemporaryDirectory(prefix="gocheck_") as temporary:
        pkg_dir = Path(temporary)
        fixture_path = pkg_dir / "fixture.go"
        fixture_path.write_text(fixture.source, encoding="utf-8")
        (pkg_dir / "go.mod").write_text(
            "module gocheckfixture\n\ngo 1.26.4\n",
            encoding="utf-8",
        )
        findings, tool_status = lint(fixture_path, enable, toolchain)
        build_status = go_build_status(pkg_dir, toolchain) if parse_ok else "NA"
    return GoCheckResult(
        parse_ok=parse_ok,
        build_status=build_status,
        findings=findings,
        tool_status=tool_status,
    )


def check(code: str, enable: tuple[str, ...], toolchain: ToolchainInfo) -> GoCheckResult:
    return run(synthesize(code), enable, toolchain)


@dataclass(frozen=True)
class GoCheck:
    toolchain: ToolchainInfo

    def parse_ok(self, source: str) -> bool:
        return gofmt_parse_ok(source, self.toolchain)

    def check(self, code: str, enable: tuple[str, ...]) -> GoCheckResult:
        return check(code, enable, self.toolchain)
