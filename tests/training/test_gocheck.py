from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

import go_ast_assistant.paper4.gocheck.go_critic as go_critic_module
import go_ast_assistant.paper4.gocheck.parse as parse_module
import go_ast_assistant.paper4.gocheck.runner as runner_module
import go_ast_assistant.paper4.gocheck.toolchain as toolchain_module
from go_ast_assistant.paper4.gocheck import (
    ENABLED_CHECKS,
    GO_CRITIC_VERSION,
    STUDIED_CHECKS,
    GoCheck,
    parse_go_critic_output,
    resolve_toolchain,
    synthesize,
)
from go_ast_assistant.paper4.gocheck.fixture import Fixture
from go_ast_assistant.paper4.gocheck.go_critic import lint
from go_ast_assistant.paper4.gocheck.parse import go_build_status, gofmt_parse_ok
from go_ast_assistant.paper4.gocheck.result import Finding, GoCheckResult
from go_ast_assistant.paper4.gocheck.toolchain import ToolchainError, ToolchainInfo, parse_go_version


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic executable\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path.resolve()


def _toolchain(tmp_path: Path) -> ToolchainInfo:
    return ToolchainInfo(
        go_binary=_executable(tmp_path / "sdk" / "bin" / "go"),
        gofmt_binary=_executable(tmp_path / "sdk" / "bin" / "gofmt"),
        go_critic_binary=_executable(tmp_path / "tools" / "go-critic"),
        go_version="go1.26.4",
        go_critic_version="v0.14.4",
    )


def _resolution_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    go = _executable(tmp_path / "sdk" / "bin" / "go")
    gofmt = _executable(tmp_path / "sdk" / "bin" / "gofmt")
    go_critic = _executable(tmp_path / "tools" / "go-critic")
    return go, gofmt, go_critic


def _module_metadata(go_critic: Path, version: str = "v0.14.4") -> str:
    return (
        f"{go_critic}: go1.26.4\n"
        "\tpath\tgithub.com/go-critic/go-critic\n"
        f"\tmod\tgithub.com/go-critic/go-critic\t{version}\th1:synthetic\n"
        "\tbuild\t-buildmode=exe\n"
    )


def _install_resolution_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    go_lookup: Path | None,
    go_critic_lookup: Path | None,
    go_raw: str,
    metadata: str,
    go_returncode: int = 0,
    metadata_returncode: int = 0,
) -> tuple[list[str], list[tuple[tuple[str, ...], dict[str, Any]]]]:
    which_calls: list[str] = []
    subprocess_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def fake_which(name: str) -> str | None:
        which_calls.append(name)
        if name == "go":
            return os.fspath(go_lookup) if go_lookup is not None else None
        if name == "go-critic":
            return os.fspath(go_critic_lookup) if go_critic_lookup is not None else None
        raise AssertionError(f"unexpected PATH lookup: {name}")

    def fake_run(command: list[object], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        normalized = tuple(os.fspath(part) for part in command)
        subprocess_calls.append((normalized, kwargs))
        if go_lookup is None:
            raise AssertionError("subprocess called without a resolved Go path")
        resolved_go = os.fspath(go_lookup.resolve())
        if normalized == (resolved_go, "version"):
            return subprocess.CompletedProcess(command, go_returncode, stdout=go_raw, stderr="")
        if go_critic_lookup is None:
            raise AssertionError("go-critic metadata requested without a resolved path")
        resolved_go_critic = os.fspath(go_critic_lookup.resolve())
        if normalized == (resolved_go, "version", "-m", resolved_go_critic):
            return subprocess.CompletedProcess(command, metadata_returncode, stdout=metadata, stderr="")
        raise AssertionError(f"unexpected subprocess command: {normalized}")

    monkeypatch.setattr(toolchain_module.shutil, "which", fake_which)
    monkeypatch.setattr(toolchain_module.subprocess, "run", fake_run)
    return which_calls, subprocess_calls


def test_checker_constants_retain_the_exact_study_scope() -> None:
    assert GO_CRITIC_VERSION == "v0.14.4"
    assert STUDIED_CHECKS == (
        "assignOp",
        "builtinShadow",
        "captLocal",
        "commentFormatting",
        "elseif",
        "ifElseChain",
        "paramTypeCombine",
        "singleCaseSwitch",
    )
    assert ENABLED_CHECKS == ("#style",)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            "package main\n\nfunc main() {}\n",
            Fixture(source="package main\n\nfunc main() {}\n", strategy="file"),
        ),
        (
            "func Add(a int, b int) int { return a + b }",
            Fixture(
                source="package p\n\nfunc Add(a int, b int) int { return a + b }\n",
                strategy="decl",
            ),
        ),
        (
            "x := 1\n_ = x",
            Fixture(
                source="package p\n\nfunc _gocheck_body() {\n\tx := 1\n\t_ = x\n}\n",
                strategy="stmt",
            ),
        ),
    ],
)
def test_synthesize_applies_the_same_fixture_policy_to_files_declarations_and_statements(
    code: str,
    expected: Fixture,
) -> None:
    assert synthesize(code) == expected


def test_go_critic_parser_filters_named_checks_and_ignores_non_findings() -> None:
    text = (
        "go: diagnostic without a finding\n"
        "/tmp/f.go:3:8: paramTypeCombine: func params can be combined\n"
        "/tmp/f.go:5:2: captLocal: local should not be capitalized\n"
    )

    assert parse_go_critic_output(text, enable=("paramTypeCombine",)) == (
        Finding(check="paramTypeCombine", line=3, col=8),
    )
    assert parse_go_critic_output("", enable=("paramTypeCombine",)) == ()


def test_go_critic_parser_lets_the_pinned_cli_define_tag_membership() -> None:
    text = "/tmp/f.go:1:1: sloppyLen: message\n/tmp/f.go:2:2: elseif: message\n"

    assert parse_go_critic_output(text, enable=("#style",)) == (
        Finding(check="sloppyLen", line=1, col=1),
        Finding(check="elseif", line=2, col=2),
    )


def test_parse_go_version_returns_only_the_normalized_version_token() -> None:
    assert parse_go_version("go version go1.26.4 darwin/arm64") == "go1.26.4"
    with pytest.raises(ToolchainError):
        parse_go_version("not a Go version line")


def test_resolve_toolchain_uses_only_path_and_go_module_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    go, gofmt, go_critic = _resolution_layout(tmp_path)
    which_calls, subprocess_calls = _install_resolution_fakes(
        monkeypatch,
        go_lookup=go,
        go_critic_lookup=go_critic,
        go_raw="go version go1.26.4 darwin/arm64\n",
        metadata=_module_metadata(go_critic),
    )

    resolved = resolve_toolchain()

    assert which_calls == ["go", "go-critic"]
    assert resolved == ToolchainInfo(
        go_binary=go,
        gofmt_binary=gofmt,
        go_critic_binary=go_critic,
        go_version="go1.26.4",
        go_critic_version="v0.14.4",
    )
    assert [command for command, _ in subprocess_calls] == [
        (os.fspath(go), "version"),
        (os.fspath(go), "version", "-m", os.fspath(go_critic)),
    ]
    assert all(call_kwargs["env"]["GOTOOLCHAIN"] == "local" for _, call_kwargs in subprocess_calls)


@pytest.mark.parametrize("token", ["go1.26.4", "go1.26.99"])
def test_resolve_toolchain_normalizes_go_version_token(
    token: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    go, _, go_critic = _resolution_layout(tmp_path)
    raw = f"go version {token} darwin/arm64"
    _install_resolution_fakes(
        monkeypatch,
        go_lookup=go,
        go_critic_lookup=go_critic,
        go_raw=raw,
        metadata=_module_metadata(go_critic),
    )

    resolved = resolve_toolchain()

    assert resolved.go_version == token
    assert resolved.go_version != raw
    assert "darwin" not in resolved.go_version
    assert os.fspath(go) not in resolved.go_version


@pytest.mark.parametrize("token", ["go1.26.3", "go1.27.0"])
def test_resolve_toolchain_rejects_go_outside_the_supported_patch_range(
    token: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    go, _, go_critic = _resolution_layout(tmp_path)
    _install_resolution_fakes(
        monkeypatch,
        go_lookup=go,
        go_critic_lookup=go_critic,
        go_raw=f"go version {token} linux/amd64",
        metadata=_module_metadata(go_critic),
    )

    with pytest.raises(ToolchainError):
        resolve_toolchain()


@pytest.mark.parametrize(
    "metadata",
    [
        "\tmod\tgithub.com/go-critic/go-critic\tv0.14.3\th1:wrong\n",
        (
            "\tmod\texample.invalid/not-go-critic\tv0.14.4\th1:wrong\n"
            "\tdep\tgithub.com/go-critic/go-critic\tv0.14.4\th1:not-the-main-module\n"
        ),
        "go-critic version v0.0.0-SNAPSHOT\n",
    ],
)
def test_resolve_toolchain_rejects_missing_or_wrong_go_critic_module_metadata(
    metadata: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    go, _, go_critic = _resolution_layout(tmp_path)
    _install_resolution_fakes(
        monkeypatch,
        go_lookup=go,
        go_critic_lookup=go_critic,
        go_raw="go version go1.26.4 linux/amd64",
        metadata=metadata,
    )

    with pytest.raises(ToolchainError):
        resolve_toolchain()


@pytest.mark.parametrize("failed_command", ["go_version", "module_metadata"])
def test_resolve_toolchain_rejects_failed_version_commands(
    failed_command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    go, _, go_critic = _resolution_layout(tmp_path)
    _install_resolution_fakes(
        monkeypatch,
        go_lookup=go,
        go_critic_lookup=go_critic,
        go_raw="go version go1.26.4 linux/amd64",
        metadata=_module_metadata(go_critic),
        go_returncode=1 if failed_command == "go_version" else 0,
        metadata_returncode=1 if failed_command == "module_metadata" else 0,
    )

    with pytest.raises(ToolchainError):
        resolve_toolchain()


@pytest.mark.parametrize("missing", ["go", "gofmt", "go-critic"])
def test_resolve_toolchain_requires_every_path(
    missing: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    go, gofmt, go_critic = _resolution_layout(tmp_path)
    go_lookup = None if missing == "go" else go
    go_critic_lookup = None if missing == "go-critic" else go_critic
    if missing == "gofmt":
        gofmt.unlink()
    _install_resolution_fakes(
        monkeypatch,
        go_lookup=go_lookup,
        go_critic_lookup=go_critic_lookup,
        go_raw="go version go1.26.4 linux/amd64",
        metadata=_module_metadata(go_critic),
    )

    with pytest.raises(ToolchainError):
        resolve_toolchain()


@pytest.mark.parametrize("target", ["go", "gofmt", "go-critic"])
@pytest.mark.parametrize("defect", ["directory", "not_executable"])
def test_resolve_toolchain_requires_regular_executable_files(
    target: str,
    defect: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    go, gofmt, go_critic = _resolution_layout(tmp_path)
    path = {"go": go, "gofmt": gofmt, "go-critic": go_critic}[target]
    if defect == "directory":
        path.unlink()
        path.mkdir()
    else:
        path.chmod(0o644)
    _install_resolution_fakes(
        monkeypatch,
        go_lookup=go,
        go_critic_lookup=go_critic,
        go_raw="go version go1.26.4 linux/amd64",
        metadata=_module_metadata(go_critic),
    )

    with pytest.raises(ToolchainError):
        resolve_toolchain()


def test_resolve_toolchain_resolves_symlinks_before_selecting_neighboring_gofmt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    go, gofmt, go_critic = _resolution_layout(tmp_path)
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    go_link = path_bin / "go"
    go_critic_link = path_bin / "go-critic"
    go_link.symlink_to(go)
    go_critic_link.symlink_to(go_critic)
    _install_resolution_fakes(
        monkeypatch,
        go_lookup=go_link,
        go_critic_lookup=go_critic_link,
        go_raw="go version go1.26.4 linux/amd64",
        metadata=_module_metadata(go_critic),
    )

    resolved = resolve_toolchain()

    assert resolved.go_binary == go
    assert resolved.gofmt_binary == gofmt
    assert resolved.go_critic_binary == go_critic


@pytest.mark.parametrize(("outcome", "expected"), [(0, True), (1, False), ("timeout", False)])
def test_gofmt_parse_uses_the_resolved_binary_and_local_toolchain_environment(
    outcome: int | str,
    expected: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def fake_run(command: list[object], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        normalized = tuple(os.fspath(part) for part in command)
        calls.append((normalized, kwargs))
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, timeout=60)
        return subprocess.CompletedProcess(command, int(outcome), stdout="", stderr="")

    monkeypatch.setattr(parse_module.subprocess, "run", fake_run)

    assert gofmt_parse_ok("package p\n", toolchain) is expected
    assert len(calls) == 1
    command, call_kwargs = calls[0]
    assert command == (os.fspath(toolchain.gofmt_binary), "-e")
    assert call_kwargs["input"] == "package p\n"
    assert call_kwargs["env"]["GOTOOLCHAIN"] == "local"
    assert call_kwargs["timeout"] == 60


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "expected"),
    [
        (0, "", "", "OK"),
        (1, "", "cannot find package example.invalid/p", "NA"),
        (1, "no required module provides package p", "", "NA"),
        (1, "", "cannot find module providing package p", "NA"),
        (1, "", "missing go.sum entry", "NA"),
        (1, "", "syntax error", "FAIL"),
    ],
)
def test_go_build_classifies_results_and_disables_toolchain_and_proxy_acquisition(
    returncode: int,
    stdout: str,
    stderr: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def fake_run(command: list[object], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        normalized = tuple(os.fspath(part) for part in command)
        calls.append((normalized, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(parse_module.subprocess, "run", fake_run)

    assert go_build_status(tmp_path, toolchain) == expected
    assert len(calls) == 1
    command, call_kwargs = calls[0]
    assert command == (os.fspath(toolchain.go_binary), "build", "./...")
    assert call_kwargs["cwd"] == tmp_path
    assert call_kwargs["env"]["GOTOOLCHAIN"] == "local"
    assert call_kwargs["env"]["GOPROXY"] == "off"
    assert call_kwargs["timeout"] == 120


def test_go_build_timeout_is_a_failed_build_not_a_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)

    def timeout(command: list[object], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=120)

    monkeypatch.setattr(parse_module.subprocess, "run", timeout)

    assert go_build_status(tmp_path, toolchain) == "FAIL"


def test_lint_uses_the_resolved_binary_and_treats_findings_on_stderr_as_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)
    source_path = tmp_path / "fixture.go"
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    stderr = f"{source_path}:3:8: captLocal: local should not be capitalized\n"

    def fake_run(command: list[object], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        normalized = tuple(os.fspath(part) for part in command)
        calls.append((normalized, kwargs))
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=stderr)

    monkeypatch.setattr(go_critic_module.subprocess, "run", fake_run)

    findings, status = lint(source_path, STUDIED_CHECKS, toolchain)

    assert findings == (Finding(check="captLocal", line=3, col=8),)
    assert status == "ok"
    assert len(calls) == 1
    command, call_kwargs = calls[0]
    assert command == (
        os.fspath(toolchain.go_critic_binary),
        "check",
        f"-enable={','.join(STUDIED_CHECKS)}",
        os.fspath(source_path),
    )
    assert call_kwargs["env"]["GOTOOLCHAIN"] == "local"
    assert call_kwargs["timeout"] == 120


def test_lint_clean_success_without_findings_is_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)

    def fake_run(command: list[object], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(go_critic_module.subprocess, "run", fake_run)

    assert lint(tmp_path / "fixture.go", STUDIED_CHECKS, toolchain) == ((), "ok")


def test_lint_combines_stdout_findings_with_stderr_degradation_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)
    source_path = tmp_path / "fixture.go"
    stdout = f"{source_path}:4:6: elseif: replace else-if nesting\n"
    stderr = "could not load export data for one imported package\n"

    def fake_run(command: list[object], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(go_critic_module.subprocess, "run", fake_run)

    assert lint(source_path, STUDIED_CHECKS, toolchain) == (
        (Finding(check="elseif", line=4, col=6),),
        "load_degraded",
    )


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected_status"),
    [
        (1, "could not load export data for package", "load_degraded"),
        (1, "type-check failed", "load_degraded"),
        (1, "no required module provides package p", "load_degraded"),
        (1, "", "load_failed"),
        (2, "unexpected tool failure", "load_failed"),
    ],
)
def test_lint_preserves_truthful_degraded_and_failed_statuses(
    returncode: int,
    stderr: str,
    expected_status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)

    def fake_run(command: list[object], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(go_critic_module.subprocess, "run", fake_run)

    findings, status = lint(tmp_path / "fixture.go", STUDIED_CHECKS, toolchain)

    assert findings == ()
    assert status == expected_status


def test_lint_timeout_scores_load_failed_instead_of_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)

    def timeout(command: list[object], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=120)

    monkeypatch.setattr(go_critic_module.subprocess, "run", timeout)

    assert lint(tmp_path / "fixture.go", STUDIED_CHECKS, toolchain) == ((), "load_failed")


def test_runner_stages_one_fixture_and_returns_parse_lint_and_build_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)
    fixture = Fixture(source="package p\n\nfunc F(a int, b int) {}\n", strategy="file")
    finding = Finding(check="paramTypeCombine", line=3, col=15)
    calls: list[str] = []

    def parse(source: str, received: ToolchainInfo) -> bool:
        calls.append("parse")
        assert source == fixture.source
        assert received is toolchain
        return True

    def lint_fixture(path: Path, enable: tuple[str, ...], received: ToolchainInfo):
        calls.append("lint")
        assert path.name == "fixture.go"
        assert path.read_text(encoding="utf-8") == fixture.source
        assert sorted(child.name for child in path.parent.iterdir()) == ["fixture.go", "go.mod"]
        module_text = (path.parent / "go.mod").read_text(encoding="utf-8")
        assert "module gocheckfixture" in module_text
        assert "toolchain " not in module_text
        assert enable == STUDIED_CHECKS
        assert received is toolchain
        return (finding,), "ok"

    def build(pkg_dir: Path, received: ToolchainInfo) -> str:
        calls.append("build")
        assert (pkg_dir / "fixture.go").is_file()
        assert (pkg_dir / "go.mod").is_file()
        assert received is toolchain
        return "OK"

    monkeypatch.setattr(runner_module, "gofmt_parse_ok", parse)
    monkeypatch.setattr(runner_module, "lint", lint_fixture)
    monkeypatch.setattr(runner_module, "go_build_status", build)

    result = runner_module.run(fixture, STUDIED_CHECKS, toolchain)

    assert result == GoCheckResult(
        parse_ok=True,
        build_status="OK",
        findings=(finding,),
        tool_status="ok",
    )
    assert calls == ["parse", "lint", "build"]


def test_runner_skips_build_for_unparseable_fixture_but_keeps_lint_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)
    fixture = Fixture(source="package p\n\nfunc F( {\n", strategy="file")
    calls: list[str] = []

    def parse(_source: str, _toolchain: ToolchainInfo) -> bool:
        calls.append("parse")
        return False

    def lint_fixture(_path: Path, _enable: tuple[str, ...], _toolchain: ToolchainInfo):
        calls.append("lint")
        return (), "load_failed"

    def unexpected_build(_pkg_dir: Path, _toolchain: ToolchainInfo) -> str:
        raise AssertionError("an unparseable fixture must not be built")

    monkeypatch.setattr(runner_module, "gofmt_parse_ok", parse)
    monkeypatch.setattr(runner_module, "lint", lint_fixture)
    monkeypatch.setattr(runner_module, "go_build_status", unexpected_build)

    result = runner_module.run(fixture, STUDIED_CHECKS, toolchain)

    assert result == GoCheckResult(
        parse_ok=False,
        build_status="NA",
        findings=(),
        tool_status="load_failed",
    )
    assert calls == ["parse", "lint"]


def test_go_check_binds_one_resolved_toolchain_for_parse_and_full_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = _toolchain(tmp_path)
    checker = GoCheck(toolchain)
    expected = GoCheckResult(parse_ok=True, build_status="OK", findings=(), tool_status="ok")
    calls: list[tuple[str, object, ToolchainInfo]] = []

    def parse(source: str, received: ToolchainInfo) -> bool:
        calls.append(("parse", source, received))
        return True

    def run(fixture: Fixture, _enable: tuple[str, ...], received: ToolchainInfo) -> GoCheckResult:
        calls.append(("run", fixture, received))
        return expected

    monkeypatch.setattr(runner_module, "gofmt_parse_ok", parse)
    monkeypatch.setattr(runner_module, "run", run)

    assert checker.parse_ok("package p\n") is True
    assert checker.check("x := 1", STUDIED_CHECKS) is expected
    assert calls == [
        ("parse", "package p\n", toolchain),
        ("run", synthesize("x := 1"), toolchain),
    ]
