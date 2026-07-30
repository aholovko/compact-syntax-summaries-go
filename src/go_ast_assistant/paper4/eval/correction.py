from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from go_ast_assistant.paper4.gocheck.result import Finding, GoCheckResult
from go_ast_assistant.paper4.gocheck.toolchain import STUDIED_CHECKS
from go_ast_assistant.paper4.eval.extract import ExtractionStatus, extract


OrigCache = dict[tuple[str, tuple[str, ...]], GoCheckResult]


class _GoCheck(Protocol):
    def parse_ok(self, source: str) -> bool: ...

    def check(self, code: str, enable: tuple[str, ...]) -> GoCheckResult: ...


@dataclass(frozen=True)
class CorrectionOutcome:
    extracted: bool
    extraction_status: ExtractionStatus
    parse_ok: bool
    build_status: Literal["OK", "FAIL", "NA"]
    lint_ok: bool
    out_tool_status: Literal["ok", "load_degraded", "load_failed"]
    orig_tool_status: Literal["ok", "load_degraded", "load_failed"]
    target_fixed: bool
    studied_regression: bool | None
    enabled_regression: bool | None
    fix_rate_hit: bool
    category: Literal["A", "B", "C", "D", "INVALID"]
    residual_findings: tuple[Finding, ...] = ()
    introduced_checks: tuple[str, ...] = ()


def classify(
    *,
    extracted: bool,
    parse_ok: bool,
    lint_ok: bool,
    build_status: Literal["OK", "FAIL", "NA"],
    out_tool_status: Literal["ok", "load_degraded", "load_failed"],
    target_checks: set[str],
    studied_orig: set[str],
    studied_out: set[str],
    extraction_status: ExtractionStatus = ExtractionStatus.FAILED,
    orig_tool_status: Literal["ok", "load_degraded", "load_failed"] = "ok",
    out_findings: tuple[Finding, ...] = (),
    enabled_orig: set[str] | None = None,
    enabled_out: set[str] | None = None,
) -> CorrectionOutcome:
    if not (extracted and parse_ok and lint_ok):
        return CorrectionOutcome(
            extracted=extracted,
            extraction_status=extraction_status,
            parse_ok=parse_ok,
            build_status=build_status,
            lint_ok=lint_ok,
            out_tool_status=out_tool_status,
            orig_tool_status=orig_tool_status,
            target_fixed=False,
            studied_regression=None,
            enabled_regression=None,
            fix_rate_hit=False,
            category="INVALID",
        )

    target_fixed = not bool(target_checks & studied_out)
    studied_regression = bool(studied_out - studied_orig)
    enabled_regression = None
    if enabled_orig is not None and enabled_out is not None:
        enabled_regression = bool(enabled_out - enabled_orig)
    fix_rate_hit = target_fixed and not studied_regression
    if target_fixed and not studied_regression:
        category = "A"
    elif target_fixed:
        category = "B"
    elif not studied_regression:
        category = "C"
    else:
        category = "D"
    return CorrectionOutcome(
        extracted=True,
        extraction_status=extraction_status,
        parse_ok=parse_ok,
        build_status=build_status,
        lint_ok=True,
        out_tool_status=out_tool_status,
        orig_tool_status=orig_tool_status,
        target_fixed=target_fixed,
        studied_regression=studied_regression,
        enabled_regression=enabled_regression,
        fix_rate_hit=fix_rate_hit,
        category=category,
        residual_findings=tuple(finding for finding in out_findings if finding.check in target_checks),
        introduced_checks=tuple(sorted(studied_out - studied_orig)),
    )


def _original_result(
    original_code: str,
    gocheck: _GoCheck,
    cache: OrigCache | None,
) -> GoCheckResult:
    key = (original_code, STUDIED_CHECKS)
    if cache is None:
        return gocheck.check(*key)
    if key not in cache:
        cache[key] = gocheck.check(*key)
    return cache[key]


def score_correction(
    original_code: str,
    model_output: str,
    target_checks: set[str],
    *,
    gocheck: _GoCheck,
    orig_cache: OrigCache | None = None,
    extracted: tuple[str | None, ExtractionStatus] | None = None,
) -> CorrectionOutcome:
    original = _original_result(original_code, gocheck, orig_cache)
    code, extraction_status = extracted or extract(model_output, is_parseable=gocheck.parse_ok)
    if code is None:
        return classify(
            extracted=False,
            extraction_status=extraction_status,
            parse_ok=False,
            lint_ok=False,
            build_status="NA",
            out_tool_status="load_failed",
            orig_tool_status=original.tool_status,
            target_checks=target_checks,
            studied_orig={finding.check for finding in original.findings},
            studied_out=set(),
        )

    output = gocheck.check(code, STUDIED_CHECKS)
    original_checks = {finding.check for finding in original.findings}
    output_checks = {finding.check for finding in output.findings}
    lint_ok = original.tool_status != "load_failed" and output.tool_status != "load_failed"
    return classify(
        extracted=True,
        extraction_status=extraction_status,
        parse_ok=output.parse_ok,
        lint_ok=lint_ok,
        build_status=output.build_status,
        out_tool_status=output.tool_status,
        orig_tool_status=original.tool_status,
        target_checks=target_checks,
        studied_orig=original_checks,
        studied_out=output_checks,
        out_findings=output.findings,
    )
