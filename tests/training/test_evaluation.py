from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest

import go_ast_assistant.paper4.eval.evaluator as evaluator_module
from analysis.inputs import (
    CorrectionRecord,
    ExplanationRecord,
    FineTunedCondition,
    JointRecord,
    RuleIdentificationRecord,
    TaskType,
)
from go_ast_assistant.paper4.config import CHECK_NAMES
from go_ast_assistant.paper4.eval.correction import classify, score_correction
from go_ast_assistant.paper4.eval.evaluator import (
    EvaluationAggregateMetrics,
    EvaluationResult,
    build_validation_composite,
    evaluate,
)
from go_ast_assistant.paper4.eval.extract import ExtractionStatus, extract
from go_ast_assistant.paper4.eval.labels import normalize
from go_ast_assistant.paper4.eval.rule_id import (
    exact_match_rate,
    macro_f1,
    micro_f1,
    parse_rule_id_output,
)
from go_ast_assistant.paper4.gocheck.toolchain import ToolchainInfo
from go_ast_assistant.paper4.prompts import registry as prompt_registry
from go_ast_assistant.paper4.records import TaskExample
from go_ast_assistant.paper4.training.conditions import SummaryRender, assemble_main_user_content
from go_ast_assistant.paper4.training.instruction_dataset import EOT_ID


_COMMON_RECORD_FIELDS = {
    "base_snippet_id",
    "condition",
    "seed",
    "task_type",
    "target_checks",
    "summary_status",
    "prompt_tokens",
    "retokenized_response_token_proxy",
    "latency_ms",
}
_ALIASES = {
    "capitalized local": "captLocal",
    "builtin shadow": "builtinShadow",
    "builtin shadowing": "builtinShadow",
    "else if": "elseif",
    "if else chain": "ifElseChain",
    "single case switch": "singleCaseSwitch",
    "param type combine": "paramTypeCombine",
    "parameter type combine": "paramTypeCombine",
    "assignment operator": "assignOp",
    "comment formatting": "commentFormatting",
}


def _snippet_id(index: int) -> str:
    return f"sha256:{hashlib.sha256(f'snippet-{index}'.encode()).hexdigest()}"


def _example(
    task_type: str,
    index: int = 0,
    *,
    target_checks: tuple[str, ...] = ("assignOp",),
    code: str | None = None,
) -> TaskExample:
    return TaskExample(
        id=_snippet_id(index),
        split="test",
        task_type=cast(TaskType, task_type),
        target_checks=target_checks,
        code=code or f"package p\n// CODE:{index}:{task_type}",
        target="private target",
        meta={},
    )


def _toolchain() -> ToolchainInfo:
    return ToolchainInfo(
        go_binary=Path("/resolved/bin/go"),
        gofmt_binary=Path("/resolved/bin/gofmt"),
        go_critic_binary=Path("/resolved/bin/go-critic"),
        go_version="go1.26.4",
        go_critic_version="v0.14.4",
    )


@dataclass(frozen=True)
class _Finding:
    check: str
    line: int
    col: int


@dataclass(frozen=True)
class _GoResult:
    parse_ok: bool = True
    build_status: Literal["OK", "FAIL", "NA"] = "NA"
    findings: tuple[_Finding, ...] = ()
    tool_status: Literal["ok", "load_degraded", "load_failed"] = "ok"


class _FakeGoCheck:
    def __init__(
        self,
        results: dict[str, _GoResult] | None = None,
        *,
        parseable: bool = True,
        events: list[tuple[str, str]] | None = None,
    ) -> None:
        self.results = results or {}
        self.parseable = parseable
        self.events = events if events is not None else []
        self.check_calls: list[tuple[str, tuple[str, ...]]] = []
        self.parse_calls: list[str] = []

    def check(self, code: str, enable: tuple[str, ...]) -> _GoResult:
        self.events.append(("check", code))
        self.check_calls.append((code, enable))
        result = self.results.get(code, _GoResult())
        return replace(result, findings=tuple(finding for finding in result.findings if finding.check in enable))

    def parse_ok(self, source: str) -> bool:
        self.events.append(("parse", source))
        self.parse_calls.append(source)
        return self.parseable


class _SummaryStore:
    def __init__(
        self,
        statuses: dict[str, Literal["present", "present_truncated", "skipped", "failed"]] | None = None,
        *,
        default: Literal["present", "present_truncated", "skipped", "failed"] = "skipped",
    ) -> None:
        self.statuses = statuses or {}
        self.default = default
        self.calls: list[str] = []
        self.code_token_counts: list[tuple[str, int]] = []

    def render_for_main(self, snippet_id: str, code_tokens: int) -> SummaryRender:
        self.calls.append(snippet_id)
        self.code_token_counts.append((snippet_id, code_tokens))
        status = self.statuses.get(snippet_id, self.default)
        text = "compact summary" if status in {"present", "present_truncated"} else ""
        return SummaryRender(text=text, attached=status)


class _FakeTokenizer:
    eos_token_id = 128001

    def __init__(self) -> None:
        self.encode_calls: list[str] = []
        self.response_counts: dict[str, int] = {}
        self._decoded: dict[tuple[int, ...], str] = {}
        self._next_response_token = 900_000

    def encode(self, text: str, *args: object, **kwargs: object) -> list[int]:
        del args, kwargs
        self.encode_calls.append(text)
        count = self.response_counts.get(text, len(text.split()))
        return list(range(count))

    def decode(self, token_ids: list[int]) -> str:
        key = tuple(token_ids)
        if key not in self._decoded:
            raise AssertionError(f"evaluation must decode only the generated response suffix, got {key!r}")
        return self._decoded[key]

    def response_token(self, response: str) -> int:
        token = self._next_response_token
        self._next_response_token += 1
        self._decoded[(token,)] = response
        return token


class _FakeChat:
    def __init__(self, *, prompt_length: int = 3) -> None:
        self.tok = _FakeTokenizer()
        self.prompt_length = prompt_length
        self.encode_calls: list[tuple[str, str | None]] = []
        self.prompt_text: dict[tuple[int, ...], str] = {}

    def encode(self, user_message: str, system_message: str | None = None) -> list[int]:
        self.encode_calls.append((user_message, system_message))
        start = 10_000 + len(self.encode_calls) * 100
        prompt = list(range(start, start + self.prompt_length))
        self.prompt_text[tuple(prompt)] = user_message
        return prompt

    def decode(self, token_ids: list[int]) -> str:
        return self.tok.decode(token_ids)


class _FakeModel:
    cfg = {"context_length": 9_305}


@dataclass(frozen=True)
class _GenerationCall:
    max_new_tokens: int
    eos_id: int
    prompt_texts: tuple[str, ...]


def _task_from_prompt(prompt: str) -> str:
    for task_type in ("rule_identification", "correction", "joint", "explanation"):
        if f"TASK: {task_type}" in prompt:
            return task_type
    raise AssertionError(f"prompt has no released task marker: {prompt!r}")


def _install_evaluation_fakes(
    monkeypatch: pytest.MonkeyPatch,
    chat: _FakeChat,
    gocheck: _FakeGoCheck,
    response_for_prompt: Any,
) -> list[_GenerationCall]:
    calls: list[_GenerationCall] = []

    def fake_generate_bucketed(
        model: object,
        prompts: tuple[tuple[int, ...], ...] | list[list[int]],
        max_new_tokens: int,
        context_size: int,
        *,
        eos_id: int,
        **kwargs: object,
    ) -> list[list[int]]:
        del kwargs
        assert isinstance(model, _FakeModel)
        assert context_size == 9_305
        texts = tuple(chat.prompt_text[tuple(prompt)] for prompt in prompts)
        calls.append(_GenerationCall(max_new_tokens=max_new_tokens, eos_id=eos_id, prompt_texts=texts))
        generated: list[list[int]] = []
        for prompt, text in zip(prompts, texts, strict=True):
            raw_response = response_for_prompt(text)
            response_token = chat.tok.response_token(raw_response)
            generated.append([*prompt, response_token])
        return generated

    monkeypatch.setattr(evaluator_module, "generate_bucketed", fake_generate_bucketed)
    monkeypatch.setattr(evaluator_module, "GoCheck", lambda toolchain: gocheck)
    return calls


def _evaluate(
    examples: tuple[TaskExample, ...],
    monkeypatch: pytest.MonkeyPatch,
    *,
    condition: FineTunedCondition = "C0",
    seed: int = 42,
    chat: _FakeChat | None = None,
    summaries: _SummaryStore | None = None,
    gocheck: _FakeGoCheck | None = None,
    excluded_ids: frozenset[str] = frozenset(),
    response_for_prompt: Any | None = None,
    toolchain: ToolchainInfo | None = None,
) -> tuple[EvaluationResult, _FakeChat, _SummaryStore, _FakeGoCheck, list[_GenerationCall]]:
    actual_chat = chat or _FakeChat()
    actual_summaries = summaries or _SummaryStore()
    actual_gocheck = gocheck or _FakeGoCheck()

    def default_response(prompt: str) -> str:
        task_type = _task_from_prompt(prompt)
        if task_type == "rule_identification":
            return '["assignOp"]'
        if task_type in {"correction", "joint"}:
            return "```go\npackage fixed\n```"
        return "private explanation response"

    calls = _install_evaluation_fakes(
        monkeypatch,
        actual_chat,
        actual_gocheck,
        response_for_prompt or default_response,
    )
    actual_toolchain = toolchain or _toolchain()
    result = evaluate(
        examples=examples,
        condition=condition,
        seed=seed,
        model=_FakeModel(),
        tokenizer=actual_chat,
        summaries=actual_summaries,
        toolchain=actual_toolchain,
        excluded_ids=excluded_ids,
    )
    return result, actual_chat, actual_summaries, actual_gocheck, calls


def _punctuated_case_variant(label: str) -> str:
    return ".-".join(
        character.upper() if index % 2 == 0 else character.lower() for index, character in enumerate(label)
    )


@pytest.mark.parametrize(
    ("raw_label", "canonical"),
    [*((name, name) for name in CHECK_NAMES), *_ALIASES.items()],
)
def test_rule_label_aliases_deduplicate_in_check_order_and_count_every_member(
    raw_label: str,
    canonical: str,
) -> None:
    variant = _punctuated_case_variant(raw_label)
    parsed_variant = parse_rule_id_output(json.dumps([variant]), frozenset(CHECK_NAMES))

    assert parsed_variant.pred == (canonical,)
    assert parsed_variant.normalization_status == "recognized_array"
    assert parsed_variant.n_emitted == 1
    assert parsed_variant.rejected_label_count == 0
    assert normalize(variant) == canonical

    mixed = [
        "SINGLE--CASE SWITCH",
        "assignOp",
        "ASSIGNMENT_OPERATOR",
        17,
        "unknown",
        {"not": "a string"},
        "else if",
        "else-if",
    ]
    parsed_mixed = parse_rule_id_output(json.dumps(mixed), frozenset(CHECK_NAMES))
    assert parsed_mixed.pred == ("assignOp", "elseif", "singleCaseSwitch")
    assert parsed_mixed.normalization_status == "recognized_array"
    assert parsed_mixed.n_emitted == len(mixed)
    assert parsed_mixed.rejected_label_count == 3


@pytest.mark.parametrize(
    ("text", "status", "pred", "n_emitted", "rejected"),
    [
        ("no JSON list here", "no_recognized_array", (), 0, 0),
        ("[]", "recognized_array", (), 0, 0),
        ('["unknown"]', "recognized_array", (), 1, 1),
        ('[["assignOp"], "captLocal"]', "recognized_array", ("captLocal",), 2, 1),
        ('first ["assignOp"] then ["elseif"]', "recognized_array", ("elseif",), 1, 0),
        ('first ["assignOp"] then []', "recognized_array", (), 0, 0),
        ('first ["assignOp"] then ["unknown"]', "recognized_array", (), 1, 1),
        ('["unknown \\"quoted\\" label", "else if"]', "recognized_array", ("elseif",), 2, 1),
        ('["captLocal]still-a-string"]', "recognized_array", (), 1, 1),
        ('truncated [garbage then ["assignOp"]', "recognized_array", ("assignOp",), 1, 0),
        ("[1, null, false]", "recognized_array", (), 3, 3),
    ],
)
def test_balanced_rule_array_scanner_handles_every_released_edge_case(
    text: str,
    status: str,
    pred: tuple[str, ...],
    n_emitted: int,
    rejected: int,
) -> None:
    parsed = parse_rule_id_output(text, frozenset(CHECK_NAMES))

    assert parsed.normalization_status == status
    assert parsed.pred == pred
    assert parsed.n_emitted == n_emitted
    assert parsed.rejected_label_count == rejected


def test_rule_id_primary_rates_match_hand_calculated_values() -> None:
    predictions = (frozenset({"assignOp"}), frozenset({"assignOp"}), frozenset())
    gold = (frozenset({"assignOp"}), frozenset({"builtinShadow"}), frozenset())

    assert macro_f1(predictions, gold, CHECK_NAMES) == pytest.approx((2.0 / 3.0) / len(CHECK_NAMES))
    assert micro_f1(predictions, gold, CHECK_NAMES) == pytest.approx(0.5)
    assert exact_match_rate(predictions, gold) == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    ("text", "parseable", "expected_code", "expected_status"),
    [
        ("```go\npackage p\n```", False, "package p", ExtractionStatus.GO_BLOCK),
        ("```\npackage p\n```", False, "package p", ExtractionStatus.FENCED_BLOCK),
        ("prose\nfunc F() {}\nmore prose", True, "prose\nfunc F() {}\nmore prose", ExtractionStatus.LARGEST_PARSEABLE),
        ("```go\n```", False, None, ExtractionStatus.FAILED),
        ("```go\n// refusal only\n```", False, None, ExtractionStatus.FAILED),
        ("```go\n/* refusal only */\n```", False, None, ExtractionStatus.FAILED),
        ("```go\n\n```\n```go\npackage later\n```", False, "package later", ExtractionStatus.GO_BLOCK),
        ('`````go\nfunc F() { s := "```" }\n`````', False, 'func F() { s := "```" }', ExtractionStatus.GO_BLOCK),
        ("```go\r\npackage p\r\n```", False, "package p", ExtractionStatus.GO_BLOCK),
    ],
)
def test_extraction_uses_safe_order_and_rejects_content_free_outputs(
    text: str,
    parseable: bool,
    expected_code: str | None,
    expected_status: ExtractionStatus,
) -> None:
    code, status = extract(text, is_parseable=lambda _candidate: parseable)

    assert code == expected_code
    assert status is expected_status


def test_largest_parseable_fallback_is_bounded_to_32_probes() -> None:
    probes: list[str] = []

    code, status = extract(
        "\n".join(f"line {index}" for index in range(20)),
        is_parseable=lambda candidate: probes.append(candidate) is None and False,
    )

    assert code is None
    assert status is ExtractionStatus.FAILED
    assert len(probes) == 32


@pytest.mark.parametrize(
    ("studied_orig", "studied_out", "expected"),
    [
        ({"assignOp"}, set(), (True, False, True, "A")),
        ({"assignOp"}, {"elseif"}, (True, True, False, "B")),
        ({"assignOp"}, {"assignOp"}, (False, False, False, "C")),
        ({"assignOp"}, {"assignOp", "elseif"}, (False, True, False, "D")),
    ],
)
def test_correction_classification_pins_all_four_valid_categories(
    studied_orig: set[str],
    studied_out: set[str],
    expected: tuple[bool, bool, bool, str],
) -> None:
    outcome = classify(
        extracted=True,
        parse_ok=True,
        lint_ok=True,
        build_status="FAIL",
        out_tool_status="ok",
        orig_tool_status="ok",
        target_checks={"assignOp"},
        studied_orig=studied_orig,
        studied_out=studied_out,
    )

    assert (outcome.target_fixed, outcome.studied_regression, outcome.fix_rate_hit, outcome.category) == expected


def test_correction_checks_and_caches_original_before_failed_extraction() -> None:
    events: list[tuple[str, str]] = []
    original = "package original"
    gocheck = _FakeGoCheck(
        {original: _GoResult(tool_status="load_failed", parse_ok=False)},
        parseable=False,
        events=events,
    )
    cache: dict[tuple[str, tuple[str, ...]], _GoResult] = {}

    first = score_correction(original, "unparseable prose", {"assignOp"}, gocheck=gocheck, orig_cache=cache)
    second = score_correction(original, "different prose", {"assignOp"}, gocheck=gocheck, orig_cache=cache)

    assert events[0] == ("check", original)
    assert sum(code == original for code, _checks in gocheck.check_calls) == 1
    assert first.orig_tool_status == "load_failed"
    assert first.out_tool_status == "load_failed"
    assert first.category == "INVALID"
    assert second.orig_tool_status == "load_failed"


def test_successful_extraction_cannot_hide_original_total_load_failure() -> None:
    original = "package original"
    output = "package output"
    gocheck = _FakeGoCheck(
        {
            original: _GoResult(tool_status="load_failed", parse_ok=False),
            output: _GoResult(),
        }
    )

    outcome = score_correction(
        original,
        f"```go\n{output}\n```",
        {"assignOp"},
        gocheck=gocheck,
    )

    assert outcome.extracted is True
    assert outcome.parse_ok is True
    assert outcome.lint_ok is False
    assert outcome.category == "INVALID"
    assert outcome.fix_rate_hit is False
    assert outcome.orig_tool_status == "load_failed"
    assert outcome.out_tool_status == "ok"


def test_correction_preserves_findings_and_truthful_tool_statuses() -> None:
    original = "package original"
    output = "package output"
    gocheck = _FakeGoCheck(
        {
            original: _GoResult(
                findings=(_Finding("assignOp", 2, 3),),
                tool_status="load_degraded",
            ),
            output: _GoResult(
                findings=(
                    _Finding("assignOp", 5, 7),
                    _Finding("elseif", 8, 9),
                ),
                build_status="FAIL",
            ),
        }
    )

    outcome = score_correction(
        original,
        "ignored raw output",
        {"assignOp"},
        gocheck=gocheck,
        extracted=(output, ExtractionStatus.GO_BLOCK),
    )

    assert outcome.orig_tool_status == "load_degraded"
    assert outcome.out_tool_status == "ok"
    assert outcome.build_status == "FAIL"
    assert outcome.category == "D"
    assert tuple(finding.check for finding in outcome.residual_findings) == ("assignOp",)
    assert outcome.introduced_checks == ("elseif",)
    assert gocheck.check_calls == [(original, CHECK_NAMES), (output, CHECK_NAMES)]


@pytest.mark.parametrize("condition", ["C1", "C2", "C2-control"])
@pytest.mark.parametrize("summary_status", ["present", "present_truncated"])
def test_evaluation_uses_exact_prompt_system_summary_and_code_token_budget(
    monkeypatch: pytest.MonkeyPatch,
    condition: FineTunedCondition,
    summary_status: Literal["present", "present_truncated"],
) -> None:
    example = _example("explanation", code="package p\nfunc ExactPrompt() {}")
    chat = _FakeChat()
    chat.tok.response_counts[example.code] = 23
    summaries = _SummaryStore({example.id: summary_status})

    result, *_ = _evaluate(
        (example,),
        monkeypatch,
        condition=condition,
        chat=chat,
        summaries=summaries,
    )

    expected_user = assemble_main_user_content(
        example.task_type,
        example.code,
        "compact summary",
        example.target_checks,
    )
    assert chat.encode_calls == [(expected_user, prompt_registry.SYSTEM_PROMPT)]
    assert example.target not in expected_user
    assert summaries.code_token_counts == [(example.id, 23)]
    assert chat.tok.encode_calls.count(example.code) == 1
    assert result.records[0].summary_status == summary_status


def test_evaluation_uses_pre_generation_prompt_and_retokenized_raw_response_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_response = '["assignOp"] with trailing prose'
    chat = _FakeChat(prompt_length=5)
    chat.tok.response_counts[raw_response] = 17
    result, _, _, _, calls = _evaluate(
        (_example("rule_identification"),),
        monkeypatch,
        chat=chat,
        response_for_prompt=lambda _prompt: raw_response,
    )

    record = cast(RuleIdentificationRecord, result.records[0])
    assert record.prompt_tokens == 5
    assert record.retokenized_response_token_proxy == 17
    assert sum(len(call.prompt_texts) for call in calls) == 1
    assert raw_response in chat.tok.encode_calls


def test_evaluation_terminates_generation_with_trained_end_of_turn_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, _, calls = _evaluate((_example("rule_identification"),), monkeypatch)

    assert [call.eos_id for call in calls] == [EOT_ID]


def test_new_attempt_latency_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    examples = tuple(
        _example(task_type, index)
        for index, task_type in enumerate(("rule_identification", "correction", "joint", "explanation"))
    )

    result, *_ = _evaluate(examples, monkeypatch)

    assert all(record.latency_ms is None for record in result.records)


def test_generation_cap_groups_scatter_back_to_preserved_source_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = ("explanation", "rule_identification", "joint", "correction", "rule_identification")
    examples = tuple(_example(task_type, index) for index, task_type in enumerate(tasks))
    chat = _FakeChat()

    def response(prompt: str) -> str:
        task_type = _task_from_prompt(prompt)
        marker = next(part for part in prompt.split() if "CODE:" in part).rstrip("`")
        source_index = int(marker.split(":")[1])
        if task_type == "rule_identification":
            raw = f'["assignOp"] // response-{source_index}'
        elif task_type in {"correction", "joint"}:
            raw = f"```go\npackage fixed // {marker}\n```"
        else:
            raw = f"explanation {marker}"
        chat.tok.response_counts[raw] = 100 + source_index
        return raw

    result, chat, _, _, calls = _evaluate(examples, monkeypatch, chat=chat, response_for_prompt=response)

    assert tuple(record.base_snippet_id for record in result.records) == tuple(example.id for example in examples)
    assert tuple(record.task_type for record in result.records) == tasks
    assert tuple(record.retokenized_response_token_proxy for record in result.records) == tuple(
        100 + index for index in range(len(examples))
    )
    assert sorted((call.max_new_tokens, len(call.prompt_texts)) for call in calls) == [(64, 2), (512, 3)]
    tasks_by_cap = {
        call.max_new_tokens: tuple(_task_from_prompt(prompt) for prompt in call.prompt_texts) for call in calls
    }
    assert tasks_by_cap == {
        64: ("rule_identification", "rule_identification"),
        512: ("explanation", "joint", "correction"),
    }
    generated_prompts = {text for call in calls for text in call.prompt_texts}
    assert generated_prompts == {text for text, _system in chat.encode_calls}


def test_evaluation_projects_exact_source_free_record_shapes_and_five_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = tuple(
        _example(task_type, index)
        for index, task_type in enumerate(("rule_identification", "correction", "joint", "explanation"))
    )
    result, _, _, gocheck, _ = _evaluate(examples, monkeypatch)

    rule, correction, joint, explanation = result.records
    assert isinstance(rule, RuleIdentificationRecord)
    assert isinstance(correction, CorrectionRecord)
    assert isinstance(joint, JointRecord)
    assert isinstance(explanation, ExplanationRecord)
    assert set(rule.model_dump()) == _COMMON_RECORD_FIELDS | {
        "gold",
        "pred",
        "rejected_label_count",
        "exact_match",
        "n_emitted",
        "normalization_status",
    }
    assert set(correction.model_dump()) == _COMMON_RECORD_FIELDS | {
        "outcome",
        "extracted_similarity",
        "sensitivity_class",
    }
    assert set(joint.model_dump()) == set(correction.model_dump())
    assert set(explanation.model_dump()) == _COMMON_RECORD_FIELDS
    assert correction.extracted_similarity is None and correction.sensitivity_class is None
    assert joint.extracted_similarity is None and joint.sensitivity_class is None
    assert all(record.summary_status == "not_applicable" for record in result.records)

    dumped = json.dumps([record.model_dump(mode="json") for record in result.records], sort_keys=True)
    for forbidden in (
        "private target",
        "private explanation response",
        "package fixed",
        "CODE:",
        "raw_output",
        "extracted_code",
        "rejected_labels",
        "/resolved/",
    ):
        assert forbidden not in dumped
    explanation_code = examples[-1].code
    assert all(code != explanation_code for code, _checks in gocheck.check_calls)
    assert "private explanation response" not in gocheck.parse_calls

    assert result.aggregate_metrics.model_dump() == {
        "rule_id_macro_f1": pytest.approx(1.0 / len(CHECK_NAMES)),
        "rule_id_micro_f1": 1.0,
        "rule_id_exact_match": 1.0,
        "correction_fix_rate": 1.0,
        "joint_fix_rate": 1.0,
    }


def test_evaluation_maps_runtime_findings_and_fix_rate_hit_to_strict_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "package original"
    output = "package output"
    gocheck = _FakeGoCheck(
        {
            original: _GoResult(findings=(_Finding("assignOp", 1, 2),)),
            output: _GoResult(
                findings=(
                    _Finding("assignOp", 4, 5),
                    _Finding("elseif", 6, 7),
                ),
            ),
        }
    )
    result, *_ = _evaluate(
        (_example("correction", code=original),),
        monkeypatch,
        gocheck=gocheck,
        response_for_prompt=lambda _prompt: f"```go\n{output}\n```",
    )

    record = cast(CorrectionRecord, result.records[0])
    assert record.outcome.overall_fixed is False
    assert record.outcome.category == "D"
    assert record.outcome.original_tool_status == "ok"
    assert record.outcome.output_tool_status == "ok"
    assert record.outcome.introduced_checks == ("elseif",)
    assert tuple(finding.model_dump() for finding in record.outcome.residual_findings) == (
        {"check": "assignOp", "line": 4, "column": 5},
    )


def test_mixed_repair_rates_and_concrete_composite_use_all_attempts_in_each_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_types = ("correction",) * 4 + ("joint",) * 4
    examples = tuple(
        _example(task_type, index, code=f"package original_{index}") for index, task_type in enumerate(task_types)
    )
    output_by_original = {example.code: f"package output_{index}" for index, example in enumerate(examples)}
    check_results: dict[str, _GoResult] = {}
    for index, example in enumerate(examples):
        check_results[example.code] = _GoResult(findings=(_Finding("assignOp", 1, 1),))
        scenario = index % 4
        if scenario == 0:
            output_result = _GoResult()
        elif scenario == 1:
            output_result = _GoResult(findings=(_Finding("elseif", 2, 2),))
        elif scenario == 2:
            output_result = _GoResult(findings=(_Finding("assignOp", 3, 3),))
        else:
            output_result = _GoResult(tool_status="load_failed")
        check_results[output_by_original[example.code]] = output_result
    gocheck = _FakeGoCheck(check_results)

    def raw_output_for_prompt(prompt: str) -> str:
        original = next(code for code in output_by_original if code in prompt)
        return f"```go\n{output_by_original[original]}\n```"

    result, *_ = _evaluate(
        examples,
        monkeypatch,
        gocheck=gocheck,
        response_for_prompt=raw_output_for_prompt,
    )

    assert tuple(record.outcome.overall_fixed for record in result.records) == (
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )
    assert result.aggregate_metrics.correction_fix_rate == pytest.approx(0.25)
    assert result.aggregate_metrics.joint_fix_rate == pytest.approx(0.25)

    rule_example = _example("rule_identification", 99)
    composite = build_validation_composite(_toolchain())

    def generate(
        selected: tuple[TaskExample, ...],
        max_new_tokens: int,
    ) -> tuple[str, ...]:
        if selected and selected[0].task_type == "rule_identification":
            assert max_new_tokens == 64
            return ('["assignOp"]',)
        assert max_new_tokens == 512
        return tuple(f"```go\n{output_by_original[example.code]}\n```" for example in selected)

    composite_result = composite.evaluate(None, (rule_example, *examples), generate)
    assert composite_result.components == {
        "rule_id_macro_f1": pytest.approx(1.0 / len(CHECK_NAMES)),
        "correction_fix_rate": pytest.approx(0.25),
        "joint_fix_rate": pytest.approx(0.25),
    }
    assert composite_result.composite == pytest.approx((1.0 / len(CHECK_NAMES) + 0.25 + 0.25) / 3.0)


def test_failed_and_skipped_summary_delivery_and_go_statuses_remain_truthful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _example("correction", 0, code="package original_degraded")
    second = _example("correction", 1, code="package original_ok")
    summaries = _SummaryStore({first.id: "failed", second.id: "skipped"})
    gocheck = _FakeGoCheck(
        {
            first.code: _GoResult(tool_status="load_degraded"),
            second.code: _GoResult(),
            "package output_good": _GoResult(),
            "package output_failed": _GoResult(tool_status="load_failed"),
        }
    )

    def response(prompt: str) -> str:
        output = "package output_good" if "original_degraded" in prompt else "package output_failed"
        return f"```go\n{output}\n```"

    result, *_ = _evaluate(
        (first, second),
        monkeypatch,
        condition="C1",
        summaries=summaries,
        gocheck=gocheck,
        response_for_prompt=response,
    )

    degraded, failed = (cast(CorrectionRecord, record) for record in result.records)
    assert degraded.summary_status == "failed"
    assert degraded.outcome.original_tool_status == "load_degraded"
    assert degraded.outcome.output_tool_status == "ok"
    assert degraded.outcome.category == "A"
    assert failed.summary_status == "skipped"
    assert failed.outcome.original_tool_status == "ok"
    assert failed.outcome.output_tool_status == "load_failed"
    assert failed.outcome.category == "INVALID"


def test_c0_never_requests_a_summary_and_uses_not_applicable(monkeypatch: pytest.MonkeyPatch) -> None:
    summaries = _SummaryStore(default="failed")

    result, *_ = _evaluate((_example("explanation"),), monkeypatch, summaries=summaries)

    assert summaries.calls == []
    assert result.records[0].summary_status == "not_applicable"


def test_evaluation_propagates_nondefault_identity_and_supplied_toolchain_without_scoring_explanations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = _example("explanation")
    summaries = _SummaryStore({example.id: "present"})
    gocheck = _FakeGoCheck()
    toolchain = _toolchain()

    result, *_ = _evaluate(
        (example,),
        monkeypatch,
        condition="C2-control",
        seed=44,
        summaries=summaries,
        gocheck=gocheck,
        toolchain=toolchain,
    )

    record = result.records[0]
    assert record.condition == "C2-control"
    assert record.seed == 44
    assert result.toolchain is toolchain
    assert gocheck.check_calls == []
    assert gocheck.parse_calls == []


def test_full_test_matrix_filters_before_prompt_encoding_generation_and_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = ("rule_identification", "correction", "joint", "explanation")
    examples = tuple(_example(task_type, index) for index in range(448) for task_type in tasks)
    excluded_ids = frozenset(_snippet_id(index) for index in range(38))
    summaries = _SummaryStore()

    result, chat, summaries, gocheck, calls = _evaluate(
        examples,
        monkeypatch,
        condition="C1",
        summaries=summaries,
        excluded_ids=excluded_ids,
    )

    assert result.n_examples == 1_640
    assert result.n_excluded == 152
    assert len(result.records) == 1_640
    assert tuple(record.base_snippet_id for record in result.records) == tuple(
        example.id for example in examples if example.id not in excluded_ids
    )
    assert len(chat.encode_calls) == 1_640
    assert len(summaries.calls) == 1_640
    assert sum(len(call.prompt_texts) for call in calls) == 1_640
    assert sorted((call.max_new_tokens, len(call.prompt_texts)) for call in calls) == [(64, 410), (512, 1_230)]

    excluded_markers = tuple(f"CODE:{index}:" for index in range(38))
    observed_text = "\n".join([*(text for text, _system in chat.encode_calls), *chat.tok.encode_calls])
    assert not any(marker in observed_text for marker in excluded_markers)
    assert not any(any(marker in code for marker in excluded_markers) for code, _checks in gocheck.check_calls)


def test_evaluation_rejects_unknown_exclusions_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = _FakeChat()
    gocheck = _FakeGoCheck()
    calls = _install_evaluation_fakes(monkeypatch, chat, gocheck, lambda _prompt: "[]")

    with pytest.raises(ValueError, match="exclusion|unknown|missing"):
        evaluate(
            examples=(_example("rule_identification"),),
            condition="C0",
            seed=42,
            model=_FakeModel(),
            tokenizer=chat,
            summaries=_SummaryStore(),
            toolchain=_toolchain(),
            excluded_ids=frozenset({_snippet_id(999)}),
        )

    assert chat.encode_calls == []
    assert calls == []


def test_evaluation_rejects_empty_retained_population_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    example = _example("rule_identification")
    chat = _FakeChat()
    gocheck = _FakeGoCheck()
    calls = _install_evaluation_fakes(monkeypatch, chat, gocheck, lambda _prompt: "[]")

    with pytest.raises(ValueError, match="empty"):
        evaluate(
            examples=(example,),
            condition="C0",
            seed=42,
            model=_FakeModel(),
            tokenizer=chat,
            summaries=_SummaryStore(),
            toolchain=_toolchain(),
            excluded_ids=frozenset({example.id}),
        )

    assert chat.encode_calls == []
    assert calls == []


def test_evaluation_rejects_generation_cardinality_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator_module, "GoCheck", lambda toolchain: _FakeGoCheck())
    monkeypatch.setattr(evaluator_module, "generate_bucketed", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="output|count|number"):
        evaluate(
            examples=(_example("rule_identification"),),
            condition="C0",
            seed=42,
            model=_FakeModel(),
            tokenizer=_FakeChat(),
            summaries=_SummaryStore(),
            toolchain=_toolchain(),
            excluded_ids=frozenset(),
        )


def _direct_rule_record() -> RuleIdentificationRecord:
    return RuleIdentificationRecord(
        base_snippet_id=_snippet_id(0),
        condition="C0",
        seed=42,
        task_type="rule_identification",
        target_checks=("assignOp",),
        summary_status="not_applicable",
        prompt_tokens=3,
        retokenized_response_token_proxy=1,
        latency_ms=None,
        gold=("assignOp",),
        pred=("assignOp",),
        rejected_label_count=0,
        exact_match=True,
        n_emitted=1,
        normalization_status="recognized_array",
    )


def _metrics() -> EvaluationAggregateMetrics:
    return EvaluationAggregateMetrics(
        rule_id_macro_f1=0.5,
        rule_id_micro_f1=0.5,
        rule_id_exact_match=0.5,
        correction_fix_rate=0.5,
        joint_fix_rate=0.5,
    )


@pytest.mark.parametrize("invalid", (-0.1, 1.1, math.inf, -math.inf, math.nan))
def test_evaluation_aggregate_rejects_non_rates(invalid: float) -> None:
    with pytest.raises(ValueError):
        EvaluationAggregateMetrics(
            rule_id_macro_f1=invalid,
            rule_id_micro_f1=0.5,
            rule_id_exact_match=0.5,
            correction_fix_rate=0.5,
            joint_fix_rate=0.5,
        )


def test_evaluation_result_validates_counts_against_records() -> None:
    valid = EvaluationResult(
        records=(_direct_rule_record(),),
        aggregate_metrics=_metrics(),
        toolchain=_toolchain(),
        n_examples=1,
        n_excluded=0,
    )

    assert valid.n_examples == len(valid.records)
    with pytest.raises(ValueError, match="n_examples|record"):
        replace(valid, n_examples=2)
    with pytest.raises(ValueError, match="n_excluded|nonnegative"):
        replace(valid, n_excluded=-1)
