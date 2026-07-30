from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from pydantic import Field, FiniteFloat

from analysis.inputs import (
    CheckName,
    CorrectionRecord,
    ExplanationRecord,
    FindingStatus,
    JointRecord,
    RepairOutcome,
    RuleIdentificationRecord,
)

from go_ast_assistant.paper4.config import CHECK_NAMES, FineTunedCondition, StrictModel
from go_ast_assistant.paper4.eval.correction import CorrectionOutcome, OrigCache, score_correction
from go_ast_assistant.paper4.eval.rule_id import exact_match_rate, macro_f1, micro_f1, parse_rule_id_output
from go_ast_assistant.paper4.gocheck.runner import GoCheck
from go_ast_assistant.paper4.gocheck.toolchain import ToolchainInfo
from go_ast_assistant.paper4.prompts.registry import SYSTEM_PROMPT
from go_ast_assistant.paper4.records import TaskExample
from go_ast_assistant.paper4.runtime.generate import generate_bucketed
from go_ast_assistant.paper4.runtime.tokenizer import ChatFormat
from go_ast_assistant.paper4.training.composite import (
    CorrectionFixRateScorer,
    JointFixRateScorer,
    RuleIdMacroF1Scorer,
    ValidationComposite,
)
from go_ast_assistant.paper4.training.conditions import CONDITIONS, SummaryStore, assemble_main_user_content
from go_ast_assistant.paper4.training.instruction_dataset import EOT_ID


EvaluationRecord = RuleIdentificationRecord | CorrectionRecord | JointRecord | ExplanationRecord


class EvaluationAggregateMetrics(StrictModel):
    rule_id_macro_f1: FiniteFloat = Field(ge=0, le=1)
    rule_id_micro_f1: FiniteFloat = Field(ge=0, le=1)
    rule_id_exact_match: FiniteFloat = Field(ge=0, le=1)
    correction_fix_rate: FiniteFloat = Field(ge=0, le=1)
    joint_fix_rate: FiniteFloat = Field(ge=0, le=1)


@dataclass(frozen=True)
class EvaluationResult:
    records: tuple[EvaluationRecord, ...]
    aggregate_metrics: EvaluationAggregateMetrics
    toolchain: ToolchainInfo
    n_examples: int
    n_excluded: int

    def __post_init__(self) -> None:
        if type(self.n_examples) is not int or self.n_examples < 0:
            raise ValueError("n_examples must be a nonnegative integer")
        if type(self.n_excluded) is not int or self.n_excluded < 0:
            raise ValueError("n_excluded must be a nonnegative integer")
        if self.n_examples != len(self.records):
            raise ValueError("n_examples must equal the record count")
        allowed = (RuleIdentificationRecord, CorrectionRecord, JointRecord, ExplanationRecord)
        if any(not isinstance(record, allowed) for record in self.records):
            raise ValueError("records must use the authoritative evaluation record schemas")


class _Tokenizer(Protocol):
    eos_token_id: int

    def encode(self, text: str, *args: object, **kwargs: object) -> list[int]: ...


class _ChatFormat(Protocol):
    tok: _Tokenizer

    def encode(self, user_message: str, system_message: str | None = None) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class _PreparedPrompt:
    example: TaskExample
    prompt_ids: list[int]
    summary_status: Literal["present", "present_truncated", "skipped", "failed", "not_applicable"]


_MAX_NEW_TOKENS = {
    "rule_identification": 64,
    "correction": 512,
    "joint": 512,
    "explanation": 512,
}


def _ordered_checks(checks: Collection[str]) -> tuple[CheckName, ...]:
    check_set = set(checks)
    return cast(tuple[CheckName, ...], tuple(check for check in CHECK_NAMES if check in check_set))


def _prepare_prompts(
    examples: tuple[TaskExample, ...],
    condition: FineTunedCondition,
    tokenizer: _ChatFormat,
    summaries: SummaryStore,
) -> tuple[_PreparedPrompt, ...]:
    configured = CONDITIONS[condition]
    prepared: list[_PreparedPrompt] = []
    for example in examples:
        summary: str | None = None
        if configured.use_summary:
            code_tokens = len(tokenizer.tok.encode(example.code))
            rendered = summaries.render_for_main(example.id, code_tokens)
            summary_status = rendered.attached
            if summary_status in {"present", "present_truncated"}:
                summary = rendered.text
        else:
            summary_status = "not_applicable"
        user_content = assemble_main_user_content(
            example.task_type,
            example.code,
            summary,
            example.target_checks,
        )
        prompt_ids = tokenizer.encode(user_content, system_message=SYSTEM_PROMPT)
        prepared.append(
            _PreparedPrompt(
                example=example,
                prompt_ids=prompt_ids,
                summary_status=summary_status,
            )
        )
    return tuple(prepared)


def _generate_responses(
    prepared: tuple[_PreparedPrompt, ...],
    model: object,
    tokenizer: _ChatFormat,
) -> tuple[str, ...]:
    try:
        context_size = model.cfg["context_length"]  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("model must expose a positive integer context_length") from error
    if type(context_size) is not int or context_size <= 0:
        raise ValueError("model context_length must be a positive integer")

    groups: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(prepared):
        groups[_MAX_NEW_TOKENS[item.example.task_type]].append(index)
    responses: list[str | None] = [None] * len(prepared)
    for max_new_tokens, indices in groups.items():
        prompts = [prepared[index].prompt_ids for index in indices]
        generated = generate_bucketed(
            model,
            prompts,
            max_new_tokens,
            context_size,
            eos_id=EOT_ID,
        )
        if len(generated) != len(indices):
            raise ValueError(f"generation output count {len(generated)} does not match prompt count {len(indices)}")
        for index, prompt, tokens in zip(indices, prompts, generated, strict=True):
            if tokens[: len(prompt)] != prompt:
                raise ValueError("generated token sequence does not preserve its prompt prefix")
            responses[index] = tokenizer.decode(tokens[len(prompt) :])
    if any(response is None for response in responses):  # pragma: no cover - guarded by group cardinality
        raise ValueError("generation did not produce every response")
    return cast(tuple[str, ...], tuple(responses))


def _repair_projection(outcome: CorrectionOutcome) -> RepairOutcome:
    return RepairOutcome(
        target_fixed=outcome.target_fixed,
        overall_fixed=outcome.fix_rate_hit,
        studied_regression=outcome.studied_regression,
        enabled_regression=outcome.enabled_regression,
        extracted=outcome.extracted,
        extraction_status=outcome.extraction_status.value,
        parse_ok=outcome.parse_ok,
        lint_ok=outcome.lint_ok,
        original_tool_status=outcome.orig_tool_status,
        output_tool_status=outcome.out_tool_status,
        build_status=outcome.build_status,
        category=outcome.category,
        introduced_checks=outcome.introduced_checks,
        residual_findings=tuple(
            FindingStatus(check=finding.check, line=finding.line, column=finding.col)
            for finding in outcome.residual_findings
        ),
    )


def _records(
    prepared: tuple[_PreparedPrompt, ...],
    responses: tuple[str, ...],
    condition: FineTunedCondition,
    seed: Literal[42, 43, 44],
    tokenizer: _ChatFormat,
    gocheck: GoCheck,
) -> tuple[EvaluationRecord, ...]:
    records: list[EvaluationRecord] = []
    original_cache: OrigCache = {}
    for item, raw_response in zip(prepared, responses, strict=True):
        example = item.example
        common: dict[str, Any] = {
            "base_snippet_id": example.id,
            "condition": condition,
            "seed": seed,
            "task_type": example.task_type,
            "target_checks": _ordered_checks(example.target_checks),
            "summary_status": item.summary_status,
            "prompt_tokens": len(item.prompt_ids),
            "retokenized_response_token_proxy": len(tokenizer.tok.encode(raw_response)),
            "latency_ms": None,
        }
        if example.task_type == "rule_identification":
            parsed = parse_rule_id_output(raw_response, frozenset(CHECK_NAMES))
            gold = _ordered_checks(example.target_checks)
            records.append(
                RuleIdentificationRecord(
                    **common,
                    gold=gold,
                    pred=parsed.pred,
                    rejected_label_count=parsed.rejected_label_count,
                    exact_match=set(parsed.pred) == set(gold),
                    n_emitted=parsed.n_emitted,
                    normalization_status=parsed.normalization_status,
                )
            )
        elif example.task_type in {"correction", "joint"}:
            scored = score_correction(
                example.code,
                raw_response,
                set(example.target_checks),
                gocheck=gocheck,
                orig_cache=original_cache,
            )
            record_type = CorrectionRecord if example.task_type == "correction" else JointRecord
            records.append(
                record_type(
                    **common,
                    outcome=_repair_projection(scored),
                    extracted_similarity=None,
                    sensitivity_class=None,
                )
            )
        else:
            records.append(ExplanationRecord(**common))
    return tuple(records)


def _mean(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(records: tuple[EvaluationRecord, ...]) -> EvaluationAggregateMetrics:
    rule_records = tuple(record for record in records if isinstance(record, RuleIdentificationRecord))
    predictions = tuple(record.pred for record in rule_records)
    gold = tuple(record.gold for record in rule_records)
    correction = tuple(record for record in records if isinstance(record, CorrectionRecord))
    joint = tuple(record for record in records if isinstance(record, JointRecord))
    return EvaluationAggregateMetrics(
        rule_id_macro_f1=macro_f1(predictions, gold, CHECK_NAMES),
        rule_id_micro_f1=micro_f1(predictions, gold, CHECK_NAMES),
        rule_id_exact_match=exact_match_rate(predictions, gold),
        correction_fix_rate=_mean(tuple(record.outcome.overall_fixed for record in correction)),
        joint_fix_rate=_mean(tuple(record.outcome.overall_fixed for record in joint)),
    )


def evaluate(
    examples: Sequence[TaskExample],
    condition: FineTunedCondition,
    seed: Literal[42, 43, 44],
    model: object,
    tokenizer: ChatFormat,
    summaries: SummaryStore,
    toolchain: ToolchainInfo,
    excluded_ids: frozenset[str],
) -> EvaluationResult:
    example_tuple = tuple(examples)
    population_ids = frozenset(example.id for example in example_tuple)
    unknown = excluded_ids - population_ids
    if unknown:
        raise ValueError(f"evaluation exclusion IDs are missing from the input population: {sorted(unknown)}")
    kept = tuple(example for example in example_tuple if example.id not in excluded_ids)
    if not kept:
        raise ValueError("empty evaluation population after exclusions")

    prepared = _prepare_prompts(kept, condition, tokenizer, summaries)
    responses = _generate_responses(prepared, model, tokenizer)
    gocheck = GoCheck(toolchain)
    records = _records(prepared, responses, condition, seed, tokenizer, gocheck)
    return EvaluationResult(
        records=records,
        aggregate_metrics=_aggregate(records),
        toolchain=toolchain,
        n_examples=len(records),
        n_excluded=len(example_tuple) - len(kept),
    )


def _rule_id_score(examples: tuple[TaskExample, ...], outputs: tuple[str, ...]) -> float | None:
    if not examples:
        return None
    predictions = tuple(parse_rule_id_output(output, frozenset(CHECK_NAMES)).pred for output in outputs)
    gold = tuple(_ordered_checks(example.target_checks) for example in examples)
    return macro_f1(predictions, gold, CHECK_NAMES)


def _repair_score(
    examples: tuple[TaskExample, ...],
    outputs: tuple[str, ...],
    gocheck: GoCheck,
) -> float | None:
    if not examples:
        return None
    cache: OrigCache = {}
    hits = tuple(
        score_correction(
            example.code,
            output,
            set(example.target_checks),
            gocheck=gocheck,
            orig_cache=cache,
        ).fix_rate_hit
        for example, output in zip(examples, outputs, strict=True)
    )
    return _mean(hits)


def build_validation_composite(toolchain: ToolchainInfo) -> ValidationComposite:
    gocheck = GoCheck(toolchain)

    def correction_score(examples: tuple[TaskExample, ...], outputs: tuple[str, ...]) -> float | None:
        return _repair_score(examples, outputs, gocheck)

    def joint_score(examples: tuple[TaskExample, ...], outputs: tuple[str, ...]) -> float | None:
        return _repair_score(examples, outputs, gocheck)

    return ValidationComposite(
        (
            RuleIdMacroF1Scorer(score_outputs_fn=_rule_id_score),
            CorrectionFixRateScorer(score_outputs_fn=correction_score),
            JointFixRateScorer(score_outputs_fn=joint_score),
        )
    )
