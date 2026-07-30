from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from go_ast_assistant.paper4.prepared_study import (
    BudgetGate,
    BudgetGuard,
    LengthBudgetPayload,
    LengthDistribution,
    PreExclusionTruncation,
    PreparedStudy,
    PreparedSummaryLine,
    PreparedSummaryRecord,
)
from go_ast_assistant.paper4.records import TaskExample
from go_ast_assistant.paper4.training.length_budget import (
    LengthRecord,
    forwarded_token_count,
    percentile,
    supervised_token_count,
    validate_lengths,
)


CONDITIONS = ("C0", "C1", "C2", "C2-control")
SEEDS = (42, 43, 44)
STREAM_TOTALS = {
    "C0": 38_400,
    "C1": 57_600,
    "C2": 57_600,
    "C2-control": 57_600,
}
SUPERVISED_TOTALS = dict.fromkeys(CONDITIONS, 38_400)


def _id(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


def _example(label: str, task_type: str = "rule_identification") -> TaskExample:
    return TaskExample(
        id=_id(label),
        split="train",
        task_type=task_type,  # type: ignore[arg-type]
        target_checks=("assignOp",) if task_type != "syntax_summary" else (),
        code=f"code:{label}",
        target=f"target:{label}",
        meta={},
    )


class _RawTokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str, allowed_special: object = None) -> list[int]:
        del allowed_special
        self.calls.append(text)
        return [1] * 40 if text.startswith("code:") else [1]


class _ChatTokenizer:
    def __init__(self) -> None:
        self.tok = _RawTokenizer()

    def encode(
        self,
        user_message: str,
        system_message: str | None = None,
        allowed_special: object = None,
    ) -> list[int]:
        del system_message, allowed_special
        return [2, 3] if "SYNTAX SUMMARY:" in user_message else [2]


def _distribution(length: int, n: int) -> LengthDistribution:
    return LengthDistribution(p50=length, p90=length, p95=length, p99=length, max=length, n=n)


def _guards(values: dict[str, int]) -> dict[str, BudgetGuard]:
    reference = values["C1"]
    result: dict[str, BudgetGuard] = {}
    for condition, value in values.items():
        delta = (value - reference) / reference
        guarded = condition in {"C2", "C2-control"}
        result[condition] = BudgetGuard(
            delta=delta,
            exceeds=guarded and abs(delta) > 0.05,
            guarded=guarded,
        )
    return result


def _budget() -> LengthBudgetPayload:
    token_matrix = {str(seed): dict(STREAM_TOTALS) for seed in SEEDS}
    guard_matrix = {str(seed): _guards(STREAM_TOTALS) for seed in SEEDS}
    supervised_tokens = {str(seed): dict(SUPERVISED_TOTALS) for seed in SEEDS}
    supervised_guards = {str(seed): _guards(SUPERVISED_TOTALS) for seed in SEEDS}
    return LengthBudgetPayload(
        allowed_max_length=9_305,
        distributions={
            "C0:rule_identification": _distribution(3, 1),
            "C1:rule_identification": _distribution(4, 1),
            "C2:rule_identification": _distribution(4, 1),
            "C2:syntax_summary": _distribution(3, 1),
            "C2-control:rule_identification": _distribution(4, 4),
        },
        pre_exclusion_truncation=PreExclusionTruncation(
            prompt_truncated={},
            response_truncated={},
            total=0,
        ),
        tokens_by_seed=token_matrix,  # type: ignore[arg-type]
        token_budget_guard_by_seed=guard_matrix,  # type: ignore[arg-type]
        supervised_tokens_by_seed=supervised_tokens,  # type: ignore[arg-type]
        supervised_token_budget_guard_by_seed=supervised_guards,  # type: ignore[arg-type]
        data_fraction=1.0,
        aux_ratio=None,
        max_steps=600,
        micro_batch_size=2,
        eff_batch=32,
        aux_stratification="response",
        budget_gate=BudgetGate(total="report_only", supervised="strict"),
        exclusion_ids_sha256=hashlib.sha256(b"").hexdigest(),
    )


def _study(condition: str) -> PreparedStudy:
    main = _example("main")
    summary = PreparedSummaryRecord(
        id=main.id,
        ok=True,
        parse_strategy="file",
        type_facts_available=False,
        lines=(
            PreparedSummaryLine(
                tier=0,
                depth=0,
                text="func main()",
                segments=(),
            ),
        ),
        excluded_constructs=(),
        parse_error=None,
    )
    auxiliary: tuple[TaskExample, ...] = ()
    if condition == "C2":
        auxiliary = (_example("syntax", "syntax_summary"),)
    elif condition == "C2-control":
        first = _example("control-first")
        second = _example("control-second")
        auxiliary = (first, second, first)
    summary_examples = (main, *auxiliary)
    return PreparedStudy(
        root=Path("must-not-be-read"),
        tasks_by_split={"train": (main,), "validation": (), "test": ()},  # type: ignore[arg-type]
        length_budget=_budget(),
        length_exclusion_ids=frozenset(),
        composite_validation_ids=frozenset(),
        quarantine_ids=frozenset(),
        adjudications={},
        summaries={example.id: summary.model_copy(update={"id": example.id}) for example in summary_examples}
        if condition != "C0"
        else None,
        auxiliary_examples=auxiliary,
    )


def _with_ignored_slice_drift(study: PreparedStudy, condition: str, seed: int) -> PreparedStudy:
    budget = study.length_budget
    distributions = dict(budget.distributions)
    ignored_conditions = tuple(candidate for candidate in CONDITIONS if candidate != condition)
    for index, ignored_condition in enumerate(ignored_conditions, start=1):
        for key in tuple(distributions):
            if key.split(":", maxsplit=1)[0] == ignored_condition:
                distributions[key] = LengthDistribution(
                    p50=10 + index,
                    p90=20 + index,
                    p95=30 + index,
                    p99=40 + index,
                    max=50 + index,
                    n=60 + index,
                )

    prompt_truncated = {
        f"{ignored_condition}:rule_identification": 10 + index
        for index, ignored_condition in enumerate(ignored_conditions)
    }
    response_truncated = {
        f"{ignored_condition}:rule_identification": 20 + index
        for index, ignored_condition in enumerate(ignored_conditions)
    }
    truncation = budget.pre_exclusion_truncation.model_copy(
        update={
            "prompt_truncated": prompt_truncated,
            "response_truncated": response_truncated,
            "total": 9_999,
        }
    )

    tokens = {matrix_seed: dict(values) for matrix_seed, values in budget.tokens_by_seed.items()}
    supervised = {matrix_seed: dict(values) for matrix_seed, values in budget.supervised_tokens_by_seed.items()}
    guards = {matrix_seed: dict(values) for matrix_seed, values in budget.token_budget_guard_by_seed.items()}
    supervised_guards = {
        matrix_seed: dict(values) for matrix_seed, values in budget.supervised_token_budget_guard_by_seed.items()
    }
    for matrix_seed in map(str, SEEDS):
        for index, matrix_condition in enumerate(CONDITIONS, start=1):
            if (matrix_seed, matrix_condition) == (str(seed), condition):
                continue
            if matrix_seed == str(seed) and matrix_condition == "C1":
                continue
            sentinel = int(matrix_seed) * 1_000 + index
            tokens[matrix_seed][matrix_condition] = sentinel
            supervised[matrix_seed][matrix_condition] = sentinel + 100
            guards[matrix_seed][matrix_condition] = BudgetGuard(delta=0.5, exceeds=True, guarded=False)
            supervised_guards[matrix_seed][matrix_condition] = BudgetGuard(
                delta=0.75,
                exceeds=True,
                guarded=True,
            )

    return replace(
        study,
        length_budget=budget.model_copy(
            update={
                "distributions": distributions,
                "pre_exclusion_truncation": truncation,
                "tokens_by_seed": tokens,
                "supervised_tokens_by_seed": supervised,
                "token_budget_guard_by_seed": guards,
                "supervised_token_budget_guard_by_seed": supervised_guards,
            }
        ),
    )


def _with_requested_mismatch(
    study: PreparedStudy,
    condition: str,
    seed: int,
    field: str,
    *,
    task_type: str = "rule_identification",
) -> PreparedStudy:
    budget = study.length_budget
    update: dict[str, Any]
    if field == "distribution":
        distributions = dict(budget.distributions)
        key = f"{condition}:{task_type}"
        distributions[key] = distributions[key].model_copy(update={"max": distributions[key].max + 1})
        update = {"distributions": distributions}
    elif field in {"prompt_truncated", "response_truncated"}:
        truncation = budget.pre_exclusion_truncation.model_copy(update={field: {f"{condition}:{task_type}": 1}})
        update = {"pre_exclusion_truncation": truncation}
    elif field in {"tokens_by_seed", "supervised_tokens_by_seed"}:
        matrix = {matrix_seed: dict(values) for matrix_seed, values in getattr(budget, field).items()}
        matrix[str(seed)][condition] += 1
        update = {field: matrix}
    elif field in {"token_budget_guard_by_seed", "supervised_token_budget_guard_by_seed"}:
        matrix = {matrix_seed: dict(values) for matrix_seed, values in getattr(budget, field).items()}
        requested = matrix[str(seed)][condition]
        matrix[str(seed)][condition] = requested.model_copy(update={"guarded": not requested.guarded})
        update = {field: matrix}
    else:  # pragma: no cover - the parametrization below is exhaustive
        raise AssertionError(field)
    return replace(study, length_budget=budget.model_copy(update=update))


def test_length_helpers_retain_fixed_record_shape_and_token_accounting() -> None:
    record = LengthRecord("rule_identification", 5, 8)

    assert record.total_len == 8
    assert percentile([10, 20, 30, 40, 100], 99.0) == 100
    assert forwarded_token_count([5, 3], micro_batch_size=2, allowed=10) == 7
    assert supervised_token_count([(10, 4), (6, 5), (20, 18)], 2, 12) == 7


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_validate_lengths_checks_every_requested_condition_seed_slice(condition: str, seed: int) -> None:
    study = _with_ignored_slice_drift(_study(condition), condition, seed)
    tokenizer = _ChatTokenizer()

    validate_lengths(study, tokenizer, condition, seed, "paper")  # type: ignore[arg-type]

    assert study.tasks_by_split["train"][0].code == "code:main"


@pytest.mark.parametrize(
    "field",
    (
        "distribution",
        "prompt_truncated",
        "response_truncated",
        "tokens_by_seed",
        "supervised_tokens_by_seed",
        "token_budget_guard_by_seed",
        "supervised_token_budget_guard_by_seed",
    ),
)
@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_validate_lengths_rejects_every_requested_slice_mismatch(
    condition: str,
    seed: int,
    field: str,
) -> None:
    study = _with_requested_mismatch(_study(condition), condition, seed, field)

    with pytest.raises(ValueError):
        validate_lengths(
            study,
            _ChatTokenizer(),
            condition,  # type: ignore[arg-type]
            seed,  # type: ignore[arg-type]
            "paper",
        )


@pytest.mark.parametrize("field", ("tokens_by_seed", "supervised_tokens_by_seed"))
@pytest.mark.parametrize("condition", ("C2", "C2-control"))
def test_validate_lengths_rejects_guard_reference_mismatch(condition: str, field: str) -> None:
    study = _study(condition)
    budget = study.length_budget
    matrix = {matrix_seed: dict(values) for matrix_seed, values in getattr(budget, field).items()}
    matrix["42"]["C1"] += 1
    study = replace(study, length_budget=budget.model_copy(update={field: matrix}))

    with pytest.raises(ValueError):
        validate_lengths(
            study,
            _ChatTokenizer(),
            condition,  # type: ignore[arg-type]
            42,
            "paper",
        )


@pytest.mark.parametrize("condition", ("C2", "C2-control"))
def test_validate_lengths_rejects_strict_supervised_token_overage(condition: str) -> None:
    study = _study(condition)
    budget = study.length_budget
    supervised = {matrix_seed: dict(values) for matrix_seed, values in budget.supervised_tokens_by_seed.items()}
    supervised["42"]["C1"] = 19_200
    supervised_guards = {
        matrix_seed: dict(values) for matrix_seed, values in budget.supervised_token_budget_guard_by_seed.items()
    }
    supervised_guards["42"][condition] = BudgetGuard(delta=1.0, exceeds=True, guarded=True)
    study = replace(
        study,
        length_budget=budget.model_copy(
            update={
                "supervised_tokens_by_seed": supervised,
                "supervised_token_budget_guard_by_seed": supervised_guards,
            }
        ),
    )

    with pytest.raises(ValueError, match="supervised token budget.*strict"):
        validate_lengths(
            study,
            _ChatTokenizer(),
            condition,  # type: ignore[arg-type]
            42,
            "paper",
        )


@pytest.mark.parametrize("condition", ("C2", "C2-control"))
def test_validate_lengths_accepts_report_only_total_token_overage(condition: str) -> None:
    study = _study(condition)
    budget = study.length_budget
    tokens = {matrix_seed: dict(values) for matrix_seed, values in budget.tokens_by_seed.items()}
    tokens["42"]["C1"] = 28_800
    guards = {matrix_seed: dict(values) for matrix_seed, values in budget.token_budget_guard_by_seed.items()}
    guards["42"][condition] = BudgetGuard(delta=1.0, exceeds=True, guarded=True)
    study = replace(
        study,
        length_budget=budget.model_copy(
            update={
                "tokens_by_seed": tokens,
                "token_budget_guard_by_seed": guards,
            }
        ),
    )

    validate_lengths(
        study,
        _ChatTokenizer(),
        condition,  # type: ignore[arg-type]
        42,
        "paper",
    )


@pytest.mark.parametrize("field", ("distribution", "prompt_truncated", "response_truncated"))
@pytest.mark.parametrize("seed", SEEDS)
def test_validate_lengths_rejects_requested_c2_syntax_auxiliary_mismatch(seed: int, field: str) -> None:
    study = _with_requested_mismatch(
        _study("C2"),
        "C2",
        seed,
        field,
        task_type="syntax_summary",
    )

    with pytest.raises(ValueError):
        validate_lengths(study, _ChatTokenizer(), "C2", seed, "paper")  # type: ignore[arg-type]


def test_validate_lengths_preserves_control_repetition_and_source_order() -> None:
    study = _study("C2-control")
    before = study.auxiliary_examples
    tokenizer = _ChatTokenizer()

    validate_lengths(study, tokenizer, "C2-control", 42, "paper")

    assert study.auxiliary_examples == before
    control_code_calls = [text for text in tokenizer.tok.calls if text.startswith("code:control-")]
    assert control_code_calls == ["code:control-first", "code:control-second", "code:control-first"]
