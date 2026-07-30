from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from go_ast_assistant.paper4.adjudication import Adjudication, load_adjudications
from go_ast_assistant.paper4.config import EXPECTED_SPLIT_SIZES, FineTunedCondition, StrictModel, TaskType
from go_ast_assistant.paper4.records import TaskExample, load_jsonl, load_task_examples, reject_duplicate_json_keys

_SPLITS = ("train", "validation", "test")
_MAIN_TASKS = ("rule_identification", "correction", "joint", "explanation")
_CONDITIONS = ("C0", "C1", "C2", "C2-control")
_SEEDS = ("42", "43", "44")
_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class AnnotationSegment(StrictModel):
    a: str


class VariableSegment(StrictModel):
    v: str


class PreparedSummaryLine(StrictModel):
    tier: Literal[0, 1, 2]
    depth: int = Field(ge=0)
    text: str
    segments: tuple[AnnotationSegment | VariableSegment, ...]

    @field_validator("tier", mode="before")
    @classmethod
    def require_integer_tier(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("summary tier must use an integer scalar")
        return value


class PreparedSummaryRecord(StrictModel):
    id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ok: bool
    parse_strategy: Literal["file", "package", "func_body", "expr"] | None
    type_facts_available: bool
    lines: tuple[PreparedSummaryLine, ...]
    excluded_constructs: tuple[str, ...]
    parse_error: str | None

    @model_validator(mode="after")
    def validate_parse_outcome(self) -> PreparedSummaryRecord:
        if self.ok:
            if self.parse_strategy is None or self.parse_error is not None:
                raise ValueError("successful summary requires a strategy and null parse_error")
        elif (
            self.parse_strategy is not None
            or self.lines
            or self.type_facts_available
            or self.parse_error is None
            or not self.parse_error.strip()
        ):
            raise ValueError("failed summary requires null strategy, empty lines, and a parse error")
        return self


class LengthDistribution(StrictModel):
    p50: int = Field(ge=0)
    p90: int = Field(ge=0)
    p95: int = Field(ge=0)
    p99: int = Field(ge=0)
    max: int = Field(ge=0)
    n: int = Field(ge=0)


class BudgetGuard(StrictModel):
    delta: FiniteFloat
    exceeds: bool
    guarded: bool


class PreExclusionTruncation(StrictModel):
    prompt_truncated: dict[str, int]
    response_truncated: dict[str, int]
    total: int = Field(ge=0)


class BudgetGate(StrictModel):
    total: Literal["report_only"]
    supervised: Literal["strict"]


_TokenMap = dict[Literal["42", "43", "44"], dict[FineTunedCondition, int]]
_GuardMap = dict[Literal["42", "43", "44"], dict[FineTunedCondition, BudgetGuard]]


class LengthBudgetPayload(StrictModel):
    allowed_max_length: Literal[9305]
    distributions: dict[str, LengthDistribution]
    pre_exclusion_truncation: PreExclusionTruncation
    tokens_by_seed: _TokenMap
    token_budget_guard_by_seed: _GuardMap
    supervised_tokens_by_seed: _TokenMap
    supervised_token_budget_guard_by_seed: _GuardMap
    data_fraction: Literal[1.0]
    aux_ratio: None
    max_steps: Literal[600]
    micro_batch_size: Literal[2]
    eff_batch: Literal[32]
    aux_stratification: Literal["response"]
    budget_gate: BudgetGate
    exclusion_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("allowed_max_length", "max_steps", "micro_batch_size", "eff_batch", mode="before")
    @classmethod
    def require_integer_scalar(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("fixed length-budget field must use an integer scalar")
        return value

    @field_validator("data_fraction", mode="before")
    @classmethod
    def require_float_data_fraction(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("data_fraction must use a floating-point scalar")
        return value

    @model_validator(mode="after")
    def validate_exact_budget_matrix(self) -> LengthBudgetPayload:
        matrices = (
            self.tokens_by_seed,
            self.token_budget_guard_by_seed,
            self.supervised_tokens_by_seed,
            self.supervised_token_budget_guard_by_seed,
        )
        for matrix in matrices:
            if set(matrix) != set(_SEEDS):
                raise ValueError("budget matrix requires the exact seed keys")
            if any(set(matrix[seed]) != set(_CONDITIONS) for seed in _SEEDS):
                raise ValueError("budget matrix requires the exact condition keys")
        return self


@dataclass(frozen=True)
class PreparedStudy:
    root: Path
    tasks_by_split: Mapping[Literal["train", "validation", "test"], tuple[TaskExample, ...]]
    length_budget: LengthBudgetPayload
    length_exclusion_ids: frozenset[str]
    composite_validation_ids: frozenset[str]
    quarantine_ids: frozenset[str]
    adjudications: Mapping[str, Adjudication]
    summaries: Mapping[str, PreparedSummaryRecord] | None
    auxiliary_examples: tuple[TaskExample, ...]


class _QuarantineProjection(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False, frozen=True, strict=True)

    id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split: Literal["train", "validation", "test"]


def _require_files(root: Path, condition: FineTunedCondition) -> None:
    required = {
        root / "tasks" / "train.jsonl",
        root / "tasks" / "validation.jsonl",
        root / "tasks" / "test.jsonl",
        root / "length_budget.json",
        root / "length_exclusion_ids.txt",
        root / "composite_val_ids.txt",
        root / "quarantine.jsonl",
        root / "oracle_adjudication.jsonl",
    }
    if condition in {"C1", "C2", "C2-control"}:
        required.add(root / "summaries.jsonl")
    if condition == "C2":
        required.add(root / "aux_pool_syntax.jsonl")
    if condition == "C2-control":
        required.add(root / "aux_pool_main_dup.jsonl")
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise ValueError(f"required prepared-study files are missing or non-regular: {missing}")


def _load_main_tasks(
    root: Path,
) -> tuple[
    dict[Literal["train", "validation", "test"], tuple[TaskExample, ...]],
    dict[Literal["train", "validation", "test"], frozenset[str]],
    dict[tuple[str, TaskType], TaskExample],
]:
    tasks_by_split: dict[Literal["train", "validation", "test"], tuple[TaskExample, ...]] = {}
    ids_by_split: dict[Literal["train", "validation", "test"], frozenset[str]] = {}
    canonical: dict[tuple[str, TaskType], TaskExample] = {}
    task_sets_by_split: dict[str, dict[str, set[str]]] = {}

    for raw_split in _SPLITS:
        split = cast(Literal["train", "validation", "test"], raw_split)
        rows = load_task_examples(root / "tasks" / f"{split}.jsonl")
        tasks_by_split[split] = rows
        tasks_by_id: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row.split != split:
                raise ValueError(f"task row {row.id} declares split {row.split!r} in {split}.jsonl")
            if row.task_type not in _MAIN_TASKS:
                raise ValueError(f"main task file contains non-main task {row.task_type!r}")
            task_type = cast(TaskType, row.task_type)
            cell = (row.id, task_type)
            if cell in canonical:
                raise ValueError(f"duplicate main task cell: {cell}")
            canonical[cell] = row
            tasks_by_id[row.id].add(task_type)
        expected_size = EXPECTED_SPLIT_SIZES[split]
        if len(tasks_by_id) != expected_size:
            raise ValueError(f"{split} requires exactly {expected_size} unique IDs")
        ids_by_split[split] = frozenset(tasks_by_id)
        task_sets_by_split[split] = dict(tasks_by_id)

    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(ids_by_split[left] & ids_by_split[right] for left, right in split_pairs):
        raise ValueError("task IDs must be disjoint across splits")

    full_tasks = set(_MAIN_TASKS)
    partial_tasks = {"rule_identification", "explanation"}
    train_task_sets = tuple(task_sets_by_split["train"].values())
    if sum(tasks == full_tasks for tasks in train_task_sets) != EXPECTED_SPLIT_SIZES["train"] - 3:
        raise ValueError("train requires all but three IDs to have the four exact main tasks")
    if sum(tasks == partial_tasks for tasks in train_task_sets) != 3:
        raise ValueError("train requires exactly three rule-identification/explanation-only IDs")
    if any(tasks != full_tasks for split in ("validation", "test") for tasks in task_sets_by_split[split].values()):
        raise ValueError("validation and test IDs require the four exact main tasks")

    return tasks_by_split, ids_by_split, canonical


def _load_plain_ids(path: Path) -> frozenset[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read required ID file: {path}") from error
    if any(not line or _ID_PATTERN.fullmatch(line) is None for line in lines):
        raise ValueError(f"invalid ID row in {path}")
    if len(lines) != len(set(lines)):
        raise ValueError(f"duplicate ID in {path}")
    return frozenset(lines)


def _load_quarantine(path: Path) -> tuple[_QuarantineProjection, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read required quarantine file: {path}") from error
    if not text:
        return ()
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"quarantine JSONL contains a blank row: {path}")
    result: list[_QuarantineProjection] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            reject_duplicate_json_keys(line)
            row = _QuarantineProjection.model_validate_json(line)
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid quarantine row {line_number} in {path}: {error}") from error
        if row.id in seen:
            raise ValueError(f"duplicate quarantine ID: {row.id}")
        seen.add(row.id)
        result.append(row)
    return tuple(result)


def _load_length_budget(path: Path) -> LengthBudgetPayload:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read required length budget: {path}") from error
    reject_duplicate_json_keys(text)
    try:
        return LengthBudgetPayload.model_validate_json(text)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid length budget {path}: {error}") from error


def _load_summaries(root: Path, all_ids: frozenset[str]) -> dict[str, PreparedSummaryRecord]:
    summaries: dict[str, PreparedSummaryRecord] = {}
    for summary in load_jsonl(root / "summaries.jsonl", PreparedSummaryRecord):
        if summary.id in summaries:
            raise ValueError(f"duplicate summary ID: {summary.id}")
        summaries[summary.id] = summary
    if frozenset(summaries) != all_ids:
        raise ValueError("summaries must cover exactly every loaded task ID")
    return summaries


def _render_summary(summary: PreparedSummaryRecord) -> str:
    return "\n".join(f"{'  ' * line.depth}{line.text}" for line in summary.lines)


def _validate_syntax_auxiliary(
    root: Path,
    train_ids: frozenset[str],
    canonical: dict[tuple[str, TaskType], TaskExample],
    summaries: dict[str, PreparedSummaryRecord],
) -> tuple[TaskExample, ...]:
    rows = load_task_examples(root / "aux_pool_syntax.jsonl")
    by_id: dict[str, TaskExample] = {}
    for row in rows:
        if row.id in by_id:
            raise ValueError(f"duplicate syntax-auxiliary ID: {row.id}")
        by_id[row.id] = row
        if row.split != "train" or row.task_type != "syntax_summary" or row.target_checks:
            raise ValueError(f"invalid syntax-auxiliary row: {row.id}")
        if row.id not in train_ids:
            raise ValueError(f"syntax-auxiliary row is not a train ID: {row.id}")
        summary = summaries[row.id]
        if not summary.ok or not summary.lines:
            raise ValueError(f"syntax-auxiliary row requires a successful nonempty summary: {row.id}")
        source = canonical[(row.id, "rule_identification")]
        if row.code != source.code or row.target != _render_summary(summary):
            raise ValueError(f"syntax-auxiliary row disagrees with its source or summary: {row.id}")
        if row.meta.get("aux_role") != "syntax":
            raise ValueError(f"syntax-auxiliary row has the wrong auxiliary role: {row.id}")
        if row.meta.get("source_revision") != source.meta.get("source_revision"):
            raise ValueError(f"syntax-auxiliary row has the wrong source revision: {row.id}")
        excluded = row.meta.get("excluded_constructs")
        if not isinstance(excluded, list) or tuple(excluded) != summary.excluded_constructs:
            raise ValueError(f"syntax-auxiliary row has inconsistent excluded constructs: {row.id}")
    if frozenset(by_id) != train_ids or len(rows) != EXPECTED_SPLIT_SIZES["train"]:
        raise ValueError("C2 syntax pool requires exactly one row for every train ID")
    return rows


def _validate_control_auxiliary(
    root: Path,
    canonical: dict[tuple[str, TaskType], TaskExample],
) -> tuple[TaskExample, ...]:
    rows = load_task_examples(root / "aux_pool_main_dup.jsonl")
    if len(rows) != EXPECTED_SPLIT_SIZES["train"]:
        raise ValueError("C2-control pool requires exactly one sampled row per train ID count")
    for row in rows:
        if row.split != "train" or row.task_type not in _MAIN_TASKS:
            raise ValueError(f"invalid C2-control auxiliary row: {row.id}")
        task_type = cast(TaskType, row.task_type)
        source = canonical.get((row.id, task_type))
        if source is None:
            raise ValueError(f"C2-control row has no canonical main cell: {(row.id, task_type)}")
        actual = row.model_dump(mode="python")
        actual_meta = dict(actual["meta"])
        if actual_meta.pop("aux_role", None) != "main_dup":
            raise ValueError(f"C2-control row has the wrong auxiliary role: {row.id}")
        actual["meta"] = actual_meta
        if actual != source.model_dump(mode="python"):
            raise ValueError(f"C2-control row differs from its canonical main cell: {(row.id, task_type)}")
    return rows


def validate_prepared_study(root: Path, condition: FineTunedCondition) -> PreparedStudy:
    if condition not in _CONDITIONS:
        raise ValueError(f"unknown fine-tuned condition: {condition!r}")
    root = Path(root)
    _require_files(root, condition)
    tasks_by_split, ids_by_split, canonical = _load_main_tasks(root)
    all_ids = frozenset().union(*ids_by_split.values())

    length_exclusion_ids = _load_plain_ids(root / "length_exclusion_ids.txt")
    if not length_exclusion_ids <= ids_by_split["train"]:
        raise ValueError("length exclusions must contain only train IDs")

    adjudications = load_adjudications(root / "oracle_adjudication.jsonl")
    for adjudication in adjudications.values():
        if adjudication.id not in ids_by_split[adjudication.split]:
            raise ValueError(f"adjudication ID does not belong to its declared split: {adjudication.id}")

    quarantine = _load_quarantine(root / "quarantine.jsonl")
    for row in quarantine:
        if row.id not in ids_by_split[row.split]:
            raise ValueError(f"quarantine ID does not belong to its declared split: {row.id}")
    quarantine_ids = frozenset(row.id for row in quarantine)

    composite_validation_ids = _load_plain_ids(root / "composite_val_ids.txt")
    excluded_validation_ids = {
        adjudication.id
        for adjudication in adjudications.values()
        if adjudication.split == "validation" and adjudication.resolution == "exclude"
    }
    retained_validation_ids = ids_by_split["validation"] - quarantine_ids - excluded_validation_ids
    if not composite_validation_ids or not composite_validation_ids <= retained_validation_ids:
        raise ValueError("composite validation IDs must be retained validation IDs")

    length_budget = _load_length_budget(root / "length_budget.json")
    exclusion_hash = hashlib.sha256("\n".join(sorted(length_exclusion_ids)).encode()).hexdigest()
    if length_budget.exclusion_ids_sha256 != exclusion_hash:
        raise ValueError("length-budget exclusion hash disagrees with length_exclusion_ids.txt")

    summaries: dict[str, PreparedSummaryRecord] | None = None
    auxiliary_examples: tuple[TaskExample, ...] = ()
    if condition in {"C1", "C2", "C2-control"}:
        summaries = _load_summaries(root, all_ids)
    if condition == "C2":
        assert summaries is not None
        auxiliary_examples = _validate_syntax_auxiliary(root, ids_by_split["train"], canonical, summaries)
    elif condition == "C2-control":
        auxiliary_examples = _validate_control_auxiliary(root, canonical)

    immutable_tasks = MappingProxyType(tasks_by_split)
    immutable_adjudications = MappingProxyType(adjudications)
    immutable_summaries = MappingProxyType(summaries) if summaries is not None else None
    return PreparedStudy(
        root=root,
        tasks_by_split=immutable_tasks,
        length_budget=length_budget,
        length_exclusion_ids=length_exclusion_ids,
        composite_validation_ids=composite_validation_ids,
        quarantine_ids=quarantine_ids,
        adjudications=immutable_adjudications,
        summaries=immutable_summaries,
        auxiliary_examples=auxiliary_examples,
    )


def evaluation_exclusion_ids(study: PreparedStudy) -> frozenset[str]:
    test_ids = frozenset(row.id for row in study.tasks_by_split["test"])
    quarantined_test_ids = study.quarantine_ids & test_ids
    adjudicated_test_ids = {
        adjudication.id
        for adjudication in study.adjudications.values()
        if adjudication.split == "test" and adjudication.resolution == "exclude"
    }
    return quarantined_test_ids | frozenset(adjudicated_test_ids)
