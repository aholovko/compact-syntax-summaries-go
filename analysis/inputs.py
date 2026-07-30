from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

Condition = Literal["C0", "C1", "C2", "C2-control", "zero-shot-raw", "zero-shot-syntax"]
FineTunedCondition = Literal["C0", "C1", "C2", "C2-control"]
TaskType = Literal["rule_identification", "correction", "joint", "explanation"]
CheckName = Literal[
    "assignOp",
    "builtinShadow",
    "captLocal",
    "commentFormatting",
    "elseif",
    "ifElseChain",
    "paramTypeCombine",
    "singleCaseSwitch",
]
RunSummaryStatus = Literal["present", "present_truncated", "skipped", "failed", "not_applicable"]
StudySummaryStatus = Literal["present", "present_truncated", "skipped", "failed"]
NormalizationStatus = Literal["recognized_array", "no_recognized_array"]
BuildStatus = Literal["OK", "NA", "FAIL"]
ToolStatus = Literal["ok", "load_degraded", "load_failed"]
OriginalToolStatus = ToolStatus
OutputToolStatus = ToolStatus
SensitivityClass = Literal["same_file_truncated", "same_pkg_no_shared_func"]
BoundedIdentifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
CheckIdentifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")]
SemanticLocation = Annotated[
    str,
    StringConstraints(
        pattern=r"^(section|appendix|table):[A-Za-z0-9.]+#[a-z0-9][a-z0-9._-]*$",
        max_length=160,
    ),
]
ResultIdentifier = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$", max_length=180),
]
OutputCoordinate = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")]
CHECKS = (
    "assignOp",
    "builtinShadow",
    "captLocal",
    "commentFormatting",
    "elseif",
    "ifElseChain",
    "paramTypeCombine",
    "singleCaseSwitch",
)
TASK_TYPES = ("rule_identification", "correction", "joint", "explanation")

_FINE_TUNED_COMMIT = "16aadb26296b291538de481265a149dcb6db8876"
_ZERO_SHOT_COMMIT = "520d5ce6c49864405c5946cae57ec794b00e4218"
_RUN_FILES = frozenset({"records.jsonl", "results.yaml", "selection_trace.json", "manifest.yaml"})
_CONDITION_ORDER = {
    "C0": 0,
    "C1": 1,
    "C2": 2,
    "C2-control": 3,
    "zero-shot-raw": 4,
    "zero-shot-syntax": 5,
}
_TASK_ORDER = {task_type: index for index, task_type in enumerate(TASK_TYPES)}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True, strict=True)


class CommonRecord(StrictModel):
    base_snippet_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    condition: Condition
    seed: Literal[42, 43, 44]
    task_type: TaskType
    target_checks: tuple[CheckName, ...]
    summary_status: RunSummaryStatus
    prompt_tokens: int = Field(ge=0)
    retokenized_response_token_proxy: int = Field(ge=0)
    latency_ms: FiniteFloat | None


class RuleIdentificationRecord(CommonRecord):
    task_type: Literal["rule_identification"]
    gold: tuple[CheckName, ...]
    pred: tuple[CheckName, ...]
    rejected_label_count: int = Field(ge=0)
    exact_match: bool
    n_emitted: int = Field(ge=0)
    normalization_status: NormalizationStatus

    @model_validator(mode="after")
    def validate_normalization(self) -> RuleIdentificationRecord:
        if self.normalization_status == "no_recognized_array":
            if self.pred or self.n_emitted or self.rejected_label_count:
                raise ValueError("no_recognized_array requires empty pred and zero emission counts")
        if self.n_emitted < len(self.pred) + self.rejected_label_count:
            raise ValueError("n_emitted cannot be smaller than normalized plus rejected members")
        return self


class FindingStatus(StrictModel):
    check: CheckIdentifier
    line: int = Field(gt=0)
    column: int = Field(gt=0)


class RepairOutcome(StrictModel):
    target_fixed: bool
    overall_fixed: bool
    studied_regression: bool | None
    enabled_regression: bool | None
    extracted: bool
    extraction_status: Literal["go_block", "fenced_block", "largest_parseable", "failed"]
    parse_ok: bool
    lint_ok: bool
    original_tool_status: OriginalToolStatus
    output_tool_status: OutputToolStatus
    build_status: BuildStatus
    category: Literal["A", "B", "C", "D", "INVALID"]
    introduced_checks: tuple[CheckIdentifier, ...]
    residual_findings: tuple[FindingStatus, ...]


class CorrectionRecord(CommonRecord):
    task_type: Literal["correction"]
    outcome: RepairOutcome
    extracted_similarity: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    sensitivity_class: SensitivityClass | None = None


class JointRecord(CommonRecord):
    task_type: Literal["joint"]
    outcome: RepairOutcome
    extracted_similarity: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    sensitivity_class: SensitivityClass | None = None


class ExplanationRecord(CommonRecord):
    task_type: Literal["explanation"]


ReleasedRecord = Annotated[
    RuleIdentificationRecord | CorrectionRecord | JointRecord | ExplanationRecord,
    Field(discriminator="task_type"),
]


class RunManifest(StrictModel):
    historical_run_id: BoundedIdentifier
    condition: Condition
    seed: Literal[42, 43, 44]
    profile: Literal["paper"]
    dataset_revision: Literal["7b951fd57d19286153b46ba219aa2cb87fcc4d2b"]
    historical_source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    configuration_id: BoundedIdentifier


class SelectionPoint(StrictModel):
    step: int = Field(gt=0)
    validation_loss: FiniteFloat
    composite_score: FiniteFloat
    rule_id_macro_f1: FiniteFloat
    correction_fix_rate: FiniteFloat
    joint_fix_rate: FiniteFloat


class SelectionMetrics(StrictModel):
    selected_step: int | None = Field(default=None, gt=0)
    best_composite: FiniteFloat | None = None
    rule_id_macro_f1: FiniteFloat | None = None
    correction_fix_rate: FiniteFloat | None = None
    joint_fix_rate: FiniteFloat | None = None


class LengthMetrics(StrictModel):
    allowed_max_length: int | None = Field(default=None, gt=0)
    realized_truncation: int | None = Field(default=None, ge=0)


class ComputeMetrics(StrictModel):
    optimizer_steps: int | None = Field(default=None, ge=0)
    examples_seen: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    supervised_tokens: int | None = Field(default=None, ge=0)
    peak_allocated_gpu_memory_gib: FiniteFloat | None = Field(default=None, ge=0)
    wall_clock_train_s: FiniteFloat | None = Field(default=None, ge=0)
    wall_clock_total_s: FiniteFloat | None = Field(default=None, ge=0)


class HistoricalRunProvenance(StrictModel):
    dataset_revision: Literal["7b951fd57d19286153b46ba219aa2cb87fcc4d2b"]
    model_identifier: Literal["meta-llama/Llama-3.2-1B-Instruct"]
    data_fraction: Literal[1.0]
    historical_source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    run_kind: Literal["fine_tuned", "zero_shot"]


class NormalizedRunMetrics(StrictModel):
    checkpoint_selection: SelectionMetrics
    compute: ComputeMetrics
    length: LengthMetrics
    provenance: HistoricalRunProvenance


class RunResults(StrictModel):
    historical_run_id: BoundedIdentifier
    condition: Condition
    seed: Literal[42, 43, 44]
    profile: Literal["paper"]
    metrics: NormalizedRunMetrics

    @model_validator(mode="after")
    def validate_condition_specific_fields(self) -> RunResults:
        selection = tuple(self.metrics.checkpoint_selection.model_dump().values())
        compute = tuple(self.metrics.compute.model_dump().values())
        length = tuple(self.metrics.length.model_dump().values())
        if self.condition in {"zero-shot-raw", "zero-shot-syntax"}:
            if any(value is not None for value in selection + compute + length):
                raise ValueError("zero-shot results require null selection, compute, and length fields")
            if self.metrics.provenance.run_kind != "zero_shot":
                raise ValueError("zero-shot results require zero_shot provenance")
        elif any(value is None for value in selection + compute + length):
            raise ValueError("fine-tuned results require complete selection, compute, and length fields")
        elif self.metrics.provenance.run_kind != "fine_tuned":
            raise ValueError("fine-tuned results require fine_tuned provenance")
        return self


class GenerationMaxNewTokens(StrictModel):
    rule_identification: Literal[64]
    explanation: Literal[512]
    correction: Literal[512]
    joint: Literal[512]

    @field_validator("*", mode="before")
    @classmethod
    def require_integer_scalar(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("generation cap must use an integer scalar")
        return value


class ProfileConfig(StrictModel):
    max_steps: Literal[600]
    allowed_max_length: Literal[9305]
    micro_batch_size: Literal[2]
    grad_accum_steps: Literal[16]
    effective_batch_size: Literal[32]
    learning_rate: Literal[2e-5]
    betas: tuple[Literal[0.9], Literal[0.999]]
    epsilon: Literal[1e-8]
    weight_decay: Literal[0.1]
    warmup_ratio: Literal[0.1]
    minimum_learning_rate_ratio: Literal[0.1]
    maximum_gradient_norm: Literal[1.0]
    checkpoint_every_steps: Literal[120]
    require_full_composite: Literal[True]
    activation_checkpointing: Literal[False]
    generation_max_new_tokens: GenerationMaxNewTokens


class ConditionConfig(StrictModel):
    kind: Literal["fine_tuned", "zero_shot"]
    path: Literal["c0", "c1", "c2", "c2-control", "zero-shot-raw", "zero-shot-syntax"]
    seeds: tuple[Literal[42, 43, 44], ...]
    configuration_id: BoundedIdentifier
    use_summary: bool | None = None
    auxiliary_pool: Literal["syntax", "duplicated_main"] | None = None
    auxiliary_ratio: FiniteFloat | None = Field(default=None, ge=0, le=1)
    prompt_form: Literal["raw", "syntax"] | None = None


class DatasetIdentity(StrictModel):
    identifier: Literal["aholovko/go-critic-style"]
    doi: Literal["10.57967/hf/5304"]
    revision: Literal["7b951fd57d19286153b46ba219aa2cb87fcc4d2b"]


class ModelIdentity(StrictModel):
    identifier: Literal["meta-llama/Llama-3.2-1B-Instruct"]


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    dataset: DatasetIdentity
    model: ModelIdentity
    profiles: dict[Literal["paper"], ProfileConfig]
    conditions: dict[Condition, ConditionConfig]


class TrainingContribution(StrictModel):
    condition: FineTunedCondition
    pool: Literal["main", "syntax_auxiliary", "duplicated_main_control"]
    task_type: TaskType | Literal["syntax_summary"]
    prompt_tokens: int = Field(ge=0)
    response_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    multiplicity: int = Field(gt=0)


class OracleIndicators(StrictModel):
    status: Literal["reproduced", "excluded", "fixture_fix", "not_applicable"]
    missing_checks: tuple[CheckName, ...]
    extra_checks: tuple[CheckName, ...]


class SerializerIndicators(StrictModel):
    parse_ok: bool
    parse_strategy: Literal["file", "package", "func_body", "expr"] | None
    excluded_construct_count: int = Field(ge=0)
    maximum_depth: int = Field(ge=0)
    summary_status: StudySummaryStatus

    @model_validator(mode="after")
    def validate_parse_strategy(self) -> SerializerIndicators:
        if self.parse_ok != (self.parse_strategy is not None):
            raise ValueError("parse_ok requires a strategy and parse failure requires null")
        if (self.summary_status == "failed") != (not self.parse_ok):
            raise ValueError("failed summary status must match parse failure")
        return self


class PairIndicator(StrictModel):
    other_base_snippet_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    kind: Literal["aggressive_normalization", "near_duplicate"]
    threshold: FiniteFloat | None = Field(default=None, ge=0, le=1)
    score: FiniteFloat | None = Field(default=None, ge=0, le=1)
    matched: bool

    @model_validator(mode="after")
    def validate_pair_semantics(self) -> PairIndicator:
        if self.kind == "near_duplicate":
            if self.threshold is None or self.score is None or self.matched != (self.score >= self.threshold):
                raise ValueError("near_duplicate requires threshold, score, and derived matched")
        elif self.threshold is not None or self.score is not None or not self.matched:
            raise ValueError("aggressive_normalization requires null values and matched=true")
        return self


class DatasetQcIndicators(StrictModel):
    exact_collision: bool
    canonical_normalization_sampled: bool
    canonical_normalization_changed: bool
    pair_indicators: tuple[PairIndicator, ...]


class ReferenceQc(StrictModel):
    correction_status: Literal["accepted", "parse_fail", "target_not_fixed", "not_applicable"]
    explanation_present: bool
    primary_category_a: bool | None
    secondary_category_a: bool | None
    normalized_fixes_equal: bool | None
    selected_generator_role: Literal["primary", "secondary", "none"]
    generation_attempts: int = Field(ge=0)
    skip_mechanism: Literal["none", "configured_but_not_applied", "generation_exclusion"]
    generated_marker_retained: bool
    accepted_correction_byte_identical: bool | None
    build_status: BuildStatus


KnownLossCategory = Literal[
    "control_transfer_dropped",
    "loop_initialization_or_label_reduced",
    "embedded_field_or_tag_reduced",
    "string_whitespace_collapsed",
]


class SerializerAudit(StrictModel):
    parse_bucket: Literal["file"]
    length_bucket: Literal["len<50", "len<200", "len>=200"]
    depth_bucket: Literal["depth<=1", "depth<=3", "depth>3"]
    summary_line_count: int = Field(ge=0)
    faithful: bool | None
    relevant_omission: bool | None
    known_loss_categories: tuple[KnownLossCategory, ...]
    has_note: bool
    closure_scored: bool


class StudyRow(StrictModel):
    base_snippet_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split: Literal["train", "validation", "test"]
    task_types: tuple[TaskType, ...]
    target_checks: tuple[CheckName, ...]
    violation_count: int = Field(ge=0)
    repository_group_id: str = Field(pattern=r"^repository-group-[0-9]{4}$")
    length_excluded: bool
    quarantined: bool
    oracle: OracleIndicators
    source_line_count: int = Field(gt=0)
    serializer: SerializerIndicators
    dataset_qc: DatasetQcIndicators
    reference_qc: ReferenceQc
    training_contributions: tuple[TrainingContribution, ...]
    serializer_audit_stage1_pool_member: bool
    serializer_audit: SerializerAudit | None
    violation_in_closure: bool | None
    license_class: Literal["permissive", "no_detected_license"]


class TestStatus(StrictModel):
    status: Literal["passed"]
    evidence_class: Literal["recovered_current"]


class ArchitectureMetadata(StrictModel):
    vocabulary_size: Literal[128_256]
    context_length: Literal[131_072]
    embedding_dimension: Literal[2_048]
    query_heads: Literal[32]
    key_value_heads: Literal[8]
    query_heads_per_key_value_head: Literal[4]
    layers: Literal[16]
    feed_forward_dimension: Literal[8_192]
    rope_base: Literal[500_000.0]
    parameter_count: Literal[1_235_814_400]
    weight_tied: Literal[True]
    compute_dtype: Literal["bfloat16"]
    rmsnorm_dtype: Literal["float32"]


class ReferenceComparisonMetadata(StrictModel):
    prompt_count: Literal[215]
    scored_position_count: Literal[5666]
    tokenizer_exact: bool
    chat_template_match: bool
    chat_template_first_divergence: int | None
    generation_exact: bool
    generation_first_divergence: int | None
    next_token_agreement_fp32: FiniteFloat
    disagreements_fp32: int = Field(ge=0)
    systematic_disagreements_fp32: int = Field(ge=0)
    next_token_agreement_bf16: FiniteFloat
    disagreements_bf16: int = Field(ge=0)
    systematic_disagreements_bf16: int = Field(ge=0)
    margin_threshold: FiniteFloat
    near_tie_epsilon: FiniteFloat
    maximum_absolute_logit_difference: FiniteFloat
    mean_absolute_logit_difference: FiniteFloat
    null_forward_tolerance: FiniteFloat | None
    cached_generation_test: TestStatus
    sdpa_manual_test: TestStatus
    loss_masking_test: TestStatus


class TrainingPathMetadata(StrictModel):
    steps: int = Field(gt=0)
    examples: int = Field(gt=0)
    validation_examples: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: FiniteFloat
    mean_relative_loss_divergence: FiniteFloat
    maximum_relative_loss_divergence: FiniteFloat
    final_loss_scratch: FiniteFloat
    final_loss_reference: FiniteFloat
    validation_macro_f1_scratch: FiniteFloat
    validation_macro_f1_reference: FiniteFloat


class TokenAccountingRow(StrictModel):
    condition: FineTunedCondition
    seed: Literal[42, 43, 44]
    pool: Literal["main", "syntax_auxiliary", "duplicated_main_control"]
    slot_count: int = Field(ge=0)
    forwarded_tokens: int = Field(ge=0)
    supervised_tokens: int = Field(ge=0)
    evidence_class: Literal["historical_run", "recovered_current"]


class OperatorLogObservation(StrictModel):
    id: Literal["device_memory_c0_gb", "device_memory_syntax_gb", "provider_credits_total"]
    approximate_value: FiniteFloat = Field(ge=0)
    unit: Literal["GB", "provider_credit"]
    evidence_class: Literal["operator_log"]
    approximate: Literal[True]


class AnalysisMetadata(StrictModel):
    architecture: ArchitectureMetadata
    reference_comparison: ReferenceComparisonMetadata
    training_path: TrainingPathMetadata
    token_accounting: tuple[TokenAccountingRow, ...]
    operator_log_observations: tuple[OperatorLogObservation, ...]


class OutputTarget(StrictModel):
    kind: Literal["json", "csv"]
    file: Literal[
        "results.json",
        "table-8-1.csv",
        "table-8-2.csv",
        "table-8-3.csv",
        "table-8-4.csv",
        "table-8-5.csv",
        "table-8-6.csv",
        "table-8-7.csv",
        "table-8-8.csv",
        "table-8-9.csv",
    ]
    identifier: ResultIdentifier | None = None
    row: OutputCoordinate | None = None
    column: OutputCoordinate | None = None

    def as_key(self) -> tuple[str, str, str | None, str | None, str | None]:
        return (self.kind, self.file, self.identifier, self.row, self.column)


class InventoryEntry(StrictModel):
    id: ResultIdentifier
    manuscript_locations: tuple[SemanticLocation, ...]
    target: OutputTarget


class ManuscriptInventory(StrictModel):
    schema_version: Literal[1]
    results: tuple[InventoryEntry, ...]


@dataclass(frozen=True)
class RunInput:
    records: tuple[ReleasedRecord, ...]
    result: RunResults
    manifest: RunManifest
    selection_trace: tuple[SelectionPoint, ...]


@dataclass(frozen=True)
class RunInputs:
    records: tuple[ReleasedRecord, ...]
    results: tuple[RunResults, ...]
    manifests: tuple[RunManifest, ...]
    selection_traces: dict[tuple[Condition, int], tuple[SelectionPoint, ...]]


@dataclass(frozen=True)
class ReleaseInputs:
    config: ExperimentConfig
    records: tuple[ReleasedRecord, ...]
    scored_records: tuple[ReleasedRecord, ...]
    results: tuple[RunResults, ...]
    manifests: tuple[RunManifest, ...]
    selection_traces: dict[tuple[Condition, int], tuple[SelectionPoint, ...]]
    study_rows: tuple[StudyRow, ...]
    metadata: AnalysisMetadata
    inventory: ManuscriptInventory


class InputValidationError(ValueError):
    def __init__(
        self,
        path: Path,
        rule: str,
        *,
        record_id: str | None = None,
        field: str | None = None,
    ) -> None:
        label = f"{path}: {rule}"
        if record_id is not None:
            label = f"{path}: record {record_id}: {rule}"
        if field is not None:
            label = f"{label}: field {field}"
        super().__init__(label)


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


class _NonFiniteJsonError(ValueError):
    pass


class _DuplicateSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _DuplicateSafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping keys must be strings",
                key_node.start_mark,
            )
        if key in mapping:
            raise _DuplicateKeyError(key)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


_RECORD_ADAPTER = TypeAdapter(ReleasedRecord)
_RUN_RESULTS_ADAPTER = TypeAdapter(RunResults)
_RUN_MANIFEST_ADAPTER = TypeAdapter(RunManifest)
_SELECTION_TRACE_ADAPTER = TypeAdapter(tuple[SelectionPoint, ...])
_EXPERIMENT_CONFIG_ADAPTER = TypeAdapter(ExperimentConfig)
_STUDY_ROW_ADAPTER = TypeAdapter(StudyRow)
_ANALYSIS_METADATA_ADAPTER = TypeAdapter(AnalysisMetadata)
_INVENTORY_ADAPTER = TypeAdapter(ManuscriptInventory)


def expected_run_keys() -> tuple[tuple[Condition, int], ...]:
    return (
        ("C0", 42),
        ("C0", 43),
        ("C0", 44),
        ("C1", 42),
        ("C1", 43),
        ("C1", 44),
        ("C2", 42),
        ("C2", 43),
        ("C2", 44),
        ("C2-control", 42),
        ("C2-control", 43),
        ("C2-control", 44),
        ("zero-shot-raw", 42),
        ("zero-shot-syntax", 42),
    )


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise InputValidationError(path, "required input is missing")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise InputValidationError(path, "input is not valid UTF-8") from error
    except OSError as error:
        raise InputValidationError(path, f"cannot read input: {error.strerror}") from error


def _reject_json_constant(value: str) -> object:
    raise _NonFiniteJsonError(value)


def _scan_json(path: Path, text: str) -> object:
    duplicates: list[str] = []

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        mapping: dict[str, object] = {}
        for key, value in pairs:
            if key in mapping:
                duplicates.append(key)
            else:
                mapping[key] = value
        return mapping

    try:
        parsed = json.loads(text, object_pairs_hook=unique_pairs, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise InputValidationError(path, f"invalid JSON at line {error.lineno}, column {error.colno}") from error
    except _NonFiniteJsonError as error:
        raise InputValidationError(path, f"non-finite numeric value {error}") from error
    record_id = _mapping_record_id(parsed)
    if duplicates:
        raise InputValidationError(
            path,
            "duplicate key",
            record_id=record_id,
            field=duplicates[0],
        )
    return parsed


def _mapping_record_id(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    record_id = value.get("base_snippet_id")
    return record_id if isinstance(record_id, str) else None


def _field_from_error(error: ValidationError) -> tuple[str, str | None]:
    detail = error.errors(include_url=False)[0]
    rule = cast(str, detail["msg"])
    location = detail["loc"]
    field = ".".join(str(part) for part in location) if location else None
    return rule, field


def _validate_json[T](path: Path, text: str, adapter: TypeAdapter[T]) -> T:
    parsed = _scan_json(path, text)
    try:
        return adapter.validate_json(text)
    except ValidationError as error:
        rule, field = _field_from_error(error)
        raise InputValidationError(
            path,
            rule,
            record_id=_mapping_record_id(parsed),
            field=field,
        ) from error


def _load_yaml[T](path: Path, adapter: TypeAdapter[T]) -> T:
    text = _read_text(path)
    try:
        parsed = yaml.load(text, Loader=_DuplicateSafeLoader)
    except _DuplicateKeyError as error:
        raise InputValidationError(path, "duplicate key", field=error.key) from error
    except yaml.YAMLError as error:
        raise InputValidationError(path, f"invalid YAML: {error}") from error
    try:
        canonical_json = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise InputValidationError(path, "non-finite or non-JSON YAML value") from error
    try:
        return adapter.validate_json(canonical_json)
    except ValidationError as error:
        rule, field = _field_from_error(error)
        raise InputValidationError(path, rule, field=field) from error


def load_experiment_config(path: Path) -> ExperimentConfig:
    config = _load_yaml(path, _EXPERIMENT_CONFIG_ADAPTER)
    if tuple(config.profiles) != ("paper",):
        raise InputValidationError(path, "profiles must contain exactly the paper profile", field="profiles")
    expected_conditions = tuple(_CONDITION_ORDER)
    if tuple(config.conditions) != expected_conditions:
        raise InputValidationError(path, "condition matrix must contain the exact ordered six conditions")
    expected = {
        "C0": ("fine_tuned", "c0", (42, 43, 44), False, None, 0.0, None),
        "C1": ("fine_tuned", "c1", (42, 43, 44), True, None, 0.0, None),
        "C2": ("fine_tuned", "c2", (42, 43, 44), True, "syntax", 0.2, None),
        "C2-control": (
            "fine_tuned",
            "c2-control",
            (42, 43, 44),
            True,
            "duplicated_main",
            0.2,
            None,
        ),
        "zero-shot-raw": ("zero_shot", "zero-shot-raw", (42,), None, None, None, "raw"),
        "zero-shot-syntax": ("zero_shot", "zero-shot-syntax", (42,), None, None, None, "syntax"),
    }
    for condition, row in config.conditions.items():
        actual = (
            row.kind,
            row.path,
            row.seeds,
            row.use_summary,
            row.auxiliary_pool,
            row.auxiliary_ratio,
            row.prompt_form,
        )
        if actual != expected[condition]:
            if row.path != expected[condition][1]:
                raise InputValidationError(
                    path,
                    f"condition path mismatch for {condition}",
                    field=f"conditions.{condition}.path",
                )
            raise InputValidationError(path, f"condition configuration mismatch for {condition}")
    return config


def _record_sort_key(record: ReleasedRecord) -> tuple[int, int, str, int]:
    return (
        _CONDITION_ORDER[record.condition],
        record.seed,
        record.base_snippet_id,
        _TASK_ORDER[record.task_type],
    )


def load_records_file(path: Path) -> tuple[ReleasedRecord, ...]:
    text = _read_text(path)
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise InputValidationError(path, "JSONL must contain non-empty rows without blank lines")
    records = tuple(_validate_json(path, line, _RECORD_ADAPTER) for line in lines)
    return tuple(sorted(records, key=_record_sort_key))


def _require_exact_run_files(run_dir: Path) -> None:
    for name in sorted(_RUN_FILES):
        if not (run_dir / name).is_file():
            raise InputValidationError(run_dir / name, "required input is missing")
    actual = {entry.name for entry in run_dir.iterdir()}
    if actual != _RUN_FILES:
        raise InputValidationError(run_dir, "run files must be exactly the four released inputs")


def _validate_record_matrix(path: Path, records: tuple[ReleasedRecord, ...], condition: Condition, seed: int) -> None:
    if len(records) != 1_792:
        raise InputValidationError(path, "record matrix requires exactly 1,792 rows")
    cells: set[tuple[str, str]] = set()
    ids: set[str] = set()
    tasks_by_id: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if (record.condition, record.seed) != (condition, seed):
            raise InputValidationError(
                path,
                "record condition/seed mismatch",
                record_id=record.base_snippet_id,
            )
        cell = (record.base_snippet_id, record.task_type)
        if cell in cells:
            raise InputValidationError(path, "duplicate record matrix cell", record_id=record.base_snippet_id)
        cells.add(cell)
        ids.add(record.base_snippet_id)
        tasks_by_id[record.base_snippet_id].add(record.task_type)
    if len(ids) != 448 or any(tasks != set(TASK_TYPES) for tasks in tasks_by_id.values()):
        raise InputValidationError(path, "record matrix requires 448 IDs with the four exact task rows")


def _validate_selection_trace(path: Path, result: RunResults, trace: tuple[SelectionPoint, ...]) -> None:
    selection = result.metrics.checkpoint_selection
    if result.condition in {"zero-shot-raw", "zero-shot-syntax"}:
        if trace:
            raise InputValidationError(path, "zero-shot selection trace must be empty")
        return
    if tuple(point.step for point in trace) != (120, 240, 360, 480, 600):
        raise InputValidationError(path, "selection trace must contain exactly steps 120/240/360/480/600")
    selected = tuple(point for point in trace if point.step == selection.selected_step)
    if len(selected) != 1:
        raise InputValidationError(path, "selection trace must contain the selected step exactly once")
    point = selected[0]
    if (
        point.composite_score,
        point.rule_id_macro_f1,
        point.correction_fix_rate,
        point.joint_fix_rate,
    ) != (
        selection.best_composite,
        selection.rule_id_macro_f1,
        selection.correction_fix_rate,
        selection.joint_fix_rate,
    ):
        raise InputValidationError(path, "selection trace selected point disagrees with checkpoint selection")


def _validate_run_identity(
    path: Path,
    config: ExperimentConfig,
    condition: Condition,
    seed: int,
    result: RunResults,
    manifest: RunManifest,
) -> None:
    expected_commit = _ZERO_SHOT_COMMIT if condition.startswith("zero-shot") else _FINE_TUNED_COMMIT
    provenance = result.metrics.provenance
    identities = (
        result.historical_run_id == manifest.historical_run_id,
        result.condition == manifest.condition == condition,
        result.seed == manifest.seed == seed,
        result.profile == manifest.profile == "paper",
        provenance.dataset_revision == manifest.dataset_revision == config.dataset.revision,
        provenance.model_identifier == config.model.identifier,
        provenance.historical_source_commit == manifest.historical_source_commit == expected_commit,
        manifest.configuration_id == config.conditions[condition].configuration_id,
    )
    if not all(identities):
        raise InputValidationError(path, "result, manifest, configuration, and provenance identities must agree")


def load_run(root: Path, condition: Condition, seed: int) -> RunInput:
    config_path = root / "config/experiments.yaml"
    config = load_experiment_config(config_path)
    if (condition, seed) not in expected_run_keys():
        raise InputValidationError(root / "data/runs", f"unexpected run key {(condition, seed)!r}")
    run_dir = root / "data/runs" / config.conditions[condition].path / f"seed-{seed}"
    if not run_dir.is_dir():
        raise InputValidationError(run_dir, "required run directory is missing")
    _require_exact_run_files(run_dir)
    records_path = run_dir / "records.jsonl"
    results_path = run_dir / "results.yaml"
    manifest_path = run_dir / "manifest.yaml"
    trace_path = run_dir / "selection_trace.json"
    records = load_records_file(records_path)
    result = _load_yaml(results_path, _RUN_RESULTS_ADAPTER)
    manifest = _load_yaml(manifest_path, _RUN_MANIFEST_ADAPTER)
    trace = _validate_json(trace_path, _read_text(trace_path), _SELECTION_TRACE_ADAPTER)
    _validate_record_matrix(records_path, records, condition, seed)
    _validate_run_identity(run_dir, config, condition, seed, result, manifest)
    _validate_selection_trace(trace_path, result, trace)
    return RunInput(records=records, result=result, manifest=manifest, selection_trace=trace)


def _validate_run_directories(root: Path, config: ExperimentConfig) -> None:
    runs_root = root / "data/runs"
    if not runs_root.is_dir():
        raise InputValidationError(runs_root, "run matrix root is missing")
    expected_paths = {row.path for row in config.conditions.values()}
    actual_paths = {entry.name for entry in runs_root.iterdir() if entry.is_dir()}
    if actual_paths != expected_paths or any(not entry.is_dir() for entry in runs_root.iterdir()):
        raise InputValidationError(runs_root, "run matrix has missing or unexpected condition directories")
    for condition, row in config.conditions.items():
        condition_dir = runs_root / row.path
        expected_seeds = {f"seed-{seed}" for seed in row.seeds}
        actual_seeds = {entry.name for entry in condition_dir.iterdir() if entry.is_dir()}
        if actual_seeds != expected_seeds or any(not entry.is_dir() for entry in condition_dir.iterdir()):
            raise InputValidationError(condition_dir, f"run matrix has missing or unexpected seeds for {condition}")


def load_runs(root: Path) -> RunInputs:
    config = load_experiment_config(root / "config/experiments.yaml")
    _validate_run_directories(root, config)
    runs = tuple(load_run(root, condition, seed) for condition, seed in expected_run_keys())
    run_ids = {run.result.historical_run_id for run in runs}
    if len(run_ids) != len(runs):
        raise InputValidationError(root / "data/runs", "historical run identifiers must be unique")
    return RunInputs(
        records=tuple(record for run in runs for record in run.records),
        results=tuple(run.result for run in runs),
        manifests=tuple(run.manifest for run in runs),
        selection_traces={(run.result.condition, run.result.seed): run.selection_trace for run in runs},
    )


def load_study_rows(path: Path) -> tuple[StudyRow, ...]:
    text = _read_text(path)
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise InputValidationError(path, "JSONL must contain non-empty rows without blank lines")
    rows = tuple(_validate_json(path, line, _STUDY_ROW_ADAPTER) for line in lines)
    identifiers = [row.base_snippet_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise InputValidationError(path, "study rows must have unique snippet identifiers")
    return tuple(sorted(rows, key=lambda row: row.base_snippet_id))


def load_metadata(path: Path) -> AnalysisMetadata:
    return _load_yaml(path, _ANALYSIS_METADATA_ADAPTER)


def load_inventory(path: Path) -> ManuscriptInventory:
    inventory = _load_yaml(path, _INVENTORY_ADAPTER)
    identifiers = [entry.id for entry in inventory.results]
    targets = [entry.target.as_key() for entry in inventory.results]
    if len(identifiers) != len(set(identifiers)):
        raise InputValidationError(path, "inventory result identifiers must be unique")
    if len(targets) != len(set(targets)):
        raise InputValidationError(path, "inventory output targets must be unique")
    return inventory


def scored_test_ids(study_rows: tuple[StudyRow, ...]) -> frozenset[str]:
    test_rows = tuple(row for row in study_rows if row.split == "test")
    test_ids = {row.base_snippet_id for row in test_rows}
    if len(test_rows) != 448 or len(test_ids) != 448:
        raise InputValidationError(
            Path("data/study/analysis_inputs.jsonl"),
            "test study matrix requires 448 unique IDs",
        )
    excluded = {row.base_snippet_id for row in test_rows if row.oracle.status == "excluded"}
    quarantined = {row.base_snippet_id for row in test_rows if row.quarantined}
    if len(excluded) != 38:
        raise InputValidationError(Path("data/study/analysis_inputs.jsonl"), "test study matrix requires 38 exclusions")
    if quarantined:
        raise InputValidationError(
            Path("data/study/analysis_inputs.jsonl"),
            "test study matrix requires zero quarantines",
        )
    scored = frozenset(
        row.base_snippet_id for row in test_rows if not row.quarantined and row.oracle.status != "excluded"
    )
    if len(scored) != 410:
        raise InputValidationError(Path("data/study/analysis_inputs.jsonl"), "scored population requires 410 IDs")
    return scored


def select_scored_records(
    records: tuple[ReleasedRecord, ...],
    study_rows: tuple[StudyRow, ...],
) -> tuple[ReleasedRecord, ...]:
    scored_ids = scored_test_ids(study_rows)
    test_ids = {row.base_snippet_id for row in study_rows if row.split == "test"}
    expected_keys = set(expected_run_keys())
    records_by_run: dict[tuple[Condition, int], list[ReleasedRecord]] = defaultdict(list)
    cells: set[tuple[Condition, int, str, str]] = set()
    for record in records:
        run_key = (record.condition, record.seed)
        if run_key not in expected_keys:
            raise InputValidationError(Path("data/runs"), "record belongs to an unexpected run")
        cell = (*run_key, record.base_snippet_id, record.task_type)
        if cell in cells:
            raise InputValidationError(
                Path("data/runs"),
                "duplicate scored-population cell",
                record_id=record.base_snippet_id,
            )
        cells.add(cell)
        records_by_run[run_key].append(record)
    if set(records_by_run) != expected_keys:
        raise InputValidationError(Path("data/runs"), "scored-population run matrix is incomplete")
    for run_key in expected_run_keys():
        run_records = records_by_run[run_key]
        run_ids = {record.base_snippet_id for record in run_records}
        if run_ids != test_ids:
            raise InputValidationError(Path("data/runs"), f"run {run_key!r} test ID set differs from study test IDs")
        tasks_by_id: dict[str, set[str]] = defaultdict(set)
        for record in run_records:
            tasks_by_id[record.base_snippet_id].add(record.task_type)
        if len(run_records) != 1_792 or any(tasks != set(TASK_TYPES) for tasks in tasks_by_id.values()):
            raise InputValidationError(Path("data/runs"), f"run {run_key!r} has an incomplete task matrix")
    selected = tuple(
        sorted(
            (record for record in records if record.base_snippet_id in scored_ids),
            key=_record_sort_key,
        )
    )
    if len(selected) != 22_960:
        raise InputValidationError(Path("data/runs"), "selected record matrix requires exactly 22,960 rows")
    for run_key in expected_run_keys():
        count = sum((record.condition, record.seed) == run_key for record in selected)
        if count != 1_640:
            raise InputValidationError(Path("data/runs"), f"run {run_key!r} requires exactly 1,640 scored rows")
    return selected


def load_release_inputs(root: Path) -> ReleaseInputs:
    config = load_experiment_config(root / "config/experiments.yaml")
    runs = load_runs(root)
    study_rows = load_study_rows(root / "data/study/analysis_inputs.jsonl")
    metadata = load_metadata(root / "data/study/analysis_metadata.yaml")
    inventory = load_inventory(root / "config/manuscript_results.yaml")
    scored_records = select_scored_records(runs.records, study_rows)
    return ReleaseInputs(
        config=config,
        records=runs.records,
        scored_records=scored_records,
        results=runs.results,
        manifests=runs.manifests,
        selection_traces=runs.selection_traces,
        study_rows=study_rows,
        metadata=metadata,
        inventory=inventory,
    )
