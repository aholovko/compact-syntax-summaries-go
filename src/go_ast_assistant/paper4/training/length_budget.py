from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from go_ast_assistant.paper4.config import FineTunedCondition
from go_ast_assistant.paper4.prepared_study import BudgetGuard, PreparedStudy
from go_ast_assistant.paper4.training.budget import token_budget_guard
from go_ast_assistant.paper4.training.conditions import AUX_ENCODING, CONDITIONS, NullSummaryStore
from go_ast_assistant.paper4.training.instruction_dataset import encode_example
from go_ast_assistant.paper4.training.sampling import build_mixture_stream, length_stratified_aux_sample
from go_ast_assistant.paper4.training.summary_store import SerializedSummaryStore

if TYPE_CHECKING:
    from go_ast_assistant.paper4.runtime.tokenizer import ChatFormat


@dataclass(frozen=True)
class LengthRecord:
    task: str
    prompt_len: int
    total_len: int


def percentile(values: list[int], p: float) -> int:
    """Return the deterministic integer nearest-rank percentile."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[rank - 1]


def _dist(values: list[int]) -> dict[str, int]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else 0,
        "n": len(values),
    }


def forwarded_token_count(lengths: list[int], micro_batch_size: int, allowed: int) -> int:
    """Mirror shifted, capped, per-micro-batch token accounting."""
    total = 0
    for offset in range(0, len(lengths), micro_batch_size):
        batch = lengths[offset : offset + micro_batch_size]
        if batch:
            forwarded_width = min(max(batch), allowed) - 1
            total += sum(min(length, forwarded_width) for length in batch)
    return total


def supervised_token_count(
    rows: list[tuple[int, int]],
    micro_batch_size: int,
    allowed: int,
) -> int:
    """Mirror the response-only target mask after per-micro-batch capping."""
    total = 0
    for offset in range(0, len(rows), micro_batch_size):
        batch = rows[offset : offset + micro_batch_size]
        if batch:
            width = min(max(total_len for total_len, _ in batch), allowed)
            total += sum(max(0, min(total_len, width) - prompt_len) for total_len, prompt_len in batch)
    return total


def _condition_entries(values: Mapping[str, object], condition: str) -> dict[str, object]:
    return {key: value for key, value in values.items() if key.partition(":")[0] == condition}


def _guard_matches(stored: BudgetGuard, expected: Mapping[str, float | bool]) -> bool:
    return (
        math.isclose(stored.delta, float(expected["delta"]), rel_tol=1e-12, abs_tol=1e-12)
        and stored.exceeds is expected["exceeds"]
        and stored.guarded is expected["guarded"]
    )


def _requested_guard(
    matrix: Mapping[str, int],
    condition: FineTunedCondition,
    realized: int,
) -> dict[str, float | bool]:
    values = {"C1": realized if condition == "C1" else matrix["C1"]}
    values[condition] = realized
    return token_budget_guard(values)[condition]


def validate_lengths(
    study: PreparedStudy,
    tokenizer: ChatFormat,
    condition: FineTunedCondition,
    seed: Literal[42, 43, 44],
    profile: Literal["paper"],
) -> None:
    """Recompute and validate only one requested paper condition and seed slice."""
    if profile != "paper":
        raise ValueError(f"unsupported training profile: {profile!r}")
    condition_spec = CONDITIONS[condition]
    if condition_spec.use_summary:
        if study.summaries is None:
            raise ValueError(f"condition {condition} requires injected prepared summaries")
        store = SerializedSummaryStore(
            study.summaries,
            lambda text: len(tokenizer.tok.encode(text)),
        )
    else:
        store = NullSummaryStore()

    main_examples = study.tasks_by_split["train"]
    auxiliary_examples = study.auxiliary_examples
    main_items = tuple(encode_example(example, tokenizer, condition_spec, store) for example in main_examples)
    auxiliary_condition = AUX_ENCODING if condition_spec.aux == "syntax" else condition_spec
    auxiliary_items = tuple(
        encode_example(example, tokenizer, auxiliary_condition, store) for example in auxiliary_examples
    )

    records: list[LengthRecord] = []
    for examples, items in ((main_examples, main_items), (auxiliary_examples, auxiliary_items)):
        for example, item in zip(examples, items):
            total_len = len(item["input_ids"])
            records.append(
                LengthRecord(
                    example.task_type,
                    item["prompt_len"],
                    total_len,
                )
            )

    expected_distributions: dict[str, dict[str, int]] = {}
    totals_by_key: dict[str, list[int]] = defaultdict(list)
    for record in records:
        totals_by_key[f"{condition}:{record.task}"].append(record.total_len)
    for key, totals in totals_by_key.items():
        expected_distributions[key] = _dist(totals)
    stored_distributions = _condition_entries(study.length_budget.distributions, condition)
    if set(stored_distributions) != set(expected_distributions) or any(
        stored_distributions[key].model_dump(mode="python") != expected
        for key, expected in expected_distributions.items()
    ):
        raise ValueError(f"length distributions do not match requested condition {condition}")

    allowed = study.length_budget.allowed_max_length
    expected_prompt_truncation: dict[str, int] = defaultdict(int)
    expected_response_truncation: dict[str, int] = defaultdict(int)
    for record in records:
        if record.total_len > allowed:
            key = f"{condition}:{record.task}"
            target = expected_prompt_truncation if record.prompt_len > allowed else expected_response_truncation
            target[key] += 1
    stored_truncation = study.length_budget.pre_exclusion_truncation
    if _condition_entries(stored_truncation.prompt_truncated, condition) != dict(
        expected_prompt_truncation
    ) or _condition_entries(stored_truncation.response_truncated, condition) != dict(expected_response_truncation):
        raise ValueError(f"condition-specific truncation does not match requested condition {condition}")

    excluded = study.length_exclusion_ids
    retained_main = tuple(
        (example, item) for example, item in zip(main_examples, main_items) if example.id not in excluded
    )
    retained_auxiliary = tuple(
        (example, item) for example, item in zip(auxiliary_examples, auxiliary_items) if example.id not in excluded
    )
    if not retained_main:
        raise ValueError(f"condition {condition} has no main examples after global exclusions")
    if condition_spec.aux is not None and not retained_auxiliary:
        raise ValueError(f"condition {condition} has no auxiliary examples after global exclusions")

    main_response_lengths = tuple(len(item["input_ids"]) - item["prompt_len"] for _, item in retained_main)
    auxiliary_response_lengths = tuple(len(item["input_ids"]) - item["prompt_len"] for _, item in retained_auxiliary)
    stream = build_mixture_stream(
        len(retained_main),
        len(retained_auxiliary),
        condition_spec.aux_ratio,
        study.length_budget.max_steps * study.length_budget.eff_batch,
        seed,
    )
    if condition_spec.aux == "syntax":
        auxiliary_count = sum(pool == "aux" for pool, _ in stream)
        selected_auxiliary = iter(
            length_stratified_aux_sample(
                auxiliary_response_lengths,
                main_response_lengths,
                auxiliary_count,
                seed,
            )
        )
        stream = [("aux", next(selected_auxiliary)) if pool == "aux" else ("main", index) for pool, index in stream]

    lengths: list[int] = []
    supervised_rows: list[tuple[int, int]] = []
    for pool, index in stream:
        item = retained_auxiliary[index][1] if pool == "aux" else retained_main[index][1]
        total_len = len(item["input_ids"])
        lengths.append(total_len)
        supervised_rows.append((total_len, item["prompt_len"]))
    realized_tokens = forwarded_token_count(
        lengths,
        study.length_budget.micro_batch_size,
        allowed,
    )
    realized_supervised = supervised_token_count(
        supervised_rows,
        study.length_budget.micro_batch_size,
        allowed,
    )

    seed_key = str(seed)
    token_matrix = study.length_budget.tokens_by_seed[seed_key]
    supervised_matrix = study.length_budget.supervised_tokens_by_seed[seed_key]
    if token_matrix[condition] != realized_tokens:
        raise ValueError(f"total tokens do not match requested condition {condition} seed {seed}")
    if supervised_matrix[condition] != realized_supervised:
        raise ValueError(f"supervised tokens do not match requested condition {condition} seed {seed}")

    expected_total_guard = _requested_guard(token_matrix, condition, realized_tokens)
    stored_total_guard = study.length_budget.token_budget_guard_by_seed[seed_key][condition]
    if not _guard_matches(stored_total_guard, expected_total_guard):
        raise ValueError(f"total token guard does not match requested condition {condition} seed {seed}")
    expected_supervised_guard = _requested_guard(
        supervised_matrix,
        condition,
        realized_supervised,
    )
    stored_supervised_guard = study.length_budget.supervised_token_budget_guard_by_seed[seed_key][condition]
    if not _guard_matches(stored_supervised_guard, expected_supervised_guard):
        raise ValueError(f"supervised token guard does not match requested condition {condition} seed {seed}")
    if expected_supervised_guard["exceeds"] is True:
        raise ValueError(f"supervised token budget exceeds strict gate for condition {condition} seed {seed}")
