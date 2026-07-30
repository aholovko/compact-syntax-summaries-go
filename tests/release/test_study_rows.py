from collections import Counter
from pathlib import Path

import pytest

from analysis.inputs import (
    expected_run_keys,
    load_runs,
    load_study_rows,
    scored_test_ids,
    select_scored_records,
)

ROOT = Path(__file__).resolve().parents[2]


def test_compact_study_counts() -> None:
    rows = load_study_rows(ROOT / "data/study/analysis_inputs.jsonl")
    assert len(rows) == 2_206
    assert Counter(row.split for row in rows) == {"train": 1_536, "validation": 222, "test": 448}
    assert Counter(row.license_class for row in rows) == {
        "permissive": 1_079,
        "no_detected_license": 1_127,
    }
    assert len({row.repository_group_id for row in rows}) == 2_131
    audit = [row for row in rows if row.serializer_audit is not None]
    assert len(audit) == 65
    assert sum(row.serializer_audit_stage1_pool_member for row in rows) == 175
    assert sum(row.dataset_qc.canonical_normalization_sampled for row in rows) == 260
    assert sum(bool(row.serializer_audit.has_note) for row in audit) == 35
    assert sum(bool(row.serializer_audit.known_loss_categories) for row in audit) == 23
    assert sum(row.violation_in_closure is not None for row in audit) == 64
    assert sum(row.violation_in_closure is True for row in audit) == 2

    accepted = [row for row in rows if row.reference_qc.correction_status == "accepted"]
    assert Counter(row.reference_qc.normalized_fixes_equal for row in accepted) == {
        True: 1_185,
        False: 344,
        None: 4,
    }
    train = [row for row in rows if row.split == "train"]
    assert Counter(row.reference_qc.normalized_fixes_equal for row in train) == {
        True: 1_185,
        False: 344,
        None: 7,
    }
    assert (
        sum(
            row.reference_qc.generated_marker_retained
            and row.reference_qc.skip_mechanism == "configured_but_not_applied"
            for row in rows
        )
        == 51
    )

    owned_pairs = [(row.base_snippet_id, pair) for row in rows for pair in row.dataset_qc.pair_indicators]
    assert len(owned_pairs) == 20
    assert all(pair.kind == "near_duplicate" for _, pair in owned_pairs)
    assert all(owner < pair.other_base_snippet_id for owner, pair in owned_pairs)
    assert all(pair.threshold == 0.7 and pair.score is not None for _, pair in owned_pairs)
    assert all(pair.matched is False for _, pair in owned_pairs)
    assert len({(owner, pair.other_base_snippet_id) for owner, pair in owned_pairs}) == 20


def test_repository_overlap_contract() -> None:
    rows = load_study_rows(ROOT / "data/study/analysis_inputs.jsonl")
    train_groups = {row.repository_group_id for row in rows if row.split == "train"}
    test_rows = [row for row in rows if row.split == "test"]
    scored_ids = scored_test_ids(rows)
    scored = [row for row in test_rows if row.base_snippet_id in scored_ids]

    assert Counter(row.split for row in rows if row.length_excluded) == {"train": 47}
    assert sum(row.quarantined for row in test_rows) == 0
    assert Counter(row.split for row in rows if row.oracle.status == "excluded") == {
        "validation": 14,
        "test": 38,
    }
    assert len(scored_ids) == len(scored) == 410
    test_groups = {row.repository_group_id for row in test_rows}
    assert len(train_groups & test_groups) == 15
    assert sum(row.repository_group_id in train_groups for row in scored) == 14
    assert len({row.repository_group_id for row in scored}) == 407
    assert Counter(Counter(row.repository_group_id for row in scored).values()) == {1: 404, 2: 3}


def test_scored_record_matrix_is_exact() -> None:
    rows = load_study_rows(ROOT / "data/study/analysis_inputs.jsonl")
    runs = load_runs(ROOT)
    test_ids = {row.base_snippet_id for row in rows if row.split == "test"}
    for condition, seed in expected_run_keys():
        run_ids = {
            record.base_snippet_id for record in runs.records if (record.condition, record.seed) == (condition, seed)
        }
        assert run_ids == test_ids

    scored = select_scored_records(runs.records, rows)
    assert len(scored) == 22_960
    by_cell = {}
    for record in scored:
        key = (record.condition, record.seed, record.task_type)
        by_cell.setdefault(key, set()).add(record.base_snippet_id)
    assert len(by_cell) == 56
    assert {len(ids) for ids in by_cell.values()} == {410}
    assert len({frozenset(ids) for ids in by_cell.values()}) == 1
    c0_seed42 = [
        record
        for record in scored
        if record.condition == "C0" and record.seed == 42 and record.task_type == "rule_identification"
    ]
    assert sum(record.exact_match for record in c0_seed42) / len(c0_seed42) == pytest.approx(274 / 410)
