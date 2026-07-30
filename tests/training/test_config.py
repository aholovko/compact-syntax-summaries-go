from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import go_ast_assistant.paper4.config as training_config
import go_ast_assistant.paper4.prepared_study as prepared_study_module
from go_ast_assistant.paper4.adjudication import Adjudication
from go_ast_assistant.paper4.preflight import RunRequest, validate_request
from go_ast_assistant.paper4.prepared_study import (
    PreparedSummaryRecord,
    evaluation_exclusion_ids,
    validate_prepared_study,
)
from go_ast_assistant.paper4.records import TaskExample


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_SPLIT_SIZES = {"train": 4, "validation": 2, "test": 2}
MAIN_TASKS = ("rule_identification", "correction", "joint", "explanation")
CONDITIONS = ("C0", "C1", "C2", "C2-control")


def _id(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _task_row(snippet_id: str, split: str, task_type: str, label: str) -> dict[str, Any]:
    return {
        "id": snippet_id,
        "split": split,
        "task_type": task_type,
        "target_checks": ["assignOp"],
        "code": f"package p\nfunc {label}() {{}}",
        "target": f"target:{label}:{task_type}",
        "meta": {"n_labels": 1, "source_revision": "synthetic-revision"},
    }


def _summary_row(
    snippet_id: str,
    label: str,
    *,
    ok: bool = True,
    nested: bool = False,
) -> dict[str, Any]:
    if not ok:
        return {
            "id": snippet_id,
            "ok": False,
            "parse_strategy": None,
            "type_facts_available": False,
            "lines": [],
            "excluded_constructs": [],
            "parse_error": "synthetic parse failure",
        }
    lines = [
        {
            "tier": 0,
            "depth": 0,
            "text": f"func {label}()",
            "segments": [{"a": "func "}, {"v": f"{label}()"}],
        }
    ]
    if nested:
        lines.append(
            {
                "tier": 1,
                "depth": 1,
                "text": "return: nil",
                "segments": [{"a": "return: "}, {"v": "nil"}],
            }
        )
    return {
        "id": snippet_id,
        "ok": True,
        "parse_strategy": "file",
        "type_facts_available": False,
        "lines": lines,
        "excluded_constructs": [],
        "parse_error": None,
    }


def _budget(exclusion_ids: list[str]) -> dict[str, Any]:
    tokens = {seed: {condition: 100 for condition in CONDITIONS} for seed in ("42", "43", "44")}
    guards = {
        seed: {
            condition: {"delta": 0.0, "exceeds": False, "guarded": condition in {"C2", "C2-control"}}
            for condition in CONDITIONS
        }
        for seed in ("42", "43", "44")
    }
    return {
        "allowed_max_length": 9305,
        "distributions": {"C0:rule_identification": {"p50": 1, "p90": 1, "p95": 1, "p99": 1, "max": 1, "n": 1}},
        "pre_exclusion_truncation": {
            "prompt_truncated": {},
            "response_truncated": {},
            "total": 0,
        },
        "tokens_by_seed": copy.deepcopy(tokens),
        "token_budget_guard_by_seed": copy.deepcopy(guards),
        "supervised_tokens_by_seed": copy.deepcopy(tokens),
        "supervised_token_budget_guard_by_seed": copy.deepcopy(guards),
        "data_fraction": 1.0,
        "aux_ratio": None,
        "max_steps": 600,
        "micro_batch_size": 2,
        "eff_batch": 32,
        "aux_stratification": "response",
        "budget_gate": {"total": "report_only", "supervised": "strict"},
        "exclusion_ids_sha256": hashlib.sha256("\n".join(sorted(set(exclusion_ids))).encode()).hexdigest(),
    }


def _prepared_tree(root: Path) -> dict[str, Any]:
    ids = {split: [_id(f"{split}-{index}") for index in range(size)] for split, size in SYNTHETIC_SPLIT_SIZES.items()}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    canonical: dict[tuple[str, str], dict[str, Any]] = {}
    for split, split_ids in ids.items():
        rows: list[dict[str, Any]] = []
        for index, snippet_id in enumerate(split_ids):
            tasks = MAIN_TASKS if split != "train" or index == 0 else ("explanation", "rule_identification")
            for task_type in reversed(tasks):
                row = _task_row(snippet_id, split, task_type, f"{split}_{index}")
                rows.append(row)
                canonical[(snippet_id, task_type)] = row
        rows_by_split[split] = rows
        _write_jsonl(root / "tasks" / f"{split}.jsonl", rows)

    all_ids = [snippet_id for split in ("train", "validation", "test") for snippet_id in ids[split]]
    labels = {
        snippet_id: f"{split}_{index}"
        for split in ("train", "validation", "test")
        for index, snippet_id in enumerate(ids[split])
    }
    summaries = [
        _summary_row(
            snippet_id,
            labels[snippet_id],
            ok=snippet_id != ids["validation"][1],
            nested=snippet_id == ids["train"][0],
        )
        for snippet_id in all_ids
    ]
    _write_jsonl(root / "summaries.jsonl", summaries)

    syntax_rows: list[dict[str, Any]] = []
    for index, snippet_id in enumerate(ids["train"]):
        syntax_source = canonical[(snippet_id, "rule_identification")]
        syntax_summary = summaries[index]
        syntax_rows.append(
            {
                "id": syntax_source["id"],
                "split": "train",
                "task_type": "syntax_summary",
                "target_checks": [],
                "code": syntax_source["code"],
                "target": "\n".join(f"{'  ' * line['depth']}{line['text']}" for line in syntax_summary["lines"]),
                "meta": {
                    "aux_role": "syntax",
                    "excluded_constructs": [],
                    "source_revision": "synthetic-revision",
                },
            }
        )
    _write_jsonl(root / "aux_pool_syntax.jsonl", syntax_rows)

    control_cells = [
        (ids["train"][0], "correction"),
        (ids["train"][1], "explanation"),
        (ids["train"][0], "correction"),
        (ids["train"][2], "rule_identification"),
    ]
    control_rows: list[dict[str, Any]] = []
    for cell in control_cells:
        row = copy.deepcopy(canonical[cell])
        row["meta"]["aux_role"] = "main_dup"
        control_rows.append(row)
    _write_jsonl(root / "aux_pool_main_dup.jsonl", control_rows)

    length_ids = [ids["train"][0]]
    (root / "length_exclusion_ids.txt").write_text(f"{length_ids[0]}\n", encoding="utf-8")
    (root / "composite_val_ids.txt").write_text(f"{ids['validation'][0]}\n", encoding="utf-8")
    _write_jsonl(
        root / "quarantine.jsonl",
        [{"id": ids["test"][0], "split": "test", "unused_diagnostic": {"ignored": True}}],
    )
    _write_jsonl(
        root / "oracle_adjudication.jsonl",
        [
            {
                "id": ids["test"][1],
                "split": "test",
                "resolution": "exclude",
                "reason": "synthetic test exclusion",
            },
            {
                "id": ids["validation"][1],
                "split": "validation",
                "resolution": "exclude",
                "reason": "synthetic validation exclusion",
            },
        ],
    )
    (root / "length_budget.json").write_text(json.dumps(_budget(length_ids), indent=2) + "\n", encoding="utf-8")
    return {
        "ids": ids,
        "rows": rows_by_split,
        "canonical": canonical,
        "summaries": summaries,
        "syntax_rows": syntax_rows,
        "control_rows": control_rows,
    }


def _validate(root: Path, condition: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(training_config, "EXPECTED_SPLIT_SIZES", SYNTHETIC_SPLIT_SIZES)
    monkeypatch.setattr(prepared_study_module, "EXPECTED_SPLIT_SIZES", SYNTHETIC_SPLIT_SIZES)
    return validate_prepared_study(root, condition)  # type: ignore[arg-type]


def test_run_request_defaults_resolve_the_bundle_config_from_outside_the_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = RunRequest(
        condition="C0",
        seed=42,
        study_data_dir=tmp_path / "missing-study-is-not-opened",
        model_dir=tmp_path / "missing-model-is-not-opened",
        output_dir=tmp_path / "new-output",
    )

    validated = validate_request(request)

    assert request.config_path == Path("config/experiments.yaml")
    assert request.device == "cuda"
    assert validated.condition == "C0"
    assert validated.seed == 42
    assert validated.profile == "paper"
    assert validated.device == "cuda"
    assert validated.config.model.identifier == "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.parametrize("condition", CONDITIONS)
def test_request_validation_accepts_each_fine_tuned_condition(tmp_path: Path, condition: str) -> None:
    request = RunRequest(
        condition=condition,
        seed=43,
        study_data_dir=tmp_path / "study",
        model_dir=tmp_path / "model",
        output_dir=tmp_path / condition,
        config_path=BUNDLE_ROOT / "config" / "experiments.yaml",
        device="cpu",
    )

    validated = validate_request(request)

    assert validated.condition == condition
    assert validated.seed == 43
    assert validated.device == "cpu"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("condition", "c0"),
        ("condition", "unknown"),
        ("seed", 41),
        ("seed", "42"),
        ("seed", 42.0),
        ("seed", True),
        ("device", "auto"),
        ("device", "gpu"),
        ("device", ""),
        ("device", None),
    ],
)
def test_request_validation_rejects_unknown_or_nonpaper_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    request = RunRequest(
        condition="C0",
        seed=42,
        study_data_dir=tmp_path / "study",
        model_dir=tmp_path / "model",
        output_dir=tmp_path / "output",
        config_path=BUNDLE_ROOT / "config" / "experiments.yaml",
    )

    with pytest.raises(ValueError):
        validate_request(replace(request, **{field: value}))


def test_request_validation_requires_an_absent_output_directory(tmp_path: Path) -> None:
    base = RunRequest(
        condition="C0",
        seed=42,
        study_data_dir=tmp_path / "study",
        model_dir=tmp_path / "model",
        output_dir=tmp_path / "absent",
        config_path=BUNDLE_ROOT / "config" / "experiments.yaml",
    )
    validate_request(base)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        validate_request(replace(base, output_dir=empty))

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "sentinel").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_request(replace(base, output_dir=nonempty))

    regular_file = tmp_path / "regular-file"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_request(replace(base, output_dir=regular_file))


def test_request_validation_rejects_a_dangling_output_symlink(tmp_path: Path) -> None:
    output = tmp_path / "dangling-output"
    try:
        output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")
    request = RunRequest(
        condition="C0",
        seed=42,
        study_data_dir=tmp_path / "study",
        model_dir=tmp_path / "model",
        output_dir=output,
        config_path=BUNDLE_ROOT / "config" / "experiments.yaml",
    )

    with pytest.raises(ValueError):
        validate_request(request)


def test_strict_records_forbid_extras_and_coercion_but_json_arrays_become_tuples() -> None:
    task_payload = _task_row(_id("strict-task"), "train", "correction", "strict_task")
    task = TaskExample.model_validate_json(json.dumps(task_payload))
    assert task.target_checks == ("assignOp",)

    with pytest.raises(ValidationError):
        TaskExample.model_validate({**task_payload, "unknown": True})
    with pytest.raises(ValidationError):
        TaskExample.model_validate({**task_payload, "target_checks": "assignOp"})

    summary_payload = _summary_row(_id("strict-summary"), "strict_summary")
    summary = PreparedSummaryRecord.model_validate_json(json.dumps(summary_payload))
    assert isinstance(summary.lines, tuple)
    assert isinstance(summary.lines[0].segments, tuple)

    coerced_ok = copy.deepcopy(summary_payload)
    coerced_ok["ok"] = "false"
    with pytest.raises(ValidationError):
        PreparedSummaryRecord.model_validate(coerced_ok)

    coerced_tier = copy.deepcopy(summary_payload)
    coerced_tier["lines"][0]["tier"] = "0"
    with pytest.raises(ValidationError):
        PreparedSummaryRecord.model_validate(coerced_tier)

    invalid_tier = copy.deepcopy(summary_payload)
    invalid_tier["lines"][0]["tier"] = 3
    with pytest.raises(ValidationError):
        PreparedSummaryRecord.model_validate(invalid_tier)

    with pytest.raises(ValidationError):
        PreparedSummaryRecord.model_validate({**summary_payload, "unknown": None})


@pytest.mark.parametrize("tier", [False, 0.0])
def test_summary_tier_requires_an_exact_json_integer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: object,
) -> None:
    _prepared_tree(tmp_path)
    path = tmp_path / "summaries.jsonl"
    rows = _read_jsonl(path)
    rows[0]["lines"][0]["tier"] = tier
    _write_jsonl(path, rows)

    with pytest.raises(ValueError):
        _validate(tmp_path, "C1", monkeypatch)


@pytest.mark.parametrize(
    "mutation",
    [
        {"resolution": "keep"},
        {"reason": "  "},
        {"split": "testing"},
        {"unknown": True},
    ],
)
def test_adjudications_are_strict(mutation: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "id": _id("adjudication"),
        "split": "test",
        "resolution": "exclude",
        "reason": "synthetic reason",
    }
    payload.update(mutation)
    with pytest.raises(ValidationError):
        Adjudication.model_validate(payload)


def test_prepared_study_preserves_task_and_c2_control_source_order_and_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)

    study = _validate(tmp_path, "C2-control", monkeypatch)

    expected_task_order = [(row["id"], row["task_type"]) for row in fixture["rows"]["train"]]
    assert [(row.id, row.task_type) for row in study.tasks_by_split["train"]] == expected_task_order
    expected_control_order = [(row["id"], row["task_type"]) for row in fixture["control_rows"]]
    assert [(row.id, row.task_type) for row in study.auxiliary_examples] == expected_control_order
    assert expected_control_order[0] == expected_control_order[2]
    assert len(study.auxiliary_examples) == SYNTHETIC_SPLIT_SIZES["train"]


def test_c0_tolerates_unconsumed_condition_specific_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepared_tree(tmp_path)

    study = _validate(tmp_path, "C0", monkeypatch)

    assert study.summaries is None
    assert study.auxiliary_examples == ()


@pytest.mark.parametrize(
    ("condition", "filename"),
    [
        ("C1", "summaries.jsonl"),
        ("C2", "aux_pool_syntax.jsonl"),
        ("C2-control", "aux_pool_main_dup.jsonl"),
    ],
)
def test_condition_specific_files_are_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    filename: str,
) -> None:
    _prepared_tree(tmp_path)
    _validate(tmp_path, condition, monkeypatch)
    (tmp_path / filename).unlink()

    with pytest.raises(ValueError):
        _validate(tmp_path, condition, monkeypatch)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate-cell",
        "cross-split-id",
        "missing-validation-task",
        "missing-test-task",
        "partial-train-extra-task",
        "syntax-main",
        "wrong-split",
    ],
)
def test_main_task_files_enforce_cell_identity_split_disjointness_and_exact_matrices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    fixture = _prepared_tree(tmp_path)
    split = "test" if case == "missing-test-task" else "validation"
    split = "train" if case == "partial-train-extra-task" else split
    path = tmp_path / "tasks" / f"{split}.jsonl"
    rows = _read_jsonl(path)
    if case == "duplicate-cell":
        rows.append(copy.deepcopy(rows[0]))
    elif case == "cross-split-id":
        for row in rows[:4]:
            row["id"] = fixture["ids"]["train"][0]
    elif case in {"missing-validation-task", "missing-test-task"}:
        rows.pop(0)
    elif case == "partial-train-extra-task":
        snippet_id = fixture["ids"]["train"][1]
        rows.append(_task_row(snippet_id, "train", "correction", "train_1"))
    elif case == "syntax-main":
        rows[0]["task_type"] = "syntax_summary"
    else:
        rows[0]["split"] = "test"
    _write_jsonl(path, rows)

    with pytest.raises(ValueError):
        _validate(tmp_path, "C0", monkeypatch)


@pytest.mark.parametrize(
    ("condition", "filename"),
    [
        ("C1", "summaries.jsonl"),
        ("C2", "aux_pool_syntax.jsonl"),
        ("C0", "oracle_adjudication.jsonl"),
        ("C0", "quarantine.jsonl"),
        ("C0", "length_exclusion_ids.txt"),
        ("C0", "composite_val_ids.txt"),
    ],
)
def test_prepared_files_reject_duplicate_record_or_plain_text_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    filename: str,
) -> None:
    _prepared_tree(tmp_path)
    path = tmp_path / filename
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _validate(tmp_path, condition, monkeypatch)


def test_jsonl_loaders_reject_duplicate_object_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    path = tmp_path / "tasks" / "train.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0][:-1] + f', "id": "{fixture["ids"]["train"][0]}"}}'
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _validate(tmp_path, "C0", monkeypatch)


@pytest.mark.parametrize("case", ["length", "composite", "quarantine", "adjudication"])
def test_prepared_cross_references_must_match_the_declared_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    fixture = _prepared_tree(tmp_path)
    ids = fixture["ids"]
    if case == "length":
        (tmp_path / "length_exclusion_ids.txt").write_text(f"{ids['test'][0]}\n", encoding="utf-8")
    elif case == "composite":
        (tmp_path / "composite_val_ids.txt").write_text(f"{ids['train'][0]}\n", encoding="utf-8")
    elif case == "quarantine":
        _write_jsonl(tmp_path / "quarantine.jsonl", [{"id": ids["test"][0], "split": "validation"}])
    else:
        rows = _read_jsonl(tmp_path / "oracle_adjudication.jsonl")
        rows[0]["split"] = "validation"
        _write_jsonl(tmp_path / "oracle_adjudication.jsonl", rows)

    with pytest.raises(ValueError):
        _validate(tmp_path, "C0", monkeypatch)


def test_summary_conditions_require_unique_coverage_for_every_loaded_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepared_tree(tmp_path)
    path = tmp_path / "summaries.jsonl"
    rows = _read_jsonl(path)
    _write_jsonl(path, rows[:-1])

    with pytest.raises(ValueError):
        _validate(tmp_path, "C1", monkeypatch)


@pytest.mark.parametrize(
    "case",
    ["wrong-code", "wrong-target", "target-check", "failed-summary", "nontrain-id", "missing-row"],
)
def test_c2_syntax_rows_must_match_successful_train_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    fixture = _prepared_tree(tmp_path)
    aux_path = tmp_path / "aux_pool_syntax.jsonl"
    aux_rows = _read_jsonl(aux_path)
    aux = aux_rows[0]
    if case == "wrong-code":
        aux["code"] = "package different"
    elif case == "wrong-target":
        assert aux["target"] == "func train_0()\n  return: nil"
        aux["target"] = "func train_0()"
    elif case == "target-check":
        aux["target_checks"] = ["assignOp"]
    elif case == "nontrain-id":
        aux["id"] = fixture["ids"]["validation"][0]
    elif case == "missing-row":
        aux_rows.pop()
    else:
        summaries = _read_jsonl(tmp_path / "summaries.jsonl")
        summaries[0] = _summary_row(aux["id"], "train_0", ok=False)
        _write_jsonl(tmp_path / "summaries.jsonl", summaries)
    _write_jsonl(aux_path, aux_rows)

    with pytest.raises(ValueError):
        _validate(tmp_path, "C2", monkeypatch)


@pytest.mark.parametrize("case", ["mismatch", "missing-row", "extra-row"])
def test_c2_control_rows_must_equal_the_fixed_main_row_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    _prepared_tree(tmp_path)
    path = tmp_path / "aux_pool_main_dup.jsonl"
    rows = _read_jsonl(path)
    if case == "mismatch":
        rows[1]["target"] = "not the canonical target"
    elif case == "missing-row":
        rows.pop()
    else:
        rows.append(copy.deepcopy(rows[0]))
    _write_jsonl(path, rows)

    with pytest.raises(ValueError):
        _validate(tmp_path, "C2-control", monkeypatch)


@pytest.mark.parametrize(
    "case",
    ["coerced-scalar", "missing-seed", "missing-condition", "wrong-hash", "unknown-field", "nonfinite"],
)
def test_length_budget_is_strict_and_has_exact_seed_condition_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    _prepared_tree(tmp_path)
    path = tmp_path / "length_budget.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if case == "coerced-scalar":
        payload["allowed_max_length"] = "9305"
    elif case == "missing-seed":
        payload["tokens_by_seed"].pop("44")
    elif case == "missing-condition":
        payload["supervised_tokens_by_seed"]["42"].pop("C2-control")
    elif case == "wrong-hash":
        payload["exclusion_ids_sha256"] = "0" * 64
    elif case == "unknown-field":
        payload["unknown"] = True
    else:
        payload["token_budget_guard_by_seed"]["42"]["C0"]["delta"] = float("nan")
    path.write_text(json.dumps(payload, allow_nan=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _validate(tmp_path, "C0", monkeypatch)


def test_length_budget_seed_and_condition_object_order_is_not_semantic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepared_tree(tmp_path)
    path = tmp_path / "length_budget.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix_names = (
        "tokens_by_seed",
        "token_budget_guard_by_seed",
        "supervised_tokens_by_seed",
        "supervised_token_budget_guard_by_seed",
    )
    for matrix_name in matrix_names:
        matrix = payload[matrix_name]
        payload[matrix_name] = {
            seed: {condition: matrix[seed][condition] for condition in reversed(tuple(matrix[seed]))}
            for seed in reversed(tuple(matrix))
        }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    study = _validate(tmp_path, "C0", monkeypatch)

    assert study.length_budget.tokens_by_seed["42"]["C0"] == 100


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allowed_max_length", 9305.0),
        ("max_steps", 600.0),
        ("micro_batch_size", 2.0),
        ("eff_batch", 32.0),
        ("data_fraction", 1),
        ("data_fraction", True),
    ],
)
def test_length_budget_fixed_numeric_literals_require_exact_json_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    _prepared_tree(tmp_path)
    path = tmp_path / "length_budget.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _validate(tmp_path, "C0", monkeypatch)


def test_length_budget_rejects_duplicate_json_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepared_tree(tmp_path)
    path = tmp_path / "length_budget.json"
    text = path.read_text(encoding="utf-8")
    duplicated = text.replace(
        '"allowed_max_length": 9305,',
        '"allowed_max_length": 9305,\n  "allowed_max_length": 9305,',
    )
    path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(ValueError):
        _validate(tmp_path, "C0", monkeypatch)


def test_evaluation_exclusions_use_only_test_quarantine_and_test_exclude_adjudications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)

    study = _validate(tmp_path, "C0", monkeypatch)

    assert evaluation_exclusion_ids(study) == frozenset(fixture["ids"]["test"])
    assert fixture["ids"]["train"][0] not in evaluation_exclusion_ids(study)
    assert fixture["ids"]["validation"][1] not in evaluation_exclusion_ids(study)
