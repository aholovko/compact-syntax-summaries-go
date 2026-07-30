from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from analysis.inputs import ReleaseInputs
    from analysis.tables import GeneratedOutputs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONDITIONS = (
    ("C0", "c0", (42, 43, 44), "paper-c0-v1"),
    ("C1", "c1", (42, 43, 44), "paper-c1-v1"),
    ("C2", "c2", (42, 43, 44), "paper-c2-v1"),
    ("C2-control", "c2-control", (42, 43, 44), "paper-c2-control-v1"),
    ("zero-shot-raw", "zero-shot-raw", (42,), "paper-zero-shot-raw-v1"),
    ("zero-shot-syntax", "zero-shot-syntax", (42,), "paper-zero-shot-syntax-v1"),
)
FINE_TUNED_COMMIT = "16aadb26296b291538de481265a149dcb6db8876"
ZERO_SHOT_COMMIT = "520d5ce6c49864405c5946cae57ec794b00e4218"


@dataclass(frozen=True)
class MiniCase:
    root: Path
    release: ReleaseInputs
    outputs: GeneratedOutputs
    expected_files: tuple[tuple[str, bytes], ...]


_MINI_EXPECTED_FILES = (
    (
        "results.json",
        b'{\n  "dataset.base_snippets.total": 2,\n  "rq1.rule.delta": 0.125\n}\n',
    ),
    ("tables/table-8-1.csv", b"label,value\nRow 1,0.0\n"),
    ("tables/table-8-2.csv", b"label,value\nRow 2,2.25\n"),
    ("tables/table-8-3.csv", b"label,value\nRow 3,3\n"),
    ("tables/table-8-4.csv", b"label,value\nRow 4,4.125\n"),
    ("tables/table-8-5.csv", b"label,value\nRow 5,5.0\n"),
    ("tables/table-8-6.csv", b"label,value\nRow 6,6\n"),
    ("tables/table-8-7.csv", b"label,value\nRow 7,7.75\n"),
    ("tables/table-8-8.csv", b"label,value\nRow 8,8.5\n"),
    ("tables/table-8-9.csv", b"label,value\nRow 9,9\n"),
)


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": {
            "identifier": "aholovko/go-critic-style",
            "doi": "10.57967/hf/5304",
            "revision": "7b951fd57d19286153b46ba219aa2cb87fcc4d2b",
        },
        "model": {"identifier": "meta-llama/Llama-3.2-1B-Instruct"},
        "profiles": {
            "paper": {
                "max_steps": 600,
                "allowed_max_length": 9305,
                "micro_batch_size": 2,
                "grad_accum_steps": 16,
                "effective_batch_size": 32,
                "learning_rate": 2e-5,
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": 0.1,
                "warmup_ratio": 0.1,
                "minimum_learning_rate_ratio": 0.1,
                "maximum_gradient_norm": 1.0,
                "checkpoint_every_steps": 120,
                "require_full_composite": True,
                "activation_checkpointing": False,
                "generation_max_new_tokens": {
                    "rule_identification": 64,
                    "explanation": 512,
                    "correction": 512,
                    "joint": 512,
                },
            }
        },
        "conditions": {
            name: _condition_config(name, path, seeds, configuration_id)
            for name, path, seeds, configuration_id in CONDITIONS
        },
    }


def _condition_config(
    name: str,
    path: str,
    seeds: tuple[int, ...],
    configuration_id: str,
) -> dict[str, object]:
    base: dict[str, object] = {
        "kind": "zero_shot" if name.startswith("zero-shot") else "fine_tuned",
        "path": path,
        "seeds": list(seeds),
        "configuration_id": configuration_id,
    }
    if name == "zero-shot-raw":
        return base | {"prompt_form": "raw"}
    if name == "zero-shot-syntax":
        return base | {"prompt_form": "syntax"}
    auxiliary_pool = {"C2": "syntax", "C2-control": "duplicated_main"}.get(name)
    return base | {
        "use_summary": name != "C0",
        "auxiliary_pool": auxiliary_pool,
        "auxiliary_ratio": 0.2 if auxiliary_pool is not None else 0.0,
    }


def _outcome() -> dict[str, object]:
    return {
        "target_fixed": True,
        "overall_fixed": True,
        "studied_regression": False,
        "enabled_regression": False,
        "extracted": True,
        "extraction_status": "go_block",
        "parse_ok": True,
        "lint_ok": True,
        "original_tool_status": "ok",
        "output_tool_status": "ok",
        "build_status": "OK",
        "category": "A",
        "introduced_checks": [],
        "residual_findings": [],
    }


def _records(condition: str, seed: int) -> Iterator[dict[str, object]]:
    for index in range(448):
        snippet_id = f"sha256:{index:064x}"
        common: dict[str, object] = {
            "base_snippet_id": snippet_id,
            "condition": condition,
            "seed": seed,
            "target_checks": ["assignOp"],
            "summary_status": "not_applicable",
            "prompt_tokens": 10,
            "retokenized_response_token_proxy": 3,
            "latency_ms": 1.0,
        }
        yield common | {
            "task_type": "rule_identification",
            "gold": ["assignOp"],
            "pred": ["assignOp"],
            "rejected_label_count": 0,
            "exact_match": True,
            "n_emitted": 1,
            "normalization_status": "recognized_array",
        }
        for task_type in ("correction", "joint"):
            yield common | {
                "task_type": task_type,
                "outcome": _outcome(),
                "extracted_similarity": 1.0,
                "sensitivity_class": None,
            }
        yield common | {"task_type": "explanation"}


def _selection_metrics(fine_tuned: bool) -> dict[str, object]:
    if not fine_tuned:
        return {
            "selected_step": None,
            "best_composite": None,
            "rule_id_macro_f1": None,
            "correction_fix_rate": None,
            "joint_fix_rate": None,
        }
    return {
        "selected_step": 120,
        "best_composite": 0.7,
        "rule_id_macro_f1": 0.8,
        "correction_fix_rate": 0.7,
        "joint_fix_rate": 0.6,
    }


def _results(condition: str, seed: int, run_id: str, fine_tuned: bool) -> dict[str, object]:
    commit = FINE_TUNED_COMMIT if fine_tuned else ZERO_SHOT_COMMIT
    compute_values = (1, 32, 100, 80, 1.0, 2.0, 3.0) if fine_tuned else (None,) * 7
    length_values = (9305, 0) if fine_tuned else (None, None)
    return {
        "historical_run_id": run_id,
        "condition": condition,
        "seed": seed,
        "profile": "paper",
        "metrics": {
            "checkpoint_selection": _selection_metrics(fine_tuned),
            "compute": dict(
                zip(
                    (
                        "optimizer_steps",
                        "examples_seen",
                        "total_tokens",
                        "supervised_tokens",
                        "peak_allocated_gpu_memory_gib",
                        "wall_clock_train_s",
                        "wall_clock_total_s",
                    ),
                    compute_values,
                    strict=True,
                )
            ),
            "length": dict(zip(("allowed_max_length", "realized_truncation"), length_values, strict=True)),
            "provenance": {
                "dataset_revision": "7b951fd57d19286153b46ba219aa2cb87fcc4d2b",
                "model_identifier": "meta-llama/Llama-3.2-1B-Instruct",
                "data_fraction": 1.0,
                "historical_source_commit": commit,
                "run_kind": "fine_tuned" if fine_tuned else "zero_shot",
            },
        },
    }


def _trace(fine_tuned: bool) -> list[dict[str, object]]:
    if not fine_tuned:
        return []
    return [
        {
            "step": step,
            "validation_loss": 1.0,
            "composite_score": 0.7 if step == 120 else 0.5,
            "rule_id_macro_f1": 0.8 if step == 120 else 0.5,
            "correction_fix_rate": 0.7 if step == 120 else 0.5,
            "joint_fix_rate": 0.6 if step == 120 else 0.5,
        }
        for step in (120, 240, 360, 480, 600)
    ]


@pytest.fixture(scope="session")
def run_release_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("run-release")
    config_path = root / "config/experiments.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")
    for condition, path, seeds, configuration_id in CONDITIONS:
        for seed in seeds:
            run_dir = root / "data/runs" / path / f"seed-{seed}"
            run_dir.mkdir(parents=True)
            run_id = f"paper-{path}-{seed}"
            fine_tuned = not condition.startswith("zero-shot")
            commit = FINE_TUNED_COMMIT if fine_tuned else ZERO_SHOT_COMMIT
            record_text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in _records(condition, seed))
            (run_dir / "records.jsonl").write_text(record_text, encoding="utf-8")
            (run_dir / "results.yaml").write_text(
                yaml.safe_dump(_results(condition, seed, run_id, fine_tuned), sort_keys=False),
                encoding="utf-8",
            )
            manifest = {
                "historical_run_id": run_id,
                "condition": condition,
                "seed": seed,
                "profile": "paper",
                "dataset_revision": "7b951fd57d19286153b46ba219aa2cb87fcc4d2b",
                "historical_source_commit": commit,
                "configuration_id": configuration_id,
            }
            (run_dir / "manifest.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False),
                encoding="utf-8",
            )
            (run_dir / "selection_trace.json").write_text(
                json.dumps(_trace(fine_tuned), separators=(",", ":")),
                encoding="utf-8",
            )
    return root


@pytest.fixture(scope="session")
def release_inputs():
    from analysis.inputs import load_release_inputs

    return load_release_inputs(ROOT)


@pytest.fixture
def mini_case(tmp_path: Path) -> MiniCase:
    from analysis.inputs import (
        ExplanationRecord,
        ManuscriptInventory,
        ReleaseInputs,
        RuleIdentificationRecord,
        load_experiment_config,
        load_metadata,
    )
    from analysis.tables import GeneratedOutputs, OutputCell, TableData, TableRow

    snippet_id = f"sha256:{1:064x}"
    rule_record = RuleIdentificationRecord(
        base_snippet_id=snippet_id,
        condition="C0",
        seed=42,
        task_type="rule_identification",
        target_checks=("assignOp",),
        summary_status="not_applicable",
        prompt_tokens=10,
        retokenized_response_token_proxy=2,
        latency_ms=1.0,
        gold=("assignOp",),
        pred=("assignOp",),
        rejected_label_count=0,
        exact_match=True,
        n_emitted=1,
        normalization_status="recognized_array",
    )
    explanation_record = ExplanationRecord(
        base_snippet_id=snippet_id,
        condition="C0",
        seed=42,
        task_type="explanation",
        target_checks=("assignOp",),
        summary_status="not_applicable",
        prompt_tokens=12,
        retokenized_response_token_proxy=4,
        latency_ms=1.5,
    )
    inventory = ManuscriptInventory.model_validate_json(
        """
        {
          "schema_version": 1,
          "results": [
            {
              "id": "table_8_9.row_9.value",
              "manuscript_locations": ["table:8.9#mini-value"],
              "target": {"kind": "csv", "file": "table-8-9.csv", "row": "row_9", "column": "value"}
            },
            {
              "id": "table_8_8.row_8.value",
              "manuscript_locations": ["table:8.8#mini-value"],
              "target": {"kind": "csv", "file": "table-8-8.csv", "row": "row_8", "column": "value"}
            },
            {
              "id": "table_8_7.row_7.value",
              "manuscript_locations": ["table:8.7#mini-value"],
              "target": {"kind": "csv", "file": "table-8-7.csv", "row": "row_7", "column": "value"}
            },
            {
              "id": "table_8_6.row_6.value",
              "manuscript_locations": ["table:8.6#mini-value"],
              "target": {"kind": "csv", "file": "table-8-6.csv", "row": "row_6", "column": "value"}
            },
            {
              "id": "table_8_5.row_5.value",
              "manuscript_locations": ["table:8.5#mini-value"],
              "target": {"kind": "csv", "file": "table-8-5.csv", "row": "row_5", "column": "value"}
            },
            {
              "id": "table_8_4.row_4.value",
              "manuscript_locations": ["table:8.4#mini-value"],
              "target": {"kind": "csv", "file": "table-8-4.csv", "row": "row_4", "column": "value"}
            },
            {
              "id": "table_8_3.row_3.value",
              "manuscript_locations": ["table:8.3#mini-value"],
              "target": {"kind": "csv", "file": "table-8-3.csv", "row": "row_3", "column": "value"}
            },
            {
              "id": "table_8_2.row_2.value",
              "manuscript_locations": ["table:8.2#mini-value"],
              "target": {"kind": "csv", "file": "table-8-2.csv", "row": "row_2", "column": "value"}
            },
            {
              "id": "table_8_1.row_1.value",
              "manuscript_locations": ["table:8.1#mini-value"],
              "target": {"kind": "csv", "file": "table-8-1.csv", "row": "row_1", "column": "value"}
            },
            {
              "id": "rq1.rule.delta",
              "manuscript_locations": ["section:8.1#mini-delta"],
              "target": {"kind": "json", "file": "results.json", "identifier": "rq1.rule.delta"}
            },
            {
              "id": "dataset.base_snippets.total",
              "manuscript_locations": ["section:5.1#mini-total"],
              "target": {
                "kind": "json",
                "file": "results.json",
                "identifier": "dataset.base_snippets.total"
              }
            }
          ]
        }
        """
    )
    table_specs = (
        ("table-8-9.csv", "row_9", "Row 9", 9.0, 0),
        ("table-8-8.csv", "row_8", "Row 8", 8.5, 1),
        ("table-8-7.csv", "row_7", "Row 7", 7.75, 2),
        ("table-8-6.csv", "row_6", "Row 6", 6, 0),
        ("table-8-5.csv", "row_5", "Row 5", 5.0, 1),
        ("table-8-4.csv", "row_4", "Row 4", 4.125, 3),
        ("table-8-3.csv", "row_3", "Row 3", 3, 0),
        ("table-8-2.csv", "row_2", "Row 2", 2.25, 2),
        ("table-8-1.csv", "row_1", "Row 1", -1.3e-16, 1),
    )
    tables = {
        filename: TableData(
            filename=filename,
            columns=("label", "value"),
            rows=(
                TableRow(
                    key=row_key,
                    cells={
                        "label": label,
                        "value": OutputCell(
                            f"table_8_{filename.removeprefix('table-8-').removesuffix('.csv')}.{row_key}.value",
                            value,
                            display_digits,
                        ),
                    },
                ),
            ),
        )
        for filename, row_key, label, value, display_digits in table_specs
    }
    outputs = GeneratedOutputs(
        results={
            "rq1.rule.delta": 0.125,
            "dataset.base_snippets.total": 2,
        },
        tables=tables,
    )
    release = ReleaseInputs(
        config=load_experiment_config(ROOT / "config/experiments.yaml"),
        records=(rule_record, explanation_record),
        scored_records=(rule_record,),
        results=(),
        manifests=(),
        selection_traces={},
        study_rows=(),
        metadata=load_metadata(ROOT / "data/study/analysis_metadata.yaml"),
        inventory=inventory,
    )

    (tmp_path / "config").mkdir()
    (tmp_path / "config/experiments.yaml").write_text("fixture: mini\n", encoding="utf-8")
    (tmp_path / "data/study").mkdir(parents=True)
    (tmp_path / "data/study/analysis_inputs.jsonl").write_text('{"fixture":"mini"}\n', encoding="utf-8")
    for relative_path, content in _MINI_EXPECTED_FILES:
        path = tmp_path / "expected" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    return MiniCase(
        root=tmp_path,
        release=release,
        outputs=outputs,
        expected_files=_MINI_EXPECTED_FILES,
    )
