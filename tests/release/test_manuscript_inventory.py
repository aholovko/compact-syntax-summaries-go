from collections import Counter
from pathlib import Path

import pytest

from analysis.inputs import ManuscriptInventory, load_inventory

ROOT = Path(__file__).resolve().parents[2]


def test_inventory_has_unique_ids_targets_and_no_values() -> None:
    inventory = load_inventory(ROOT / "config/manuscript_results.yaml")
    ids = [row.id for row in inventory.results]
    targets = [row.target.model_dump_json() for row in inventory.results]
    assert len(ids) == len(set(ids))
    assert len(targets) == len(set(targets))
    assert len(inventory.results) == 982
    assert all(row.manuscript_locations for row in inventory.results)
    assert sum(row.target.kind == "json" for row in inventory.results) == 491
    assert Counter(row.target.file for row in inventory.results if row.target.kind == "csv") == {
        "table-8-1.csv": 50,
        "table-8-2.csv": 56,
        "table-8-3.csv": 18,
        "table-8-4.csv": 40,
        "table-8-5.csv": 79,
        "table-8-6.csv": 24,
        "table-8-7.csv": 72,
        "table-8-8.csv": 120,
        "table-8-9.csv": 32,
    }
    for row in inventory.results:
        if row.target.kind == "json":
            assert row.target.file == "results.json"
            assert row.target.identifier == row.id
            assert row.target.row is None and row.target.column is None
        else:
            assert row.target.file != "results.json"
            assert row.target.identifier is None
            assert row.target.row is not None and row.target.column is not None
    serialized = (ROOT / "config/manuscript_results.yaml").read_text(encoding="utf-8").lower()
    assert "expected_value" not in serialized and "result_value" not in serialized


def test_inventory_covers_every_required_family() -> None:
    inventory = load_inventory(ROOT / "config/manuscript_results.yaml")
    ids = {row.id for row in inventory.results}
    required_prefixes = {
        "dataset.",
        "reference_qc.",
        "oracle.",
        "serializer.",
        "repository.",
        "architecture.",
        "verification.",
        "training.",
        "run.",
        "rq1.",
        "rq2.",
        "rq3.",
        "sensitivity.",
        "section_8_10.",
    }
    assert all(any(result_id.startswith(prefix) for result_id in ids) for prefix in required_prefixes)
    for number in range(1, 10):
        assert any(result_id.startswith(f"table_8_{number}.") for result_id in ids)
    assert "section_8_10.c2_only.at_least_two_seeds.files" in ids
    assert "sensitivity.task_rate.c1.correction_minus_joint.approx_points" in ids


def test_inventory_rejects_embedded_result_value() -> None:
    with pytest.raises(Exception):
        ManuscriptInventory.model_validate(
            {
                "schema_version": 1,
                "results": (
                    {
                        "id": "dataset.total",
                        "manuscript_locations": ("section:5.1#corpus-count",),
                        "target": {
                            "kind": "json",
                            "file": "results.json",
                            "identifier": "dataset.total",
                        },
                        "value": 2206,
                    },
                ),
            }
        )
