from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from analysis.compare import (
    ComparisonError,
    ComparisonResult,
    CoverageError,
    compare_artifacts,
    validate_inventory_coverage,
    write_artifacts,
)
from analysis.inputs import ManuscriptInventory
from analysis.tables import GeneratedOutputs, OutputCell, TableData, TableRow, output_registry

EXPECTED_PATHS = (
    "results.json",
    "tables/table-8-1.csv",
    "tables/table-8-2.csv",
    "tables/table-8-3.csv",
    "tables/table-8-4.csv",
    "tables/table-8-5.csv",
    "tables/table-8-6.csv",
    "tables/table-8-7.csv",
    "tables/table-8-8.csv",
    "tables/table-8-9.csv",
)


def _copy_expected(mini_case, tmp_path: Path) -> tuple[Path, Path]:
    expected = mini_case.root / "expected"
    actual = tmp_path / "actual"
    shutil.copytree(expected, actual)
    return actual, expected


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_write_artifacts_emits_exact_ten_files_in_canonical_order(mini_case, tmp_path: Path) -> None:
    destination = tmp_path / "written"

    written = write_artifacts(mini_case.outputs, destination)

    assert tuple(path.relative_to(destination).as_posix() for path in written) == EXPECTED_PATHS
    assert tuple(relative_path for relative_path, _content in mini_case.expected_files) == EXPECTED_PATHS
    for relative_path, content in mini_case.expected_files:
        assert (destination / relative_path).read_bytes() == content


def test_write_artifacts_preserves_declared_csv_column_and_row_order(mini_case, tmp_path: Path) -> None:
    table = TableData(
        filename="table-8-1.csv",
        columns=("value", "label"),
        rows=(
            TableRow(
                key="second",
                cells={"label": "Second", "value": OutputCell("table_8_1.second.value", 2, 0)},
            ),
            TableRow(
                key="first",
                cells={"label": "First", "value": OutputCell("table_8_1.first.value", 1, 0)},
            ),
        ),
    )
    outputs = GeneratedOutputs(
        results=mini_case.outputs.results,
        tables=mini_case.outputs.tables | {table.filename: table},
    )

    write_artifacts(outputs, tmp_path / "written")

    content = (tmp_path / "written/tables/table-8-1.csv").read_bytes()
    assert content == b"value,label\n2,Second\n1,First\n"
    assert b"\r\n" not in content


def test_write_artifacts_normalizes_displayed_negative_zero_without_mutating_raw_cell(
    mini_case,
    tmp_path: Path,
) -> None:
    table = mini_case.outputs.tables["table-8-1.csv"]
    cell = table.rows[0].cells["value"]
    assert isinstance(cell, OutputCell)
    assert cell.value < 0.0
    assert f"{cell.value:.1f}" == "-0.0"

    write_artifacts(mini_case.outputs, tmp_path / "written")

    assert (tmp_path / "written/tables/table-8-1.csv").read_bytes().endswith(b"Row 1,0.0\n")
    assert cell.value == -1.3e-16


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_write_artifacts_rejects_nonfinite_json_numbers(mini_case, tmp_path: Path, value: float) -> None:
    outputs = GeneratedOutputs(
        results=mini_case.outputs.results | {"rq1.nonfinite": value},
        tables=mini_case.outputs.tables,
    )

    with pytest.raises(ValueError, match="finite|range|JSON"):
        write_artifacts(outputs, tmp_path / "written")


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_write_artifacts_requires_the_exact_nine_tables(mini_case, tmp_path: Path, change: str) -> None:
    tables = dict(mini_case.outputs.tables)
    if change == "missing":
        tables.pop("table-8-9.csv")
    else:
        tables["table-8-10.csv"] = TableData(
            filename="table-8-10.csv",
            columns=("label", "value"),
            rows=(
                TableRow(
                    key="extra",
                    cells={"label": "Extra", "value": OutputCell("table_8_10.extra.value", 1, 0)},
                ),
            ),
        )
    outputs = GeneratedOutputs(results=mini_case.outputs.results, tables=tables)

    with pytest.raises(ValueError, match="table|artifact"):
        write_artifacts(outputs, tmp_path / "written")


@pytest.mark.parametrize(
    "content",
    [
        b"value,label\n4.125,Row 4\n",
        b"label,value\nChanged label,4.125\n",
        b"label,value\nRow 4,4.12\n",
        b"label,value\r\nRow 4,4.125\r\n",
        b"label,value\nRow 4,4.125",
    ],
    ids=("column-order", "label", "display", "line-ending", "final-newline"),
)
def test_compare_artifacts_requires_byte_identical_csv(mini_case, tmp_path: Path, content: bytes) -> None:
    actual, expected = _copy_expected(mini_case, tmp_path)
    (actual / "tables/table-8-4.csv").write_bytes(content)

    with pytest.raises(ComparisonError, match="table-8-4.csv"):
        compare_artifacts(actual, expected)


@pytest.mark.parametrize(
    ("delta", "accepted"),
    [(9e-8, True), (1.1e-7, False)],
    ids=("inside-absolute-tolerance", "outside-absolute-tolerance"),
)
def test_compare_artifacts_uses_only_absolute_1e_7_for_json_floats(
    mini_case,
    tmp_path: Path,
    delta: float,
    accepted: bool,
) -> None:
    actual, expected = _copy_expected(mini_case, tmp_path)
    payload = json.loads((actual / "results.json").read_text(encoding="utf-8"))
    payload["rq1.rule.delta"] += delta
    _write_json(actual / "results.json", payload)

    if accepted:
        assert compare_artifacts(actual, expected)[0] == ComparisonResult(relative_path="results.json")
    else:
        with pytest.raises(ComparisonError, match="results.json"):
            compare_artifacts(actual, expected)


def test_compare_artifacts_does_not_widen_float_tolerance_relative_to_magnitude(mini_case, tmp_path: Path) -> None:
    actual, expected = _copy_expected(mini_case, tmp_path)
    _write_json(expected / "results.json", {"large": 1_000_000_000_000.0})
    _write_json(actual / "results.json", {"large": 1_000_000_000_000.0001})

    with pytest.raises(ComparisonError, match="results.json"):
        compare_artifacts(actual, expected)


@pytest.mark.parametrize("actual_value", [2.0, True, "2", None], ids=("float", "bool", "string", "null"))
def test_compare_artifacts_requires_exact_json_runtime_types(
    mini_case,
    tmp_path: Path,
    actual_value: object,
) -> None:
    actual, expected = _copy_expected(mini_case, tmp_path)
    payload = json.loads((actual / "results.json").read_text(encoding="utf-8"))
    payload["dataset.base_snippets.total"] = actual_value
    _write_json(actual / "results.json", payload)

    with pytest.raises(ComparisonError, match="results.json"):
        compare_artifacts(actual, expected)


def test_compare_artifacts_requires_exact_json_object_keys(mini_case, tmp_path: Path) -> None:
    actual, expected = _copy_expected(mini_case, tmp_path)
    _write_json(expected / "results.json", {"payload": {"kept": 1}})
    _write_json(actual / "results.json", {"payload": {"extra": 2, "kept": 1}})

    with pytest.raises(ComparisonError, match="results.json"):
        compare_artifacts(actual, expected)


def test_compare_artifacts_recurses_through_json_objects_and_ordered_lists(mini_case, tmp_path: Path) -> None:
    actual, expected = _copy_expected(mini_case, tmp_path)
    expected_payload = {"payload": {"items": [1, 2.0, True, None, "same"]}}
    actual_payload = {"payload": {"items": [1, 2.0 + 9e-11, True, None, "same"]}}
    _write_json(expected / "results.json", expected_payload)
    _write_json(actual / "results.json", actual_payload)
    assert compare_artifacts(actual, expected)[0].relative_path == "results.json"

    actual_payload["payload"]["items"] = [1, 2.0 + 9e-11, None, True, "same"]
    _write_json(actual / "results.json", actual_payload)
    with pytest.raises(ComparisonError, match="results.json"):
        compare_artifacts(actual, expected)


@pytest.mark.parametrize("side", ["actual", "expected"])
@pytest.mark.parametrize(
    "content",
    [
        b'{"duplicate": 1, "duplicate": 2}\n',
        b'{"value": NaN}\n',
        b'{"value": Infinity}\n',
        b'{"value": -Infinity}\n',
    ],
    ids=("duplicate-key", "nan", "positive-infinity", "negative-infinity"),
)
def test_compare_artifacts_rejects_duplicate_and_nonfinite_json(
    mini_case,
    tmp_path: Path,
    side: str,
    content: bytes,
) -> None:
    actual, expected = _copy_expected(mini_case, tmp_path)
    root = actual if side == "actual" else expected
    (root / "results.json").write_bytes(content)

    with pytest.raises(ComparisonError, match="results.json"):
        compare_artifacts(actual, expected)


@pytest.mark.parametrize("side", ["actual", "expected"])
def test_compare_artifacts_rejects_a_missing_canonical_file(mini_case, tmp_path: Path, side: str) -> None:
    actual, expected = _copy_expected(mini_case, tmp_path)
    root = actual if side == "actual" else expected
    (root / "tables/table-8-9.csv").unlink()

    with pytest.raises(ComparisonError, match="table-8-9.csv"):
        compare_artifacts(actual, expected)


@pytest.mark.parametrize("side", ["actual", "expected"])
def test_compare_artifacts_rejects_an_extra_numbered_table(mini_case, tmp_path: Path, side: str) -> None:
    actual, expected = _copy_expected(mini_case, tmp_path)
    root = actual if side == "actual" else expected
    (root / "tables/table-8-10.csv").write_text("label,value\nExtra,1\n", encoding="utf-8")

    with pytest.raises(ComparisonError, match="table-8-10.csv"):
        compare_artifacts(actual, expected)


def test_inventory_coverage_is_exact_but_inventory_order_is_immaterial(mini_case) -> None:
    registry = output_registry(mini_case.outputs)
    reordered = mini_case.release.inventory.model_copy(
        update={"results": tuple(reversed(mini_case.release.inventory.results))}
    )

    assert validate_inventory_coverage(mini_case.release.inventory, registry) is None
    assert validate_inventory_coverage(reordered, registry) is None


def test_empty_inventory_cannot_satisfy_coverage(mini_case) -> None:
    inventory = ManuscriptInventory(schema_version=1, results=())

    with pytest.raises(CoverageError):
        validate_inventory_coverage(inventory, output_registry(mini_case.outputs))


def test_duplicate_inventory_result_id_fails_coverage(mini_case) -> None:
    inventory = mini_case.release.inventory
    duplicated = inventory.model_copy(update={"results": (*inventory.results, inventory.results[0])})

    with pytest.raises(CoverageError, match="duplicate result ID"):
        validate_inventory_coverage(duplicated, output_registry(mini_case.outputs))


def test_duplicate_inventory_target_fails_coverage(mini_case) -> None:
    inventory = mini_case.release.inventory
    duplicate_target = inventory.results[0].model_copy(update={"id": "table_8_9.duplicate.value"})
    duplicated = inventory.model_copy(update={"results": (*inventory.results, duplicate_target)})

    with pytest.raises(CoverageError, match="duplicate target"):
        validate_inventory_coverage(duplicated, output_registry(mini_case.outputs))


def test_inventory_target_mismatch_fails_coverage(mini_case) -> None:
    inventory = mini_case.release.inventory
    original = inventory.results[0]
    changed = original.model_copy(update={"target": original.target.model_copy(update={"row": "wrong_row"})})
    mutated = inventory.model_copy(update={"results": (changed, *inventory.results[1:])})

    with pytest.raises(CoverageError, match="target|coverage"):
        validate_inventory_coverage(mutated, output_registry(mini_case.outputs))


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_inventory_and_registry_result_ids_must_be_identical(mini_case, change: str) -> None:
    registry = output_registry(mini_case.outputs)
    if change == "missing":
        registry.pop("rq1.rule.delta")
    else:
        registry["rq1.uninventoried"] = ("json", "results.json", "rq1.uninventoried", None, None)

    with pytest.raises(CoverageError, match="coverage"):
        validate_inventory_coverage(mini_case.release.inventory, registry)
