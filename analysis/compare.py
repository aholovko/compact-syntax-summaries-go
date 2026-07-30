from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import NoReturn, cast

from analysis.inputs import ManuscriptInventory
from analysis.tables import GeneratedOutputs, OutputCell, OutputTargetKey, TableData, output_registry

_TABLE_FILENAMES = tuple(f"table-8-{number}.csv" for number in range(1, 10))
_TABLE_FILENAME_SET = frozenset(_TABLE_FILENAMES)
_ARTIFACT_PATHS = ("results.json", *(f"tables/{filename}" for filename in _TABLE_FILENAMES))
_JSON_ABSOLUTE_TOLERANCE = 1e-7


@dataclass(frozen=True)
class ComparisonResult:
    relative_path: str


class ComparisonError(ValueError):
    pass


class CoverageError(ValueError):
    pass


class _DuplicateJsonKeyError(ValueError):
    pass


class _NonFiniteJsonError(ValueError):
    pass


def _canonical_json_bytes(outputs: GeneratedOutputs) -> bytes:
    try:
        text = json.dumps(
            outputs.results,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot serialize results.json: {error}") from error
    return f"{text}\n".encode()


def _render_cell(cell: OutputCell | str) -> str:
    if isinstance(cell, str):
        return cell
    if cell.display_digits is None:
        return "" if cell.value is None else str(cell.value)
    if type(cell.value) not in {int, float}:
        raise ValueError("a displayed numeric table cell requires an integer or float value")
    rendered = f"{cell.value:.{cell.display_digits}f}"
    if rendered.startswith("-") and float(rendered) == 0.0:
        return rendered[1:]
    return rendered


def _canonical_csv_bytes(table: TableData) -> bytes:
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(table.columns)
    for row in table.rows:
        writer.writerow(_render_cell(row.cells[column]) for column in table.columns)
    return stream.getvalue().encode()


def _require_exact_output_tables(outputs: GeneratedOutputs) -> None:
    actual = set(outputs.tables)
    if actual == _TABLE_FILENAME_SET:
        return
    missing = sorted(_TABLE_FILENAME_SET - actual)
    extra = sorted(actual - _TABLE_FILENAME_SET)
    details: list[str] = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if extra:
        details.append(f"unexpected {', '.join(extra)}")
    raise ValueError(f"artifact table set mismatch: {'; '.join(details)}")


def write_artifacts(outputs: GeneratedOutputs, destination: Path) -> tuple[Path, ...]:
    _require_exact_output_tables(outputs)
    output_registry(outputs)
    payloads = (
        ("results.json", _canonical_json_bytes(outputs)),
        *((f"tables/{filename}", _canonical_csv_bytes(outputs.tables[filename])) for filename in _TABLE_FILENAMES),
    )
    try:
        for relative_path, payload in payloads:
            path = destination / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    except OSError as error:
        raise ValueError(f"cannot write artifact {relative_path}: {error}") from error
    return tuple(destination / relative_path for relative_path, _payload in payloads)


def validate_inventory_coverage(
    inventory: ManuscriptInventory,
    registry: dict[str, OutputTargetKey],
) -> None:
    inventory_ids = [entry.id for entry in inventory.results]
    inventory_targets = [entry.target.as_key() for entry in inventory.results]
    if len(inventory_ids) != len(set(inventory_ids)):
        raise CoverageError("inventory coverage contains duplicate result IDs")
    if len(inventory_targets) != len(set(inventory_targets)):
        raise CoverageError("inventory coverage contains duplicate targets")

    inventory_registry = {entry.id: entry.target.as_key() for entry in inventory.results}
    inventory_id_set = set(inventory_registry)
    registry_id_set = set(registry)
    missing = sorted(inventory_id_set - registry_id_set)
    unexpected = sorted(registry_id_set - inventory_id_set)
    mismatched = sorted(
        result_id
        for result_id in inventory_id_set & registry_id_set
        if inventory_registry[result_id] != registry[result_id]
    )
    if not (missing or unexpected or mismatched):
        return

    details: list[str] = []
    if missing:
        details.append(f"not generated: {', '.join(missing)}")
    if unexpected:
        details.append(f"not inventoried: {', '.join(unexpected)}")
    if mismatched:
        details.append(f"target mismatch: {', '.join(mismatched)}")
    raise CoverageError(f"inventory coverage mismatch: {'; '.join(details)}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise _NonFiniteJsonError(value)


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _NonFiniteJsonError(value)
    return parsed


def _read_json(path: Path, relative_path: str, tree_name: str) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ComparisonError(f"{relative_path}: cannot read {tree_name} JSON: {error}") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
    except json.JSONDecodeError as error:
        raise ComparisonError(
            f"{relative_path}: invalid {tree_name} JSON at line {error.lineno}, column {error.colno}"
        ) from error
    except _DuplicateJsonKeyError as error:
        raise ComparisonError(f"{relative_path}: duplicate key in {tree_name} JSON: {error}") from error
    except _NonFiniteJsonError as error:
        raise ComparisonError(f"{relative_path}: non-finite number in {tree_name} JSON: {error}") from error
    if type(value) is not dict:
        raise ComparisonError(f"{relative_path}: {tree_name} JSON must be an object")
    return value


def _raise_json_mismatch(relative_path: str, location: str, detail: str) -> NoReturn:
    raise ComparisonError(f"{relative_path}: JSON mismatch at {location}: {detail}")


def _compare_json_values(actual: object, expected: object, relative_path: str, location: str = "$") -> None:
    if type(actual) is not type(expected):
        _raise_json_mismatch(
            relative_path,
            location,
            f"actual type {type(actual).__name__} differs from expected type {type(expected).__name__}",
        )
    if type(actual) is dict:
        actual_object = cast(dict[str, object], actual)
        expected_object = cast(dict[str, object], expected)
        if set(actual_object) != set(expected_object):
            _raise_json_mismatch(relative_path, location, "object keys differ")
        for key in sorted(expected_object):
            _compare_json_values(actual_object[key], expected_object[key], relative_path, f"{location}.{key}")
        return
    if type(actual) is list:
        actual_list = cast(list[object], actual)
        expected_list = cast(list[object], expected)
        if len(actual_list) != len(expected_list):
            _raise_json_mismatch(relative_path, location, "list lengths differ")
        for index, (actual_item, expected_item) in enumerate(zip(actual_list, expected_list, strict=True)):
            _compare_json_values(actual_item, expected_item, relative_path, f"{location}[{index}]")
        return
    if type(actual) is float:
        if not math.isclose(
            cast(float, actual),
            cast(float, expected),
            rel_tol=0.0,
            abs_tol=_JSON_ABSOLUTE_TOLERANCE,
        ):
            _raise_json_mismatch(relative_path, location, f"actual {actual!r} differs from expected {expected!r}")
        return
    if actual != expected:
        _raise_json_mismatch(relative_path, location, f"actual {actual!r} differs from expected {expected!r}")


def _require_artifact_tree(root: Path, tree_name: str) -> None:
    for relative_path in _ARTIFACT_PATHS:
        if not (root / relative_path).is_file():
            raise ComparisonError(f"{relative_path}: missing from {tree_name} artifacts")
    table_directory = root / "tables"
    numbered_tables = {path.name for path in table_directory.glob("table-8-*.csv") if path.is_file()}
    extras = sorted(numbered_tables - _TABLE_FILENAME_SET)
    if extras:
        raise ComparisonError(f"tables/{extras[0]}: unexpected numbered table in {tree_name} artifacts")


def _read_bytes(path: Path, relative_path: str, tree_name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ComparisonError(f"{relative_path}: cannot read {tree_name} artifact: {error}") from error


def compare_artifacts(actual: Path, expected: Path) -> tuple[ComparisonResult, ...]:
    _require_artifact_tree(actual, "actual")
    _require_artifact_tree(expected, "expected")

    actual_json = _read_json(actual / "results.json", "results.json", "actual")
    expected_json = _read_json(expected / "results.json", "results.json", "expected")
    _compare_json_values(actual_json, expected_json, "results.json")
    results: list[ComparisonResult] = [ComparisonResult(relative_path="results.json")]

    for filename in _TABLE_FILENAMES:
        relative_path = f"tables/{filename}"
        actual_bytes = _read_bytes(actual / relative_path, relative_path, "actual")
        expected_bytes = _read_bytes(expected / relative_path, relative_path, "expected")
        if actual_bytes != expected_bytes:
            raise ComparisonError(f"{relative_path}: CSV bytes differ")
        results.append(ComparisonResult(relative_path=relative_path))
    return tuple(results)
