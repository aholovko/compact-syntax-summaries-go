from pathlib import Path

from analysis.compare import compare_artifacts, validate_inventory_coverage, write_artifacts
from analysis.inputs import load_release_inputs
from analysis.tables import build_outputs, output_registry

ROOT = Path(__file__).resolve().parents[2]


def test_inventory_registry_and_expected_artifacts_are_complete(tmp_path: Path) -> None:
    release = load_release_inputs(ROOT)
    outputs = build_outputs(
        experiment_config=release.config,
        scored_records=release.scored_records,
        run_results=release.results,
        selection_traces=release.selection_traces,
        study_rows=release.study_rows,
        metadata=release.metadata,
    )
    registry = output_registry(outputs)
    inventory = {entry.id: entry.target.as_key() for entry in release.inventory.results}
    validate_inventory_coverage(release.inventory, registry)
    assert inventory == registry

    actual = tmp_path / "reproduced"
    written = write_artifacts(outputs, actual)
    assert len(written) == 10
    comparisons = compare_artifacts(actual, ROOT / "expected")
    assert [result.relative_path for result in comparisons] == [
        "results.json",
        *(f"tables/table-8-{number}.csv" for number in range(1, 10)),
    ]
