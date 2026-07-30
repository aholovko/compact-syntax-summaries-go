from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.compare import compare_artifacts, validate_inventory_coverage, write_artifacts  # noqa: E402
from analysis.inputs import load_release_inputs  # noqa: E402
from analysis.tables import build_outputs, output_registry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        release = load_release_inputs(ROOT)
        outputs = build_outputs(
            experiment_config=release.config,
            scored_records=release.scored_records,
            run_results=release.results,
            selection_traces=release.selection_traces,
            study_rows=release.study_rows,
            metadata=release.metadata,
        )
        validate_inventory_coverage(release.inventory, output_registry(outputs))
        write_artifacts(outputs, ROOT / "reproduced")
        comparisons = compare_artifacts(ROOT / "reproduced", ROOT / "expected")
    except ValueError as error:
        print(f"FAIL: {error}")
        return 1
    print(f"PASS: validated {len(release.manifests)} evaluation runs ({len(release.records):,} task records)")
    print(f"PASS: validated {len(release.study_rows):,} study rows")
    for result in comparisons:
        print(f"PASS: {result.relative_path}")
    print(
        f"PASS: {len(comparisons)} outputs recomputed from released records "
        "(JSON floats within 1e-7 of expected; CSVs byte-identical)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
