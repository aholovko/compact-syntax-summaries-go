from decimal import Decimal
from pathlib import Path

from analysis.inputs import expected_run_keys, load_runs

ROOT = Path(__file__).resolve().parents[2]


def test_complete_run_projection() -> None:
    runs = load_runs(ROOT)
    assert len(runs.manifests) == 14
    assert len(runs.records) == 25_088
    for manifest in runs.manifests:
        assert set(manifest.model_dump()) == {
            "historical_run_id",
            "condition",
            "seed",
            "profile",
            "dataset_revision",
            "historical_source_commit",
            "configuration_id",
        }
    for result in runs.results:
        assert set(result.model_dump()) == {
            "historical_run_id",
            "condition",
            "seed",
            "profile",
            "metrics",
        }
        assert set(result.metrics.model_dump()) == {
            "checkpoint_selection",
            "compute",
            "length",
            "provenance",
        }
    training_values = [
        r.metrics.compute.wall_clock_train_s for r in runs.results if r.metrics.compute.wall_clock_train_s is not None
    ]
    training = sum((Decimal(str(value)) for value in training_values), start=Decimal(0))
    total = sum(
        (
            Decimal(str(r.metrics.compute.wall_clock_total_s))
            for r in runs.results
            if r.metrics.compute.wall_clock_total_s is not None
        ),
        start=Decimal(0),
    )
    assert training == Decimal("92216.1416017580135")
    assert str(sum(training_values)) == "92216.14160175802"
    assert total == Decimal("127510.859002")
    for key in expected_run_keys():
        records = [r for r in runs.records if (r.condition, r.seed) == key]
        assert len(records) == 1_792
        assert len({r.base_snippet_id for r in records}) == 448
        assert {r.task_type for r in records} == {"rule_identification", "correction", "joint", "explanation"}
    results = {(result.condition, result.seed): result for result in runs.results}
    for key, trace in runs.selection_traces.items():
        selection = results[key].metrics.checkpoint_selection
        if key[0] in {"C0", "C1", "C2", "C2-control"}:
            assert tuple(point.step for point in trace) == (120, 240, 360, 480, 600)
            selected = [point for point in trace if point.step == selection.selected_step]
            assert len(selected) == 1
            point = selected[0]
            assert (
                point.composite_score,
                point.rule_id_macro_f1,
                point.correction_fix_rate,
                point.joint_fix_rate,
            ) == (
                selection.best_composite,
                selection.rule_id_macro_f1,
                selection.correction_fix_rate,
                selection.joint_fix_rate,
            )
        else:
            assert trace == ()
            assert all(value is None for value in selection.model_dump().values())


def test_run_tree_has_only_the_four_public_files() -> None:
    for directory in sorted((ROOT / "data/runs").glob("*/seed-*")):
        assert {p.name for p in directory.iterdir()} == {
            "records.jsonl",
            "results.yaml",
            "selection_trace.json",
            "manifest.yaml",
        }
