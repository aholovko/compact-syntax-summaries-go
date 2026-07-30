from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from analysis.inputs import load_records_file
import analysis.metrics as metrics

FIXTURE = Path(__file__).with_name("fixtures") / "rq3-records.jsonl"


def _loaded_fixture():
    return load_records_file(FIXTURE)


def test_rq3_fixture_is_source_free_and_rectangular() -> None:
    records = _loaded_fixture()
    raw_rows = tuple(json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines())

    assert len(records) == len(raw_rows) == 720
    assert {record.base_snippet_id for record in records} == {f"sha256:{index:064x}" for index in range(40)}
    assert {(record.base_snippet_id, record.seed, record.condition, record.task_type) for record in records} == {
        (f"sha256:{index:064x}", seed, condition, task)
        for index in range(40)
        for seed in (42, 43, 44)
        for condition in ("C0", "C1")
        for task in ("rule_identification", "correction", "joint")
    }
    forbidden = {"source", "source_text", "source_path", "prompt", "response", "private_payload"}
    assert all(not (set(row) & forbidden) for row in raw_rows)


def test_build_rq3_frame_has_fixed_columns_categories_and_order() -> None:
    records = _loaded_fixture()
    original = tuple(record.model_dump_json() for record in records)
    frame = metrics.build_rq3_frame(tuple(reversed(records)))

    assert tuple(record.model_dump_json() for record in records) == original
    assert tuple(frame.columns) == ("base_snippet_id", "seed", "condition", "task", "success")
    assert list(frame["condition"].cat.categories) == ["C0", "C1"]
    assert frame["condition"].cat.ordered is True
    assert list(frame["task"].cat.categories) == ["rule_identification", "correction", "joint"]
    assert frame["task"].cat.ordered is True
    assert frame.equals(
        frame.sort_values(["base_snippet_id", "seed", "condition", "task"], kind="mergesort").reset_index(drop=True)
    )
    assert set(frame["success"]) <= {0, 1}


def test_rq3_fit_is_byte_identical_across_fresh_processes(tmp_path: Path) -> None:
    del tmp_path
    script = (
        "from pathlib import Path; import json; "
        "from analysis.inputs import load_records_file; "
        "from analysis.metrics import build_rq3_frame, fit_rq3_interaction; "
        f"r=load_records_file(Path({str(FIXTURE)!r})); "
        "print(json.dumps(fit_rq3_interaction(build_rq3_frame(r)).as_dict(), "
        "sort_keys=True, separators=(',', ':')))"
    )
    outputs = []
    for hash_seed in ("1", "987654"):
        env = os.environ | {"PYTHONHASHSEED": hash_seed}
        outputs.append(subprocess.check_output([sys.executable, "-c", script], text=True, env=env))
    assert outputs[0] == outputs[1]


def test_build_rq3_frame_rejects_duplicate_and_incomplete_grids() -> None:
    records = _loaded_fixture()
    with pytest.raises(ValueError, match="nonempty"):
        metrics.build_rq3_frame(())
    with pytest.raises(ValueError, match="duplicate"):
        metrics.build_rq3_frame((*records, records[0]))
    with pytest.raises(ValueError, match="incomplete"):
        metrics.build_rq3_frame(records[:-1])


def test_build_rq3_frame_rejects_malformed_success() -> None:
    records = list(_loaded_fixture())
    record = records[0]
    records[0] = SimpleNamespace(
        base_snippet_id=record.base_snippet_id,
        seed=record.seed,
        condition=record.condition,
        task_type=record.task_type,
        exact_match=1,
    )

    with pytest.raises(ValueError, match="success"):
        metrics.build_rq3_frame(records)


def test_fit_uses_full_explicit_starts_and_full_interaction_term_name(monkeypatch, release_inputs) -> None:
    fixture_frame = metrics.build_rq3_frame(_loaded_fixture())
    released_frame = metrics.build_rq3_frame(release_inputs.scored_records)
    target = "C(condition, Treatment('C0'))[T.C1]:C(task, Treatment('rule_identification'))[T.joint]"
    names = (
        "Intercept",
        "C(task, Treatment('rule_identification'))[T.correction]",
        target,
        "C(condition, Treatment('C0'))[T.C1]",
        "C(task, Treatment('rule_identification'))[T.joint]",
        "C(condition, Treatment('C0'))[T.C1]:C(task, Treatment('rule_identification'))[T.correction]",
    )
    calls: list[tuple[str, dict[str, str], pd.DataFrame, np.ndarray, np.ndarray]] = []
    target_coefficient = [2.0]

    class FakeModel:
        k_fep = 6
        k_vcp = 1

        def __init__(self, formula: str, vc_formulas: dict[str, str], frame: pd.DataFrame) -> None:
            self.formula = formula
            self.vc_formulas = vc_formulas
            self.frame = frame
            self.k_vc = frame["base_snippet_id"].nunique()

        def fit_vb(self, *, mean: np.ndarray, sd: np.ndarray):
            calls.append((self.formula, self.vc_formulas, self.frame, mean.copy(), sd.copy()))
            return SimpleNamespace(
                model=SimpleNamespace(exog_names=list(names)),
                fe_mean=np.array([-5.0, -4.0, target_coefficient[0], -3.0, -2.0, -1.0]),
                fe_sd=np.array([1.0, 1.0, 0.25, 1.0, 1.0, 1.0]),
            )

    def fake_from_formula(formula: str, vc_formulas: dict[str, str], frame: pd.DataFrame) -> FakeModel:
        return FakeModel(formula, vc_formulas, frame)

    monkeypatch.setattr(
        metrics,
        "BinomialBayesMixedGLM",
        SimpleNamespace(from_formula=fake_from_formula),
    )
    altered_fixture = fixture_frame.iloc[::-1].copy()
    altered_fixture["condition"] = altered_fixture["condition"].cat.reorder_categories(["C1", "C0"])
    altered_fixture["task"] = altered_fixture["task"].cat.reorder_categories(
        ["joint", "correction", "rule_identification"]
    )
    original = altered_fixture.copy(deep=True)

    fixture_result = metrics.fit_rq3_interaction(altered_fixture)
    released_result = metrics.fit_rq3_interaction(released_frame)
    target_coefficient[0] = -2.0
    negative_result = metrics.fit_rq3_interaction(fixture_frame)

    pd.testing.assert_frame_equal(altered_fixture, original)
    assert fixture_result.term_name == target
    assert fixture_result.coefficient == 2.0
    assert fixture_result.sd == 0.25
    assert fixture_result.ci_low == 2.0 - float(norm.ppf(0.975)) * 0.25
    assert fixture_result.ci_high == 2.0 + float(norm.ppf(0.975)) * 0.25
    assert fixture_result.excludes_zero is True
    assert fixture_result.reject is True
    assert released_result == fixture_result
    assert negative_result.excludes_zero is True
    assert negative_result.reject is False
    assert len(calls) == 3
    assert [len(mean) for _, _, _, mean, _ in calls] == [47, 417, 47]
    for formula, vc_formulas, fitted_frame, mean, sd in calls:
        assert formula == ("success ~ C(condition, Treatment('C0')) * C(task, Treatment('rule_identification'))")
        assert vc_formulas == {"snippet": "0 + C(base_snippet_id)"}
        assert np.array_equal(mean, np.zeros(len(mean), dtype=float))
        assert np.array_equal(sd, np.full(len(sd), np.exp(-0.5), dtype=float))
        assert list(fitted_frame["condition"].cat.categories) == ["C0", "C1"]
        assert list(fitted_frame["task"].cat.categories) == ["rule_identification", "correction", "joint"]
        assert fitted_frame.equals(
            fitted_frame.sort_values(["base_snippet_id", "seed", "condition", "task"], kind="mergesort").reset_index(
                drop=True
            )
        )


def test_fit_is_repeated_and_reversed_input_deterministic_without_mutation() -> None:
    frame = metrics.build_rq3_frame(_loaded_fixture())
    original = frame.copy(deep=True)
    reversed_frame = frame.iloc[::-1].copy()

    first = metrics.fit_rq3_interaction(frame)
    repeated = metrics.fit_rq3_interaction(frame)
    reversed_result = metrics.fit_rq3_interaction(reversed_frame)

    pd.testing.assert_frame_equal(frame, original)
    assert first == repeated == reversed_result


def test_rq3_task_contrasts_have_fixed_order_and_model_free_definitions() -> None:
    records = _loaded_fixture()
    n_boot = 300
    seed = 42
    joint = metrics.seed_averaged_snippet_differences(records, condition_a="C1", condition_b="C0", task="joint")
    expected = []
    for reference_task in ("correction", "rule_identification"):
        reference = metrics.seed_averaged_snippet_differences(
            records,
            condition_a="C1",
            condition_b="C0",
            task=reference_task,
        )
        expected.append(
            metrics.cluster_bootstrap_interval(
                {snippet_id: joint[snippet_id] - reference[snippet_id] for snippet_id in sorted(joint)},
                n_boot=n_boot,
                seed=seed,
            )
        )

    contrasts = metrics.rq3_task_contrasts(records, n_boot=n_boot, seed=seed)
    reversed_contrasts = metrics.rq3_task_contrasts(tuple(reversed(records)), n_boot=n_boot, seed=seed)

    assert reversed_contrasts == contrasts
    assert tuple(contrast.name for contrast in contrasts) == (
        "joint_minus_correction",
        "joint_minus_rule_identification",
    )
    assert tuple(contrast.reference_task for contrast in contrasts) == ("correction", "rule_identification")
    assert all(
        (contrast.condition_a, contrast.condition_b, contrast.focal_task) == ("C1", "C0", "joint")
        for contrast in contrasts
    )
    assert tuple(contrast.interval for contrast in contrasts) == tuple(expected)
    assert tuple(contrast.reject for contrast in contrasts) == tuple(interval.ci_low > 0.0 for interval in expected)


def test_rq3_task_contrasts_require_identical_task_snippet_sets() -> None:
    records = tuple(
        record
        for record in _loaded_fixture()
        if not (record.base_snippet_id == f"sha256:{0:064x}" and record.task_type == "joint")
    )

    with pytest.raises(ValueError, match="incomplete"):
        metrics.rq3_task_contrasts(records, n_boot=10, seed=42)


def test_rq3_task_contrasts_reject_different_internally_rectangular_task_seed_sets() -> None:
    seeds_by_task = {
        "rule_identification": {42, 43},
        "correction": {42, 43},
        "joint": {43, 44},
    }
    records = tuple(record for record in _loaded_fixture() if record.seed in seeds_by_task[record.task_type])

    assert {
        task: {record.base_snippet_id for record in records if record.task_type == task} for task in seeds_by_task
    } == {task: {f"sha256:{index:064x}" for index in range(40)} for task in seeds_by_task}
    assert {
        task: {record.seed for record in records if record.task_type == task} for task in seeds_by_task
    } == seeds_by_task
    with pytest.raises(ValueError, match="incomplete"):
        metrics.rq3_task_contrasts(records, n_boot=10, seed=42)


def test_released_rq3_values_preserve_every_displayed_manuscript_value(release_inputs) -> None:
    frame = metrics.build_rq3_frame(release_inputs.scored_records)
    fit = metrics.fit_rq3_interaction(frame)

    assert len(frame) == 7_380
    assert fit == metrics.Rq3Result(
        coefficient=0.15655574535370093,
        sd=0.0777587076881755,
        ci_low=0.004151478800499153,
        ci_high=0.3089600119069027,
        excludes_zero=True,
        reject=True,
        term_name=("C(condition, Treatment('C0'))[T.C1]:C(task, Treatment('rule_identification'))[T.joint]"),
    )
    contrasts = metrics.rq3_task_contrasts(release_inputs.scored_records, n_boot=10_000, seed=42)
    assert contrasts == (
        metrics.Rq3TaskContrast(
            name="joint_minus_correction",
            condition_a="C1",
            condition_b="C0",
            focal_task="joint",
            reference_task="correction",
            interval=metrics.Interval(
                point=0.0040650406504065045,
                ci_low=-0.018699186991869916,
                ci_high=0.027642276422764223,
                p_value=0.7304,
                n_boot=10_000,
                seed=42,
                alpha=0.05,
                n_units=410,
            ),
            reject=False,
        ),
        metrics.Rq3TaskContrast(
            name="joint_minus_rule_identification",
            condition_a="C1",
            condition_b="C0",
            focal_task="joint",
            reference_task="rule_identification",
            interval=metrics.Interval(
                point=0.022764227642276424,
                ci_low=-0.005691056910569105,
                ci_high=0.05121951219512195,
                p_value=0.1206,
                n_boot=10_000,
                seed=42,
                alpha=0.05,
                n_units=410,
            ),
            reject=False,
        ),
    )
    assert (
        f"{fit.coefficient:+.3f}",
        f"{fit.sd:.3f}",
        f"[{fit.ci_low:.3f}, {fit.ci_high:.3f}]",
    ) == ("+0.157", "0.078", "[0.004, 0.309]")
    assert all(contrast.reject is False for contrast in contrasts)
