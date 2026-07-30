from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import analysis.rerun_analysis as runner
from analysis.inputs import ManuscriptInventory

ROOT = Path(__file__).resolve().parents[2]


def tree_hash(root: Path) -> str:
    digest = sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_main_writes_only_reproduced_and_preserves_inputs(mini_case, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", mini_case.root)
    monkeypatch.setattr(runner, "load_release_inputs", lambda root: mini_case.release)
    build_arguments: dict[str, object] = {}

    def build_outputs(**kwargs):
        build_arguments.update(kwargs)
        return mini_case.outputs

    monkeypatch.setattr(runner, "build_outputs", build_outputs)
    before_data = tree_hash(mini_case.root / "data")
    before_config = tree_hash(mini_case.root / "config")
    before_expected = tree_hash(mini_case.root / "expected")
    sentinel = mini_case.root / "reproduced/keep.txt"
    nested_sentinel = mini_case.root / "reproduced/notes/reviewer.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    nested_sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("keep\n", encoding="utf-8")
    nested_sentinel.write_text("reviewer note\n", encoding="utf-8")

    assert runner.main(["ignored-by-the-fixed-entry-point"]) == 0

    assert build_arguments == {
        "experiment_config": mini_case.release.config,
        "scored_records": mini_case.release.scored_records,
        "run_results": mini_case.release.results,
        "selection_traces": mini_case.release.selection_traces,
        "study_rows": mini_case.release.study_rows,
        "metadata": mini_case.release.metadata,
    }
    assert len(mini_case.release.records) == 2
    assert len(mini_case.release.scored_records) == 1
    assert tree_hash(mini_case.root / "data") == before_data
    assert tree_hash(mini_case.root / "config") == before_config
    assert tree_hash(mini_case.root / "expected") == before_expected
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert nested_sentinel.read_text(encoding="utf-8") == "reviewer note\n"
    for relative_path, content in mini_case.expected_files:
        assert (mini_case.root / "reproduced" / relative_path).read_bytes() == content


def test_expected_mismatch_is_concise_and_preserves_inputs(mini_case, monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner, "ROOT", mini_case.root)
    monkeypatch.setattr(runner, "load_release_inputs", lambda root: mini_case.release)
    monkeypatch.setattr(runner, "build_outputs", lambda **kwargs: mini_case.outputs)
    expected_path = mini_case.root / "expected/results.json"
    payload = json.loads(expected_path.read_text(encoding="utf-8"))
    payload["dataset.base_snippets.total"] += 1
    expected_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    before_data = tree_hash(mini_case.root / "data")
    before_config = tree_hash(mini_case.root / "config")
    before_expected = tree_hash(mini_case.root / "expected")

    assert runner.main() == 1

    captured = capsys.readouterr()
    assert captured.out.startswith("FAIL: ")
    assert "results.json" in captured.out
    assert "Traceback" not in captured.out + captured.err
    assert tree_hash(mini_case.root / "data") == before_data
    assert tree_hash(mini_case.root / "config") == before_config
    assert tree_hash(mini_case.root / "expected") == before_expected


def test_coverage_failure_is_concise_and_happens_before_writing(mini_case, monkeypatch, capsys) -> None:
    empty_inventory = ManuscriptInventory(schema_version=1, results=())
    release = replace(mini_case.release, inventory=empty_inventory)
    monkeypatch.setattr(runner, "ROOT", mini_case.root)
    monkeypatch.setattr(runner, "load_release_inputs", lambda root: release)
    monkeypatch.setattr(runner, "build_outputs", lambda **kwargs: mini_case.outputs)

    assert runner.main() == 1

    captured = capsys.readouterr()
    assert captured.out.startswith("FAIL: ")
    assert "coverage" in captured.out.lower()
    assert "Traceback" not in captured.out + captured.err
    assert not (mini_case.root / "reproduced/results.json").exists()


def test_real_entrypoint_is_path_stable_and_preserves_inputs(tmp_path: Path) -> None:
    before_config = tree_hash(ROOT / "config")
    before_data = tree_hash(ROOT / "data")
    before_expected = tree_hash(ROOT / "expected")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "analysis/rerun_analysis.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stdout + proc.stderr
    assert (ROOT / "reproduced/results.json").is_file()
    assert len(list((ROOT / "reproduced/tables").glob("table-8-*.csv"))) == 9
    assert tree_hash(ROOT / "config") == before_config
    assert tree_hash(ROOT / "data") == before_data
    assert tree_hash(ROOT / "expected") == before_expected
