from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
TITLE = "Compact Syntax Summaries for Low-Resource Go Style Diagnosis and Repair — Replication Package"
PROJECT = "compact-syntax-summaries-go"
VERSION = "1.0.0"
REPOSITORY = "https://github.com/aholovko/compact-syntax-summaries-go"


def _nested_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_keys(child)


def _data_documents(path: Path) -> Iterator[Any]:
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                yield json.loads(line)
    elif path.suffix == ".json":
        yield json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix in {".yaml", ".yml"}:
        yield yaml.safe_load(path.read_text(encoding="utf-8"))


def _release_files() -> Iterator[Path]:
    ignored = {".git", ".venv", "__pycache__", ".pytest_cache", "reproduced"}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and not ignored.intersection(path.relative_to(ROOT).parts):
            yield path


def test_release_tree_contains_no_model_artifacts_or_workstation_paths() -> None:
    artifact_suffixes = {".bin", ".ckpt", ".model", ".pt", ".pth", ".safetensors"}
    text_suffixes = {".cff", ".go", ".json", ".jsonl", ".md", ".mod", ".py", ".toml", ".txt", ".yaml", ".yml"}
    workstation_roots = tuple("/" + root + "/" for root in ("Users", "home", "teamspace"))

    for path in _release_files():
        assert path.suffix not in artifact_suffixes, path
        if path.suffix in text_suffixes or path.name == "LICENSE":
            assert not any(root in path.read_text(encoding="utf-8") for root in workstation_roots), path


def test_project_identity_is_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    root_lock = next(package for package in lock["package"] if package["name"] == PROJECT)

    assert project["project"]["name"] == PROJECT
    assert project["project"]["version"] == VERSION
    assert project["project"]["requires-python"] == ">=3.12,<3.14"
    assert project["tool"]["uv"]["default-groups"] == []
    assert root_lock["version"] == VERSION
    assert root_lock["source"] == {"editable": "."}
    assert citation == {
        "cff-version": "1.2.0",
        "message": "If you use this replication package, please cite it using this metadata.",
        "title": TITLE,
        "type": "software",
        "authors": [{"family-names": "Holovko", "given-names": "Andrii"}],
        "repository-code": REPOSITORY,
        "version": VERSION,
    }


def test_repository_dataset_and_module_identities_are_consistent() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    experiments = yaml.safe_load((ROOT / "config/experiments.yaml").read_text(encoding="utf-8"))
    go_mod = (ROOT / "serializer/go.mod").read_text(encoding="utf-8")

    assert REPOSITORY in readme and REPOSITORY in citation
    dois = re.findall(r"10\.\d{4,9}/[-._;()/:A-Z0-9]*[A-Z0-9]", readme, flags=re.IGNORECASE)
    assert dois == ["10.5281/zenodo.21698768", "10.57967/hf/5304"]
    assert "doi" not in {key.lower() for key in _nested_keys(yaml.safe_load(citation))}
    assert experiments["dataset"] == {
        "identifier": "aholovko/go-critic-style",
        "doi": "10.57967/hf/5304",
        "revision": "7b951fd57d19286153b46ba219aa2cb87fcc4d2b",
    }
    assert experiments["model"] == {"identifier": "meta-llama/Llama-3.2-1B-Instruct"}
    assert go_mod == f"module github.com/aholovko/{PROJECT}/serializer\n\ngo 1.26.4\n"


def test_combined_license_has_the_required_scopes() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "Project-authored software and documentation" in text
    assert text.count("MIT License") == 1
    assert "Copyright (c) 2026 Andrii Holovko" in text
    assert "Permission is hereby granted, free of charge, to any person obtaining a copy" in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR' in text
    assert "Project-generated analysis data, annotations, metadata, and numerical outputs" in text
    assert "Creative Commons Attribution 4.0 International" in text
    assert "https://creativecommons.org/licenses/by/4.0/" in text
    assert "Third-party material" in text
    assert (
        "Neither license grant relicenses third-party model material, reference material, or code. "
        "Those materials remain subject to their own terms."
    ) in text


def test_released_data_has_no_source_generated_or_private_payloads() -> None:
    forbidden_keys = {
        "blob_id",
        "code",
        "completion",
        "explanation",
        "extracted_code",
        "note",
        "outcome_text",
        "path",
        "prompt",
        "raw_output",
        "rejected_labels",
        "repo_name",
        "source",
        "target",
    }
    posix_absolute_path = re.compile(r"(?:^|[\s\"'=])/(?!/)[^\s\"']+")
    windows_absolute_path = re.compile(r"(?:^|[\s\"'=])[A-Z]:[\\/][^\s\"']+", flags=re.IGNORECASE)
    unc_absolute_path = re.compile(r"(?:^|[\s\"'=])\\\\[^\s\"']+")

    for path in sorted((ROOT / "data").rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert posix_absolute_path.search(text) is None, path
        assert windows_absolute_path.search(text) is None, path
        assert unc_absolute_path.search(text) is None, path
        for document in _data_documents(path):
            assert forbidden_keys.isdisjoint(_nested_keys(document)), path
