from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import go_ast_assistant.paper4.run_experiment as run_module


FOUR_OUTPUTS = {"manifest.yaml", "records.jsonl", "results.yaml", "selection_trace.json"}


class StageFailure(RuntimeError):
    pass


class _RawTokenizer:
    def __init__(self) -> None:
        self.encode_calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encode_calls.append(text)
        return [101, 102, 103, 104, 105, 106, 107]


class _Tokenizer:
    def __init__(self) -> None:
        self.tok = _RawTokenizer()


class _Model:
    def __init__(self, calls: list[str], output_parent: Path) -> None:
        weight = object()
        self.tok_emb = SimpleNamespace(weight=weight)
        self.out_head = SimpleNamespace(weight=weight)
        self.calls = calls
        self.output_parent = output_parent
        self.transfer_calls = 0
        self.eval_calls = 0
        self.transferred: _Model | None = None

    def to(self, device: object) -> _Model:
        assert not self.output_parent.exists()
        self.calls.append("transfer")
        self.transfer_calls += 1
        assert device == "resolved-device"
        self.transferred = _Model(self.calls, self.output_parent)
        return self.transferred

    def eval(self) -> _Model:
        self.calls.append("model-eval")
        self.eval_calls += 1
        return self


def _argv(tmp_path: Path, *, condition: str = "C1", device: str | None = "cpu") -> list[str]:
    result = [
        "--condition",
        condition,
        "--seed",
        "42",
        "--study-data-dir",
        str(tmp_path / "study"),
        "--model-dir",
        str(tmp_path / "model"),
        "--output-dir",
        str(tmp_path / "published" / "C1" / "seed-42"),
    ]
    if device is not None:
        result.extend(("--device", device))
    return result


def _checkpoint_dirs(output_parent: Path) -> tuple[Path, ...]:
    if not output_parent.exists():
        return ()
    return tuple(output_parent.glob(".paper4-checkpoints-*"))


def _install_success_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    condition: str = "C1",
    fail_at: str | None = None,
) -> tuple[list[str], dict[str, object], _Model]:
    output_dir = tmp_path / "published" / "C1" / "seed-42"
    output_parent = output_dir.parent
    calls: list[str] = []
    captured: dict[str, object] = {}
    model = _Model(calls, output_parent)
    tokenizer = _Tokenizer()
    summaries = {"snippet": object()} if condition != "C0" else None
    study = SimpleNamespace(
        tasks_by_split={"test": ("test-0", "test-1")},
        summaries=summaries,
    )
    validated = SimpleNamespace(
        condition=condition,
        seed=42,
        profile="paper",
        output_dir=output_dir,
        device="cpu",
    )
    toolchain = SimpleNamespace(go_version="go1.26.4", go_critic_version="v0.14.4")
    preflight = SimpleNamespace(
        request=validated,
        study=study,
        tokenizer=tokenizer,
        toolchain=toolchain,
        model=model,
        device="resolved-device",
    )
    hooks = object()
    captured["validated"] = validated
    captured["toolchain"] = toolchain
    captured["tokenizer"] = tokenizer

    def stage(name: str, value: object = None):
        def call(*args: object, **kwargs: object) -> object:
            calls.append(name)
            captured[name] = (args, kwargs)
            if fail_at == name:
                raise StageFailure(name)
            return value

        return call

    def build_hooks() -> object:
        calls.append("build-hooks")
        if fail_at == "build-hooks":
            raise StageFailure("build-hooks")
        return hooks

    def preflight_run(request: object, supplied_hooks: object) -> object:
        calls.append("preflight")
        captured["request"] = request
        assert supplied_hooks is hooks
        if fail_at == "preflight":
            raise StageFailure("preflight")
        return preflight

    def exclusions(supplied_study: object) -> frozenset[str]:
        calls.append("exclusions")
        assert supplied_study is study
        if fail_at == "exclusions":
            raise StageFailure("exclusions")
        return frozenset({"excluded-id"})

    original_to = model.to

    def transfer(device: object) -> _Model:
        if fail_at == "transfer":
            calls.append("transfer")
            assert not output_parent.exists()
            raise StageFailure("transfer")
        transferred = original_to(device)
        captured["transferred_model"] = transferred
        if fail_at == "untie-transfer":
            transferred.out_head.weight = object()
        return transferred

    model.to = transfer  # type: ignore[method-assign]

    def config_factory(request: object) -> object:
        calls.append("training-config")
        assert request is validated
        assert len(_checkpoint_dirs(output_parent)) == 1
        if fail_at == "training-config":
            raise StageFailure("training-config")
        return "training-config"

    def null_store() -> object:
        calls.append("summary-store")
        if fail_at == "summary-store":
            raise StageFailure("summary-store")
        return "null-store"

    def serialized_store(supplied: object, count_tokens: object) -> object:
        calls.append("summary-store")
        assert supplied is summaries
        assert callable(count_tokens)
        assert count_tokens("one two three") == 7  # type: ignore[operator]
        if fail_at == "summary-store":
            raise StageFailure("summary-store")
        return "serialized-store"

    def training(
        config: object,
        supplied_study: object,
        supplied_model: object,
        supplied_tokenizer: object,
        device: object,
        composite: object,
        *,
        checkpoint_dir: Path,
    ) -> object:
        calls.append("training")
        captured["checkpoint_dir"] = checkpoint_dir
        assert config == "training-config"
        assert supplied_study is study
        assert supplied_model is model.transferred
        assert supplied_tokenizer is tokenizer
        assert device == "resolved-device"
        assert composite == "composite"
        assert checkpoint_dir.is_dir()
        assert not tuple(checkpoint_dir.iterdir())
        if fail_at == "training":
            (checkpoint_dir / "partial.pt").write_bytes(b"private")
            raise StageFailure("training")
        best = checkpoint_dir / "best.pt"
        best.write_bytes(b"private checkpoint")
        return SimpleNamespace(best_checkpoint=best)

    def reload(supplied_model: object, path: Path) -> None:
        calls.append("reload")
        assert supplied_model is model.transferred
        assert path.is_file()
        if fail_at == "reload":
            raise StageFailure("reload")
        if fail_at == "retie":
            assert model.transferred is not None
            model.transferred.out_head.weight = object()

    def evaluation(*args: object, **kwargs: object) -> object:
        calls.append("evaluation")
        captured["evaluation"] = (args, kwargs)
        if fail_at == "evaluation":
            raise StageFailure("evaluation")
        return "evaluation-result"

    def completed(**kwargs: object) -> object:
        calls.append("completed-run")
        captured["completed"] = kwargs
        if fail_at == "completed-run":
            raise StageFailure("completed-run")
        return SimpleNamespace(**kwargs)

    def publish(run: object, target: Path) -> None:
        calls.append("publication")
        captured["published_run"] = run
        assert target == output_dir
        assert run.training.best_checkpoint.is_file()
        if fail_at is not None and fail_at.startswith("publication-"):
            raise StageFailure(fail_at)
        target.mkdir()
        for name in FOUR_OUTPUTS:
            (target / name).write_text("released\n", encoding="utf-8")

    clock_values = iter((10.0, 13.25))

    def clock() -> float:
        name = "timer-start" if "timer-start" not in calls else "timer-stop"
        calls.append(name)
        return next(clock_values)

    monkeypatch.setattr(run_module, "build_runtime_hooks", build_hooks)
    monkeypatch.setattr(run_module, "preflight_run", preflight_run)
    monkeypatch.setattr(run_module, "evaluation_exclusion_ids", exclusions)
    monkeypatch.setattr(run_module, "training_config_for", config_factory)
    monkeypatch.setattr(run_module, "NullSummaryStore", null_store)
    monkeypatch.setattr(run_module, "SerializedSummaryStore", serialized_store)
    monkeypatch.setattr(run_module, "build_validation_composite", stage("composite", "composite"))
    monkeypatch.setattr(run_module, "run_training", training)
    monkeypatch.setattr(run_module, "load_best_checkpoint", reload)
    monkeypatch.setattr(run_module, "evaluate", evaluation)
    monkeypatch.setattr(run_module, "CompletedTrainingRun", completed)
    monkeypatch.setattr(run_module, "write_training_run", publish)
    monkeypatch.setattr(run_module.time, "perf_counter", clock)
    return calls, captured, model


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_cli_accepts_only_explicit_supported_devices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    device: str,
) -> None:
    _calls, captured, _model = _install_success_pipeline(monkeypatch, tmp_path)

    assert run_module.main(_argv(tmp_path, device=device)) == 0

    request = captured["request"]
    assert request.device == device  # type: ignore[attr-defined]


def test_cli_defaults_to_cuda_without_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _calls, captured, _model = _install_success_pipeline(monkeypatch, tmp_path)

    assert run_module.main(_argv(tmp_path, device=None)) == 0

    request = captured["request"]
    assert request.device == "cuda"  # type: ignore[attr-defined]


def test_validate_config_exits_before_runtime_hooks_and_prints_one_safe_line(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "private" / "experiments.yaml"
    config_calls: list[Path] = []

    def validate(path: Path) -> object:
        config_calls.append(path)
        return object()

    monkeypatch.setattr(run_module, "load_experiment_config", validate)
    monkeypatch.setattr(
        run_module,
        "build_runtime_hooks",
        lambda: pytest.fail("configuration validation must not build runtime hooks"),
    )
    monkeypatch.setattr(
        run_module,
        "RunRequest",
        lambda **_kwargs: pytest.fail("configuration validation must not construct a run request"),
    )
    monkeypatch.setattr(
        run_module,
        "preflight_run",
        lambda *_args, **_kwargs: pytest.fail("configuration validation must not enter preflight"),
    )

    assert run_module.main(["--validate-config", str(config)]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert "valid" in lines[0].lower()
    assert str(config) not in lines[0]
    assert "model" not in lines[0].lower()
    assert "gpu" not in lines[0].lower()
    assert config_calls == [config]


def test_validate_config_form_rejects_run_arguments(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        run_module.main(["--validate-config", str(tmp_path / "config.yaml"), *_argv(tmp_path)])

    assert error.value.code == 2


def test_success_publishes_reviewer_outputs_and_meaningful_handoffs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _calls, captured, model = _install_success_pipeline(monkeypatch, tmp_path)
    output_dir = tmp_path / "published" / "C1" / "seed-42"

    assert run_module.main(_argv(tmp_path)) == 0

    assert model.transferred is not None
    assert {path.name for path in output_dir.iterdir()} == FOUR_OUTPUTS
    assert not _checkpoint_dirs(output_dir.parent)
    checkpoint_dir = captured["checkpoint_dir"]
    assert not checkpoint_dir.exists()  # type: ignore[union-attr]
    completed = captured["completed"]
    assert completed["request"] is captured["validated"]  # type: ignore[index]
    assert captured["composite"] == ((captured["toolchain"],), {})
    evaluation_args, evaluation_kwargs = captured["evaluation"]  # type: ignore[misc]
    assert evaluation_args == (
        ("test-0", "test-1"),
        "C1",
        42,
        model.transferred,
        captured["tokenizer"],
        "serialized-store",
        captured["toolchain"],
        frozenset({"excluded-id"}),
    )
    assert evaluation_kwargs == {}
    tokenizer = captured["tokenizer"]
    assert tokenizer.tok.encode_calls == ["one two three"]  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("condition", "expected_store"),
    [("C0", "null-store"), ("C1", "serialized-store")],
)
def test_summary_store_is_condition_specific_and_passed_to_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    condition: str,
    expected_store: str,
) -> None:
    _calls, captured, _model = _install_success_pipeline(monkeypatch, tmp_path, condition=condition)

    assert run_module.main(_argv(tmp_path, condition=condition)) == 0

    evaluation_args, evaluation_kwargs = captured["evaluation"]  # type: ignore[misc]
    assert evaluation_args[0] == ("test-0", "test-1")
    assert evaluation_args[1:3] == (condition, 42)
    assert evaluation_args[5] == expected_store
    assert evaluation_args[7] == frozenset({"excluded-id"})
    assert evaluation_kwargs == {}


@pytest.mark.parametrize(
    "fail_at",
    [
        "build-hooks",
        "preflight",
        "exclusions",
        "transfer",
        "training-config",
        "summary-store",
        "composite",
        "training",
        "reload",
        "evaluation",
        "completed-run",
        "publication-validation",
        "publication-serialization",
        "publication-flush",
    ],
)
def test_every_lifecycle_failure_removes_private_checkpoints_and_leaves_no_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_at: str,
) -> None:
    _calls, captured, _model = _install_success_pipeline(monkeypatch, tmp_path, fail_at=fail_at)
    output_dir = tmp_path / "published" / "C1" / "seed-42"

    assert run_module.main(_argv(tmp_path)) == 1

    assert not output_dir.exists()
    assert not _checkpoint_dirs(output_dir.parent)
    if "checkpoint_dir" in captured:
        checkpoint_dir = captured["checkpoint_dir"]
        assert not checkpoint_dir.exists()  # type: ignore[union-attr]
    if fail_at in {"build-hooks", "preflight", "exclusions", "transfer"}:
        assert not output_dir.parent.exists()


def test_tied_weights_are_checked_after_transfer_and_after_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls, _captured, model = _install_success_pipeline(monkeypatch, tmp_path, fail_at="retie")
    output_dir = tmp_path / "published" / "C1" / "seed-42"

    assert run_module.main(_argv(tmp_path)) == 1

    assert calls[-1] == "reload"
    assert model.transferred is not None
    assert model.transferred.eval_calls == 0
    assert not output_dir.exists()
    assert not _checkpoint_dirs(output_dir.parent)


def test_untied_transferred_model_fails_before_output_or_checkpoint_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls, _captured, model = _install_success_pipeline(monkeypatch, tmp_path, fail_at="untie-transfer")
    output_dir = tmp_path / "published" / "C1" / "seed-42"

    assert run_module.main(_argv(tmp_path)) == 1

    assert calls[-1] == "transfer"
    assert model.transfer_calls == 1
    assert not output_dir.parent.exists()
