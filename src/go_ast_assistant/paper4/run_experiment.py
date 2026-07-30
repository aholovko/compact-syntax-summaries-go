"""Standalone configuration validation and one fixed paper-profile retraining attempt."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

from go_ast_assistant.paper4.config import load_experiment_config
from go_ast_assistant.paper4.eval.evaluator import build_validation_composite, evaluate
from go_ast_assistant.paper4.gocheck.toolchain import resolve_toolchain
from go_ast_assistant.paper4.preflight import (
    RunRequest,
    RuntimeHooks,
    preflight_run,
    validate_model_layout,
    validate_request,
)
from go_ast_assistant.paper4.prepared_study import evaluation_exclusion_ids, validate_prepared_study
from go_ast_assistant.paper4.results import CompletedTrainingRun, write_training_run
from go_ast_assistant.paper4.runtime.device import resolve_device
from go_ast_assistant.paper4.runtime.tokenizer import load_local_tokenizer
from go_ast_assistant.paper4.runtime.weights import load_fixed_local_model
from go_ast_assistant.paper4.training.checkpoint import load_best_checkpoint
from go_ast_assistant.paper4.training.conditions import NullSummaryStore
from go_ast_assistant.paper4.training.config import training_config_for
from go_ast_assistant.paper4.training.driver import run_training
from go_ast_assistant.paper4.training.length_budget import validate_lengths
from go_ast_assistant.paper4.training.summary_store import SerializedSummaryStore


_CONDITIONS = ("C0", "C1", "C2", "C2-control")
_DEVICES = ("cuda", "mps", "cpu")


def build_runtime_hooks() -> RuntimeHooks:
    """Bind the eight local-only preflight operations."""
    return RuntimeHooks(
        validate_request=validate_request,
        validate_study=validate_prepared_study,
        validate_model_layout=validate_model_layout,
        load_tokenizer=load_local_tokenizer,
        validate_lengths=validate_lengths,
        resolve_toolchain=resolve_toolchain,
        resolve_device=resolve_device,
        load_model_tensors=load_fixed_local_model,
    )


def _require_tied_model(model: object) -> None:
    try:
        tied = model.out_head.weight is model.tok_emb.weight  # type: ignore[attr-defined]
    except AttributeError as error:
        raise ValueError("model must expose tied tok_emb and out_head weights") from error
    if not tied:
        raise ValueError("model token embedding and output-head weights must remain tied")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Retain no model weights. Run one fixed paper-profile attempt, publish records and metrics only, and "
            "discard the temporary best checkpoint."
        ),
    )
    parser.add_argument("--validate-config", type=Path, metavar="PATH")
    parser.add_argument("--condition", choices=_CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--study-data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=_DEVICES)
    return parser


def _require_form(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Literal["validate", "run"]:
    run_values = (
        args.condition,
        args.seed,
        args.study_data_dir,
        args.model_dir,
        args.output_dir,
        args.device,
    )
    if args.validate_config is not None:
        if any(value is not None for value in run_values):
            parser.error("--validate-config cannot be combined with run arguments")
        return "validate"
    labels = ("--condition", "--seed", "--study-data-dir", "--model-dir", "--output-dir")
    missing = [label for label, value in zip(labels, run_values[:5], strict=True) if value is None]
    if missing:
        parser.error(f"run form requires {', '.join(missing)}")
    return "run"


def _run(args: argparse.Namespace) -> None:
    hooks = build_runtime_hooks()
    started_at = time.perf_counter()
    request = RunRequest(
        condition=args.condition,
        seed=args.seed,
        study_data_dir=args.study_data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device or "cuda",
    )
    preflight = preflight_run(request, hooks)
    excluded_ids = evaluation_exclusion_ids(preflight.study)

    model = preflight.model.to(preflight.device)
    _require_tied_model(model)

    output_parent = preflight.request.output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if output_parent.is_symlink() or not output_parent.is_dir():
        raise ValueError(f"output parent must be a non-symlink directory: {output_parent}")

    with TemporaryDirectory(prefix=".paper4-checkpoints-", dir=output_parent) as temporary:
        checkpoint_dir = Path(temporary)
        config = training_config_for(preflight.request)
        if preflight.request.condition == "C0":
            summaries = NullSummaryStore()
        else:
            if preflight.study.summaries is None:
                raise ValueError(f"condition {preflight.request.condition} requires prepared summaries")
            summaries = SerializedSummaryStore(
                preflight.study.summaries,
                count_tokens=lambda text: len(preflight.tokenizer.tok.encode(text)),
            )
        composite = build_validation_composite(preflight.toolchain)
        training = run_training(
            config,
            preflight.study,
            model,
            preflight.tokenizer,
            preflight.device,
            composite,
            checkpoint_dir=checkpoint_dir,
        )
        load_best_checkpoint(model, training.best_checkpoint)
        _require_tied_model(model)
        model.eval()
        evaluation = evaluate(
            preflight.study.tasks_by_split["test"],
            preflight.request.condition,
            preflight.request.seed,
            model,
            preflight.tokenizer,
            summaries,
            preflight.toolchain,
            excluded_ids,
        )
        wall_clock_total_s = time.perf_counter() - started_at
        completed = CompletedTrainingRun(
            request=preflight.request,
            training=training,
            evaluation=evaluation,
            wall_clock_total_s=wall_clock_total_s,
        )
        write_training_run(completed, preflight.request.output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    form = _require_form(parser, args)
    try:
        if form == "validate":
            load_experiment_config(cast(Path, args.validate_config))
            print("Release experiment configuration is valid.")
            return 0
        _run(args)
    except Exception as error:
        print(f"paper4 run failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module command
    raise SystemExit(main())
