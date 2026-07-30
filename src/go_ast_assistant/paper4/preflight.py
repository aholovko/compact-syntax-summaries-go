from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from go_ast_assistant.paper4.config import ExperimentConfig, FineTunedCondition, load_experiment_config
from go_ast_assistant.paper4.prepared_study import PreparedStudy
from go_ast_assistant.paper4.records import reject_duplicate_json_keys

if TYPE_CHECKING:
    import torch

    from go_ast_assistant.paper4.gocheck.toolchain import ToolchainInfo
    from go_ast_assistant.paper4.runtime.model.llama3_model import Llama3Model
    from go_ast_assistant.paper4.runtime.tokenizer import ChatFormat

_BUNDLE_ROOT = Path(__file__).resolve().parents[3]
_FINE_TUNED_CONDITIONS = ("C0", "C1", "C2", "C2-control")
_DEVICES = ("cuda", "mps", "cpu")
_SIMPLE_SHARD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors$")


@dataclass(frozen=True)
class RunRequest:
    condition: str
    seed: int
    study_data_dir: Path
    model_dir: Path
    output_dir: Path
    config_path: Path = Path("config/experiments.yaml")
    device: str = "cuda"


@dataclass(frozen=True)
class ValidatedRequest:
    config: ExperimentConfig
    condition: FineTunedCondition
    seed: Literal[42, 43, 44]
    profile: Literal["paper"]
    study_data_dir: Path
    model_dir: Path
    output_dir: Path
    device: Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class LocalModelLayout:
    root: Path
    tokenizer_path: Path
    weight_paths: tuple[Path, ...]
    weight_index_path: Path | None


@dataclass(frozen=True)
class RuntimeHooks:
    validate_request: Callable[[RunRequest], ValidatedRequest]
    validate_study: Callable[[Path, FineTunedCondition], PreparedStudy]
    validate_model_layout: Callable[[Path], LocalModelLayout]
    load_tokenizer: Callable[[Path], ChatFormat]
    validate_lengths: Callable[
        [PreparedStudy, ChatFormat, FineTunedCondition, Literal[42, 43, 44], Literal["paper"]],
        None,
    ]
    resolve_toolchain: Callable[[], ToolchainInfo]
    resolve_device: Callable[[Literal["cuda", "mps", "cpu"]], torch.device]
    load_model_tensors: Callable[[LocalModelLayout], Llama3Model]


@dataclass(frozen=True)
class PreflightResult:
    request: ValidatedRequest
    study: PreparedStudy
    tokenizer: ChatFormat
    toolchain: ToolchainInfo
    model: Llama3Model
    device: torch.device


def validate_request(request: RunRequest) -> ValidatedRequest:
    path_fields = (request.config_path, request.study_data_dir, request.model_dir, request.output_dir)
    if any(not isinstance(path, Path) for path in path_fields):
        raise ValueError("config, study, model, and output paths must be pathlib.Path values")
    config_path = request.config_path if request.config_path.is_absolute() else _BUNDLE_ROOT / request.config_path
    config = load_experiment_config(config_path)

    if type(request.condition) is not str or request.condition not in _FINE_TUNED_CONDITIONS:
        raise ValueError(f"unknown fine-tuned condition: {request.condition!r}")
    condition = cast(FineTunedCondition, request.condition)
    condition_config = config.conditions[condition]
    if condition_config.kind != "fine_tuned":
        raise ValueError(f"condition is not fine-tuned: {condition!r}")
    if type(request.seed) is not int or request.seed not in condition_config.seeds:
        raise ValueError(f"seed is not configured for {condition}: {request.seed!r}")
    if type(request.device) is not str or request.device not in _DEVICES:
        raise ValueError(f"unsupported device: {request.device!r}")
    if request.output_dir.exists() or request.output_dir.is_symlink():
        raise ValueError(f"output path must be absent: {request.output_dir}")

    return ValidatedRequest(
        config=config,
        condition=condition,
        seed=cast(Literal[42, 43, 44], request.seed),
        profile="paper",
        study_data_dir=request.study_data_dir,
        model_dir=request.model_dir,
        output_dir=request.output_dir,
        device=cast(Literal["cuda", "mps", "cpu"], request.device),
    )


def _path_is_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _load_weight_index(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read model weight index: {path}") from error
    parsed = reject_duplicate_json_keys(text)
    if not isinstance(parsed, dict) or not set(parsed) <= {"metadata", "weight_map"}:
        raise ValueError("model weight index has unexpected top-level fields")
    if "metadata" in parsed and not isinstance(parsed["metadata"], dict):
        raise ValueError("model weight index metadata must be an object")
    weight_map = parsed.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model weight index requires a nonempty weight_map")
    result: dict[str, str] = {}
    for tensor_name, shard_name in weight_map.items():
        if type(tensor_name) is not str or not tensor_name.strip():
            raise ValueError("model weight index tensor names must be nonempty strings")
        if type(shard_name) is not str or _SIMPLE_SHARD.fullmatch(shard_name) is None:
            raise ValueError("model weight index shard names must be simple .safetensors filenames")
        result[tensor_name] = shard_name
    return result


def validate_model_layout(root: Path) -> LocalModelLayout:
    root = Path(root)
    tokenizer_path = root / "original" / "tokenizer.model"
    monolithic_path = root / "model.safetensors"
    index_path = root / "model.safetensors.index.json"
    if not tokenizer_path.is_file():
        raise ValueError(f"local tokenizer is missing or non-regular: {tokenizer_path}")

    monolithic_present = _path_is_present(monolithic_path)
    index_present = _path_is_present(index_path)
    if monolithic_present and index_present:
        raise ValueError("monolithic and indexed model layouts are mutually exclusive")
    if monolithic_present:
        if not monolithic_path.is_file():
            raise ValueError(f"model weight is non-regular: {monolithic_path}")
        actual_names = {entry.name for entry in root.glob("*.safetensors")}
        if actual_names != {monolithic_path.name}:
            raise ValueError("monolithic model layout contains unexpected .safetensors paths")
        return LocalModelLayout(
            root=root,
            tokenizer_path=tokenizer_path,
            weight_paths=(monolithic_path,),
            weight_index_path=None,
        )
    if not index_present or not index_path.is_file():
        raise ValueError("model directory requires a regular monolithic weight or weight index")

    weight_map = _load_weight_index(index_path)
    shard_names = set(weight_map.values())
    actual_names = {entry.name for entry in root.glob("*.safetensors")}
    if actual_names != shard_names:
        raise ValueError("indexed model layout has missing or extra .safetensors paths")
    weight_paths = tuple(root / name for name in sorted(shard_names))
    if any(not path.is_file() for path in weight_paths):
        raise ValueError("indexed model layout contains a non-regular shard")
    return LocalModelLayout(
        root=root,
        tokenizer_path=tokenizer_path,
        weight_paths=weight_paths,
        weight_index_path=index_path,
    )


def preflight_run(request: RunRequest, hooks: RuntimeHooks) -> PreflightResult:
    validated = hooks.validate_request(request)
    study = hooks.validate_study(validated.study_data_dir, validated.condition)
    layout = hooks.validate_model_layout(validated.model_dir)
    tokenizer = hooks.load_tokenizer(layout.tokenizer_path)
    hooks.validate_lengths(study, tokenizer, validated.condition, validated.seed, validated.profile)
    toolchain = hooks.resolve_toolchain()
    device = hooks.resolve_device(validated.device)
    model = hooks.load_model_tensors(layout)
    return PreflightResult(validated, study, tokenizer, toolchain, model, device)
