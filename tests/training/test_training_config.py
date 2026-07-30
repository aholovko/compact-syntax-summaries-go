from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from analysis.inputs import load_experiment_config
from go_ast_assistant.paper4.preflight import ValidatedRequest
from go_ast_assistant.paper4.training.config import training_config_for


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
CONDITION_FIELDS = {
    "C0": (False, None, 0.0),
    "C1": (True, None, 0.0),
    "C2": (True, "syntax", 0.2),
    "C2-control": (True, "duplicated_main", 0.2),
}


def _request(condition: str, seed: int) -> ValidatedRequest:
    return ValidatedRequest(
        config=load_experiment_config(BUNDLE_ROOT / "config" / "experiments.yaml"),
        condition=condition,  # type: ignore[arg-type]
        seed=seed,  # type: ignore[arg-type]
        profile="paper",
        study_data_dir=Path("ignored-study"),
        model_dir=Path("ignored-model"),
        output_dir=Path("ignored-output"),
        device="cpu",
    )


@pytest.mark.parametrize("condition", tuple(CONDITION_FIELDS))
@pytest.mark.parametrize("seed", (42, 43, 44))
def test_training_config_projects_every_condition_and_seed_exactly(condition: str, seed: int) -> None:
    config = training_config_for(_request(condition, seed))
    use_summary, auxiliary_pool, auxiliary_ratio = CONDITION_FIELDS[condition]
    payload = asdict(config)
    generation_caps = payload.pop("generation_max_new_tokens")

    assert payload == {
        "condition": condition,
        "seed": seed,
        "use_summary": use_summary,
        "auxiliary_pool": auxiliary_pool,
        "auxiliary_ratio": auxiliary_ratio,
        "max_steps": 600,
        "allowed_max_length": 9_305,
        "micro_batch_size": 2,
        "grad_accum_steps": 16,
        "effective_batch_size": 32,
        "learning_rate": 2e-5,
        "betas": (0.9, 0.999),
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "minimum_learning_rate_ratio": 0.1,
        "maximum_gradient_norm": 1.0,
        "checkpoint_every_steps": 120,
        "require_full_composite": True,
        "activation_checkpointing": False,
    }
    assert generation_caps.model_dump(mode="python") == {
        "rule_identification": 64,
        "explanation": 512,
        "correction": 512,
        "joint": 512,
    }


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("seed", True),
        ("seed", 42.0),
        ("max_steps", True),
        ("max_steps", 600.0),
        ("allowed_max_length", 9_305.0),
        ("micro_batch_size", 2.0),
        ("grad_accum_steps", 16.0),
        ("effective_batch_size", 32.0),
        ("checkpoint_every_steps", 120.0),
        ("auxiliary_ratio", 0),
        ("maximum_gradient_norm", 1),
        ("require_full_composite", 1),
        ("activation_checkpointing", 0),
    ],
)
def test_training_config_rejects_numeric_type_drift(field_name: str, invalid: object) -> None:
    config = training_config_for(_request("C0", 42))

    with pytest.raises(ValueError, match=field_name):
        replace(config, **{field_name: invalid})


@pytest.mark.parametrize(
    ("field_name", "drifted_value"),
    [
        ("max_steps", 601),
        ("allowed_max_length", 9_304),
        ("micro_batch_size", 1),
        ("grad_accum_steps", 15),
        ("effective_batch_size", 31),
        ("learning_rate", 3e-5),
        ("betas", (0.8, 0.999)),
        ("epsilon", 2e-8),
        ("weight_decay", 0.2),
        ("warmup_ratio", 0.2),
        ("minimum_learning_rate_ratio", 0.2),
        ("maximum_gradient_norm", 2.0),
        ("checkpoint_every_steps", 100),
        ("require_full_composite", False),
        ("activation_checkpointing", True),
    ],
)
def test_training_config_factory_rejects_every_profile_field_drift(
    field_name: str,
    drifted_value: object,
) -> None:
    request = _request("C0", 42)
    paper = request.config.profiles["paper"].model_copy(update={field_name: drifted_value})
    drifted = request.config.model_copy(update={"profiles": {"paper": paper}})

    with pytest.raises(ValueError, match=field_name):
        training_config_for(replace(request, config=drifted))


@pytest.mark.parametrize(
    ("cap_name", "drifted_value"),
    [
        ("rule_identification", 65),
        ("explanation", 511),
        ("correction", 511),
        ("joint", 511),
    ],
)
def test_training_config_factory_rejects_every_generation_cap_drift(
    cap_name: str,
    drifted_value: int,
) -> None:
    request = _request("C0", 42)
    paper = request.config.profiles["paper"]
    generation_caps = paper.generation_max_new_tokens.model_copy(update={cap_name: drifted_value})
    drifted_paper = paper.model_copy(update={"generation_max_new_tokens": generation_caps})
    drifted = request.config.model_copy(update={"profiles": {"paper": drifted_paper}})

    with pytest.raises(ValueError, match=cap_name):
        training_config_for(replace(request, config=drifted))


@pytest.mark.parametrize(
    ("field_name", "drifted_value"),
    [
        ("use_summary", True),
        ("auxiliary_pool", "syntax"),
        ("auxiliary_ratio", 0.2),
    ],
)
def test_training_config_factory_rejects_every_projected_condition_field_drift(
    field_name: str,
    drifted_value: object,
) -> None:
    request = _request("C0", 42)
    condition = request.config.conditions["C0"].model_copy(update={field_name: drifted_value})
    conditions = dict(request.config.conditions)
    conditions["C0"] = condition
    drifted = request.config.model_copy(update={"conditions": conditions})

    with pytest.raises(ValueError, match=field_name):
        training_config_for(replace(request, config=drifted))
