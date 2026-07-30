from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from analysis.inputs import GenerationMaxNewTokens
from go_ast_assistant.paper4.config import FineTunedCondition
from go_ast_assistant.paper4.preflight import ValidatedRequest


_CONDITION_FIELDS = {
    "C0": (False, None, 0.0),
    "C1": (True, None, 0.0),
    "C2": (True, "syntax", 0.2),
    "C2-control": (True, "duplicated_main", 0.2),
}
_FIXED_INTS = {
    "max_steps": 600,
    "allowed_max_length": 9_305,
    "micro_batch_size": 2,
    "grad_accum_steps": 16,
    "effective_batch_size": 32,
    "checkpoint_every_steps": 120,
}
_FIXED_FLOATS = {
    "learning_rate": 2e-5,
    "epsilon": 1e-8,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "minimum_learning_rate_ratio": 0.1,
    "maximum_gradient_norm": 1.0,
}
_GENERATION_CAPS = {
    "rule_identification": 64,
    "explanation": 512,
    "correction": 512,
    "joint": 512,
}


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    condition: FineTunedCondition
    seed: Literal[42, 43, 44]
    use_summary: bool
    auxiliary_pool: Literal["syntax", "duplicated_main"] | None
    auxiliary_ratio: Literal[0.0, 0.2]
    max_steps: Literal[600]
    allowed_max_length: Literal[9305]
    micro_batch_size: Literal[2]
    grad_accum_steps: Literal[16]
    effective_batch_size: Literal[32]
    learning_rate: Literal[2e-5]
    betas: tuple[Literal[0.9], Literal[0.999]]
    epsilon: Literal[1e-8]
    weight_decay: Literal[0.1]
    warmup_ratio: Literal[0.1]
    minimum_learning_rate_ratio: Literal[0.1]
    maximum_gradient_norm: Literal[1.0]
    checkpoint_every_steps: Literal[120]
    require_full_composite: Literal[True]
    activation_checkpointing: Literal[False]
    generation_max_new_tokens: GenerationMaxNewTokens

    def __post_init__(self) -> None:
        if type(self.condition) is not str or self.condition not in _CONDITION_FIELDS:
            raise ValueError(f"condition must be one fixed fine-tuned condition: {self.condition!r}")
        if type(self.seed) is not int or self.seed not in {42, 43, 44}:
            raise ValueError(f"seed must be one of 42, 43, and 44: {self.seed!r}")
        for field_name, expected in _FIXED_INTS.items():
            value = getattr(self, field_name)
            if type(value) is not int or value != expected:
                raise ValueError(f"{field_name} must be the exact integer {expected}")
        for field_name, expected in _FIXED_FLOATS.items():
            value = getattr(self, field_name)
            if type(value) is not float or value != expected:
                raise ValueError(f"{field_name} must be the exact float {expected!r}")
        if (
            type(self.betas) is not tuple
            or len(self.betas) != 2
            or any(type(value) is not float for value in self.betas)
            or self.betas != (0.9, 0.999)
        ):
            raise ValueError("betas must be the exact floating-point tuple (0.9, 0.999)")
        if type(self.require_full_composite) is not bool or self.require_full_composite is not True:
            raise ValueError("require_full_composite must be exactly true")
        if type(self.activation_checkpointing) is not bool or self.activation_checkpointing is not False:
            raise ValueError("activation_checkpointing must be exactly false")
        expected_summary, expected_pool, expected_ratio = _CONDITION_FIELDS[self.condition]
        if type(self.use_summary) is not bool or self.use_summary is not expected_summary:
            raise ValueError(f"use_summary does not match condition {self.condition}")
        if self.auxiliary_pool != expected_pool:
            raise ValueError(f"auxiliary_pool does not match condition {self.condition}")
        if type(self.auxiliary_ratio) is not float or self.auxiliary_ratio != expected_ratio:
            raise ValueError(f"auxiliary_ratio does not match condition {self.condition}")
        for cap_name, expected in _GENERATION_CAPS.items():
            value = getattr(self.generation_max_new_tokens, cap_name, None)
            if type(value) is not int or value != expected:
                raise ValueError(f"{cap_name} generation cap must be the exact integer {expected}")


def training_config_for(request: ValidatedRequest) -> TrainingConfig:
    profile = request.config.profiles["paper"]
    condition = request.config.conditions[request.condition]
    return TrainingConfig(
        condition=request.condition,
        seed=request.seed,
        use_summary=condition.use_summary,
        auxiliary_pool=condition.auxiliary_pool,
        auxiliary_ratio=condition.auxiliary_ratio,
        max_steps=profile.max_steps,
        allowed_max_length=profile.allowed_max_length,
        micro_batch_size=profile.micro_batch_size,
        grad_accum_steps=profile.grad_accum_steps,
        effective_batch_size=profile.effective_batch_size,
        learning_rate=profile.learning_rate,
        betas=profile.betas,
        epsilon=profile.epsilon,
        weight_decay=profile.weight_decay,
        warmup_ratio=profile.warmup_ratio,
        minimum_learning_rate_ratio=profile.minimum_learning_rate_ratio,
        maximum_gradient_norm=profile.maximum_gradient_norm,
        checkpoint_every_steps=profile.checkpoint_every_steps,
        require_full_composite=profile.require_full_composite,
        activation_checkpointing=profile.activation_checkpointing,
        generation_max_new_tokens=profile.generation_max_new_tokens,
    )
