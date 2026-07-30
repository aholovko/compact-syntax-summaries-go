from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from go_ast_assistant.paper4.records import TaskExample


ComponentScoreFn = Callable[[tuple[TaskExample, ...], tuple[str, ...]], float | None]
_COMPONENT_NAMES = (
    "rule_id_macro_f1",
    "correction_fix_rate",
    "joint_fix_rate",
)


@runtime_checkable
class ComponentScorer(Protocol):
    name: str
    task_type: str
    max_new_tokens: int

    def score_outputs(
        self,
        examples: tuple[TaskExample, ...],
        outputs: tuple[str, ...],
    ) -> float | None: ...


@dataclass(frozen=True)
class RuleIdMacroF1Scorer:
    score_outputs_fn: ComponentScoreFn
    name: ClassVar[str] = "rule_id_macro_f1"
    task_type: ClassVar[str] = "rule_identification"
    max_new_tokens: ClassVar[int] = 64

    def score_outputs(
        self,
        examples: tuple[TaskExample, ...],
        outputs: tuple[str, ...],
    ) -> float | None:
        return self.score_outputs_fn(examples, outputs)


@dataclass(frozen=True)
class CorrectionFixRateScorer:
    score_outputs_fn: ComponentScoreFn
    name: ClassVar[str] = "correction_fix_rate"
    task_type: ClassVar[str] = "correction"
    max_new_tokens: ClassVar[int] = 512

    def score_outputs(
        self,
        examples: tuple[TaskExample, ...],
        outputs: tuple[str, ...],
    ) -> float | None:
        return self.score_outputs_fn(examples, outputs)


@dataclass(frozen=True)
class JointFixRateScorer:
    score_outputs_fn: ComponentScoreFn
    name: ClassVar[str] = "joint_fix_rate"
    task_type: ClassVar[str] = "joint"
    max_new_tokens: ClassVar[int] = 512

    def score_outputs(
        self,
        examples: tuple[TaskExample, ...],
        outputs: tuple[str, ...],
    ) -> float | None:
        return self.score_outputs_fn(examples, outputs)


@dataclass(frozen=True)
class CompositeResult:
    composite: float
    components: dict[str, float]


class ValidationComposite:
    def __init__(self, scorers: Iterable[ComponentScorer]) -> None:
        scorer_tuple = tuple(scorers)
        names = tuple(scorer.name for scorer in scorer_tuple)
        if len(names) != len(_COMPONENT_NAMES) or set(names) != set(_COMPONENT_NAMES):
            raise ValueError(
                f"composite requires exactly one component for each of {_COMPONENT_NAMES}; received {names}"
            )
        by_name = {scorer.name: scorer for scorer in scorer_tuple}
        self._scorers = tuple(by_name[name] for name in _COMPONENT_NAMES)

    def evaluate(
        self,
        model: object,
        val_examples: tuple[TaskExample, ...],
        generate_fn: Callable[[tuple[TaskExample, ...], int], tuple[str, ...]],
    ) -> CompositeResult:
        del model
        components: dict[str, float] = {}
        for scorer in self._scorers:
            examples = tuple(example for example in val_examples if example.task_type == scorer.task_type)
            outputs = tuple(generate_fn(examples, scorer.max_new_tokens))
            if len(outputs) != len(examples):
                raise ValueError(
                    f"generate_fn returned {len(outputs)} outputs for {len(examples)} {scorer.task_type} examples"
                )
            value = scorer.score_outputs(examples, outputs)
            if value is None:
                raise ValueError(f"required component {scorer.name} returned None")
            components[scorer.name] = value
        return CompositeResult(
            composite=sum(components.values()) / len(components),
            components=components,
        )
