from __future__ import annotations

from typing import Literal

from analysis.inputs import (
    ExperimentConfig,
    FineTunedCondition,
    StrictModel,
    TaskType,
    load_experiment_config,
)

TrainingTaskType = TaskType | Literal["syntax_summary"]

CHECK_NAMES = (
    "assignOp",
    "builtinShadow",
    "captLocal",
    "commentFormatting",
    "elseif",
    "ifElseChain",
    "paramTypeCombine",
    "singleCaseSwitch",
)

EXPECTED_SPLIT_SIZES = {"train": 1_536, "validation": 222, "test": 448}

__all__ = [
    "CHECK_NAMES",
    "EXPECTED_SPLIT_SIZES",
    "ExperimentConfig",
    "FineTunedCondition",
    "StrictModel",
    "TaskType",
    "TrainingTaskType",
    "load_experiment_config",
]
