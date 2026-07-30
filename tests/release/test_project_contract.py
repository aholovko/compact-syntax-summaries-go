from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_project_dependency_boundary() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["requires-python"] == ">=3.12,<3.14"
    base = "\n".join(project["project"]["dependencies"]).lower()
    training = "\n".join(project["dependency-groups"]["training"]).lower()
    assert "torch" not in base and "tiktoken" not in base and "safetensors" not in base
    assert "torch==2.6.0" in training
    assert "huggingface" not in base + training
    assert project["tool"]["uv"]["build-backend"]["module-name"] == "go_ast_assistant"


def test_exact_condition_and_profile_contract() -> None:
    from analysis.inputs import load_experiment_config

    config = load_experiment_config(ROOT / "config/experiments.yaml")
    assert list(config.conditions) == ["C0", "C1", "C2", "C2-control", "zero-shot-raw", "zero-shot-syntax"]
    expected = {
        "C0": ("c0", [42, 43, 44]),
        "C1": ("c1", [42, 43, 44]),
        "C2": ("c2", [42, 43, 44]),
        "C2-control": ("c2-control", [42, 43, 44]),
        "zero-shot-raw": ("zero-shot-raw", [42]),
        "zero-shot-syntax": ("zero-shot-syntax", [42]),
    }
    assert {name: (row.path, list(row.seeds)) for name, row in config.conditions.items()} == expected
    assert config.dataset.revision == "7b951fd57d19286153b46ba219aa2cb87fcc4d2b"
    assert config.model.identifier == "meta-llama/Llama-3.2-1B-Instruct"
    assert config.profiles["paper"].model_dump(mode="json") == {
        "max_steps": 600,
        "allowed_max_length": 9305,
        "micro_batch_size": 2,
        "grad_accum_steps": 16,
        "effective_batch_size": 32,
        "learning_rate": 2e-5,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "minimum_learning_rate_ratio": 0.1,
        "maximum_gradient_norm": 1.0,
        "checkpoint_every_steps": 120,
        "require_full_composite": True,
        "activation_checkpointing": False,
        "generation_max_new_tokens": {
            "rule_identification": 64,
            "explanation": 512,
            "correction": 512,
            "joint": 512,
        },
    }
