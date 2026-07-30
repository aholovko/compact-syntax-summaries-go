from pathlib import Path

from analysis.inputs import load_metadata, load_runs

ROOT = Path(__file__).resolve().parents[2]


def test_architecture_and_reference_comparison_are_complete() -> None:
    metadata = load_metadata(ROOT / "data/study/analysis_metadata.yaml")
    assert metadata.architecture.model_dump() == {
        "vocabulary_size": 128_256,
        "context_length": 131_072,
        "embedding_dimension": 2_048,
        "query_heads": 32,
        "key_value_heads": 8,
        "query_heads_per_key_value_head": 4,
        "layers": 16,
        "feed_forward_dimension": 8_192,
        "rope_base": 500_000.0,
        "parameter_count": 1_235_814_400,
        "weight_tied": True,
        "compute_dtype": "bfloat16",
        "rmsnorm_dtype": "float32",
    }
    assert metadata.reference_comparison.model_dump() == {
        "prompt_count": 215,
        "scored_position_count": 5_666,
        "tokenizer_exact": True,
        "chat_template_match": False,
        "chat_template_first_divergence": 5,
        "generation_exact": True,
        "generation_first_divergence": None,
        "next_token_agreement_fp32": 0.9811154253441582,
        "disagreements_fp32": 107,
        "systematic_disagreements_fp32": 0,
        "next_token_agreement_bf16": 0.9811154253441582,
        "disagreements_bf16": 107,
        "systematic_disagreements_bf16": 1,
        "margin_threshold": 0.5,
        "near_tie_epsilon": 0.5,
        "maximum_absolute_logit_difference": 1.5130138397216797,
        "mean_absolute_logit_difference": 0.0295319463667828,
        "null_forward_tolerance": None,
        "cached_generation_test": {"status": "passed", "evidence_class": "recovered_current"},
        "sdpa_manual_test": {"status": "passed", "evidence_class": "recovered_current"},
        "loss_masking_test": {"status": "passed", "evidence_class": "recovered_current"},
    }
    assert metadata.reference_comparison.margin_threshold == metadata.reference_comparison.near_tie_epsilon


def test_training_path_retains_only_aggregate_scalars() -> None:
    metadata = load_metadata(ROOT / "data/study/analysis_metadata.yaml")
    assert metadata.training_path.model_dump() == {
        "steps": 50,
        "examples": 64,
        "validation_examples": 32,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "learning_rate": 0.00002,
        "mean_relative_loss_divergence": 0.05377453590358823,
        "maximum_relative_loss_divergence": 0.13970443349753695,
        "final_loss_scratch": 0.00229644775390625,
        "final_loss_reference": 0.002493143081665039,
        "validation_macro_f1_scratch": 0.2890151515151515,
        "validation_macro_f1_reference": 0.25667388167388167,
    }


def test_token_accounting_matches_released_run_totals() -> None:
    metadata = load_metadata(ROOT / "data/study/analysis_metadata.yaml")
    runs = load_runs(ROOT)
    rows_by_run = {}
    for row in metadata.token_accounting:
        rows_by_run.setdefault((row.condition, row.seed), []).append(row)

    assert len(metadata.token_accounting) == 18
    assert set(rows_by_run) == {
        (condition, seed) for condition in ("C0", "C1", "C2", "C2-control") for seed in (42, 43, 44)
    }
    result_by_run = {(result.condition, result.seed): result for result in runs.results}
    for key, rows in rows_by_run.items():
        compute = result_by_run[key].metrics.compute
        assert sum(row.slot_count for row in rows) == compute.examples_seen == 19_200
        assert sum(row.forwarded_tokens for row in rows) == compute.total_tokens
        assert sum(row.supervised_tokens for row in rows) == compute.supervised_tokens
        if key[0] in {"C0", "C1"}:
            assert [(row.pool, row.evidence_class) for row in rows] == [("main", "historical_run")]
        else:
            expected_aux = "syntax_auxiliary" if key[0] == "C2" else "duplicated_main_control"
            assert [(row.pool, row.evidence_class) for row in rows] == [
                ("main", "recovered_current"),
                (expected_aux, "recovered_current"),
            ]


def test_operator_observations_are_qualified_not_billing_records() -> None:
    metadata = load_metadata(ROOT / "data/study/analysis_metadata.yaml")
    observations = {row.id: row for row in metadata.operator_log_observations}
    assert observations["device_memory_c0_gb"].approximate_value == 81
    assert observations["device_memory_syntax_gb"].approximate_value == 72
    assert observations["provider_credits_total"].approximate_value == 205
    assert all(row.evidence_class == "operator_log" and row.approximate for row in observations.values())
