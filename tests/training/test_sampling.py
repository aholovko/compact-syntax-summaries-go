from __future__ import annotations

from go_ast_assistant.paper4.training.sampling import build_mixture_stream, length_stratified_aux_sample


def test_paper_stream_has_exact_600_step_effective_batch_and_80_20_order() -> None:
    stream = build_mixture_stream(
        n_main=6_138,
        n_aux=1_536,
        aux_ratio=0.2,
        total=600 * 32,
        seed=42,
    )

    assert len(stream) == 19_200
    assert sum(pool == "main" for pool, _ in stream) == 15_360
    assert sum(pool == "aux" for pool, _ in stream) == 3_840
    assert tuple(pool for pool, _ in stream[:10]) == (
        "main",
        "main",
        "main",
        "main",
        "aux",
        "main",
        "main",
        "main",
        "main",
        "aux",
    )


def test_stream_is_seeded_and_does_not_mutate_pool_lengths() -> None:
    main_lengths = [10, 20, 30]
    aux_lengths = [9, 21]
    original = (tuple(main_lengths), tuple(aux_lengths))

    first = build_mixture_stream(len(main_lengths), len(aux_lengths), 0.2, 100, 42)
    replay = build_mixture_stream(len(main_lengths), len(aux_lengths), 0.2, 100, 42)
    other_seed = build_mixture_stream(len(main_lengths), len(aux_lengths), 0.2, 100, 43)

    assert first == replay
    assert first != other_seed
    assert (tuple(main_lengths), tuple(aux_lengths)) == original
    assert all(0 <= index < len(main_lengths) for pool, index in first if pool == "main")
    assert all(0 <= index < len(aux_lengths) for pool, index in first if pool == "aux")


def test_length_stratified_sampling_is_deterministic_and_preserves_inputs() -> None:
    aux_lengths = [9, 1_000]
    main_lengths = [10, 11, 10]
    original = (tuple(aux_lengths), tuple(main_lengths))

    picks = length_stratified_aux_sample(aux_lengths, main_lengths, k=20, seed=44)

    assert picks == length_stratified_aux_sample(aux_lengths, main_lengths, k=20, seed=44)
    assert picks == [0] * 20
    assert (tuple(aux_lengths), tuple(main_lengths)) == original
