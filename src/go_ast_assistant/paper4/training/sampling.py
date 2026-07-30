from __future__ import annotations

import math

import torch

from go_ast_assistant.paper4.training.seeding import seeded_generator


def build_mixture_stream(
    n_main: int,
    n_aux: int,
    aux_ratio: float,
    total: int,
    seed: int,
) -> list[tuple[str, int]]:
    """Build the deterministic fixed-budget main/auxiliary stream."""
    generator = seeded_generator(seed)
    buffers: dict[str, list[int]] = {"main": [], "aux": []}
    sizes = {"main": n_main, "aux": n_aux}

    def draw(pool: str) -> int:
        if not buffers[pool]:
            buffers[pool] = torch.randperm(sizes[pool], generator=generator).tolist()
        return buffers[pool].pop()

    use_auxiliary = n_aux > 0 and aux_ratio > 0.0
    stream: list[tuple[str, int]] = []
    for position in range(total):
        wants_auxiliary = use_auxiliary and math.floor((position + 1) * aux_ratio) > math.floor(position * aux_ratio)
        pool = "aux" if wants_auxiliary else "main"
        stream.append((pool, draw(pool)))
    return stream


def length_stratified_aux_sample(
    aux_lengths: list[int] | tuple[int, ...],
    main_lengths: list[int] | tuple[int, ...],
    k: int,
    seed: int,
) -> list[int]:
    """Select C2 auxiliaries nearest seeded main response lengths."""
    generator = seeded_generator(seed)
    main_draw = torch.randint(0, len(main_lengths), (k,), generator=generator).tolist()
    auxiliary_order = range(len(aux_lengths))
    return [
        min(
            auxiliary_order,
            key=lambda auxiliary_index: (
                abs(aux_lengths[auxiliary_index] - main_lengths[main_index]),
                auxiliary_index,
            ),
        )
        for main_index in main_draw
    ]
