from __future__ import annotations

import random

import numpy as np
import torch

from go_ast_assistant.paper4.training.seeding import seed_everything, seeded_generator


def test_seed_everything_replays_python_numpy_and_torch() -> None:
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        seed_everything(42)
        first = (random.random(), np.random.random(3), torch.rand(3))

        seed_everything(42)
        second = (random.random(), np.random.random(3), torch.rand(3))

        assert first[0] == second[0]
        assert np.array_equal(first[1], second[1])
        assert torch.equal(first[2], second[2])
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)


def test_seeded_generators_cover_all_paper_seeds_without_global_state() -> None:
    previous_state = torch.random.get_rng_state().clone()
    try:
        torch.manual_seed(8_675_309)
        state_before = torch.random.get_rng_state().clone()

        generators = {seed: seeded_generator(seed) for seed in (42, 43, 44)}
        state_after_construction = torch.random.get_rng_state().clone()
        draws = {seed: torch.randperm(64, generator=generator) for seed, generator in generators.items()}
        state_after_use = torch.random.get_rng_state().clone()

        assert torch.equal(state_after_construction, state_before)
        assert torch.equal(state_after_use, state_before)
        for seed, draw in draws.items():
            assert torch.equal(draw, torch.randperm(64, generator=seeded_generator(seed)))
        assert not torch.equal(draws[42], draws[43])
        assert not torch.equal(draws[43], draws[44])
        assert torch.equal(torch.random.get_rng_state(), state_before)
    finally:
        torch.random.set_rng_state(previous_state)
