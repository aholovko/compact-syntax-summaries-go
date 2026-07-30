from __future__ import annotations

import torch

from go_ast_assistant.paper4.runtime.collate import PAD_ID, instruction_collate_fn, response_mask_targets


def test_pad_id_is_the_official_finetune_right_pad_token() -> None:
    assert PAD_ID == 128004


def test_response_mask_targets_masks_prompt_and_padding_only() -> None:
    targets = torch.tensor(
        [
            [11, 12, 20, 21],
            [11, 30, 128004, 128004],
        ]
    )

    masked = response_mask_targets(targets, prompt_lens=[3, 2])

    assert masked.tolist() == [
        [-100, -100, 20, 21],
        [-100, 30, -100, -100],
    ]


def test_response_mask_targets_honors_nondefault_pad_and_ignore_ids() -> None:
    targets = torch.tensor([[11, 12, 20, 0]])

    masked = response_mask_targets(targets, prompt_lens=[3], pad_token_id=0, ignore_index=-7)

    assert masked.tolist() == [[-7, -7, 20, -7]]


def test_instruction_collate_keeps_response_and_final_eot_targets() -> None:
    batch = [
        {"input_ids": [10, 11, 12, 20, 128009], "prompt_len": 3},
        {"input_ids": [10, 11, 30], "prompt_len": 2},
    ]

    inputs, targets = instruction_collate_fn(batch)

    assert inputs.tolist() == [
        [10, 11, 12, 20],
        [10, 11, 30, 128004],
    ]
    assert targets.tolist() == [
        [-100, -100, 20, 128009],
        [-100, 30, -100, -100],
    ]


def test_instruction_collate_applies_the_fixed_length_before_shifting() -> None:
    batch = [{"input_ids": [10, 11, 12, 20, 21], "prompt_len": 3}]

    inputs, targets = instruction_collate_fn(batch, allowed_max_length=4)

    assert inputs.tolist() == [[10, 11, 12]]
    assert targets.tolist() == [[-100, -100, 20]]


def test_instruction_collate_honors_nondefault_pad_and_ignore_ids() -> None:
    batch = [
        {"input_ids": [10, 11, 20, 21], "prompt_len": 2},
        {"input_ids": [10, 30], "prompt_len": 1},
    ]

    inputs, targets = instruction_collate_fn(batch, pad_token_id=0, ignore_index=-7)

    assert inputs.tolist() == [[10, 11, 20], [10, 30, 0]]
    assert targets.tolist() == [[-7, 20, 21], [30, -7, -7]]


def test_instruction_collate_places_tensors_on_the_requested_device() -> None:
    batch = [{"input_ids": [10, 20], "prompt_len": 1}]

    inputs, targets = instruction_collate_fn(batch, device=torch.device("meta"))

    assert inputs.device.type == "meta"
    assert targets.device.type == "meta"
    assert inputs.shape == targets.shape == (1, 1)
