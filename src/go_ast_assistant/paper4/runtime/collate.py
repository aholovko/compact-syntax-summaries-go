from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


PAD_ID = 128004


def response_mask_targets(
    targets: torch.Tensor,
    prompt_lens: Sequence[int],
    pad_token_id: int = PAD_ID,
    ignore_index: int = -100,
) -> torch.Tensor:
    masked = targets.masked_fill(targets == pad_token_id, ignore_index)
    target_count = targets.shape[1]
    for row, prompt_length in enumerate(prompt_lens):
        prompt_targets = max(0, min(prompt_length - 1, target_count))
        masked[row, :prompt_targets] = ignore_index
    return masked


def instruction_collate_fn(
    batch: Sequence[Mapping[str, object]],
    pad_token_id: int = PAD_ID,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [list(item["input_ids"]) for item in batch]  # type: ignore[arg-type]
    prompt_lengths = [int(item["prompt_len"]) for item in batch]  # type: ignore[arg-type]
    maximum_length = max(map(len, sequences))
    if allowed_max_length is not None:
        maximum_length = min(maximum_length, allowed_max_length)

    padded = [
        sequence[:maximum_length] + [pad_token_id] * max(0, maximum_length - len(sequence)) for sequence in sequences
    ]
    inputs = torch.tensor([sequence[:-1] for sequence in padded], dtype=torch.long, device=device)
    targets = torch.tensor([sequence[1:] for sequence in padded], dtype=torch.long, device=device)
    return inputs, response_mask_targets(targets, prompt_lengths, pad_token_id, ignore_index)
