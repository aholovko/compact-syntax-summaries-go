from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .model.kv_cache import KVCache


def generate_batched(
    model: Any,
    prompts: Sequence[Sequence[int]],
    max_new_tokens: int,
    context_size: int,
    eos_id: int = 128009,
) -> list[list[int]]:
    if not prompts:
        return []
    device = next(model.parameters()).device
    lengths = [len(prompt) for prompt in prompts]
    maximum_length = max(lengths)
    for prompt in prompts:
        assert len(prompt) + max_new_tokens <= context_size, "prompt + max_new_tokens exceeds context_size"

    batch_size = len(prompts)
    pad_id = 0
    token_ids = torch.full((batch_size, maximum_length), pad_id, dtype=torch.long, device=device)
    position_ids = torch.zeros((batch_size, maximum_length), dtype=torch.long, device=device)
    real_keys = torch.zeros((batch_size, maximum_length), dtype=torch.bool, device=device)
    for row, prompt in enumerate(prompts):
        length = len(prompt)
        token_ids[row, maximum_length - length :] = torch.tensor(prompt, dtype=torch.long, device=device)
        position_ids[row, maximum_length - length :] = torch.arange(length, device=device)
        real_keys[row, maximum_length - length :] = True

    cache = KVCache(n_layers=model.cfg["n_layers"])
    with torch.no_grad():
        causal = torch.tril(torch.ones(maximum_length, maximum_length, dtype=torch.bool, device=device))
        prefill_mask = (causal.unsqueeze(0) & real_keys.unsqueeze(1)).unsqueeze(1)
        logits = model(
            token_ids,
            cache=cache,
            start_pos=0,
            attn_mask=prefill_mask,
            position_ids=position_ids,
        )[:, -1, :]

        additions: list[list[int]] = [[] for _ in prompts]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        next_positions = torch.tensor(lengths, dtype=torch.long, device=device)
        cache_length = maximum_length
        for step in range(max_new_tokens):
            next_tokens = torch.argmax(logits, dim=-1)
            for row in range(batch_size):
                if not bool(finished[row]):
                    token = int(next_tokens[row])
                    if token == eos_id:
                        finished[row] = True
                    else:
                        additions[row].append(token)
            if bool(torch.all(finished)) or step + 1 == max_new_tokens:
                break

            step_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens).unsqueeze(1)
            key_length = cache_length + 1
            generated_keys = torch.ones((batch_size, key_length - maximum_length), dtype=torch.bool, device=device)
            step_mask = torch.cat((real_keys, generated_keys), dim=1).view(batch_size, 1, 1, key_length)
            logits = model(
                step_tokens,
                cache=cache,
                start_pos=cache_length,
                attn_mask=step_mask,
                position_ids=next_positions.unsqueeze(1),
            )[:, -1, :]
            cache_length += 1
            next_positions += (~finished).long()
    return [list(prompt) + additions[row] for row, prompt in enumerate(prompts)]


def generate_bucketed(
    model: Any,
    prompts: Sequence[Sequence[int]],
    max_new_tokens: int,
    context_size: int,
    eos_id: int = 128009,
    batch_size: int = 16,
    token_cap: int = 32768,
) -> list[list[int]]:
    order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
    results: list[list[int] | None] = [None] * len(prompts)

    def flush(bucket: list[int]) -> None:
        if not bucket:
            return
        generated = generate_batched(
            model,
            [prompts[index] for index in bucket],
            max_new_tokens,
            context_size,
            eos_id=eos_id,
        )
        for index, tokens in zip(bucket, generated, strict=True):
            results[index] = tokens

    bucket: list[int] = []
    for index in order:
        candidate = [*bucket, index]
        padded_length = len(candidate) * max(len(prompts[item]) for item in candidate)
        if bucket and (len(candidate) > batch_size or padded_length > token_cap):
            flush(bucket)
            bucket = [index]
        else:
            bucket = candidate
    flush(bucket)
    assert all(result is not None for result in results)
    return [result for result in results if result is not None]


def strip_assistant_header(
    text: str,
    header_end: str = "assistant<|end_header_id|>\n\n",
) -> str:
    index = text.find(header_end)
    if index < 0:
        return text.strip()
    return text[index + len(header_end) :].strip()
