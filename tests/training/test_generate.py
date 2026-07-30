from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from go_ast_assistant.paper4.runtime import generate as generate_module
from go_ast_assistant.paper4.runtime.generate import (
    generate_batched,
    generate_bucketed,
    strip_assistant_header,
)
from go_ast_assistant.paper4.runtime.model.llama3_model import Llama3Model


_TINY_CONFIG = {
    "vocab_size": 64,
    "context_length": 64,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 64,
    "n_kv_groups": 2,
    "rope_base": 10000.0,
    "rope_freq": None,
    "dtype": torch.float32,
}


def _tiny_model(seed: int = 0) -> Llama3Model:
    torch.manual_seed(seed)
    model = Llama3Model(dict(_TINY_CONFIG))
    model.eval()
    return model


def _uncached_reference(
    model: Llama3Model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    eos_id: int,
) -> torch.Tensor:
    result = prompt
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(result[:, -context_size:])
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token == eos_id:
            break
        result = torch.cat((result, next_token), dim=1)
    return result


def _unbatched(model: Llama3Model, prompt: list[int], max_new_tokens: int, eos_id: int) -> list[int]:
    return generate_batched(
        model,
        [prompt],
        max_new_tokens,
        _TINY_CONFIG["context_length"],
        eos_id=eos_id,
    )[0]


def test_cached_generation_matches_uncached_reference() -> None:
    model = _tiny_model()
    for prompt_length in (1, 5, 58):
        prompt = torch.randint(0, _TINY_CONFIG["vocab_size"], (1, prompt_length))
        expected = _uncached_reference(model, prompt.clone(), 6, _TINY_CONFIG["context_length"], eos_id=-1)
        actual = generate_batched(
            model,
            [prompt[0].tolist()],
            6,
            _TINY_CONFIG["context_length"],
            eos_id=-1,
        )
        assert actual == [expected[0].tolist()], f"prompt_length={prompt_length}"


def test_batched_generation_matches_unbatched_for_mixed_prompt_lengths() -> None:
    model = _tiny_model()
    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

    actual = generate_batched(model, prompts, 6, _TINY_CONFIG["context_length"], eos_id=63)
    expected = [_unbatched(model, prompt, 6, eos_id=63) for prompt in prompts]

    assert actual == expected


class _ScriptedBatchModel(torch.nn.Module):
    def __init__(self, scripts: Sequence[Sequence[int]], vocab_size: int) -> None:
        super().__init__()
        self.scripts = scripts
        self.vocab_size = vocab_size
        self.cfg = {"n_layers": 1, "context_length": 32}
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, token_ids: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        logits = torch.full(
            (token_ids.shape[0], token_ids.shape[1], self.vocab_size),
            -1000.0,
            device=token_ids.device,
        )
        for row in range(token_ids.shape[0]):
            script = self.scripts[row]
            token = script[min(self.calls, len(script) - 1)]
            logits[row, -1, token] = 1000.0
        self.calls += 1
        return logits


def test_batched_generation_stops_each_sequence_at_its_own_eos() -> None:
    eos = 9
    model = _ScriptedBatchModel(((4, eos, eos), (5, 6, eos)), vocab_size=10)
    prompts = [[1, 2, 3], [7]]

    actual = generate_batched(model, prompts, 6, context_size=32, eos_id=eos)

    assert actual == [[1, 2, 3, 4], [7, 5, 6]]


def test_bucketed_generation_honors_batch_and_token_caps_and_scatters_to_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [[9, 9, 9, 9], [1], [5, 5, 5], [2, 2]]
    calls: list[list[list[int]]] = []

    def fake_generate_batched(
        _model: object,
        batch: list[list[int]],
        _max_new_tokens: int,
        _context_size: int,
        *,
        eos_id: int,
    ) -> list[list[int]]:
        assert eos_id == 63
        calls.append(batch)
        return [prompt + [100 + len(prompt)] for prompt in batch]

    monkeypatch.setattr(generate_module, "generate_batched", fake_generate_batched)

    actual = generate_bucketed(object(), prompts, 4, 64, eos_id=63, batch_size=2, token_cap=6)

    assert calls == [[[1], [2, 2]], [[5, 5, 5]], [[9, 9, 9, 9]]]
    assert actual == [prompt + [100 + len(prompt)] for prompt in prompts]
    assert all(len(batch) == 1 or len(batch) * max(map(len, batch)) <= 6 for batch in calls)


def test_generation_rejects_a_prompt_plus_budget_beyond_context() -> None:
    model = _tiny_model()
    with pytest.raises(AssertionError, match="exceeds context_size"):
        generate_batched(model, [[1] * 60], 5, context_size=64, eos_id=-1)


def test_strip_assistant_header_returns_only_the_response() -> None:
    raw = "prefix assistant<|end_header_id|>\n\nanswer"
    assert strip_assistant_header(raw) == "answer"
