from __future__ import annotations

import torch


class KVCache:
    def __init__(self, n_layers: int) -> None:
        self.cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers

    def get(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self.cache[layer_index]

    def update(self, layer_index: int, key_values: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.cache[layer_index] = key_values

    def reset(self) -> None:
        for layer_index in range(len(self.cache)):
            self.cache[layer_index] = None
