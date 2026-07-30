from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from go_ast_assistant.paper4.runtime.loss import calc_loss_batch, calc_loss_loader


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[:2] == self.fixed_logits.shape[:2]
        return self.fixed_logits


class _TransferSpy:
    def __init__(
        self,
        label: str,
        tensor: torch.Tensor,
        transfers: list[tuple[str, torch.device]],
    ) -> None:
        self.label = label
        self.tensor = tensor
        self.transfers = transfers

    def to(self, device: torch.device, *_args: object, **_kwargs: object) -> torch.Tensor:
        self.transfers.append((self.label, device))
        return self.tensor


def test_calc_loss_batch_uses_only_nonignored_response_targets() -> None:
    inputs = torch.tensor([[10, 11, 12]])
    targets = torch.tensor([[-100, -100, 1]])
    logits = torch.tensor(
        [
            [
                [1000.0, -1000.0, 0.0],
                [-1000.0, 1000.0, 0.0],
                [0.0, 2.0, -1.0],
            ]
        ]
    )

    actual = calc_loss_batch(inputs, targets, _FixedLogitModel(logits), torch.device("cpu"))
    expected = functional.cross_entropy(logits[:, 2, :], torch.tensor([1]))

    torch.testing.assert_close(actual, expected)


def test_calc_loss_batch_moves_inputs_and_targets_to_the_requested_device() -> None:
    inputs = torch.tensor([[10, 11, 12]])
    targets = torch.tensor([[-100, -100, 1]])
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 2.0]]])
    transfers: list[tuple[str, torch.device]] = []
    requested_device = torch.device("cuda")

    actual = calc_loss_batch(
        _TransferSpy("inputs", inputs, transfers),
        _TransferSpy("targets", targets, transfers),
        _FixedLogitModel(logits),
        requested_device,
    )

    assert transfers == [("inputs", requested_device), ("targets", requested_device)]
    expected = functional.cross_entropy(logits[:, 2, :], torch.tensor([1]))
    torch.testing.assert_close(actual, expected)


def test_calc_loss_loader_averages_the_requested_batches_and_handles_empty() -> None:
    inputs = torch.tensor([[10]])
    targets = torch.tensor([[1]])
    logits = torch.tensor([[[0.0, 2.0]]])
    batch = (inputs, targets)
    model = _FixedLogitModel(logits)

    one_batch = calc_loss_loader([batch, batch], model, torch.device("cpu"), num_batches=1)
    expected = functional.cross_entropy(logits.flatten(0, 1), targets.flatten()).item()

    assert one_batch == pytest.approx(expected)
    assert math.isnan(calc_loss_loader([], model, torch.device("cpu")))
