from __future__ import annotations

import pytest
import torch

from go_ast_assistant.paper4.runtime.device import resolve_device


def test_explicit_cpu_is_available_without_an_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_device("cpu") == torch.device("cpu")


def test_explicit_cuda_requires_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_explicit_mps_requires_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps") == torch.device("mps")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


@pytest.mark.parametrize("kind", ["auto", "cuda:0", "gpu", ""])
def test_device_resolution_rejects_implicit_or_unknown_kinds(kind: str) -> None:
    with pytest.raises(ValueError):
        resolve_device(kind)  # type: ignore[arg-type]
