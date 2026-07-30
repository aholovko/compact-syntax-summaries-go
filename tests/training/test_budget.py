from __future__ import annotations

import pytest
import torch

import go_ast_assistant.paper4.training.budget as budget_module
from go_ast_assistant.paper4.training.budget import BudgetMeter, token_budget_guard


@pytest.mark.parametrize("device_type", ["cpu", "mps"])
def test_non_cuda_meter_never_touches_cuda_memory(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda: pytest.fail("non-CUDA meter reset CUDA memory stats"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda: pytest.fail("non-CUDA meter read CUDA memory stats"),
    )
    times = iter((10.0, 12.5))
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: next(times))

    meter = BudgetMeter(torch.device(device_type))
    meter.start()
    meter.add_micro_batch(n_main=2, n_aux=0, n_real_tokens=20, n_supervised_tokens=8)
    meter.add_micro_batch(n_main=0, n_aux=2, n_real_tokens=18, n_supervised_tokens=6)
    meter.add_step()
    meter.stop()

    assert meter.report() == {
        "main_examples": 2,
        "aux_examples": 2,
        "tokens_processed": 38,
        "supervised_tokens": 14,
        "optimizer_steps": 1,
        "wall_clock_s": 2.5,
        "peak_gpu_mem_gib": None,
    }


def test_cuda_meter_resets_and_reads_only_its_resolved_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, torch.device | None]] = []
    device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda target=None: calls.append(("reset", target)))
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda target=None: calls.append(("read", target)) or 3 * 1024**3,
    )

    meter = BudgetMeter(device)
    meter.start()
    report = meter.report()

    assert calls == [("reset", device), ("read", device)]
    assert report["peak_gpu_mem_gib"] == 3.0


def test_token_budget_guard_uses_c1_only_for_rq2_conditions() -> None:
    guarded = token_budget_guard(
        {"C0": 800, "C1": 1_000, "C2": 1_051, "C2-control": 950},
        reference="C1",
        tol=0.05,
    )

    assert guarded["C0"]["guarded"] is False
    assert guarded["C1"]["guarded"] is False
    assert guarded["C2"] == {"delta": pytest.approx(0.051), "guarded": True, "exceeds": True}
    assert guarded["C2-control"] == {"delta": pytest.approx(-0.05), "guarded": True, "exceeds": False}


def test_token_budget_guard_requires_reference_when_a_guarded_condition_is_present() -> None:
    with pytest.raises(ValueError, match="reference.*absent"):
        token_budget_guard({"C2": 100})

    assert token_budget_guard({"C0": 100})["C0"]["guarded"] is False


@pytest.mark.parametrize("reference_tokens", (0, -1))
def test_token_budget_guard_rejects_nonpositive_guard_reference(reference_tokens: int) -> None:
    with pytest.raises(ValueError, match="reference.*positive"):
        token_budget_guard({"C1": reference_tokens, "C2": 100})
