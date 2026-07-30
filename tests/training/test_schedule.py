from __future__ import annotations

import math

from go_ast_assistant.paper4.training.schedule import cosine_with_warmup_lambda


def test_paper_schedule_has_60_step_warmup_and_fixed_floor() -> None:
    schedule = cosine_with_warmup_lambda(total_steps=600, warmup_steps=60, min_lr_ratio=0.1)

    assert schedule(0) == 0.0
    assert math.isclose(schedule(30), 0.5, abs_tol=1e-12)
    assert math.isclose(schedule(60), 1.0, abs_tol=1e-12)
    assert schedule(300) > schedule(480) > 0.1
    assert math.isclose(schedule(600), 0.1, abs_tol=1e-12)
    assert math.isclose(schedule(900), 0.1, abs_tol=1e-12)


def test_zero_warmup_starts_at_peak() -> None:
    schedule = cosine_with_warmup_lambda(total_steps=600, warmup_steps=0, min_lr_ratio=0.0)

    assert math.isclose(schedule(0), 1.0, abs_tol=1e-12)
    assert math.isclose(schedule(600), 0.0, abs_tol=1e-12)
