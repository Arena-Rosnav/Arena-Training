import pytest

from arena_training.environments.base_env import gate_period

STEP = 0.0999


def test_a_steady_stream_gates_at_its_period_with_margin() -> None:
    stamps = [round(0.1 * k, 3) for k in range(20)]
    assert gate_period(stamps, STEP) == pytest.approx(1.5 * 0.1)


def test_a_stream_faster_than_the_step_gates_at_the_step() -> None:
    stamps = [round(0.02 * k, 3) for k in range(50)]
    assert gate_period(stamps, STEP) == STEP


def test_one_long_gap_from_a_reset_does_not_widen_the_gate() -> None:
    stamps = [round(0.1 * k, 3) for k in range(10)] + [round(30.0 + 0.1 * k, 3) for k in range(10)]
    assert gate_period(stamps, STEP) == pytest.approx(1.5 * 0.1)


def test_jitter_of_one_physics_step_stays_inside_the_gate() -> None:
    stamps, t = [], 0.0
    for k in range(20):
        t += 0.1 if k % 3 else 0.11
        stamps.append(round(t, 3))
    period = gate_period(stamps, STEP)
    assert period is not None and period > 0.11


def test_too_few_stamps_are_not_gated() -> None:
    assert gate_period([1.0, 2.0, 3.0, 4.0], STEP) is None
    assert gate_period([], STEP) is None
