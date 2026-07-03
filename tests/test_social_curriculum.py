"""Unit tests for SocialCurriculumCallback (M6.1).

Tests the stage-advance logic with synthetic metric streams.
No ROS, no torch, no GPU required.
"""

import sys
import types
import pytest

# Stub pydantic dependency chain to load social_curriculum without the full stack.
sys.path.insert(0, "/home/tuananhroman/arena_ws/src/Arena/arena_training")
from arena_training.arena_rosnav_rl.social_curriculum import (
    SocialCurriculumCallback,
    _STAGE_S0,
    _STAGE_S1,
    _STAGE_S2,
    _STAGE_S3,
    _STAGE_DONE,
)


def _make_cfg(
    warmup_steps=10,
    gat_steps=10,
    dali_steps=10,
    density_steps=10,
    recon_mse_gate=0.10,
    probe_acc_gate=0.70,
    success_gate=0.0,
    ct_ablation_drop_gate=0.03,
    collision_gate=1.0,
):
    """Minimal curriculum config namespace matching SocialCurriculumCfg fields."""
    return types.SimpleNamespace(
        warmup_steps=warmup_steps,
        gat_steps=gat_steps,
        dali_steps=dali_steps,
        density_steps=density_steps,
        recon_mse_gate=recon_mse_gate,
        probe_acc_gate=probe_acc_gate,
        success_gate=success_gate,
        ct_ablation_drop_gate=ct_ablation_drop_gate,
        collision_gate=collision_gate,
    )


class _FakeAgent:
    def __init__(self, step=0, metrics=None):
        self._step = step
        self.metrics = metrics or {}


class _FakeLogger:
    def __init__(self, scalars=None):
        self._scalars = scalars or {}


# ── S0 tests ─────────────────────────────────────────────────────────────────

def test_s0_advances_when_gate_passes():
    """S0 advances to S1 when ped_loss ≤ MSE gate AND probe_acc ≥ probe gate."""
    cfg = _make_cfg(probe_acc_gate=0.70)
    agent = _FakeAgent(step=5, metrics={"PedestrianNodeSetSpace_loss": 0.08})
    logger = _FakeLogger({"social_probe_acc": 0.75})
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb(eval_return=0.5)
    assert cb.stage == _STAGE_S1


def test_s0_blocked_when_mse_too_high():
    """S0 stays at S0 when ped_loss > gate, even if probe_acc passes."""
    cfg = _make_cfg(probe_acc_gate=0.70)
    agent = _FakeAgent(step=5, metrics={"PedestrianNodeSetSpace_loss": 0.25})
    logger = _FakeLogger({"social_probe_acc": 0.90})
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb(eval_return=0.5)
    assert cb.stage == _STAGE_S0


def test_s0_blocked_when_probe_too_low():
    """S0 stays at S0 when probe_acc < gate, even if MSE passes."""
    cfg = _make_cfg(probe_acc_gate=0.70)
    agent = _FakeAgent(step=5, metrics={"PedestrianNodeSetSpace_loss": 0.05})
    logger = _FakeLogger({"social_probe_acc": 0.60})
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb(eval_return=0.5)
    assert cb.stage == _STAGE_S0


def test_s0_probe_gate_skipped_when_zero():
    """probe_acc_gate=0.0 skips the probe check (treats as passing)."""
    cfg = _make_cfg(probe_acc_gate=0.0)
    agent = _FakeAgent(step=5, metrics={"PedestrianNodeSetSpace_loss": 0.05})
    logger = _FakeLogger({})  # no probe_acc logged
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb(eval_return=0.5)
    assert cb.stage == _STAGE_S1


def test_s0_fails_at_cap_when_gate_not_met():
    """S0 raises RuntimeError when step ≥ warmup_steps and gate not met."""
    cfg = _make_cfg(warmup_steps=10, probe_acc_gate=0.70)
    agent = _FakeAgent(step=10, metrics={"PedestrianNodeSetSpace_loss": 0.99})
    logger = _FakeLogger({"social_probe_acc": 0.10})
    cb = SocialCurriculumCallback(cfg, agent, logger)
    with pytest.raises(RuntimeError, match="S0"):
        cb(eval_return=0.5)


# ── S1 tests ─────────────────────────────────────────────────────────────────

def test_s1_advances_when_gates_pass():
    """S1 advances to S2 when success_gate (0.0) and ct_ablation_drop_gate pass."""
    cfg = _make_cfg(success_gate=0.0, ct_ablation_drop_gate=0.03)
    agent = _FakeAgent(step=15, metrics={})
    logger = _FakeLogger({"eval_success_rate": 0.60, "social_ct_ablation_drop": 0.05,
                          "eval_collision_rate": 0.10})
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb._stage = _STAGE_S1  # skip S0
    cb(eval_return=0.6)
    assert cb.stage == _STAGE_S2


def test_s1_records_collision_baseline():
    """S1→S2 advance records S1 collision rate as baseline for S2 gate."""
    cfg = _make_cfg()
    agent = _FakeAgent(step=15, metrics={})
    logger = _FakeLogger({"eval_collision_rate": 0.12, "eval_success_rate": 0.0,
                          "social_ct_ablation_drop": 0.05})
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb._stage = _STAGE_S1
    cb(eval_return=0.0)
    assert cb._s1_collision_rate == pytest.approx(0.12)


def test_s1_fails_at_cap():
    """S1 raises RuntimeError when step ≥ cap and gate not met."""
    cfg = _make_cfg(warmup_steps=5, gat_steps=5, ct_ablation_drop_gate=0.10)
    agent = _FakeAgent(step=10, metrics={})
    logger = _FakeLogger({"social_ct_ablation_drop": 0.01})  # below gate
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb._stage = _STAGE_S1
    with pytest.raises(RuntimeError, match="S1"):
        cb(eval_return=0.5)


# ── S2 tests ─────────────────────────────────────────────────────────────────

def test_s2_advances_when_collision_stable():
    """S2 advances to S3 when collision ratio ≤ 1.0 (not worse than S1)."""
    cfg = _make_cfg(collision_gate=1.0)
    agent = _FakeAgent(step=25, metrics={})
    logger = _FakeLogger({"eval_collision_rate": 0.10})
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb._stage = _STAGE_S2
    cb._s1_collision_rate = 0.12  # S2 collision ≤ S1 → ratio 0.83 ≤ 1.0
    cb(eval_return=0.0)
    assert cb.stage == _STAGE_S3


def test_s2_blocked_when_collision_increases():
    """S2 stays at S2 when collision rate increased beyond S1 baseline."""
    cfg = _make_cfg(collision_gate=1.0)
    agent = _FakeAgent(step=25, metrics={})
    logger = _FakeLogger({"eval_collision_rate": 0.20})
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb._stage = _STAGE_S2
    cb._s1_collision_rate = 0.10  # ratio 2.0 > 1.0
    cb(eval_return=0.0)
    assert cb.stage == _STAGE_S2


# ── S3 / DONE tests ──────────────────────────────────────────────────────────

def test_s3_completes_at_density_cap():
    """S3 transitions to DONE when step ≥ density cap."""
    cfg = _make_cfg(warmup_steps=5, gat_steps=5, dali_steps=5, density_steps=5)
    agent = _FakeAgent(step=20, metrics={})
    logger = _FakeLogger({})
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb._stage = _STAGE_S3
    cb(eval_return=0.0)
    assert cb.stage == _STAGE_DONE


def test_done_stage_is_noop():
    """Callback in DONE stage is a no-op (no error, no stage change)."""
    cfg = _make_cfg()
    agent = _FakeAgent(step=9999, metrics={})
    logger = _FakeLogger({})
    cb = SocialCurriculumCallback(cfg, agent, logger)
    cb._stage = _STAGE_DONE
    cb(eval_return=0.0)  # should not raise
    assert cb.stage == _STAGE_DONE


if __name__ == "__main__":
    tests = [
        test_s0_advances_when_gate_passes,
        test_s0_blocked_when_mse_too_high,
        test_s0_blocked_when_probe_too_low,
        test_s0_probe_gate_skipped_when_zero,
        test_s0_fails_at_cap_when_gate_not_met,
        test_s1_advances_when_gates_pass,
        test_s1_records_collision_baseline,
        test_s1_fails_at_cap,
        test_s2_advances_when_collision_stable,
        test_s2_blocked_when_collision_increases,
        test_s3_completes_at_density_cap,
        test_done_stage_is_noop,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1; print(f"  OK  {t.__name__}")
        except Exception as e:
            failed += 1; print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{'PASSED' if not failed else 'FAILED'} {passed}/{len(tests)}")
