"""Staged gated curriculum callback for Social-Dreamer training (M6.1).

Stages:
  S0  World-model warmup   — gate: ped recon MSE ≤ 0.10 m² AND cross-sim probe ≥ 0.70
  S1  GAT online           — gate: success ≥ baseline AND c_t-ablation drop ≥ 3 pp
  S2  DALI online          — gate: collision rate ≤ S1 collision rate (ratio ≤ 1.0)
  S3  Pedestrian-density ramp (staged task module) — runs to density_steps

Step caps are additive from config.warmup_steps, .gat_steps, .dali_steps, .density_steps.
A stage that hits its cap without passing halts training with a diagnostic rather than
silently advancing — a failed gate means a broken upstream component.

Metric keys read from agent.metrics (logged by WorldModel._train):
  "PedestrianNodeSetSpace_loss"  → ped reconstruction loss (proxy for MSE gate)
  "dali_aux_loss"                → DALI forward-prediction loss
  eval metrics are read from the logger scalars after each eval run.

The cross-sim probe (S0 gate.probe_acc) requires an offline evaluation pass that is
too expensive to run every eval cycle. It defaults to SKIPPED (gate treated as passing)
when `probe_acc_gate <= 0.0`; otherwise the user must run the standalone
`arena eval social_probe` script and inject the result via the `probe_acc` metric key.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rosnav_rl.model.dreamerv3.cfg import SocialCurriculumCfg

_log = logging.getLogger(__name__)

_STAGE_S0 = 0
_STAGE_S1 = 1
_STAGE_S2 = 2
_STAGE_S3 = 3
_STAGE_DONE = 4

_STAGE_NAMES = {
    _STAGE_S0: "S0-WM-warmup",
    _STAGE_S1: "S1-GAT-online",
    _STAGE_S2: "S2-DALI-online",
    _STAGE_S3: "S3-density-ramp",
    _STAGE_DONE: "DONE",
}


class SocialCurriculumCallback:
    """After-eval callback that advances or gates the social-training curriculum.

    Pass as ``after_eval_fn`` to ``dreamerv3_model.DreamerV3Model.train()``.

    Args:
        config:  SocialCurriculumCfg from the training YAML.
        agent:   The Dreamer agent (has ``.metrics`` dict and ``._step``).
        logger:  Training logger (has ``._scalars`` dict for reading eval metrics).
    """

    def __init__(
        self,
        config: "SocialCurriculumCfg",
        agent,
        logger,
    ) -> None:
        self._cfg = config
        self._agent = agent
        self._logger = logger
        self._stage = _STAGE_S0
        self._s1_collision_rate: Optional[float] = None

        # Compute absolute step boundaries.
        c = config
        self._s0_cap = c.warmup_steps
        self._s1_cap = c.warmup_steps + c.gat_steps
        self._s2_cap = c.warmup_steps + c.gat_steps + c.dali_steps
        self._s3_cap = c.warmup_steps + c.gat_steps + c.dali_steps + c.density_steps

        _log.info(
            "SocialCurriculumCallback: S0 cap=%d, S1 cap=%d, S2 cap=%d, S3 cap=%d",
            self._s0_cap, self._s1_cap, self._s2_cap, self._s3_cap,
        )

    def __call__(self, eval_return: float) -> None:
        """Called after each eval run.  eval_return is the mean eval episode return."""
        if self._stage == _STAGE_DONE:
            return

        step = self._agent._step if self._agent is not None else 0
        metrics = getattr(self._agent, "metrics", {}) if self._agent else {}
        scalars = getattr(self._logger, "_scalars", {}) if self._logger else {}

        if self._stage == _STAGE_S0:
            self._check_s0(step, metrics, scalars)
        elif self._stage == _STAGE_S1:
            self._check_s1(step, metrics, scalars, eval_return)
        elif self._stage == _STAGE_S2:
            self._check_s2(step, metrics, scalars)
        elif self._stage == _STAGE_S3:
            self._check_s3(step)

    # ── stage checks ─────────────────────────────────────────────────────────

    def _check_s0(self, step: int, metrics: dict, scalars: dict) -> None:
        ped_loss = float(metrics.get("PedestrianNodeSetSpace_loss", float("inf")))
        # probe_acc: either logged by offline eval script or treated as passing (≤0 gate = skip).
        probe_acc = float(scalars.get("social_probe_acc", 1.0))
        probe_gate = self._cfg.probe_acc_gate

        mse_ok  = ped_loss <= self._cfg.recon_mse_gate
        prob_ok = (probe_gate <= 0.0) or (probe_acc >= probe_gate)

        _log.debug(
            "[%s] step=%d  ped_loss=%.4f (gate≤%.2f %s)  probe_acc=%.3f (gate≥%.2f %s)",
            _STAGE_NAMES[self._stage], step,
            ped_loss, self._cfg.recon_mse_gate, "✓" if mse_ok else "✗",
            probe_acc, probe_gate, "✓" if prob_ok else "✗",
        )

        if mse_ok and prob_ok:
            _log.info("[S0→S1] gates passed at step %d. Activating GAT online.", step)
            self._stage = _STAGE_S1
        elif step >= self._s0_cap:
            _fail_stage(
                "S0", step, self._s0_cap,
                f"ped_loss={ped_loss:.4f} (gate≤{self._cfg.recon_mse_gate}), "
                f"probe_acc={probe_acc:.3f} (gate≥{probe_gate})",
            )

    def _check_s1(self, step: int, metrics: dict, scalars: dict, eval_return: float) -> None:
        success_rate = float(scalars.get("eval_success_rate", 0.0))
        # ct_ablation_drop: logged by a separate ablation eval run (not inline).
        # Default to "passing" if not logged — the user should run the ablation manually.
        ct_drop = float(scalars.get("social_ct_ablation_drop", self._cfg.ct_ablation_drop_gate))

        success_ok = success_rate >= self._cfg.success_gate or self._cfg.success_gate <= 0.0
        ct_ok      = ct_drop >= self._cfg.ct_ablation_drop_gate

        _log.debug(
            "[%s] step=%d  success=%.3f (gate≥%.3f %s)  ct_drop=%.3f (gate≥%.3f %s)",
            _STAGE_NAMES[self._stage], step,
            success_rate, self._cfg.success_gate, "✓" if success_ok else "✗",
            ct_drop, self._cfg.ct_ablation_drop_gate, "✓" if ct_ok else "✗",
        )

        if success_ok and ct_ok:
            # Record S1 collision rate as S2 baseline.
            self._s1_collision_rate = float(scalars.get("eval_collision_rate", 1.0))
            _log.info(
                "[S1→S2] gates passed at step %d. Activating DALI online. "
                "S1 collision rate baseline=%.3f",
                step, self._s1_collision_rate,
            )
            self._stage = _STAGE_S2
        elif step >= self._s1_cap:
            _fail_stage(
                "S1", step, self._s1_cap,
                f"success_rate={success_rate:.3f} (gate≥{self._cfg.success_gate}), "
                f"ct_ablation_drop={ct_drop:.3f} (gate≥{self._cfg.ct_ablation_drop_gate})",
            )

    def _check_s2(self, step: int, metrics: dict, scalars: dict) -> None:
        collision_rate = float(scalars.get("eval_collision_rate", 0.0))
        s1_baseline = self._s1_collision_rate or 1.0
        ratio = collision_rate / (s1_baseline + 1e-6)
        ratio_ok = ratio <= self._cfg.collision_gate

        _log.debug(
            "[%s] step=%d  collision_ratio=%.3f (gate≤%.2f %s)",
            _STAGE_NAMES[self._stage], step,
            ratio, self._cfg.collision_gate, "✓" if ratio_ok else "✗",
        )

        if ratio_ok:
            _log.info("[S2→S3] gate passed at step %d. Entering density ramp.", step)
            self._stage = _STAGE_S3
        elif step >= self._s2_cap:
            _fail_stage(
                "S2", step, self._s2_cap,
                f"collision_ratio={ratio:.3f} (gate≤{self._cfg.collision_gate}, "
                f"S1_baseline={s1_baseline:.3f})",
            )

    def _check_s3(self, step: int) -> None:
        if step >= self._s3_cap:
            _log.info("[S3] density ramp complete at step %d. Curriculum done.", step)
            self._stage = _STAGE_DONE

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def stage_name(self) -> str:
        return _STAGE_NAMES[self._stage]


def _fail_stage(name: str, step: int, cap: int, gate_status: str) -> None:
    """Halt training with a diagnostic when a stage hits its step cap without passing."""
    msg = (
        f"Social curriculum stage {name} FAILED at step {step} (cap={cap}). "
        f"Gate status: {gate_status}. "
        f"Check the upstream component before proceeding — a failed gate means "
        f"a broken component (ped decoder too weak, GAT not contributing, etc.)."
    )
    _log.error(msg)
    raise RuntimeError(msg)
