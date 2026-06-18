"""DreamerV3-specific curriculum learning adapter.

Mirrors the StableBaselines3 ``StagedTrainCallback`` interface but hooks into
the DreamerV3 training loop via the ``after_eval_fn`` parameter of
``DreamerV3Model.train()``.

Usage (handled automatically by ``DreamerV3Trainer`` when the config contains a
``task.staged.curriculum_definition``):

    curriculum = DreamerV3Curriculum(node=node, train_stages=stages, ...)
    model.train(train_envs=..., eval_envs=..., after_eval_fn=curriculum.after_eval_hook)

After every evaluation phase ``helper.train()`` calls
``after_eval_fn(metrics)`` with a dict containing ``eval_return`` and
``eval_success_rate``.  ``DreamerV3Curriculum.after_eval_hook`` picks the
right value based on ``threshold_type`` and delegates to
``CurriculumBase.check_thresholds_and_update()``.
"""

import logging
from typing import Any, Dict, Optional

from rclpy.node import Node

from rosnav_rl.utils.curriculum.curriculum_base import CurriculumBase
from arena_training.arena_rosnav_rl.cfg.arena_cfg.task import StagedCfg

_log = logging.getLogger(__name__)

# Maps threshold_type config values to the key present in the metrics dict
# passed by helper.train() → after_eval_fn.
_METRIC_KEY: Dict[str, str] = {
    "rew":  "eval_return",
    "succ": "eval_success_rate",
}


class DreamerV3Curriculum(CurriculumBase):
    """Curriculum learning adapter for the DreamerV3 training pipeline.

    Accepts a ``StagedCfg`` pydantic object directly so callers do not need
    to unpack every field.

    Implements the two abstract methods of ``CurriculumBase``:

    * ``get_current_performance()``  — returns the most recent metric value
      (eval_return or eval_success_rate, depending on ``threshold_type``)
      logged after an evaluation phase, or *None* if no evaluation has run yet.
    * ``reset_performance_tracking()`` — resets the tracked metric to *-inf* so
      the next stage starts fresh.

    The bridge between the DreamerV3 training loop and this class is the thin
    ``after_eval_hook(metrics)`` method.  Pass it as::

        model.train(..., after_eval_fn=curriculum.after_eval_hook)
    """

    def __init__(
        self,
        node: Node,
        staged_cfg: StagedCfg,
        num_envs: int,
        verbose: int = 0,
        *,
        tm_dict: Optional[Dict[str, Any]] = None,
    ):
        """Construct from a ``StagedCfg`` config object.

        Args:
            node:       ROS2 node used for parameter service calls.
            staged_cfg: ``StagedCfg`` instance from ``arena_cfg.task.staged``.
            num_envs:   Number of parallel environments (for parameter broadcast).
            verbose:    Verbosity level (0=WARNING, 1=INFO, 2=DEBUG).
            tm_dict:    Optional ``{tm_robots, tm_obstacles, tm_modules}``.
        """
        # Must be set before CurriculumBase.__init__ because the base
        # constructor calls _apply_curriculum() which may trigger
        # get_current_performance() indirectly through hooks.
        self._last_performance: float = float("-inf")
        train_stages = [
            s.model_dump(by_alias=True, exclude_none=True)
            for s in staged_cfg.curriculum_definition
        ]
        super().__init__(
            node=node,
            train_stages=train_stages,
            threshold_type=staged_cfg.threshold_type,
            upper_threshold=staged_cfg.upper_threshold,
            lower_threshold=staged_cfg.lower_threshold,
            num_envs=num_envs,
            parameter_node_template=staged_cfg.parameter_node_template,
            timeout=staged_cfg.timeout,
            starting_stage=staged_cfg.starting_stage,
            verbose=verbose,
        )

        if tm_dict:
            self._queue_episode(tm_dict)

    # ── CurriculumBase abstract interface ──────────────────────────────────

    def get_current_performance(self) -> Optional[float]:
        """Return the last recorded performance metric, or *None* before first eval."""
        if self._last_performance == float("-inf"):
            return None
        return self._last_performance

    def reset_performance_tracking(self) -> None:
        """Reset metric so thresholds are evaluated fresh in the new stage."""
        self._last_performance = float("-inf")

    # ── DreamerV3 hook ─────────────────────────────────────────────────────

    def after_eval_hook(self, metrics: Dict[str, float]) -> None:
        """Called by ``helper.train()`` after every evaluation phase.

        Picks the right metric from *metrics* based on ``self.threshold_type``
        and immediately checks whether the curriculum should advance or retreat.

        Args:
            metrics: Dict with at least ``eval_return`` and ``eval_success_rate``.
        """
        metric_key = _METRIC_KEY.get(self.threshold_type, "eval_return")
        value = metrics.get(metric_key, float("-inf"))
        self._last_performance = value
        _log.info(
            "[Curriculum] stage=%d/%d  %s=%.3f  (advance≥%.2f  retreat≤%.2f)",
            self.curriculum_index,
            self.max_index - 1,
            metric_key,
            value,
            self.upper_threshold,
            self.lower_threshold,
        )
        self.check_thresholds_and_update()
