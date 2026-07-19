"""DreamerV3-specific curriculum learning adapter.

Mirrors the StableBaselines3 ``StagedTrainCallback`` interface but hooks into
the DreamerV3 training loop via the ``after_eval_fn`` parameter of
``DreamerV3Model.train()``.

Usage (handled automatically by ``DreamerV3Trainer`` when the config contains a
``task.staged.curriculum_definition``):

    curriculum = DreamerV3Curriculum(node=node, train_stages=stages, ...)
    model.train(train_envs=..., eval_envs=..., after_eval_fn=curriculum.after_eval_hook)

After every evaluation phase ``helper.train()`` calls
``after_eval_fn({"eval_return": ..., "eval_success_rate": ...})`` with the
metrics measured during that evaluation.  ``DreamerV3Curriculum.after_eval_hook``
stores both and delegates to ``CurriculumBase.check_thresholds_and_update()``,
which reads whichever one matches ``threshold_type`` ('rew' or 'succ') as
defined in the base class.
"""

import logging
from typing import Any, Dict, Optional

from rclpy.node import Node

from rosnav_rl.utils.curriculum.curriculum_base import CurriculumBase
from arena_training.arena_rosnav_rl.cfg.arena_cfg.task import StagedCfg

_log = logging.getLogger(__name__)


class DreamerV3Curriculum(CurriculumBase):
    """Curriculum learning adapter for the DreamerV3 training pipeline.

    Accepts a ``StagedCfg`` pydantic object directly so callers do not need
    to unpack every field.

    Implements the two abstract methods of ``CurriculumBase``:

    * ``get_current_performance()``  — returns the most recent metric matching
      ``threshold_type`` ('rew' -> eval_return, 'succ' -> eval_success_rate)
      logged after an evaluation phase, or *None* if no evaluation has run yet.
    * ``reset_performance_tracking()`` — resets the tracked metrics to *-inf* so
      the next stage starts fresh.

    The bridge between the DreamerV3 training loop and this class is the thin
    ``after_eval_hook(eval_metrics)`` method.  Pass it as::

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
        self._last_eval_return: float = float("-inf")
        self._last_eval_success_rate: float = float("-inf")
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
            tm_dict=tm_dict,
        )

    # ── CurriculumBase abstract interface ──────────────────────────────────

    def get_current_performance(self) -> Optional[float]:
        """Return the metric matching ``threshold_type``, or *None* before first eval."""
        value = (
            self._last_eval_success_rate
            if self.threshold_type == "succ"
            else self._last_eval_return
        )
        if value == float("-inf"):
            return None
        return value

    def reset_performance_tracking(self) -> None:
        """Reset metrics so thresholds are evaluated fresh in the new stage."""
        self._last_eval_return = float("-inf")
        self._last_eval_success_rate = float("-inf")

    # ── DreamerV3 hook ─────────────────────────────────────────────────────

    def after_eval_hook(self, eval_metrics: Dict[str, float]) -> None:
        """Called by ``helper.train()`` after every evaluation phase.

        Stores *eval_return* / *eval_success_rate* and immediately checks
        whether the curriculum should advance or retreat.

        Args:
            eval_metrics: Dict with ``eval_return`` and ``eval_success_rate``
                keys, as passed by ``rosnav_rl.model.dreamerv3.helper.train()``.
        """
        self._last_eval_return = eval_metrics["eval_return"]
        self._last_eval_success_rate = eval_metrics["eval_success_rate"]
        _log.info(
            "[Curriculum] stage=%d/%d  eval_return=%.3f  eval_success_rate=%.3f  "
            "(threshold_type=%s  advance≥%.2f  retreat≤%.2f)",
            self.curriculum_index,
            self.max_index - 1,
            self._last_eval_return,
            self._last_eval_success_rate,
            self.threshold_type,
            self.upper_threshold,
            self.lower_threshold,
        )
        self.check_thresholds_and_update()
