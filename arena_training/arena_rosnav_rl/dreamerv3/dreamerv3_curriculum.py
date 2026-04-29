"""DreamerV3-specific curriculum learning adapter.

Mirrors the StableBaselines3 ``StagedTrainCallback`` interface but hooks into
the DreamerV3 training loop via the ``after_eval_fn`` parameter of
``DreamerV3Model.train()``.

Usage (handled automatically by ``DreamerV3Trainer`` when the config contains a
``task.staged.curriculum_definition``):

    curriculum = DreamerV3Curriculum(node=node, train_stages=stages, ...)
    model.train(train_envs=..., eval_envs=..., after_eval_fn=curriculum.after_eval_hook)

After every evaluation phase ``helper.train()`` calls
``after_eval_fn(eval_return)`` with the mean episodic return measured during
that evaluation.  ``DreamerV3Curriculum.after_eval_hook`` stores the metric and
delegates to ``CurriculumBase.check_thresholds_and_update()`` which handles
stage advance / retreat logic as defined in the base class.
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

    * ``get_current_performance()``  — returns the most recent ``eval_return``
      logged after an evaluation phase, or *None* if no evaluation has run yet.
    * ``reset_performance_tracking()`` — resets the tracked metric to *-inf* so
      the next stage starts fresh.

    The bridge between the DreamerV3 training loop and this class is the thin
    ``after_eval_hook(eval_return)`` method.  Pass it as::

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
        """Return the last recorded ``eval_return``, or *None* before first eval."""
        if self._last_eval_return == float("-inf"):
            return None
        return self._last_eval_return

    def reset_performance_tracking(self) -> None:
        """Reset metric so thresholds are evaluated fresh in the new stage."""
        self._last_eval_return = float("-inf")

    # ── DreamerV3 hook ─────────────────────────────────────────────────────

    def after_eval_hook(self, eval_return: float) -> None:
        """Called by ``helper.train()`` after every evaluation phase.

        Stores *eval_return* and immediately checks whether the curriculum
        should advance or retreat.

        Args:
            eval_return: Mean episodic return measured in the latest evaluation.
        """
        self._last_eval_return = eval_return
        _log.info(
            "[Curriculum] stage=%d/%d  eval_return=%.3f  "
            "(advance≥%.2f  retreat≤%.2f)",
            self.curriculum_index,
            self.max_index - 1,
            eval_return,
            self.upper_threshold,
            self.lower_threshold,
        )
        self.check_thresholds_and_update()
