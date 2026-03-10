"""Trainer-side logging helpers for arena_rosnav_rl.

Responsibilities split from ``rosnav_rl.cfg.logging``:

* ``rosnav_rl`` package owns ``configure_rosnav_rl_logging()`` — it sets stdlib
  log levels for every ``rosnav_rl.*`` namespace and the ``ErrorCollector``
  min-severity threshold.  That function knows nothing about trainers.

* This module owns ``configure_trainer_logging()`` — it calls the above and
  additionally adjusts the *framework* logger (e.g. the DreamerV3 helper
  banner logger) and the *trainer* module logger, both of which live outside
  the ``rosnav_rl`` namespace and are therefore arena_training concerns.

Usage in a trainer::

    from ..tools.log_utils import configure_trainer_logging

    configure_trainer_logging(
        logging_cfg=self.config.arena_cfg.logging,
        verbose=self.config.arena_cfg.general.verbose,
        framework_logger=_DREAMER_HELPER_LOGGER,   # optional
        trainer_logger=logger,                      # optional
    )
"""

from __future__ import annotations

import logging
from typing import Optional

from rosnav_rl.cfg.logging import (
    LoggingCfg,
    VERBOSE_TO_LEVEL,
    configure_rosnav_rl_logging,
)


def configure_trainer_logging(
    logging_cfg: Optional[LoggingCfg],
    verbose: int,
    *,
    framework_logger: Optional[logging.Logger] = None,
    trainer_logger: Optional[logging.Logger] = None,
) -> None:
    """Apply log levels for rosnav_rl namespaces AND the calling trainer.

    1. Delegates to ``configure_rosnav_rl_logging`` for all ``rosnav_rl.*``
       namespaces and the ``ErrorCollector`` threshold.
    2. Sets *framework_logger* to the level derived from *verbose* — useful
       for RL-framework helper/banner loggers that live outside ``rosnav_rl``
       but should match the user-visible verbosity intent.
    3. Sets *trainer_logger* to at most that same level, so the trainer itself
       is never more verbose than the framework it wraps.

    Args:
        logging_cfg:       ``LoggingCfg`` from ``arena_cfg.logging``, or None
                           to fall back to the *verbose* integer mapping.
        verbose:           Integer verbose level (0=WARNING, 1=INFO, 2=DEBUG).
        framework_logger:  Logger for the RL framework helper module (optional).
        trainer_logger:    Root logger of the calling trainer module (optional).
    """
    # ── rosnav_rl namespaces + ErrorCollector ──────────────────────────────
    configure_rosnav_rl_logging(logging_cfg, verbose)

    # ── Trainer / framework loggers (arena_training concern) ───────────────
    framework_level = VERBOSE_TO_LEVEL.get(
        int(verbose), logging.INFO if verbose else logging.WARNING
    )
    if framework_logger is not None:
        framework_logger.setLevel(framework_level)
    if trainer_logger is not None:
        trainer_logger.setLevel(
            min(trainer_logger.level or logging.WARNING, framework_level)
        )
