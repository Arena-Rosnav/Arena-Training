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


# Mapping: LoggingCfg string level → rclpy LoggingSeverity attribute name.
# Kept as strings so that rclpy is imported lazily (not all users have it).
_STDLIB_STR_TO_ROS_SEVERITY: dict = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARN",
    "ERROR": "ERROR",
    "CRITICAL": "FATAL",
}

# rcutils env-var name and its expected token values (DEBUG/INFO/WARN/ERROR/FATAL)
_RCUTILS_ENV_VAR = "RCUTILS_LOGGING_MIN_SEVERITY_LEVEL"


def configure_ros_logging(logging_cfg: Optional[LoggingCfg]) -> None:
    """Apply ROS 2 log levels via ``rclpy.logging`` and env vars.

    Two mechanisms are used so both the current process and any child
    processes (parallel env workers, which fork/spawn after this call) see the
    configured severity:

    1. **Live rclpy call** — ``rclpy.logging.set_logger_level()`` affects rcutils
       loggers that are already initialised in the current process.
    2. **Environment variable** — sets ``RCUTILS_LOGGING_MIN_SEVERITY_LEVEL``
       which rcutils reads when *initialising* a new logging context.  Forked /
       spawned worker processes that call ``rclpy.init()`` internally will
       therefore start at the configured level.

    No-ops gracefully when *rclpy* is not initialised or not installed.

    Args:
        logging_cfg: ``LoggingCfg`` from ``arena_cfg.logging``, or None for
                     a no-op.
    """
    if logging_cfg is None:
        return
    if logging_cfg.ros_level is None and not logging_cfg.ros_overrides:
        return

    ros_level = logging_cfg.ros_level  # e.g. "WARNING"

    # ── 1. Propagate to child processes via environment variable ───────────
    if ros_level is not None:
        import os as _os
        rcutils_token = _STDLIB_STR_TO_ROS_SEVERITY.get(ros_level, "INFO")
        _os.environ[_RCUTILS_ENV_VAR] = rcutils_token

    # ── 2. Apply to currently-initialised rclpy context ───────────────────
    try:
        import rclpy.logging as _ros_log

        def _severity(key: str):
            attr = _STDLIB_STR_TO_ROS_SEVERITY.get(key, "INFO")
            return getattr(_ros_log.LoggingSeverity, attr)

        if ros_level is not None:
            # Empty string "" is the rcutils global/root logger name.
            _ros_log.set_logger_level("", _severity(ros_level))

        for name, lvl_str in logging_cfg.ros_overrides.items():
            _ros_log.set_logger_level(name, _severity(lvl_str))

    except Exception as exc:  # rclpy not available or context not initialised
        logging.getLogger(__name__).debug(
            "configure_ros_logging: skipped (%s)", exc
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
    2. Applies ROS 2 system log levels via ``configure_ros_logging``.
    3. Sets *framework_logger* to the level derived from *verbose* — useful
       for RL-framework helper/banner loggers that live outside ``rosnav_rl``
       but should match the user-visible verbosity intent.
    4. Sets *trainer_logger* to at most that same level, so the trainer itself
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

    # ── ROS 2 system logging ───────────────────────────────────────────────
    configure_ros_logging(logging_cfg)

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
