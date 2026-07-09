#!/usr/bin/env python3
"""
Arena Training Script - Main entry point for DRL training pipeline.

This script orchestrates the training process for deep reinforcement learning agents
in the Arena-Rosnav environment. It supports multiple RL frameworks including
Stable Baselines3 and DreamerV3.

Usage:
    ros2 run arena_training train_agent --config <config_file.yaml>

    # Or with absolute path:
    python3 train_agent.py --config /path/to/config.yaml
"""

import sys
import logging
import time
from contextlib import contextmanager
from pathlib import Path

# ros2 launch pipes this process's stdout/stderr through a non-tty pipe, which
# forces Python's default block-buffering; without this, log lines written
# shortly before exit can be delayed or lost in the launch log.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# `launch` (pulled in transitively via arena_rclpy_mixins.Async, and by every
# arena_training/rosnav_rl module imported below) calls
# logging.setLoggerClass(LaunchLogger) as an import-time side effect, and
# LaunchLogger.__init__ hardcodes propagate=False. Every logger.getLogger(name)
# call anywhere in THIS process born after that point silently loses all its
# INFO/DEBUG output (only WARNING+ survives, via logging.lastResort). This
# process never runs an actual launch description, so `launch`'s own console
# formatting is irrelevant here — neutralize the mutation before it can fire.
logging.setLoggerClass = lambda cls: None

import torch
import rclpy

# Import arena_training subpackages
from arena_training.arena_rosnav_rl.utils.argsparser import parse_training_args
from arena_training.arena_rosnav_rl.utils.config import load_training_config
from arena_training.arena_rosnav_rl.utils.sim_bootstrap import (
    build_namespace_fn,
    spawn_envs,
    wait_for_simulation,
)
from arena_training.arena_rosnav_rl.trainer import get_trainer
from arena_training.arena_rosnav_rl.social_curriculum import SocialCurriculumCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@contextmanager
def _stage(label: str):
    t0 = time.monotonic()
    try:
        yield
    finally:
        logger.info("%s done in %.1fs", label, time.monotonic() - t0)

# Disable compilation features that conflict with multiprocessing
# Disable dynamo entirely to avoid issues with parallel environments
# torch._dynamo.config.disable = True


def get_config_path(args) -> Path:
    """
    Get the full path to the training configuration file.

    Args:
        args: Parsed command-line arguments containing config filename

    Returns:
        Path: Full path to the configuration file

    Raises:
        FileNotFoundError: If the config file cannot be found

    Notes:
        If the config argument is an absolute path, it is used directly.
        Otherwise, it looks for the config in the arena_bringup package's config directory.
    """
    config_file = args.config

    # Convert to Path object
    config_path = Path(config_file)

    # If absolute path is provided, verify it exists
    if config_path.is_absolute():
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.info(f"Using config file: {config_path}")
        return config_path

    # Try current working directory first
    if config_path.exists():
        logger.info(
            f"Using config file from current directory: {config_path.absolute()}"
        )
        return config_path.absolute()

    # Try to find the config in the arena_training package share
    from ament_index_python.packages import get_package_share_directory

    try:
        arena_training_dir = get_package_share_directory("arena_training")
        package_config_path = Path(arena_training_dir) / "configs" / config_file

        if package_config_path.exists():
            logger.info(f"Using config file from arena_training: {package_config_path}")
            return package_config_path
        else:
            logger.warning(f"Config file not found at {package_config_path}")
    except Exception as e:
        logger.warning(f"Could not locate arena_training package: {e}")

    raise FileNotFoundError(
        f"Config file '{config_file}' not found in:\n"
        f"  - Current directory\n"
        f"  - arena_training package share configs\n"
        f"  - arena_bringup package configs\n"
        f"Please provide an absolute path or ensure the file exists in one of these locations."
    )


def validate_environment():
    """
    Validate the training environment setup.

    Checks:
    - CUDA availability
    - ROS 2 setup
    - Required packages
    """
    logger.info("Validating training environment...")

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("CUDA is not available. Training will use CPU (slower)")

    # Check ROS 2 context
    if not rclpy.ok():
        logger.warning("ROS 2 not properly initialized")
    else:
        logger.info("ROS 2 context initialized")


def main():
    """
    Main entry point for the training script.

    This function:
    1. Parses command-line arguments
    2. Validates environment
    3. Loads the training configuration
    4. Creates the appropriate trainer
    5. Starts the training process

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    overall_t0 = time.monotonic()
    try:
        args, _ = parse_training_args()

        with _stage("validate_environment"):
            validate_environment()

        with _stage("wait_for_simulation"):
            wait_for_simulation(timeout=120.0)

        config_path = get_config_path(args)

        logger.info(f"\n{'='*70}")
        logger.info("  Arena Training Pipeline")
        logger.info(f"{'='*70}")
        logger.info(f"  Configuration: {config_path}")
        logger.info(f"{'='*70}\n")

        with _stage("load_training_config"):
            config = load_training_config(str(config_path))

            if args.robot:
                config.arena_cfg.robot.robot_model = args.robot
                config.arena_cfg.robot.robot_description = None
                config.arena_cfg.robot.model_post_init(None)

        assert config.arena_cfg.general is not None
        n_envs: int = config.arena_cfg.general.n_envs
        per_env_launch_args = [["train_mode:=true", "auto_reset:=false"] for _ in range(n_envs)]

        with _stage(f"spawn_envs (n={n_envs})"):
            env_map = spawn_envs(n_envs, per_env_launch_args)

        logger.info("building trainer for framework=%s", config.agent_config.framework.name)

        namespace_fn = build_namespace_fn(env_map)

        with _stage("get_trainer (agent + envs + model)"):
            trainer = get_trainer(config, namespace_fn=namespace_fn)

        logger.info(
            "full init complete in %.1fs, entering train loop",
            time.monotonic() - overall_t0,
        )
        # Staged gated curriculum for Social-Dreamer (M6.1).
        # Only active when social.curriculum.enabled=true in config.
        _curriculum_cb = None
        _social_cfg = getattr(config.agent_config.framework, "model", None)
        _social_cfg = getattr(_social_cfg, "social", None) if _social_cfg else None
        if _social_cfg and getattr(_social_cfg, "enabled", False):
            _curriculum_cfg = getattr(_social_cfg, "curriculum", None)
            if _curriculum_cfg and getattr(_curriculum_cfg, "enabled", False):
                _curriculum_cb = SocialCurriculumCallback(
                    config=_curriculum_cfg,
                    agent=trainer._model if hasattr(trainer, "_model") else None,
                    logger=logger,
                )
        trainer.train(after_eval_fn=_curriculum_cb)

        logger.info("Training completed successfully!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = 1

    try:
        # Initialize ROS 2 Python client library
        logger.info("Initializing ROS 2...")
        rclpy.init()

        # Run main training function
        exit_code = main()

    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user (Ctrl+C)")
        exit_code = 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        exit_code = 1

    finally:
        # Ensure proper shutdown
        try:
            if rclpy.ok():
                logger.info("Shutting down ROS 2...")
                rclpy.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        logger.info(f"Exiting with code {exit_code}")
        sys.exit(exit_code)
