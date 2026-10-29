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

import torch
import rclpy

# Import arena_training subpackages
from arena_training.arena_rosnav_rl.cfg import RobotCfg
from arena_training.arena_rosnav_rl.utils.argsparser import parse_training_args
from arena_training.arena_rosnav_rl.utils.config import get_config_path, load_training_config
from arena_training.arena_rosnav_rl.utils.sim_bootstrap import spawn_envs, wait_for_simulation
from arena_training.arena_rosnav_rl.trainer import get_trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

        config_path = get_config_path(args.config)

        logger.info(f"\n{'='*70}")
        logger.info("  Arena Training Pipeline")
        logger.info(f"{'='*70}")
        logger.info(f"  Configuration: {config_path}")
        logger.info(f"{'='*70}\n")

        with _stage("load_training_config"):
            config = load_training_config(str(config_path))

            if args.robot:
                config.arena_cfg.robot = RobotCfg(robot_model=args.robot)

        assert config.arena_cfg.general is not None
        n_envs: int = config.arena_cfg.general.n_envs
        env_seed = config.arena_cfg.general.env_seed
        extra_args = config.arena_cfg.general.env_launch_args
        robot_launch_arg = config.arena_cfg.robot.launch_arg
        clearance = f"task.episode.spawn.robot_clearance:={config.arena_cfg.general.safety_distance}"
        per_env_launch_args = [["robot.train:=true", "task.episode.auto_reset:=false", robot_launch_arg, clearance, *([f"run_seed:={env_seed}-{i}"] if env_seed is not None else []), *(extra_args[i % len(extra_args)] if extra_args else [])] for i in range(n_envs)]

        with _stage(f"spawn_envs (n={n_envs})"):
            env_map = spawn_envs(n_envs, per_env_launch_args)

        logger.info("building trainer for framework=%s", config.agent_config.framework.name)

        def namespace_fn(idx: int, m: dict[int, str] = env_map) -> str:
            return m[idx]

        with _stage("get_trainer (agent + envs + model)"):
            trainer = get_trainer(config, namespace_fn=namespace_fn)

        logger.info(
            "full init complete in %.1fs, entering train loop",
            time.monotonic() - overall_t0,
        )
        try:
            trainer.train()
        finally:
            trainer.close()

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
