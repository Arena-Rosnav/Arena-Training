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
import threading
from pathlib import Path

import torch
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

# Import arena_training subpackages
from arena_training.arena_rosnav_rl.cfg import TrainingCfg
from arena_training.arena_rosnav_rl.utils.argsparser import parse_training_args
from arena_training.arena_rosnav_rl.utils.config import load_training_config
from arena_training.arena_rosnav_rl.trainer import get_trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Disable compilation features that conflict with multiprocessing
# Disable dynamo entirely to avoid issues with parallel environments
# torch._dynamo.config.disable = True


def wait_for_simulation(timeout: float = 120.0) -> bool:
    """Block until the simulation is fully loaded.

    Waits for the first message on ``/clock`` which Gazebo (and other
    simulators) only starts publishing once the physics engine is ready.

    Args:
        timeout: Maximum seconds to wait before giving up.

    Returns:
        ``True`` if the clock was received, ``False`` on timeout.
    """
    logger.info("Waiting for simulation (listening for /clock)...")
    event = threading.Event()

    node = Node("_wait_for_sim")

    def _cb(msg: Clock):
        event.set()

    sub = node.create_subscription(Clock, "/clock", _cb, 1)

    # Spin in a background thread so the subscription can receive
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    received = event.wait(timeout=timeout)

    executor.shutdown()
    node.destroy_node()

    if received:
        logger.info("Simulation is ready (/clock received).")
    else:
        logger.warning(
            f"Timed out after {timeout}s waiting for /clock. "
            "Proceeding anyway — the simulation may not be fully loaded."
        )
    return received


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
    try:
        # Parse command-line arguments
        args, _ = parse_training_args()

        # Validate environment
        validate_environment()

        # Wait for the simulation to be fully loaded before proceeding
        wait_for_simulation(timeout=120.0)

        # Get the full path to the configuration file
        config_path = get_config_path(args)

        logger.info(f"\n{'='*70}")
        logger.info(f"  Arena Training Pipeline")
        logger.info(f"{'='*70}")
        logger.info(f"  Configuration: {config_path}")
        logger.info(f"{'='*70}\n")

        # Load training configuration from YAML file
        logger.info("Loading training configuration...")
        config = load_training_config(str(config_path))

        # Override robot model if provided via CLI
        if args.robot:
            config.arena_cfg.robot.robot_model = args.robot
            config.arena_cfg.robot.robot_description = None
            config.arena_cfg.robot.model_post_init(None)

        logger.info("Configuration loaded successfully")

        # Create the appropriate trainer based on the framework
        logger.info("Initializing trainer for framework: %s", config.agent_config.framework.name)
        trainer = get_trainer(config)

        # Start training
        logger.info("Starting training process...")
        trainer.train()

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
