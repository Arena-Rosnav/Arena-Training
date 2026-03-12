#!/usr/bin/env python3
"""
Test script for rosnav_rl.RL_Agent and environment stepping.

This script tests the RL_Agent class initialization, observation collection,
action sampling, and environment stepping in the Arena training framework.

Usage:
    python3 test_agent.py
    # Or from ROS 2:
    ros2 run arena_training test_agent
"""

import logging
from pathlib import Path
import numpy as np
import yaml
import rclpy
import rosnav_rl
from rosnav_rl.observations import ObservationManager
from rosnav_rl.utils.logging import configure_logger


# These imports require full ROS workspace - use only when running via ros2 run
# from arena_training.arena_rosnav_rl.cfg import RobotCfg, TaskCfg
# from arena_training.arena_rosnav_rl.node import SupervisorNode
# from arena_training.arena_rosnav_rl.tools.states import get_arena_states
# from arena_training.environments import GazeboEnv
# from arena_training.environments.wrappers.time_sync_wrapper import TimeSyncWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_observation_manager():
    """Test observation manager creation and data collection."""
    logger.info("=" * 70)
    logger.info("Testing Observation Manager")
    logger.info("=" * 70)

    # Import here to avoid circular dependencies and allow running without full ROS workspaced
    from arena_training.arena_rosnav_rl.node import SupervisorNode

    rclpy.init()

    node = SupervisorNode("observation_test")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])
    node.start_spinning()

    try:
        # Load observation configuration
        obs_config_path = (
            Path(__file__).resolve().parents[1]
            / "deps" / "rosnav_rl" / "rosnav_rl"
            / "rosnav_rl" / "observations" / "observations.yaml"
        )

        logger.info(f"Loading observation config from: {obs_config_path}")
        with open(obs_config_path, "r") as file:
            config = yaml.safe_load(file)

        # Create the observation manager
        observation_manager_1 = ObservationManager.from_config(
            config=config,
            node=node,
            ns="/task_generator_node/env0/jackal",
            simulation_state_container=None,
            wait_for_obs=False,
            enable_synchronization=True,
            # sync_tolerance_seconds=0.3,
        )

        logger.info("Observation manager created successfully")

        # Collect observations
        logger.info("Collecting 10 observation samples...")
        for i in range(10):
            obs1 = observation_manager_1.get_observations()

            logger.info(f"Sample {i + 1}: {list(obs1.keys())}")

        logger.info("✓ Observation manager test passed")

    except Exception as e:
        logger.error(f"✗ Observation manager test failed: {e}", exc_info=True)
        raise

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_multi_environment_observations(num_envs=2, num_samples=5):
    """
    Test observation collection from a variable number of environments.

    Args:
        num_envs: Number of environments to create observation managers for
        num_samples: Number of observation samples to collect from each environment
    """
    logger.info("=" * 70)
    logger.info(f"Testing Multi-Environment Observations ({num_envs} environments)")
    logger.info("=" * 70)

    # Import here to avoid circular dependencies
    from arena_training.arena_rosnav_rl.node import SupervisorNode

    rclpy.init()

    node = SupervisorNode("multi_env_observation_test")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])
    node.start_spinning()

    try:
        # Load observation configuration
        obs_config_path = (
            Path(__file__).resolve().parents[1]
            / "deps" / "rosnav_rl" / "rosnav_rl"
            / "rosnav_rl" / "observations" / "observations.yaml"
        )

        logger.info(f"Loading observation config from: {obs_config_path}")
        with open(obs_config_path, "r") as file:
            config = yaml.safe_load(file)

        # Create observation managers for each environment
        observation_managers = []
        for env_idx in range(num_envs):
            env_namespace = f"/task_generator_node/env{env_idx}/jackal"
            logger.info(f"Creating observation manager for environment {env_idx}: {env_namespace}")

            obs_manager = ObservationManager.from_config(
                config=config,
                node=node,
                ns=env_namespace,
                simulation_state_container=None,
                wait_for_obs=False,
                enable_synchronization=True,
            )
            observation_managers.append(obs_manager)

        logger.info(f"✓ Successfully created {len(observation_managers)} observation managers")

        # Collect observations from all environments
        logger.info(f"\nCollecting {num_samples} observation samples from {num_envs} environments...\n")

        for sample_idx in range(num_samples):
            logger.info(f"{'=' * 60}")
            logger.info(f"Sample {sample_idx + 1}/{num_samples}")
            logger.info(f"{'=' * 60}")

            for env_idx, obs_manager in enumerate(observation_managers):
                try:
                    observations = obs_manager.get_observations()

                    logger.info(f"\nEnvironment {env_idx}:")
                    logger.info(f"  Observation keys: {list(observations.keys())}")

                    # Print details for each observation component
                    for obs_key, obs_value in observations.items():
                        if hasattr(obs_value, "shape"):
                            logger.info(f"    - {obs_key}: shape={obs_value.shape}, dtype={obs_value.dtype}")
                        elif hasattr(obs_value, "__len__"):
                            logger.info(f"    - {obs_key}: length={len(obs_value)}, type={type(obs_value).__name__}")
                        else:
                            logger.info(f"    - {obs_key}: value={obs_value}, type={type(obs_value).__name__}")

                except Exception as e:
                    logger.warning(f"  Failed to get observations from env{env_idx}: {e}")

            logger.info("")  # Empty line for readability

        logger.info("=" * 70)
        logger.info(f"✓ Multi-environment observation test passed")
        logger.info(f"  - Tested {num_envs} environments")
        logger.info(f"  - Collected {num_samples} samples per environment")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"✗ Multi-environment observation test failed: {e}", exc_info=True)
        raise

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_rl_agent_and_env(num_envs=1, use_multiproc=True):
    """
    Test RL_Agent initialization and environment stepping with variable number of parallel environments.

    Args:
        num_envs: Number of parallel environments to create and test (default: 1)
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"Testing RL_Agent and Environment Stepping ({num_envs} environments)")
    logger.info("=" * 70)

    # Import here to ensure ROS workspace is properly set up
    from arena_training.arena_rosnav_rl.cfg import RobotCfg, TaskCfg
    from arena_training.arena_rosnav_rl.node import SupervisorNode
    from arena_training.arena_rosnav_rl.tools.states import get_arena_states
    from arena_training.environments.wrappers.time_sync_wrapper import TimeSyncWrapper
    from arena_training.arena_rosnav_rl.tools.env_utils import make_envs
    from functools import partial
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from arena_training.arena_rosnav_rl.tools.env_utils import DelayedSubprocVecEnv
    from stable_baselines3.common.vec_env import DummyVecEnv

    rclpy.init()

    try:
        # Create simulation state container
        logger.info("Creating simulation state container...")
        simulation_state_container = get_arena_states(
            goal_radius=0.33,
            max_steps=350,
            is_discrete=False,
            safety_distance=1.0,
            robot_cfg=RobotCfg(),
            task_modules_cfg=TaskCfg(),
        )
        logger.info("✓ Simulation state container created")

        # Create agent state container
        logger.info("Creating agent state container...")
        agent_state_cont = simulation_state_container.to_agent_state_container()
        agent_state_cont.action_space.actions = {
            "linear_range": [-2.0, 2.0],
            "angular_range": [-4.0, 4.0],
        }
        logger.info("✓ Agent state container created")

        # Load agent configuration
        logger.info("Loading agent configuration...")
        agent_config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "sb_training_config.yaml"
        )

        with open(agent_config_path, "r") as f:
            training_config = yaml.safe_load(f)

        agent_cfg = rosnav_rl.AgentCfg.model_validate(training_config["agent_cfg"])
        logger.info(f"✓ Agent config loaded: {agent_cfg.name}")

        # Create RL Agent
        logger.info("Creating RL_Agent...")
        rl_agent = rosnav_rl.RL_Agent(
            agent_cfg=agent_cfg,
            agent_state_container=agent_state_cont,
        )
        logger.info("✓ RL_Agent created successfully")
        logger.info(f"  - Observation space: {rl_agent.observation_space}")
        logger.info(f"  - Action space: {rl_agent.action_space}")

        # Create supervisor node
        logger.info("Creating supervisor node...")
        node = SupervisorNode("gazebo_env_test")
        node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])
        node.start_spinning()
        configure_logger(node=node)
        logger.info("✓ Supervisor node created and spinning")

        # Create environments using make_envs
        logger.info(f"Creating {num_envs} parallel environment(s) using make_envs...")

        # Define namespace function for multi-environment support
        # if num_envs == 1:
        #     namespace_fn = lambda idx: "/task_generator_node/jackal"
        # else:
        namespace_fn = lambda idx: f"/task_generator_node/env{idx}/jackal"

        env_fncs = make_envs(
            node=node if not use_multiproc else None,  # Pass node only if not using multiprocessing
            rl_agent=rl_agent,
            simulation_state_container=simulation_state_container,
            n_envs=num_envs,
            max_steps=50,
            init_env_by_call=False,  # Must init by call for multiprocessing to avoid pickling issues
            namespace_fn=namespace_fn,
            wrappers=[partial(TimeSyncWrapper, control_hz=1.0)],
        )

        if use_multiproc:

            envs = DelayedSubprocVecEnv(env_fncs, start_method="forkserver")
        else:
            envs = DummyVecEnv(env_fncs)

        # Test environment episodes
        num_episodes = 10
        max_steps_per_episode = 50

        logger.info(f"✓ {envs.num_envs} environment(s) created with time sync wrapper")
        logger.info(f"\nRunning {num_episodes} test episodes on {num_envs} environment(s)...")

        for episode in range(num_episodes):
            # Reset all environments at once (VecEnv batch API)
            observations = envs.reset()
            logger.info(f"  All envs reset - observation type: {type(observations)}")

            logger.info(f"{'=' * 70}")
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"{'=' * 70}")
            # Track rewards for each environment
            episode_rewards = np.zeros(num_envs)
            env_dones = [False] * num_envs

            for step in range(max_steps_per_episode):
                # Sample actions for all environments at once
                actions = np.array(
                    [
                        rl_agent.space_manager.action_space.sample()
                        for _ in range(num_envs)
                    ]
                )

                # Step all environments at once (VecEnv batch API)
                # dones includes both terminated and truncated (SB3 VecEnv collapses them)
                observations, rewards, dones, infos = envs.step(actions)
                episode_rewards += rewards

                for env_idx, (done, reward) in enumerate(zip(dones, rewards)):
                    if done and not env_dones[env_idx]:
                        env_dones[env_idx] = True
                        # VecEnv auto-resets on done; terminal info is in infos[env_idx]
                        truncated = infos[env_idx].get("TimeLimit.truncated", False)
                        logger.info(
                            f"  Env {env_idx} terminated at step {step + 1}: "
                            f"done={done}, truncated={truncated}, total_reward={episode_rewards[env_idx]:.3f}"
                        )

                # Log progress every 10 steps
                if (step + 1) % 10 == 0:
                    active_envs = sum(1 for done in env_dones if not done)
                    total_reward = float(episode_rewards.sum())
                    logger.info(
                        f"  Step {step + 1}: active_envs={active_envs}/{num_envs}, total_reward={total_reward:.3f}"
                    )

                # Stop if all environments are done
                if all(env_dones):
                    logger.info(f"  All environments completed at step {step + 1}")
                    break

            # Log episode summary
            logger.info(f"\nEpisode {episode + 1} complete:")
            for env_idx in range(num_envs):
                logger.info(f"  Env {env_idx} total reward: {episode_rewards[env_idx]:.3f}")
            logger.info(f"  Average reward: {float(episode_rewards.mean()):.3f}")

        logger.info("\n✓ All tests passed successfully!")

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}", exc_info=True)
        raise

    finally:
        try:
            if "node" in locals():
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main test runner."""
    logger.info("Starting main test runner...")
    # import argparse

    # parser = argparse.ArgumentParser(description="Test RL_Agent and environment observations")
    # parser.add_argument(
    #     "--test",
    #     choices=["observation", "multi-env", "agent", "all"],
    #     default="multi-env",
    #     help="Which test to run (default: multi-env)",
    # )
    # parser.add_argument(
    #     "--num-envs",
    #     type=int,
    #     default=2,
    #     help="Number of environments to test (default: 2)",
    # )
    # parser.add_argument(
    #     "--num-samples",
    #     type=int,
    #     default=5,
    #     help="Number of observation samples to collect (default: 5)",
    # )

    # args = parser.parse_args()

    try:
        # if args.test in ["observation", "all"]:
        #     test_observation_manager()

        # if args.test in ["multi-env", "all"]:
        # test_observation_manager()

        # if args.test in ["agent", "all"]:
        test_rl_agent_and_env(4)

        # logger.info("All tests completed successfully!")
        # return 0

    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    logger.info("Script starting...")
    exit_code = main()
    logger.info(f"Script exiting with code: {exit_code}")
    exit(exit_code)
