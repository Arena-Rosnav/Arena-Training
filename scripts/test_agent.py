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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
    node.set_parameters(
        [
            rclpy.parameter.Parameter(
                "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True
            )
        ]
    )
    node.start_spinning()

    try:
        # Load observation configuration
        obs_config_path = (
            "/home/le/arena5_ws/src/Arena/arena_training/deps/rosnav_rl/rosnav_rl/"
            "rosnav_rl/observations/observations.yaml"
        )

        logger.info(f"Loading observation config from: {obs_config_path}")
        with open(obs_config_path, "r") as file:
            config = yaml.safe_load(file)

        # Create the observation manager
        observation_manager = ObservationManager.from_config(
            config=config,
            node=node,
            ns="/task_generator_node/jackal",
            simulation_state_container=None,
            wait_for_obs=False,
            enable_synchronization=True,
            # sync_tolerance_seconds=0.3,
        )

        logger.info("Observation manager created successfully")

        # Collect observations
        logger.info("Collecting 10 observation samples...")
        for i in range(10):
            obs = observation_manager.get_observations()

            logger.info(f"Sample {i + 1}: {list(obs.keys())}")

        logger.info("✓ Observation manager test passed")

    except Exception as e:
        logger.error(f"✗ Observation manager test failed: {e}", exc_info=True)
        raise

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_rl_agent_and_env():
    """Test RL_Agent initialization and environment stepping."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing RL_Agent and Environment Stepping")
    logger.info("=" * 70)

    # Import here to ensure ROS workspace is properly set up
    from arena_training.arena_rosnav_rl.cfg import RobotCfg, TaskCfg
    from arena_training.arena_rosnav_rl.node import SupervisorNode
    from arena_training.arena_rosnav_rl.tools.states import get_arena_states
    from arena_training.environments import GazeboEnv
    from arena_training.environments.wrappers.time_sync_wrapper import TimeSyncWrapper

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
        agent_config_path = "/home/le/arena5_ws/src/Arena/arena_bringup/configs/training/sb_training_config.yaml"

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
        node.start_spinning()
        configure_logger(node=node)
        logger.info("✓ Supervisor node created and spinning")

        # Create environment
        logger.info("Creating Gazebo environment...")
        env = GazeboEnv(
            ns="/task_generator_node/jackal",
            node=node,
            space_manager=rl_agent.space_manager,
            reward_function=rl_agent.reward_function,
            simulation_state_container=simulation_state_container,
            max_steps_per_episode=100,
        )
        env = TimeSyncWrapper(env, control_hz=10.0)
        logger.info("✓ Environment created with time sync wrapper")

        # Test environment episodes
        num_episodes = 2
        max_steps_per_episode = 50

        logger.info(f"\nRunning {num_episodes} test episodes...")

        for episode in range(num_episodes):
            logger.info(f"\n--- Episode {episode + 1}/{num_episodes} ---")

            obs = env.reset()
            logger.info(
                f"Episode reset - Initial observation keys: {list(obs[0].keys())}"
            )

            episode_reward = 0.0

            for step in range(max_steps_per_episode):
                # Sample action from agent's action space
                action = rl_agent.space_manager.action_space.sample()

                # Step environment
                obs, reward, done, truncated, info = env.step(action=action)
                episode_reward += reward

                if (step + 1) % 10 == 0:
                    logger.info(
                        f"  Step {step + 1}: reward={reward:.3f}, "
                        f"cumulative={episode_reward:.3f}, done={done}"
                    )

                if done or truncated:
                    logger.info(
                        f"  Episode terminated at step {step + 1}: "
                        f"done={done}, truncated={truncated}"
                    )
                    break

            logger.info(
                f"Episode {episode + 1} complete - Total reward: {episode_reward:.3f}"
            )

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

    # parser = argparse.ArgumentParser(description="Test RL_Agent and environment")
    # parser.add_argument(
    #     "--test",
    #     choices=["observation", "agent", "all"],
    #     default="all",
    #     help="Which test to run (default: all)",
    # )

    # args = parser.parse_args()

    # try:
    #     if args.test in ["observation", "all"]:
    #         test_observation_manager()

    #     if args.test in ["agent", "all"]:
    #         test_rl_agent_and_env()
    # logger.info("Calling test_observation_manager()...")
    # test_observation_manager()
    # logger.info("test_observation_manager() completed successfully")
    # test_rl_agent_and_env()
    test_observation_manager()
    logger.info("All tests completed successfully!")
    return 0


if __name__ == "__main__":
    logger.info("Script starting...")
    exit_code = main()
    logger.info(f"Script exiting with code: {exit_code}")
    exit(exit_code)
