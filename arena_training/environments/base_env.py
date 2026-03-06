import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist

from rosnav_rl.observations.factory.factory import (
    create_observation_manager_from_config,
)
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.spaces import BaseSpaceManager
from rosnav_rl.states import SimulationStateContainer
from rosnav_rl.utils.rostopic import Namespace
from rosnav_rl.utils.type_aliases import EncodedObservationDict, ObservationDict
from std_srvs.srv import Empty as EmptySrv

from arena_training.arena_rosnav_rl.node import SupervisorNode
from arena_training.arena_rosnav_rl.utils.envs import (
    determine_termination,
    get_twist_from_action,
)
from arena_training.arena_rosnav_rl.utils.type_alias.observation import InformationDict

from rosnav_rl.utils.logging import flush_errors_decorator


class ArenaBaseEnv(ABC, gymnasium.Env):
    """Abstract base class for Arena reinforcement learning environments.

    This class provides a foundational structure for creating Gymnasium-compliant
    environments that interact with a ROS2-based simulation.  It handles
    observation collection, reward computation, and episode management.

    In **train_mode** (default) the environment publishes velocity commands
    directly to the ``cmd_vel`` topic, completely bypassing the nav2
    controller_server.  This decouples the RL training loop from the nav2
    planning/control pipeline so that planner failures, ``follow_path``
    aborts, or bt_navigator restarts never stall training.

    Attributes:
        node (SupervisorNode): The ROS2 node used for communication.
        ns (Namespace): The ROS2 namespace for the agent.
        action_space (gymnasium.spaces.Box): The action space, defined by the space manager.
        observation_space (gymnasium.spaces.Dict): The observation space, defined by the space manager.
        is_train_mode (bool): Flag indicating if the environment is in training mode.
        observation_collector (ObservationManager): Manages the collection of observations.
        metadata (dict): Standard Gymnasium metadata.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ns: Union[str, Namespace],
        space_manager: Union[BaseSpaceManager, Dict[str, Any]],
        reward_function: Union[RewardFunction, Dict[str, Any]],
        node: Optional[SupervisorNode] = None,
        simulation_state_container: Optional[SimulationStateContainer] = None,
        max_steps_per_episode: int = 100,
        init_by_call: bool = False,
        wait_for_obs: bool = False,
        obs_unit_kwargs: Optional[Dict[str, Any]] = None,
        train_mode: bool = True,
        observations_config: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Initializes the ArenaBaseEnv.

        Args:
            node (SupervisorNode): The ROS2 node instance.
            ns (Union[str, Namespace]): The namespace for ROS2 topics and services.
            space_manager (Union[BaseSpaceManager, Dict[str, Any]]): An instance or configuration
                dict for a class that defines the action and observation spaces.
            reward_function (Union[RewardFunction, Dict[str, Any]]): An instance or configuration
                dict for a class that calculates the reward.
            simulation_state_container (Optional[SimulationStateContainer]): A container for sharing
                state data across different components. Defaults to None.
            max_steps_per_episode (int): The maximum number of steps before an episode is
                truncated. Defaults to 100.
            init_by_call (bool): If True, ROS-dependent components are not initialized in the
                constructor but must be initialized by a manual call to `_initialize_environment()`.
                Defaults to False.
            wait_for_obs (bool): If True, the ObservationManager will wait for all observation
                sources to publish at least once before proceeding. Defaults to False.
            obs_unit_kwargs (Optional[Dict[str, Any]]): A dictionary of keyword arguments to be
                passed to the constructors of individual observation units. Defaults to None.
            observations_config (Optional[str]): Path to the observations YAML config file.
                If None, uses the default bundled config.
        """
        super().__init__()
        self.node = node
        self.ns = Namespace(ns) if isinstance(ns, str) else ns

        self._is_train_mode = train_mode
        if self.is_train_mode and reward_function is None:
            raise ValueError("A reward function is required for training mode.")

        self._initialize_agent_components(space_manager, reward_function)
        self.__simulation_state_container = simulation_state_container

        self._obs_unit_kwargs = obs_unit_kwargs or {}
        self.__wait_for_obs = wait_for_obs
        self.__observations_config_path = observations_config

        self._steps_curr_episode = 0
        self._episode = 0
        self._max_steps_per_episode = max_steps_per_episode
        self.__is_first_step = True

        # ROS interfaces (initialised in _initialize_environment)
        self._reset_task_srv = None
        self._cmd_vel_pub = None
        self._initialized = False

        if not init_by_call:
            self._initialize_environment()

    def _initialize_environment(self):
        """Initializes ROS-dependent components and the observation manager."""
        if self._initialized:
            return  # Already initialized

        if self.node is None:
            if not rclpy.ok():
                rclpy.init()  # Initialize ROS in worker process (e.g. Parallel daemon subprocess)
            env_node_name = f"{self.ns.to_string()}_env".replace("/", "_")
            self.node = SupervisorNode(node_name=env_node_name)
            self.node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])
            self.node.start_spinning()

        if self.is_train_mode:
            self._setup_ros_services()

        self._setup_observation_manager()
        self._initialized = True

    def _setup_ros_services(self):
        """Creates ROS2 services and clients required for training."""
        task_srv_name = (self.ns[0] / self.ns[1] / "reset_task").to_string()
        self._reset_task_srv = self.node.create_client(
            EmptySrv,
            task_srv_name,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )

        if not self._reset_task_srv.wait_for_service(timeout_sec=10.0):
            self.node.get_logger().warn(
                f"Service '{task_srv_name}' not available after 10 seconds. "
                f"Ensure the simulation and task_generator are running."
            )

        # Direct cmd_vel publisher — the sole source of velocity commands
        # during training.  Completely bypasses the nav2 controller_server.
        cmd_vel_topic = self.ns("cmd_vel").to_string()
        self._cmd_vel_pub = self.node.create_publisher(Twist, cmd_vel_topic, 10)
        self.node.get_logger().info(
            f"Created direct cmd_vel publisher at: {cmd_vel_topic}"
        )

    def _setup_observation_manager(self):
        """Configures and initializes the ObservationManager."""
        import importlib.resources

        # Use configured path, or fall back to bundled default
        obs_config_path = self.__observations_config_path
        if obs_config_path is None:
            obs_config_path = str(
                importlib.resources.files("rosnav_rl")
                / "observations"
                / "observations.yaml"
            )

        with open(obs_config_path, "r") as file:
            config = yaml.safe_load(file)

        # Create the observation manager from the configuration
        self.observation_collector = create_observation_manager_from_config(
            config=config,
            node=self.node,
            ns=self.ns.to_string(),
            simulation_state_container=self.simulation_state_container,
            wait_for_obs=self.__wait_for_obs,  # Wait for topics to be available
        )

    @property
    def action_space(self) -> gymnasium.spaces.Box:
        return self._model_space_manager.action_space

    @property
    def observation_space(self) -> gymnasium.spaces.Dict:
        return self._model_space_manager.observation_space

    @property
    def simulation_state_container(self) -> SimulationStateContainer:
        return self.__simulation_state_container

    @property
    def is_train_mode(self) -> bool:
        return self._is_train_mode

    def _initialize_agent_components(
        self,
        space_manager: Union[BaseSpaceManager, Dict[str, Any]],
        reward_function: Union[RewardFunction, Dict[str, Any]],
    ):
        """Initializes space manager and reward function from instances or dicts."""
        self._model_space_manager = (
            BaseSpaceManager(**space_manager)
            if isinstance(space_manager, dict)
            else space_manager
        )
        self._reward_function = (
            RewardFunction(**reward_function)
            if isinstance(reward_function, dict)
            else reward_function
        )

        assert isinstance(self._model_space_manager, BaseSpaceManager)
        assert isinstance(self._reward_function, RewardFunction)

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        """Decodes the given action using the model space encoder."""
        return self._model_space_manager.decode_action(action)

    def _encode_observation(
        self, observation: ObservationDict
    ) -> EncodedObservationDict:
        """Encodes the given observation using the model space encoder."""
        return self._model_space_manager.encode_observation(observation)

    @flush_errors_decorator
    def step(
        self, action: np.ndarray
    ) -> Tuple[EncodedObservationDict, float, bool, bool, InformationDict]:
        """
        Processes a single step in the environment.

        1. Decodes the action provided by the RL agent.
        2. Publishes the velocity command directly to ``cmd_vel``.
        3. Collects the new observation from the simulation.
        4. Calculates the reward and determines if the episode has terminated.
        5. Returns the standard Gymnasium step tuple.
        """
        decoded_action = self._decode_action(action)

        # Publish velocity command directly — no nav2 controller dependency.
        self._cmd_vel_pub.publish(get_twist_from_action(decoded_action))

        obs_dict = self.observation_collector.get_observations(
            simulation_state_container=self.__simulation_state_container,
            is_first=self.__is_first_step,
        )

        reward, reward_info = self._reward_function.get_reward(
            obs_dict=obs_dict,
            simulation_state_container=self.__simulation_state_container,
        )

        self._steps_curr_episode += 1
        info, done = determine_termination(
            reward_info=reward_info,
            curr_steps=self._steps_curr_episode,
            max_steps=self._max_steps_per_episode,
        )
        obs_dict["is_terminal"] = done
        self.__is_first_step = False

        if done:
            done_reason = info.get("done_reason", "unknown")
            is_success = info.get("is_success", False)
            msg = (
                f"[{self.ns.to_string()}] Episode {self._episode} ended — "
                f"reason: {done_reason}, steps: {self._steps_curr_episode}, "
                f"reward: {reward:.3f}"
            )
            if is_success:
                # Log world-frame goal position if available in the observation dict.
                goal, subgoal, robot = obs_dict.get("goal_pose"), obs_dict.get("subgoal_pose"), obs_dict.get("robot_pose")
                if goal is not None:
                    msg += f", goal: {goal}"
                if subgoal is not None:
                    msg += f", subgoal: {subgoal}"
                if robot is not None:
                    msg += f", robot: {robot}"
            self.node.get_logger().info(msg)

        return (
            self._encode_observation(obs_dict),
            reward,
            done,
            False,
            info,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[EncodedObservationDict, InformationDict]:
        """
        Resets the environment to its initial state and returns an initial observation.
        """
        # Ensure initialization has happened (for init_by_call=True case)
        if not self._initialized:
            self._initialize_environment()

        super().reset(seed=seed)

        # Stop the robot immediately.
        if self._cmd_vel_pub is not None:
            self._cmd_vel_pub.publish(Twist())

        try:
            # Reset episode-specific variables
            self._episode += 1
            self._steps_curr_episode = 0
            self.__is_first_step = True

            self.node.get_logger().info(
                f"[{self.ns.to_string()}] Resetting environment for episode {self._episode}..."
            )

            self._before_task_reset()
            self.reset_task()
            self._after_task_reset()

            self._reward_function.reset()

            # Get the initial observation after reset.
            obs_dict = self.observation_collector.get_observations(
                is_terminal=False, is_first=True
            )
            obs_dict["is_first"] = True  # Indicate it's the first observation

        finally:
            pass

        info = {}
        return self._encode_observation(obs_dict), info

    def close(self):
        """Cleans up ROS2 resources."""
        self.node.get_logger().info(
            "Closing environment and shutting down ROS components."
        )

        self.observation_collector.shutdown()
        if getattr(self, "_cmd_vel_pub", None):
            self.node.destroy_publisher(self._cmd_vel_pub)
        if getattr(self, "_reset_task_srv", None):
            self._reset_task_srv.destroy()

    def reset_task(self, timeout: float = 10.0, retries: int = 2):
        """Call the task-generator's reset_task service and **block** until it
        completes (or *timeout* seconds elapse).  Without a successful reset
        the goal/obstacle positions are stale and the episode will terminate
        immediately, so we retry on failure.

        Args:
            timeout:  Seconds to wait for the service response per attempt.
            retries:  Number of additional attempts after the first failure.
        """
        if not getattr(self, "_reset_task_srv", None):
            self.node.get_logger().warn("Reset task service client is not available (client not created).")
            return False

        if not self._reset_task_srv.service_is_ready():
            self.node.get_logger().warn(
                "Reset task service not ready — waiting up to 10 s for task_generator..."
            )
            if not self._reset_task_srv.wait_for_service(timeout_sec=10.0):
                self.node.get_logger().error(
                    "Reset task service still unavailable after 10 s. "
                    "Episode will use stale goal. "
                    "Verify that the simulation and task_generator are running."
                )
                return False

        for attempt in range(1 + retries):
            completion_event = threading.Event()
            result_container = {"success": False, "exception": None}

            def done_callback(future):
                try:
                    result = future.result()
                    result_container["success"] = result is not None
                except Exception as e:
                    result_container["exception"] = e
                finally:
                    completion_event.set()

            future = self._reset_task_srv.call_async(EmptySrv.Request())
            future.add_done_callback(done_callback)

            if completion_event.wait(timeout=timeout):
                if result_container["success"]:
                    self.node.get_logger().debug("Service call successful.")
                    return True
                else:
                    self.node.get_logger().error(
                        f"Service call failed (attempt {attempt + 1}): "
                        f"{result_container['exception']}"
                    )
            else:
                self.node.get_logger().error(
                    f"Service call timeout after {timeout}s (attempt {attempt + 1}/{1 + retries})."
                )
                future.cancel()

        self.node.get_logger().error(
            "reset_task failed after all retries – episode will use stale goals."
        )
        return False

    def _before_task_reset(self):
        """Hook for executing actions before the task is reset."""
        pass

    def _after_task_reset(self):
        """Hook for executing actions after the task is reset."""
        pass
