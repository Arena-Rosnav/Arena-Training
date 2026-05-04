import posixpath
import threading
import time
from abc import ABC
from typing import Any, Dict, Optional, Tuple, Union

import arena_robots.Robot
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
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.utils.rostopic import Namespace
from rosnav_rl.utils.type_aliases import EncodedObservationDict, ObservationDict
from arena_runtime_msgs.srv import LifecycleHold as LifecycleHoldSrv
from task_generator_msgs.msg import EpisodeRecord, RobotFleet
from task_generator_msgs.srv import ResetEpisode

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
        env_ns (Namespace): The env's task_generator_node namespace (lifecycle/state).
        robot_ns (Namespace): The robot's namespace (cmd_vel, sensor topics). Set by `_resolve_robot`.
        action_space (gymnasium.spaces.Box): The action space, defined by the space manager.
        observation_space (gymnasium.spaces.Dict): The observation space, defined by the space manager.
        is_train_mode (bool): Flag indicating if the environment is in training mode.
        observation_collector (ObservationManager): Manages the collection of observations.
        metadata (dict): Standard Gymnasium metadata.
    """

    metadata = {"render_modes": ["human"]}

    _TIMEOUT_PARAM_DEFAULTS = {
        "reset_task_timeout": -1.0,
        "service_wait_timeout": -1.0,
        "episode_wait_timeout": -1.0,
        "fleet_wait_timeout": -1.0,
    }

    def __init__(
        self,
        ns: Union[str, Namespace],
        space_manager: Union[BaseSpaceManager, Dict[str, Any]],
        reward_function: Union[RewardFunction, Dict[str, Any]],
        node: Optional[SupervisorNode] = None,
        simulation_state_container: Optional[AgentParameters] = None,
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
            simulation_state_container (Optional[AgentParameters]): A container for sharing
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
        self.env_ns = Namespace(ns) if isinstance(ns, str) else ns

        self._is_train_mode = train_mode
        if self.is_train_mode and reward_function is None:
            raise ValueError("A reward function is required for training mode.")

        self._initialize_agent_components(space_manager, reward_function)
        self.__agent_parameters = simulation_state_container

        self._obs_unit_kwargs = obs_unit_kwargs or {}
        self.__wait_for_obs = wait_for_obs
        self.__observations_config_path = observations_config

        self._steps_curr_episode = 0
        self._episode = 0
        self._max_steps_per_episode = max_steps_per_episode
        self.__is_first_step = True

        # ROS interfaces (initialised in _initialize_environment)
        self._reset_task_srv = None
        self._pause_srv = None
        self._cmd_vel_pub = None
        self._episode_state_sub = None
        self._latest_episode: Optional[EpisodeRecord] = None
        self._episode_event = threading.Event()
        self._fleet_sub = None
        self._latest_fleet: Optional[RobotFleet] = None
        self._fleet_event = threading.Event()
        self.robot_ns: Optional[Namespace] = None
        self.robot_source_frame: Optional[str] = None
        # Set on first step(); reset() blocks here so we don't fire a fresh
        # task_generator reset until the model has cleared its first inference.
        self._ready_event = threading.Event()
        self._first_reset_done = False
        # Lazy-pause: pause(True) only takes effect if inference exceeds threshold.
        self._pause_pending_timer: Optional[threading.Timer] = None
        self._pause_pending_token: object = None
        self._pause_lock = threading.Lock()
        self._pause_lazy_threshold: float = 5.0
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
            env_node_name = f"{self.env_ns.to_string()}_env".replace("/", "_")
            self.node = SupervisorNode(node_name=env_node_name)
            self.node.set_parameters(
                [rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)]
            )
            self.node.start_spinning()

        self._declare_timeout_params()

        self._setup_fleet_subscription()
        self._resolve_robot()

        if self.is_train_mode:
            self._setup_ros_services()

        self._setup_observation_manager()
        self._initialized = True

    def _setup_fleet_subscription(self) -> None:
        robots_topic = (self.env_ns / "state" / "robots").to_string()
        self._fleet_sub = self.node.create_subscription(
            RobotFleet,
            robots_topic,
            self._on_fleet,
            qos_profile=rclpy.qos.QoSProfile(
                depth=1,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

    def _on_fleet(self, msg: RobotFleet) -> None:
        self._latest_fleet = msg
        self._fleet_event.set()

    def _resolve_robot(self) -> None:
        robots_topic = (self.env_ns / "state" / "robots").to_string()
        if not self._fleet_event.wait(timeout=self._timeout_param("fleet_wait_timeout")):
            raise RuntimeError(
                f"No robots fleet snapshot on '{robots_topic}'; task_generator did not publish state/robots."
            )
        if not self._latest_fleet.robots:
            raise RuntimeError(f"Robots fleet on '{robots_topic}' is empty; training requires at least one robot.")

        robot = self._latest_fleet.robots[0]
        self.robot_ns = Namespace(robot.ns)
        base_frame = arena_robots.Robot.RobotIdentifier(robot.model).resolve_sync().model_params.base_frame
        self.robot_source_frame = posixpath.join(robot.frame, base_frame)

    def _setup_ros_services(self):
        """Creates ROS2 services and clients required for training."""
        task_srv_name = (self.env_ns / "lifecycle/reset_episode").to_string()
        self._reset_task_srv = self.node.create_client(
            ResetEpisode,
            task_srv_name,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )

        service_wait = self._timeout_param("service_wait_timeout")
        if not self._reset_task_srv.wait_for_service(timeout_sec=service_wait):
            self.node.get_logger().warn(
                f"Service '{task_srv_name}' not available. "
                f"Ensure the simulation and task_generator are running."
            )

        pause_srv_name = "/arena/sim_lifecycle/hold"
        self._pause_srv = self.node.create_client(
            LifecycleHoldSrv,
            pause_srv_name,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )
        if not self._pause_srv.wait_for_service(timeout_sec=10.0):
            self.node.get_logger().warn(
                f"Service '{pause_srv_name}' not available after 10s. "
                f"Simulation pause/unpause will be disabled."
            )
            self._pause_srv = None

        # Direct cmd_vel publisher, the sole source of velocity commands
        # during training.  Completely bypasses the nav2 controller_server.
        cmd_vel_topic = self.robot_ns("cmd_vel").to_string()
        self._cmd_vel_pub = self.node.create_publisher(Twist, cmd_vel_topic, 10)
        self.node.get_logger().info(
            f"Created direct cmd_vel publisher at: {cmd_vel_topic}"
        )

        # Episode-state feed from task_generator (TRANSIENT_LOCAL).
        episode_topic = (self.env_ns / "state" / "episode").to_string()
        self._episode_state_sub = self.node.create_subscription(
            EpisodeRecord,
            episode_topic,
            self._on_episode_state,
            qos_profile=rclpy.qos.QoSProfile(
                depth=1,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )

    def _declare_timeout_params(self) -> None:
        for name, default in self._TIMEOUT_PARAM_DEFAULTS.items():
            if not self.node.has_parameter(name):
                self.node.declare_parameter(name, default)

    def _timeout_param(self, name: str) -> Optional[float]:
        """Read a timeout ROS param. Negative sentinel means infinite (None)."""
        val = self.node.get_parameter(name).value
        return None if val < 0 else float(val)

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

        for ds in config.get("datasources", {}).values():
            if ds.get("type") == "RobotPoseTFGenerator":
                ds.setdefault("params", {})["source_frame"] = self.robot_source_frame

        # Create the observation manager from the configuration
        self.observation_collector = create_observation_manager_from_config(
            config=config,
            node=self.node,
            ns=self.robot_ns.to_string(),
            simulation_state_container=self.__agent_parameters,
            wait_for_obs=self.__wait_for_obs,  # Wait for topics to be available
        )

    @property
    def action_space(self) -> gymnasium.spaces.Box:
        return self._model_space_manager.action_space

    @property
    def observation_space(self) -> gymnasium.spaces.Dict:
        return self._model_space_manager.observation_space

    @property
    def agent_parameters(self) -> AgentParameters:
        return self.__agent_parameters

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

        # First step() means agent inference returned at least once: model is loaded.
        if not self._ready_event.is_set():
            self._ready_event.set()

        # Publish velocity command directly — no nav2 controller dependency.
        self._cmd_vel_pub.publish(get_twist_from_action(decoded_action))

        obs_dict = self.observation_collector.get_observations(
            simulation_state_container=self.__agent_parameters,
            is_first=self.__is_first_step,
        )

        reward, reward_info = self._reward_function.get_reward(
            obs_dict=obs_dict,
            simulation_state_container=self.__agent_parameters,
        )

        self._steps_curr_episode += 1
        info, done = determine_termination(
            reward_info=reward_info,
            curr_steps=self._steps_curr_episode,
            max_steps=self._max_steps_per_episode,
        )
        tg = self._latest_episode
        if tg is not None and tg.outcome_state not in (EpisodeRecord.QUEUED, EpisodeRecord.RUNNING):
            done = True
            info["done_reason"] = tg.outcome_info or f"task_generator:{tg.outcome_state}"
            info["is_success"] = int(tg.outcome_state == EpisodeRecord.SUCCESS)
            info["episode_length"] = self._steps_curr_episode
        obs_dict["is_terminal"] = done
        self.__is_first_step = False

        if done:
            done_reason = info.get("done_reason", "unknown")
            is_success = info.get("is_success", False)
            episode_id = self._latest_episode.episode_id if self._latest_episode else self._episode
            msg = (
                f"[{self.env_ns.to_string()}] Episode {episode_id} ended — "
                f"reason: {done_reason}, steps: {self._steps_curr_episode}, "
                f"reward: {reward:.3f}"
            )
            if is_success:
                # Log world-frame goal position if available in the observation dict.
                goal = obs_dict.get("goal_pose")
                subgoal = obs_dict.get("subgoal_pose")
                robot = obs_dict.get("robot_pose")
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
            steps_this_episode = self._steps_curr_episode
            self._steps_curr_episode = 0
            self.__is_first_step = True

            if not self._first_reset_done:
                # Managed mode: task_generator does not auto-spawn episode 1
                # on activate (auto_reset=false), drive the first reset here.
                self._first_reset_done = True
                self._before_task_reset()
                self.reset_task()
                self._wait_for_new_episode(prev_id=0)
                self._after_task_reset()
                self._episode = (
                    self._latest_episode.episode_id if self._latest_episode is not None else 1
                )
            elif (
                self._latest_episode is not None
                and self._latest_episode.outcome_state == EpisodeRecord.QUEUED
                and steps_this_episode == 0
            ):
                # No-op fresh reset (e.g. simulate-loop's iter-1 double reset).
                pass
            else:
                # Hold off real resets until the model has finished its first
                # inference (compile pass), so the pause/unpause loop is reliable.
                if not self._ready_event.is_set():
                    self.node.get_logger().info(
                        f"[{self.env_ns.to_string()}] Holding reset until model is loaded..."
                    )
                    if not self._ready_event.wait(timeout=120.0):
                        self.node.get_logger().warn(
                            "Ready signal timed out at 120 s, proceeding anyway."
                        )

                prev_id = self._latest_episode.episode_id if self._latest_episode is not None else 0

                self.node.get_logger().info(
                    f"[{self.env_ns.to_string()}] Resetting environment after episode {prev_id}..."
                )

                self._before_task_reset()
                self.reset_task()
                self._wait_for_new_episode(prev_id)
                self._after_task_reset()
                self._episode = (
                    self._latest_episode.episode_id if self._latest_episode is not None else prev_id + 1
                )

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
        if self._cmd_vel_pub is not None:
            self.node.destroy_publisher(self._cmd_vel_pub)
        if self._reset_task_srv is not None:
            self._reset_task_srv.destroy()
        if self._pause_srv is not None:
            self._pause_srv.destroy()
        if self._episode_state_sub is not None:
            self.node.destroy_subscription(self._episode_state_sub)

    def pause(self, paused: bool) -> None:
        """Lazy pause/unpause: pause(True) only fires if inference takes longer
        than ``_pause_lazy_threshold``; pause(False) cancels any pending pause
        and unconditionally issues an unpause. Race-safe: a token is checked in
        the timer callback so a late-firing timer that lost the race against
        ``pause(False)`` does not send a stray PAUSE."""
        if self._pause_srv is None:
            return
        with self._pause_lock:
            if self._pause_pending_timer is not None:
                self._pause_pending_timer.cancel()
                self._pause_pending_timer = None
            self._pause_pending_token = None
            if paused:
                token = object()
                self._pause_pending_token = token
                timer = threading.Timer(
                    self._pause_lazy_threshold,
                    self._maybe_fire_pause,
                    args=(token,),
                )
                timer.daemon = True
                self._pause_pending_timer = timer
                timer.start()
                fire_unpause_now = False
            else:
                fire_unpause_now = True
        if fire_unpause_now:
            self._fire_pause_request(False)

    def _maybe_fire_pause(self, token: object) -> None:
        with self._pause_lock:
            if self._pause_pending_token is not token:
                return
            self._pause_pending_token = None
            self._pause_pending_timer = None
        self._fire_pause_request(True)

    def _fire_pause_request(self, paused: bool) -> None:
        if self._pause_srv is None:
            return
        req = LifecycleHoldSrv.Request()
        req.caller_id = self.node.get_fully_qualified_name()
        req.reason = "training_env_pause"
        req.action = LifecycleHoldSrv.Request.ACQUIRE if paused else LifecycleHoldSrv.Request.RELEASE
        self._pause_srv.call_async(req)

    def reset_task(self, timeout: Optional[float] = None, retries: int = 2):
        """Call task-generator's lifecycle/reset_episode and block until it completes.

        Args:
            timeout:  Per-attempt seconds; None reads `reset_task_timeout` param.
            retries:  Number of additional attempts after the first failure.
        """
        if timeout is None:
            timeout = self._timeout_param("reset_task_timeout")
        if self._reset_task_srv is None:
            self.node.get_logger().warn("Reset task service client is not available (client not created).")
            return False

        if not self._reset_task_srv.service_is_ready():
            self.node.get_logger().warn(
                "Reset task service not ready, waiting for task_generator..."
            )
            if not self._reset_task_srv.wait_for_service(
                timeout_sec=self._timeout_param("service_wait_timeout")
            ):
                self.node.get_logger().error(
                    "Reset task service still unavailable. "
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
                    result_container["success"] = result is not None and result.success
                    if result is not None and not result.success:
                        result_container["exception"] = result.error_msg
                except Exception as e:
                    result_container["exception"] = e
                finally:
                    completion_event.set()

            req = ResetEpisode.Request()
            req.world = ""
            req.seed = -1
            future = self._reset_task_srv.call_async(req)
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
            "reset_episode failed after all retries – episode will use stale goals."
        )
        return False

    def _before_task_reset(self):
        """Hook for executing actions before the task is reset."""
        pass

    def _after_task_reset(self):
        """Hook for executing actions after the task is reset."""
        pass

    def _on_episode_state(self, msg: EpisodeRecord) -> None:
        self._latest_episode = msg
        self._episode_event.set()

    def _wait_for_new_episode(self, prev_id: int, timeout: Optional[float] = None) -> bool:
        """Block until task_generator publishes a fresh QUEUED episode."""
        if timeout is None:
            timeout = self._timeout_param("episode_wait_timeout")
        deadline = float("inf") if timeout is None else time.monotonic() + timeout
        while time.monotonic() < deadline:
            tg = self._latest_episode
            if (
                tg is not None
                and tg.episode_id > prev_id
                and tg.outcome_state == EpisodeRecord.QUEUED
            ):
                return True
            self._episode_event.clear()
            self._episode_event.wait(timeout=0.1)
        self.node.get_logger().warn(
            f"Timeout waiting for new episode after id={prev_id} ({timeout}s)"
        )
        return False
