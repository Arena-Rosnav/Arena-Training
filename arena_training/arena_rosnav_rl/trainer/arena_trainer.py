from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

import rosnav_rl
import arena_training.arena_rosnav_rl.cfg as arena_cfg

from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.type_aliases import SupportedRLFrameworks


from ..node import SupervisorNode
from ..utils.training import (
    print_base_model,
    setup_paths_dictionary,
)
from ..utils import paths as Paths
from ..utils.hooks import HookManager, TrainingHookStages, bind_hooks
from ..utils.type_alias.observation import EnvironmentType, PathsDict

@dataclass
class TrainingArguments:
    """Arguments for training."""

    def to_dict(self) -> dict:
        return self.__dict__


class ArenaTrainer(ABC):
    """
    Abstract base class for reinforcement learning trainers in the Arena-Rosnav framework.

    ArenaTrainer handles the setup, training, and management of RL agents and their environments.
    It uses a hook-based system to manage the training workflow and provides a structured approach
    to implementing different RL frameworks.

    Attributes:
        __framework (SupportedRLFrameworks): The RL framework being used (e.g., SB3, RLlib).
        config_cls (TrainingCfg): Configuration settings for the training process.
        paths (PathsDict): Dictionary containing paths for model saving, logging, etc.
        agent (RL_Agent): The reinforcement learning agent being trained.
        environment (EnvironmentType): The training environment for the agent.
        hook_manager (HookManager): Manager for registering and executing hooks at various training stages.

    Methods:
        train: Executes the training process for the RL agent.
        save: Saves the trained model to a specified checkpoint.
        close: Cleans up resources and exits gracefully.

    The class uses a hook system through the @bind_hooks decorator to allow for
    custom callbacks at different stages of the training process.

    Implementing classes must provide concrete implementations for:
    - _setup_agent: Initialize the RL agent
    - _setup_environment: Create the training environment
    - _setup_monitoring: Set up monitoring and logging tools
    """

    _framework: SupportedRLFrameworks
    _config_type: Type[arena_cfg.ArenaBaseCfg]

    config: arena_cfg.TrainingCfg
    paths: PathsDict

    agent: RL_Agent
    environment: EnvironmentType

    hook_manager: HookManager = HookManager()

    @bind_hooks(before_stage=TrainingHookStages.ON_INIT)
    def __init__(self, config: arena_cfg.TrainingCfg, resume: bool = False) -> None:
        """
        Initializes the ArenaTrainer with the given configuration.

        Args:
            config (TrainingCfg): The configuration object for training.
        """
        self._validate_config(config)
        self.config = config
        self.__resume = resume

        self._setup_supervisor_node()

        self._register_default_hooks()
        self._register_framework_specific_hooks()
        self._setup_trainer()

    def _validate_config(self, config: arena_cfg.TrainingCfg) -> None:
        if not isinstance(config.arena_cfg, self._config_type):
            raise TypeError(
                f"Invalid configuration type: {type(config.arena_cfg)} for {self._framework}. "
                f"Expected one of: {self._config_type}"
            )

    def _setup_supervisor_node(self):
        self._supervisor_node = SupervisorNode(node_name="Arena_Trainer")
        self._supervisor_node.start_spinning()

    @bind_hooks(
        before_stage=TrainingHookStages.BEFORE_SETUP,
        after_stage=TrainingHookStages.AFTER_SETUP,
    )
    def _setup_trainer(self) -> None:
        """
        Sets up the trainer by sequentially executing the following setup steps:
        1. Sets up simulation state container
        2. Sets up agent state container
        3. Sets up the agent
        4. Sets up the environment

        This method orchestrates the initialization process in a structured order
        to ensure all components are properly initialized before training.

        Returns:
            None
        """
        setup_steps = [
            self._populate_agent_spec,
            self._setup_agent_parameters,
            lambda: setup_paths_dictionary(self, self.is_debug_mode),
            lambda: self._write_config(),
            self._setup_agent,
            self._setup_environment,
        ]

        for step in setup_steps:
            step()

    @bind_hooks(
        before_stage=TrainingHookStages.BEFORE_TRAINING,
        after_stage=TrainingHookStages.AFTER_TRAINING,
    )
    def train(self, *args, **kwargs) -> None:
        """
        Execute the training process for the reinforcement learning agent.

        This method acts as a wrapper that calls the internal implementation
        of the training procedure through the `_train_impl` method.

        Args:
            *args: Variable length argument list to be passed to the training implementation.
            **kwargs: Arbitrary keyword arguments to be passed to the training implementation.

        Returns:
            None
        """
        self._train_impl()

    def save(self, checkpoint: str, *args, **kwargs) -> None:
        """
        Save the model checkpoint.
        This method saves the model to the specified checkpoint path.
        Args:
            checkpoint (str): Path where the model checkpoint will be saved.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            None
        """
        self._save_model(checkpoint=checkpoint)

    @bind_hooks(before_stage=TrainingHookStages.ON_CLOSE)
    def close(self):
        self.environment.close()
        self._supervisor_node.stop_spinning()
        self._supervisor_node.destroy_node()

    def _register_default_hooks(self) -> None:
        """Register default hooks common across implementations."""
        default_hooks = {
            TrainingHookStages.BEFORE_SETUP: [
                lambda _: print_base_model(self.config),
            ],
            TrainingHookStages.AFTER_SETUP: [
                lambda _: self._set_resume_true(),
                # lambda _: self.simulation_state_container.distribute(),
            ],
            TrainingHookStages.BEFORE_TRAINING: [lambda _: self._setup_monitoring()],
        }

        for hook, callbacks in default_hooks.items():
            self.hook_manager.register(hook, callbacks)

    def _set_resume_true(self):
        """Set resume to True to make sure one can directly load and resume training."""
        self.config.resume = True

    def _write_config(self):
        """Write configuration to file if not in debug mode."""
        if not self.is_debug_mode:
            self.config.to_yaml(
                self.paths[Paths.Agent].path / "training_config.yaml"
            )

    @bind_hooks(before_stage=TrainingHookStages.ON_SAVE)
    def _save_model(self, checkpoint: str) -> None:
        """Save the trained model."""
        if not self.is_debug_mode:
            self.agent.model.save(
                dirpath=self.paths[Paths.Agent].path, file_name=checkpoint
            )

    def _register_framework_specific_hooks(self):
        """Register hooks that are specific to the framework."""
        pass

    def _train_impl(self, *args, **kwargs) -> None:
        """Implementation of training logic."""
        self.agent.model.train()
        self._save_model(checkpoint="last_model")

    def _setup_agent_parameters(self, *args, **kwargs) -> None:
        """Store the fully-populated AgentParameters on the trainer."""
        self.agent_parameters = self.config.agent_config.parameters

    def _populate_agent_spec(self) -> None:
        """Derive action_space and parameters from the robot description.

        The robot description (loaded from ``arena_robots``) is the
        single source of truth for kinematics and sensor geometry.
        :class:`~rosnav_rl.cfg.parameters.AgentParameters` is built
        directly from the robot description and arena config.
        """
        from rosnav_rl.cfg.action_spaces import (
            DifferentialDriveActionSpace,
            OmnidirectionalActionSpace,
        )

        robot_desc = self.config.arena_cfg.robot.robot_description
        general = self.config.arena_cfg.general
        cont = robot_desc.actions.continuous

        # --- action space from robot description ---
        if robot_desc.is_holonomic:
            action_space = OmnidirectionalActionSpace(
                linear_range_x=tuple(cont.linear_range),
                linear_range_y=tuple(cont.linear_range),
                angular_range=tuple(cont.angular_range),
            )
        else:
            action_space = DifferentialDriveActionSpace(
                linear_range=tuple(cont.linear_range),
                angular_range=tuple(cont.angular_range),
            )

        # Carry over the discretization config from the agent_config YAML, then
        # resolve it immediately using the robot's built-in discrete action list.
        discretization_cfg = self.config.agent_config.discretization
        if discretization_cfg is not None:
            action_space = action_space.model_copy(
                update={"discretization": discretization_cfg}
            )
            action_space = action_space.resolve_discretization(
                robot_discrete_actions=robot_desc.actions.discrete
            )

        # --- unified parameters from robot description + arena config ---
        # Start from whatever the user set in the YAML (preserves normalize,
        # normalizer, and any other overrides), then stamp in the
        # hardware-derived values which must always come from the robot desc.
        parameters = self.config.agent_config.parameters.model_copy(
            update=dict(
                laser_num_beams=robot_desc.laser.num_beams,
                laser_max_range=robot_desc.laser.range,
                min_linear_vel=cont.linear_range[0],
                max_linear_vel=cont.linear_range[1],
                min_angular_vel=cont.angular_range[0],
                max_angular_vel=cont.angular_range[1],
                min_translational_vel=cont.linear_range[0],
                max_translational_vel=cont.linear_range[1],
                robot_radius=robot_desc.robot_radius,
                safety_distance=general.safety_distance,
                goal_radius=general.goal_radius,
                max_steps=general.max_num_moves_per_eps,
            )
        )

        self.config.agent_config = self.config.agent_config.model_copy(
            update={
                "robot": robot_desc.robot_model,
                "action_space": action_space,
                "parameters": parameters,
            }
        )

    @abstractmethod
    def _setup_agent(self, *args, **kwargs) -> None:
        """Initialize the RL agent."""
        raise NotImplementedError()

    @abstractmethod
    def _setup_environment(self, *args, **kwargs) -> None:
        """Setup training Gym environment."""
        raise NotImplementedError()

    @abstractmethod
    def _setup_monitoring(self, *args, **kwargs) -> None:
        """Setup monitoring tools."""
        raise NotImplementedError()

    @property
    def is_debug_mode(self):
        return self.node.get_parameter_or("debug_mode", False)

    @property
    def is_resume(self):
        return self.__resume

    @property
    def node(self) -> SupervisorNode:
        """Return the supervisor node for this trainer."""
        return self._supervisor_node
