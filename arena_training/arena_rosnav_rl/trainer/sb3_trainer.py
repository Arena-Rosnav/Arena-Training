import logging
import time
from dataclasses import dataclass
from functools import partial

from rosnav_rl.model.stable_baselines3 import StableBaselinesModel
from rosnav_rl.model.stable_baselines3.cfg import StableBaselinesCfg
from rosnav_rl.utils.stable_baselines3.tick_callbacks import TickCheckpointCallback, TickScheduledCurriculum
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from ...environments.wrappers import TimeSyncWrapper
from ..cfg import sb3_cfg as arena_cfg
from ..stable_baselines3.eval_callbacks.initialization import curriculum_stage_lists, init_sb3_callbacks
from ..stable_baselines3.vec_wrapper import VecActionChunk
from ..utils import paths as Paths
from ..utils.env_factory import make_envs, sb3_wrap_env
from ..utils.monitoring import setup_wandb
from .arena_trainer import (
    ArenaTrainer,
    SupportedRLFrameworks,
    TrainingHookStages,
)

_init_log = logging.getLogger("arena_training.init")


@dataclass
class SB3Environment:
    train_env: VecEnv
    eval_env: VecEnv

    def close(self) -> None:
        """Close the training and evaluation environments."""
        self.train_env.close()
        # eval_env may be the same object as train_env (shared-env mode)
        if self.eval_env is not self.train_env:
            self.eval_env.close()


class StableBaselines3Trainer(ArenaTrainer[SB3Environment]):
    """
    StableBaselines3Trainer class for managing RL training with the Stable Baselines 3 framework.

    This trainer handles the setup and execution of RL training processes using the
    Stable Baselines 3 framework in the arena-rosnav environment. It manages configuration,
    environment setup, agent initialization, and training workflow.

    Attributes:
        __framework (SupportedRLFrameworks): Identifier for the SB3 framework
        config_manager (SB3ConfigManager): Manages the training configuration
        environment (SB3Environment): Contains training and evaluation environments
        general_cfg: General training configuration parameters
        monitoring_cfg: Configuration for monitoring tools
        task_cfg: Task-specific configuration
        normalization_cfg: Configuration for observation normalization
        callbacks_cfg: Configuration for training callbacks
        profiling_cfg: Configuration for performance profiling
        robot_cfg: Robot configuration parameters
        agent_cfg: Agent-specific configuration

    Methods:
        __init__(config): Initialize the trainer with configuration
        __unpack_config(): Extract SB3-specific configuration fields
        _register_framework_specific_hooks(): Register SB3-specific training hooks
        _setup_agent(): Initialize the RL agent
        _setup_environment(): Set up training and evaluation environments
        _create_environments(): Create the environments for training and evaluation
        _setup_callbacks(environment): Initialize training callbacks
        _setup_simulation_state_container(): Initialize simulation state container
        _setup_agent_state_container(): Set up the agent state container
        _setup_monitoring(): Configure monitoring tools like W&B
        _train_impl(*args, **kwargs): Core training implementation
        _complete_model_initialization(train_env): Finalize model initialization
    """

    _framework = SupportedRLFrameworks.STABLE_BASELINES3
    _config_type = arena_cfg.ArenaSB3Cfg
    environment: SB3Environment

    def _register_framework_specific_hooks(self) -> None:
        """Register framework specific hooks for training.

        This method sets up the following hooks:
            - BEFORE_SETUP:
                - Set the curriculum file parameter in the task generator server
                - Set the current curriculum stage in the task generator server
            - AFTER_SETUP:
                - Transfer weights from a source model if configured

        The hooks are executed at specific stages during the training process.
        """

        def _transfer_weights(_: ArenaTrainer | None):
            transfer_cfg = self._framework_cfg().algorithm.transfer_weights
            if transfer_cfg:
                self.agent.model.transfer_weights(
                    source_dir=transfer_cfg.source_dir,
                    source_checkpoint=transfer_cfg.source_checkpoint,
                    include=transfer_cfg.include,
                    exclude=transfer_cfg.exclude,
                )

        self.hook_manager.register(
            TrainingHookStages.AFTER_SETUP,
            [_transfer_weights],
        )

    def _setup_agent(self, *args: object, **kwargs: object) -> None:
        """Initializes the reinforcement learning agent.

        This method creates an instance of the RL_Agent class using the
        configuration parameters specified in the config object and the
        current agent state container.

        Returns:
            None
        """
        import rosnav_rl

        t0 = time.monotonic()
        self.agent = rosnav_rl.RL_Agent(self.config.agent_config)
        _init_log.debug("RL_Agent ready in %.1fs", time.monotonic() - t0)

    def _setup_environment(self, *args: object, **kwargs: object) -> None:
        """Sets up the training environment.

        This method performs the following steps:
            1. Applies logging verbosity configuration.
            2. Creates the necessary environments.
            3. Sets up callbacks for the environment.
            4. Completes the model initialization using the training environment.

        Returns:
            None
        """
        self._configure_verbose(self.config.arena_cfg.general.verbose)

        t0 = time.monotonic()
        self._create_environments()
        _init_log.debug("_create_environments done in %.1fs", time.monotonic() - t0)

        t0 = time.monotonic()
        self._setup_callbacks(self.environment)
        _init_log.debug("_setup_callbacks done in %.1fs", time.monotonic() - t0)

        t0 = time.monotonic()
        self._complete_model_initialization(self.environment.train_env)
        _init_log.debug("_complete_model_initialization done in %.1fs", time.monotonic() - t0)

    def _configure_verbose(self, verbose: int | bool) -> None:
        """Apply log levels for rosnav_rl namespaces and this trainer."""
        import logging as _logging

        from ..utils.log_utils import configure_trainer_logging

        configure_trainer_logging(
            logging_cfg=self.config.arena_cfg.logging,
            verbose=int(verbose),
            trainer_logger=_logging.getLogger(__name__),
        )

    def _create_environments(self) -> None:
        """Creates training and evaluation environments for the RL agent.

        This method initializes the training and evaluation environments using the specified configurations.
        It first creates environment function factories for both training and evaluation environments
        using the `make_envs` function. Then, it wraps these environments for compatibility with
        Stable Baselines 3 (SB3) using the `sb3_wrap_env` function. Finally, it sets up the environments
        for the agent and stores them in an SB3Environment container.
        The environments are configured with parameters from the trainer's config, including the number
        of environments, maximum steps per episode, and whether to initialize environments on call.

        Returns:
            None

        Note:
            Training environments use the TRAIN_NS namespace, while evaluation environments use EVAL_NS.
        """

        train_env_fncs = make_envs(
            node=self.node if self.config.arena_cfg.general.debug_mode else None,  # Each worker creates its own node after rclpy.init()
            rl_agent=self.agent,
            n_envs=self.config.arena_cfg.general.n_envs,
            max_steps=self.config.arena_cfg.general.max_num_moves_per_eps,
            init_env_by_call=False,
            namespace_fn=self.namespace_fn,
            simulation_state_container=self.agent_parameters,
            wrappers=[partial(TimeSyncWrapper, control_hz=self.config.arena_cfg.general.control_hz)],
            observations_config=self.config.agent_config.observations_config,
            lockstep=self.config.arena_cfg.general.lockstep,
            lockstep_step_seconds=self.config.arena_cfg.general.lockstep_step_seconds,
        )
        t0 = time.monotonic()
        env = sb3_wrap_env(
            node=self._supervisor_node,
            env_fncs=train_env_fncs,
            general_cfg=self.config.arena_cfg.general,
            monitoring_cfg=self.config.arena_cfg.monitoring,
            profiling_cfg=self.config.arena_cfg.profiling,
        )
        _init_log.debug("sb3_wrap_env done in %.1fs", time.monotonic() - t0)

        chunk = self.config.agent_config.action_chunk
        if chunk > 1:
            gamma = self._framework_cfg().algorithm.parameters.gamma
            env = VecActionChunk(env, horizon=chunk, gamma=gamma)
            _init_log.info(
                "action chunk %d: total_timesteps, eval_freq and max steps count control ticks, the policy decides every %d ticks with gamma %.4f per decision (%.4f per tick)",
                chunk,
                chunk,
                gamma**chunk,
                gamma,
            )

        t0 = time.monotonic()
        env = self._sb3_model().setup_environment(env)
        _init_log.debug("agent.model.setup_environment done in %.1fs", time.monotonic() - t0)
        self.environment = SB3Environment(train_env=env, eval_env=env)

    def _setup_callbacks(self, environment: SB3Environment) -> None:
        """Initialize training callbacks."""
        arena_sb3_cfg = self._arena_sb3_cfg()
        callback_cfg = arena_sb3_cfg.callbacks
        if callback_cfg.periodic_evaluation is None:
            raise ValueError("callbacks.periodic_evaluation is required, the shared-env eval callback runs on it.")
        eval_max_steps = callback_cfg.periodic_evaluation.max_num_moves_per_eps
        chunk = self.config.agent_config.action_chunk
        if chunk > 1 and callback_cfg.periodic_evaluation is not None:
            callback_cfg = callback_cfg.model_copy(update={"periodic_evaluation": callback_cfg.periodic_evaluation.model_copy(update={"eval_freq": max(1, callback_cfg.periodic_evaluation.eval_freq // chunk)})})
        self.eval_cb = init_sb3_callbacks(
            node=self._supervisor_node,
            eval_env=environment.eval_env,
            n_envs=self.config.arena_cfg.general.n_envs,
            tm_modules=self.config.arena_cfg.task.tm_modules,
            callback_cfg=callback_cfg,
            task_cfg=self.config.arena_cfg.task,
            model_save_path=str(self.paths[Paths.Agent].path),
            eval_log_path=str(self.paths[Paths.AgentEval].path),
            debug_mode=self.config.arena_cfg.general.debug_mode,
            train_max_steps=self.config.arena_cfg.general.max_num_moves_per_eps,
            eval_max_steps=eval_max_steps,
        )
        self.tick_callbacks: list[BaseCallback] = []
        task_cfg = self.config.arena_cfg.task
        if "staged" in task_cfg.tm_modules and task_cfg.staged is not None and task_cfg.staged.stage_ticks is not None:
            self.tick_callbacks.append(
                TickScheduledCurriculum(
                    node=self._supervisor_node,
                    train_stages=curriculum_stage_lists(task_cfg),
                    stage_ticks=task_cfg.staged.stage_ticks,
                    ticks_per_step=chunk,
                    num_envs=self.config.arena_cfg.general.n_envs,
                    parameter_node_template=task_cfg.staged.parameter_node_template,
                    timeout=task_cfg.staged.timeout,
                    verbose=1,
                    tm_dict={"tm_robots": task_cfg.tm_robots, "tm_obstacles": task_cfg.tm_obstacles, "tm_modules": task_cfg.tm_modules},
                )
            )
        if arena_sb3_cfg.callbacks.checkpoint_ticks and not self.config.arena_cfg.general.debug_mode:
            self.tick_callbacks.append(TickCheckpointCallback(self.paths[Paths.Agent].path, arena_sb3_cfg.callbacks.checkpoint_ticks, ticks_per_step=chunk))

    def _setup_monitoring(self, *args: object, **kwargs: object) -> None:
        """Set up monitoring tools if not in debug mode."""
        if not self.config.arena_cfg.general.debug_mode and self.config.arena_cfg.monitoring is not None and self.config.arena_cfg.monitoring.wandb is not None and self.config.arena_cfg.monitoring.wandb.enabled:
            setup_wandb(
                run_name=self.config.agent_config.name,
                group=self._framework_cfg().algorithm.architecture_name,
                config=self.config,
                to_watch=[self._sb3_algorithm().policy],
                agent_id=self.config.agent_config.name,
            )

    def _train_impl(self, *args: object, **kwargs: object) -> None:
        """Implementation of training logic."""
        algorithm_cfg = self._framework_cfg().algorithm
        steps = algorithm_cfg.parameters.total_timesteps // self.config.agent_config.action_chunk
        done = self._sb3_algorithm().num_timesteps if self.is_resume else 0
        self.agent.train(
            total_timesteps=max(steps - done, 0),
            reset_num_timesteps=not self.is_resume,
            callback=CallbackList([self.eval_cb, *self.tick_callbacks]),
            progress_bar=algorithm_cfg.parameters.show_progress_bar,
        )
        self._save_model(checkpoint="last_model")

    def _complete_model_initialization(self, train_env: VecEnv) -> None:
        """Complete the initialization of the RL model.

        This method sets up the tensorboard logging directory and checkpoint path for model resumption,
        then initializes the model with the provided training environment.

        Args:
            train_env (VecEnv): The vectorized training environment to use for model initialization.

        Returns:
            None

        Note:
            - Tensorboard logs are disabled in debug mode.
            - Checkpoint path is only used when resuming training from a previous checkpoint.
            - After initialization, the training environment is attached to the model.
        """

        tensorboard_log_path = self.paths[Paths.AgentTensorboard].path if not self.config.arena_cfg.general.debug_mode else None
        checkpoint_path = None
        if self.is_resume:
            checkpoint = self._framework_cfg().algorithm.checkpoint
            if checkpoint is None:
                raise ValueError("Checkpoint must be provided for resume!")
            checkpoint_path = self.paths[Paths.Agent].path / checkpoint

        self.agent.initialize_model(
            env=train_env,
            no_gpu=self.config.arena_cfg.general.no_gpu,
            tensorboard_log_path=tensorboard_log_path,
            checkpoint_path=checkpoint_path,
        )
        self._sb3_model().environment = train_env

    def _arena_sb3_cfg(self) -> arena_cfg.ArenaSB3Cfg:
        cfg = self.config.arena_cfg
        assert isinstance(cfg, arena_cfg.ArenaSB3Cfg)
        return cfg

    def _framework_cfg(self) -> StableBaselinesCfg:
        framework = self.config.agent_config.framework
        assert isinstance(framework, StableBaselinesCfg)
        return framework

    def _sb3_model(self) -> StableBaselinesModel:
        model = self.agent.model
        assert isinstance(model, StableBaselinesModel)
        return model

    def _sb3_algorithm(self) -> BaseAlgorithm:
        algorithm = self._sb3_model().model
        assert isinstance(algorithm, BaseAlgorithm)
        return algorithm
