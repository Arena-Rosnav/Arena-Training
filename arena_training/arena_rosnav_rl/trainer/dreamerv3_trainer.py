import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import rosnav_rl
import rosnav_rl.model.dreamerv3 as dreamerv3
import arena_training.arena_rosnav_rl.cfg as arena_cfg

from rosnav_rl import SupportedRLFrameworks

from ..utils.env_factory import make_envs
from ..utils.monitoring import setup_wandb
from ..trainer.arena_trainer import ArenaTrainer
from ...environments.wrappers import TimeSyncWrapper

logger = logging.getLogger(__name__)

# Logger used inside the DreamerV3 helper / tools modules
_HELPER_LOGGER = logging.getLogger("rosnav_rl.model.dreamerv3.helper")


@dataclass
class DreamerV3Environment:
    """Container for DreamerV3 training/eval environment lists."""

    train_envs: List = field(default_factory=list)
    eval_envs: List = field(default_factory=list)

    def close(self) -> None:
        """Safely close all environment instances."""
        seen = set()
        for env in self.train_envs + self.eval_envs:
            if id(env) in seen:
                continue
            seen.add(id(env))
            try:
                env.close()
            except Exception:
                pass


class DreamerV3Trainer(ArenaTrainer):
    """Trainer for DreamerV3 reinforcement learning framework.

    Handles agent setup, environment configuration, and training
    for the DreamerV3 world-model-based RL approach.

    Attributes:
        environment (DreamerV3Environment): Container with train/eval environment lists.
    """

    _framework = SupportedRLFrameworks.DREAMER_V3
    _config_type = arena_cfg.ArenaBaseCfg
    environment: DreamerV3Environment
    _curriculum: Optional["DreamerV3Curriculum"] = None

    def __init__(self, config: arena_cfg.TrainingCfg, namespace_fn: Callable[[int], str]):
        super().__init__(config, config.resume, namespace_fn=namespace_fn)

    def _setup_monitoring(self, *args, **kwargs):
        """Set up monitoring tools (Weights & Biases) if enabled."""
        if (
            not self.config.arena_cfg.general.debug_mode
            and self.config.arena_cfg.monitoring is not None
            and self.config.arena_cfg.monitoring.wandb is not None
            and self.config.arena_cfg.monitoring.wandb.enabled
        ):
            setup_wandb(
                run_name=self.config.agent_config.name,
                group=self.config.arena_cfg.monitoring.wandb.group,
                config=self.config,
                to_watch=[self.agent.model.model],
                agent_id=self.config.agent_config.name,
            )

    def _setup_agent(self, *args, **kwargs):
        """Set up the DreamerV3 RL agent.

        Sets the framework logdir from the trainer's paths dictionary (if
        not already specified in the config) so that checkpoints, episodes,
        and logs land next to the training_config.yaml.
        """
        from ..utils import paths as Paths

        # Ensure logdir is set before DreamerV3Model.__init__ runs
        fw_cfg = self.config.agent_config.framework
        if fw_cfg.general.logdir is None and hasattr(self, "paths"):
            fw_cfg.general.logdir = str(self.paths[Paths.Agent].path)

        self.agent = rosnav_rl.RL_Agent(self.config.agent_config)
        self.agent.initialize_model()

    def _setup_environment(self, *args, **kwargs):
        """Set up the training and evaluation environments for DreamerV3.

        Creates gym environments wrapped with DreamerV3-specific wrappers and
        wraps them in either Parallel (daemon) or Damy (debug) executors.

        Wrapper chain (innermost to outermost):
            GazeboEnv -> TimeSyncWrapper -> WoTruncatedFlag -> TimeLimit
            -> SelectAction -> UUID -> ResetWoInfo -> ChannelFirsttoLast
        """
        general_cfg = self.config.arena_cfg.general
        self._configure_verbose(general_cfg.verbose)

        logger.info(
            "[Setup] Creating %d DreamerV3 environment(s) "
            "(control_hz=%.1f, max_steps=%d, debug=%s)",
            general_cfg.n_envs,
            general_cfg.control_hz,
            general_cfg.max_num_moves_per_eps,
            general_cfg.debug_mode,
        )

        train_env_fncs = make_envs(
            node=self.node if general_cfg.debug_mode else None,
            rl_agent=self.agent,
            n_envs=general_cfg.n_envs,
            max_steps=general_cfg.max_num_moves_per_eps,
            init_env_by_call=False,
            namespace_fn=self.namespace_fn,
            simulation_state_container=self.agent_parameters,
            wrappers=[
                partial(TimeSyncWrapper, control_hz=general_cfg.control_hz),
                dreamerv3.WoTruncatedFlag,
                partial(
                    dreamerv3.TimeLimit,
                    duration=general_cfg.max_num_moves_per_eps,
                ),
                partial(dreamerv3.SelectAction, key="action"),
                dreamerv3.UUID,
                dreamerv3.ResetWoInfo,
                dreamerv3.ChannelFirsttoLast,
                dreamerv3.RenameObsForDreamer,
            ],
            observations_config=self.config.agent_config.observations_config,
        )

        if general_cfg.debug_mode:
            train_envs = [dreamerv3.Damy(init_fnc()) for init_fnc in train_env_fncs]
        else:
            # Capture via default argument to avoid late-binding closure bug
            train_envs = [
                dreamerv3.Parallel(lambda _f=fnc: _f(), "daemon")
                for fnc in train_env_fncs
            ]
        # Shared envs for train and eval (same pattern as SB3 trainer)
        self.environment = DreamerV3Environment(
            train_envs=train_envs, eval_envs=train_envs
        )
        self._setup_curriculum()

    def _setup_curriculum(self) -> None:
        """Instantiate DreamerV3Curriculum from task config, if configured."""
        from ..dreamerv3.dreamerv3_curriculum import DreamerV3Curriculum

        self._curriculum = None
        task_cfg = self.config.arena_cfg.task
        if not task_cfg.has_curriculum():
            return

        general_cfg = self.config.arena_cfg.general
        self._curriculum = DreamerV3Curriculum(
            node=self.node,
            staged_cfg=task_cfg.staged,
            num_envs=general_cfg.n_envs,
            verbose=int(general_cfg.verbose),
            tm_dict={
                "tm_robots": task_cfg.tm_robots,
                "tm_obstacles": task_cfg.tm_obstacles,
                "tm_modules": task_cfg.tm_modules,
            },
        )
        staged = task_cfg.staged
        logger.info(
            "[Setup] Curriculum learning enabled — %d stages, "
            "threshold_type=%s, advance\u2265%.2f, retreat\u2264%.2f",
            len(staged.curriculum_definition),
            staged.threshold_type,
            staged.upper_threshold,
            staged.lower_threshold,
        )

    def _configure_verbose(self, verbose) -> None:
        """Apply log levels for rosnav_rl namespaces and this trainer."""
        from ..utils.log_utils import configure_trainer_logging

        configure_trainer_logging(
            logging_cfg=getattr(self.config.arena_cfg, "logging", None),
            verbose=int(verbose),
            framework_logger=_HELPER_LOGGER,
            trainer_logger=logger,
        )

    def _train_impl(self, *args, **kwargs):
        """Run the DreamerV3 training loop via the model's own train() method."""
        fw_cfg = self.config.agent_config.framework
        logger.info(
            "[Train] Starting DreamerV3 training — "
            "total_steps=%d  eval_every=%d  batch=%dx%d  device=%s",
            fw_cfg.training.steps,
            fw_cfg.training.eval_every,
            fw_cfg.training.batch_size,
            fw_cfg.training.batch_length,
            fw_cfg.general.device,
        )
        self.agent.model.train(
            train_envs=self.environment.train_envs,
            eval_envs=self.environment.eval_envs,
            after_eval_fn=self._curriculum.after_eval_hook if self._curriculum else None,
        )
        logger.info("[Train] Training complete.")

