import logging
from dataclasses import dataclass, field
from functools import partial
from typing import List

import rosnav_rl
import rosnav_rl.model.dreamerv3 as dreamerv3
from rosnav_rl import SupportedRLFrameworks
import arena_training.arena_rosnav_rl.cfg as arena_cfg

from ..tools.config import load_training_config
from ..tools.env_utils import make_envs
from ..tools.model_utils import setup_wandb
from ..trainer.arena_trainer import ArenaTrainer

logger = logging.getLogger(__name__)


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

    def __init__(self, config: arena_cfg.TrainingCfg):
        super().__init__(config, config.resume)

    def _setup_monitoring(self, *args, **kwargs):
        """Set up monitoring tools (Weights & Biases) if enabled."""
        if (
            not self.config.arena_cfg.general.debug_mode
            and self.config.arena_cfg.monitoring is not None
            and self.config.arena_cfg.monitoring.wandb is not None
            and self.config.arena_cfg.monitoring.wandb.enabled
        ):
            setup_wandb(
                run_name=self.config.agent_cfg.name,
                group=self.config.arena_cfg.monitoring.wandb.group,
                config=self.config,
                to_watch=[self.agent.model.model],
                agent_id=self.config.agent_cfg.name,
            )

    def _setup_agent(self, *args, **kwargs):
        """Set up the DreamerV3 RL agent.

        Sets the framework logdir from the trainer's paths dictionary (if
        not already specified in the config) so that checkpoints, episodes,
        and logs land next to the training_config.yaml.
        """
        from ..utils import paths as Paths

        # Ensure logdir is set before DreamerV3Model.__init__ runs
        fw_cfg = self.config.agent_cfg.framework
        if fw_cfg.general.logdir is None and hasattr(self, "paths"):
            fw_cfg.general.logdir = str(self.paths[Paths.Agent].path)

        self.agent = rosnav_rl.RL_Agent(
            agent_cfg=self.config.agent_cfg,
            agent_state_container=self.agent_state_container,
        )
        self.agent.initialize_model()

    def _setup_environment(self, *args, **kwargs):
        """Set up the training and evaluation environments for DreamerV3.

        Creates gym environments wrapped with DreamerV3-specific wrappers and
        wraps them in either Parallel (daemon) or Damy (debug) executors.

        Wrapper chain (innermost to outermost):
            GazeboEnv -> WoTruncatedFlag -> TimeLimit -> SelectAction
            -> UUID -> ResetWoInfo -> ChannelFirsttoLast
        """
        general_cfg = self.config.arena_cfg.general

        train_env_fncs = make_envs(
            node=self.node if general_cfg.debug_mode else None,
            rl_agent=self.agent,
            n_envs=general_cfg.n_envs,
            max_steps=general_cfg.max_num_moves_per_eps,
            init_env_by_call=False,
            namespace_fn=lambda idx: f"/task_generator_node/env{idx}/jackal",
            simulation_state_container=self.simulation_state_container,
            wrappers=[
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
            observations_config=general_cfg.observations_config,
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

    def _train_impl(self, *args, **kwargs):
        """Run the DreamerV3 training loop via the model's own train() method."""
        self.agent.model.train(
            train_envs=self.environment.train_envs,
            eval_envs=self.environment.eval_envs,
        )


if __name__ == "__main__":
    config = load_training_config("dreamer_training_config.yaml")

    trainer = DreamerV3Trainer(config)
    trainer.train()
