"""Monitoring / experiment-tracking helpers (Weights & Biases)."""

import re
from typing import List, TYPE_CHECKING

import torch

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.utils import configure_logger, constant_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import wandb

if TYPE_CHECKING:
    from ..cfg import TrainingCfg


def setup_wandb(
    run_name: str = None,
    group: str = None,
    config: "TrainingCfg" = None,
    agent_id: str = None,
    to_watch: List[torch.nn.Module] = [],
) -> None:
    """Set up Weights and Biases (wandb) for training tracking."""
    wandb.login()
    wandb.init(
        name=run_name if run_name else config.arena_cfg.monitoring.wandb.run_name,
        group=group if group else config.arena_cfg.monitoring.wandb.group,
        project=config.arena_cfg.monitoring.wandb.project_name,
        tags=config.arena_cfg.monitoring.wandb.tags,
        entity=None,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
        config=config.model_dump(),
        id=agent_id,
    )
    for module in to_watch:
        wandb.watch(module, log_graph=True)
