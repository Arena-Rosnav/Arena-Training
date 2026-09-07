"""Monitoring / experiment-tracking helpers (Weights & Biases)."""

import logging
from typing import TYPE_CHECKING

import torch
import wandb

if TYPE_CHECKING:
    from ..cfg import TrainingCfg


def setup_wandb(
    run_name: str = None,
    group: str = None,
    config: "TrainingCfg" = None,
    agent_id: str = None,
    to_watch: list[torch.nn.Module] | None = None,
) -> None:
    """Set up Weights and Biases (wandb) for training tracking."""
    logger = logging.getLogger(__name__)
    try:
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
        for module in to_watch or ():
            wandb.watch(module, log_graph=True)
    except Exception as e:
        logger.warning(f"[W&B] Failed to initialize (no network?): {e}. Continuing without W&B.")
