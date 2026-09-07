"""Trainer setup helpers, path management, config serialization, display."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name

if TYPE_CHECKING:
    from ruamel.yaml.comments import CommentedMap

    from ..cfg.train import TrainingCfg
    from ..trainer.arena_trainer import ArenaTrainer

from .paths import PathDictionary, PathFactory


def write_config_yaml(config: dict, path: str) -> None:
    with open(path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def build_training_commented_map(cfg: TrainingCfg) -> CommentedMap:
    """Build a structured ruamel.yaml CommentedMap for the full training config."""
    from ruamel.yaml.comments import CommentedMap

    root = CommentedMap()

    # Agent
    root["agent_config"] = cfg.agent_config._to_commented_map()
    root.yaml_set_comment_before_after_key("agent_config", before="Agent configuration")

    # agents_dir (usually null, keep it visible)
    root["agents_dir"] = str(cfg.agents_dir) if cfg.agents_dir is not None else None

    # Arena / simulation
    root["arena_cfg"] = cfg.arena_cfg.model_dump(mode="json")
    root.yaml_set_comment_before_after_key("arena_cfg", before="\nArena / simulation configuration")

    # Training metadata
    root["resume"] = cfg.resume
    root.yaml_set_comment_before_after_key("resume", before="\nTraining metadata")

    return root


def print_dict(hyperparams: dict) -> None:
    print("\n--------------------------------")
    print("         HYPERPARAMETERS         \n")
    for param, param_val in hyperparams.items():
        print("{:30s}{:<10s}".format(f"{param}:", str(param_val)))
    print("--------------------------------\n\n")


def print_base_model(hyperparams: BaseModel) -> None:
    print("\n--------------------------------")
    print("         HYPERPARAMETERS         \n")
    yaml_str = yaml.dump(hyperparams.model_dump(), default_flow_style=False, sort_keys=False)
    colorful_yaml = highlight(yaml_str, get_lexer_by_name("yaml"), TerminalFormatter())
    print(colorful_yaml)
    print("--------------------------------\n\n")


def create_directories(
    paths: dict,
    resume_name: str,
    log_evaluation: bool,
    use_wandb: bool,
) -> None:
    create_model_directory(paths, resume_name)
    paths["eval"] = create_evaluation_directory(paths, log_evaluation)
    paths["tb"] = create_tensorboard_directory(paths, use_wandb)


def create_model_directory(paths: dict, resume_name: str) -> None:
    if resume_name is None:
        os.makedirs(paths["model"])


def create_evaluation_directory(paths: dict, log_evaluation: bool) -> str:
    if log_evaluation:
        if not os.path.exists(paths["eval"]):
            os.makedirs(paths["eval"])
        return paths["eval"]
    return None


def create_tensorboard_directory(paths: dict, use_wandb: bool) -> str:
    if use_wandb:
        if not os.path.exists(paths["tb"]):
            os.makedirs(paths["tb"])
        return paths["tb"]
    return None


def load_config(file_path: str) -> dict:
    """Load config parameters from config file."""
    with open(file_path, encoding="utf-8") as target:
        config = yaml.load(target, Loader=yaml.FullLoader)
    return config


def setup_paths_dictionary(trainer: ArenaTrainer, is_debug_mode: bool = False) -> PathDictionary:
    agents_dir = trainer.config.resolved_agents_dir
    trainer.paths = PathFactory.get_paths(
        trainer.config.agent_config.name,
        trainer.config.arena_cfg.robot.robot_model,
        agents_dir=agents_dir,
    )
    if not is_debug_mode:
        trainer.paths.create_all()


def load_yaml(file_path: str) -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)
