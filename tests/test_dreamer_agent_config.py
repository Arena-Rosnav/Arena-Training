"""Dreamer trains a fresh continuous agent on normalized actions and leaves a resumed one as it was saved."""

from __future__ import annotations

import pathlib

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace, DiscretizationCfg, DiscretizationStrategy

from arena_training.arena_rosnav_rl.trainer.dreamerv3_trainer import DreamerV3Trainer, dreamer_agent_config
from arena_training.arena_rosnav_rl.utils.config import load_training_config

CONFIGS = pathlib.Path(__file__).parents[1] / "configs"
FORWARD = DifferentialDriveActionSpace(linear_range=(0.0, 0.5), angular_range=(-1.5, 1.5))


def _agent_config(**update: object):
    config = load_training_config(str(CONFIGS / "dreamer_training_config.yaml"))
    return config.agent_config.model_copy(update={"action_space": FORWARD, "normalized_actions": False, **update})


def test_a_fresh_continuous_training_acts_on_normalized_actions():
    assert dreamer_agent_config(_agent_config(), resume=False).normalized_actions


def test_a_resumed_training_keeps_its_saved_action_mapping():
    assert not dreamer_agent_config(_agent_config(), resume=True).normalized_actions


def test_a_discrete_action_space_keeps_its_action_mapping():
    uniform = DiscretizationCfg(strategy=DiscretizationStrategy.UNIFORM, buckets_linear=3, buckets_angular=3)
    discrete = FORWARD.model_copy(update={"discretization": uniform}).resolve_discretization()

    assert not dreamer_agent_config(_agent_config(action_space=discrete), resume=False).normalized_actions


def test_the_saved_training_config_records_the_normalized_actions(tmp_path):
    config = load_training_config(str(CONFIGS / "dreamer_training_config.yaml"))
    config.agent_config = config.agent_config.model_copy(update={"action_space": FORWARD, "normalized_actions": False})
    trainer = object.__new__(DreamerV3Trainer)
    trainer.config = config

    trainer._populate_agent_spec()
    config.to_yaml(tmp_path / "training_config.yaml")

    assert "normalized_actions: true" in (tmp_path / "training_config.yaml").read_text()
