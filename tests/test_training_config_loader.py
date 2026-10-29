from __future__ import annotations

import pathlib

import pytest
import yaml

from arena_training.arena_rosnav_rl.utils.config import load_training_config

CONFIGS = pathlib.Path(__file__).parents[1] / "configs"
TRAINING_CONFIGS = sorted(CONFIGS.glob("*.yaml"))


@pytest.mark.parametrize("config", TRAINING_CONFIGS, ids=lambda p: p.stem)
def test_every_shipped_training_config_loads(config: pathlib.Path):
    load_training_config(str(config))


@pytest.mark.parametrize(
    ("action_space", "parsed"),
    [
        (
            {"type": "differential_drive", "linear_range": [0.0, 0.5], "angular_range": [-1.5, 1.5]},
            {"linear_range": (0.0, 0.5), "angular_range": (-1.5, 1.5)},
        ),
        (
            {"type": "omnidirectional", "linear_range_x": [0.0, 0.5], "linear_range_y": [-0.5, 0.5], "angular_range": [-1.5, 1.5]},
            {"linear_range_x": (0.0, 0.5), "linear_range_y": (-0.5, 0.5), "angular_range": (-1.5, 1.5)},
        ),
        (
            {"type": "arm", "joint_names": ["shoulder"], "position_limits": [[-1.0, 1.0]], "controller": "arm_controller", "tip_link": "tool"},
            {"position_limits": [(-1.0, 1.0)]},
        ),
    ],
)
def test_yaml_lists_load_into_action_space_tuples(tmp_path: pathlib.Path, action_space: dict, parsed: dict):
    raw = yaml.safe_load((CONFIGS / "sb_training_config.yaml").read_text())
    raw["agent_config"]["action_space"] = action_space
    config = tmp_path / "training_config.yaml"
    config.write_text(yaml.safe_dump(raw))
    space = load_training_config(str(config)).agent_config.action_space
    assert {key: getattr(space, key) for key in parsed} == parsed
