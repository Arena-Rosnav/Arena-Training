"""A configured agent_config action_space bounds the policy's commands, and an arm space is refused."""

from __future__ import annotations

import pathlib
import types

import pytest
from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace, ManipulatorActionSpace, OmnidirectionalActionSpace

from arena_training.arena_rosnav_rl.cfg.arena_cfg.robot import RobotCfg
from arena_training.arena_rosnav_rl.trainer.arena_trainer import ArenaTrainer, action_ranges, mobile_action_space
from arena_training.arena_rosnav_rl.utils.config import load_training_config

CONFIGS = pathlib.Path(__file__).parents[1] / "configs"


def _velocity_bounds(**agent_update: object) -> dict[str, float]:
    config = load_training_config(str(CONFIGS / "sb_training_config.yaml"))
    config.agent_config = config.agent_config.model_copy(update=agent_update)
    ArenaTrainer._populate_agent_spec(types.SimpleNamespace(config=config))
    return config.agent_config.parameters.model_dump(include={"min_linear_vel", "max_linear_vel", "min_translational_vel", "max_translational_vel", "min_angular_vel", "max_angular_vel"})


def test_a_configured_forward_only_action_space_replaces_the_rated_limits():
    forward = DifferentialDriveActionSpace(linear_range=(0.0, 0.5), angular_range=(-1.5, 1.5))
    space = mobile_action_space(RobotCfg(robot_model="jackal").robot_description, forward)
    assert space is forward
    assert space.get_gym_space().low.tolist() == [0.0, -1.5]


def test_a_configured_arm_action_space_is_rejected():
    arm = ManipulatorActionSpace(joint_names=["shoulder"], position_limits=[(-1.0, 1.0)], controller="arm_controller", tip_link="tool")
    with pytest.raises(ValueError, match="differential_drive or omnidirectional"):
        mobile_action_space(RobotCfg(robot_model="jackal").robot_description, arm)


def test_the_observation_velocity_bounds_follow_a_configured_action_space():
    forward = DifferentialDriveActionSpace(linear_range=(0.0, 0.5), angular_range=(-1.5, 1.5))
    assert _velocity_bounds(action_space=forward) == {
        "min_linear_vel": 0.0,
        "max_linear_vel": 0.5,
        "min_translational_vel": 0.0,
        "max_translational_vel": 0.5,
        "min_angular_vel": -1.5,
        "max_angular_vel": 1.5,
    }


def test_the_observation_velocity_bounds_follow_the_rated_limits_without_a_configured_action_space():
    config = load_training_config(str(CONFIGS / "sb_training_config.yaml"))
    assert config.arena_cfg.robot.robot_description is not None
    linear, _lateral, angular = action_ranges(config.arena_cfg.robot.robot_description)
    bounds = _velocity_bounds(action_space=None)
    assert (bounds["min_linear_vel"], bounds["max_linear_vel"]) == linear
    assert (bounds["min_angular_vel"], bounds["max_angular_vel"]) == angular


def test_the_translational_bounds_follow_the_lateral_range_of_an_omnidirectional_action_space():
    omni = OmnidirectionalActionSpace(linear_range_x=(-0.4, 0.8), linear_range_y=(-0.3, 0.3), angular_range=(-1.0, 1.0))
    bounds = _velocity_bounds(action_space=omni)
    assert (bounds["min_linear_vel"], bounds["max_linear_vel"]) == (-0.4, 0.8)
    assert (bounds["min_translational_vel"], bounds["max_translational_vel"]) == (-0.3, 0.3)
