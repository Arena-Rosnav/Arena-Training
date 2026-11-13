"""A configured agent_config action_space bounds the policy's commands, and an arm space is refused."""

from __future__ import annotations

import pytest
from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace, ManipulatorActionSpace

from arena_training.arena_rosnav_rl.cfg.arena_cfg.robot import RobotCfg
from arena_training.arena_rosnav_rl.trainer.arena_trainer import mobile_action_space


def test_a_configured_forward_only_action_space_replaces_the_rated_limits():
    forward = DifferentialDriveActionSpace(linear_range=(0.0, 0.5), angular_range=(-1.5, 1.5))
    space = mobile_action_space(RobotCfg(robot_model="jackal").robot_description, forward)
    assert space is forward
    assert space.get_gym_space().low.tolist() == [0.0, -1.5]


def test_a_configured_arm_action_space_is_rejected():
    arm = ManipulatorActionSpace(joint_names=["shoulder"], position_limits=[(-1.0, 1.0)], controller="arm_controller", tip_link="tool")
    with pytest.raises(ValueError, match="differential_drive or omnidirectional"):
        mobile_action_space(RobotCfg(robot_model="jackal").robot_description, arm)
