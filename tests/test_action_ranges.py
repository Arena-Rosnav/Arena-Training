"""The policy's command bounds are the robot's rated velocity limits from its mobile caps."""

from __future__ import annotations

import pytest

from arena_training.arena_rosnav_rl.cfg.arena_cfg.robot import RobotCfg
from arena_training.arena_rosnav_rl.trainer.arena_trainer import action_ranges


def test_jackal_commands_are_bounded_by_its_rated_limits():
    jackal = RobotCfg(robot_model="jackal").robot_description
    linear, lateral, angular = action_ranges(jackal)
    assert linear == pytest.approx((jackal.velocity_limits.linear.min, jackal.velocity_limits.linear.max))
    assert angular == pytest.approx((jackal.velocity_limits.angular.min, jackal.velocity_limits.angular.max))
    assert lateral == pytest.approx(linear)
    assert (linear, angular) == ((-0.5, 0.5), (-1.5, 1.5))
