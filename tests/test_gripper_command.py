"""A gripper action becomes a GripperCommand goal at the decoded position with the caps' effort."""

from __future__ import annotations

import numpy as np
from rosnav_rl.cfg.action_spaces import GripperActionSpace

from arena_training.arena_rosnav_rl.utils.envs import get_gripper_command


def test_the_goal_carries_the_position_and_the_effort():
    goal = get_gripper_command(0.8, 50.0)
    assert (goal.command.position, goal.command.max_effort) == (0.8, 50.0)


def test_a_full_opening_action_commands_the_open_position():
    spec = GripperActionSpace(joint="knuckle", controller="gripper_controller", positions={"open": 0.0, "close": 0.8}, max_effort=50.0)
    goal = get_gripper_command(float(spec.decode(np.array([1.0]))[0]), spec.max_effort)
    assert (goal.command.position, goal.command.max_effort) == (0.0, 50.0)
