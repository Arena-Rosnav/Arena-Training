"""An arm command is a one-point trajectory reaching its positions after one step."""

from __future__ import annotations

import numpy as np

from arena_training.arena_rosnav_rl.utils.envs import get_joint_trajectory_from_action


def test_the_point_reaches_the_positions_after_the_step_time():
    trajectory = get_joint_trajectory_from_action(np.array([0.5, -1.25]), ["shoulder", "elbow"], step_time=1.25)
    assert trajectory.joint_names == ["shoulder", "elbow"]
    assert len(trajectory.points) == 1
    assert list(trajectory.points[0].positions) == [0.5, -1.25]
    assert (trajectory.points[0].time_from_start.sec, trajectory.points[0].time_from_start.nanosec) == (1, 250_000_000)


def test_a_sub_second_step_has_no_whole_seconds():
    start = get_joint_trajectory_from_action(np.zeros(1), ["shoulder"], step_time=0.1).points[0].time_from_start
    assert (start.sec, start.nanosec) == (0, 100_000_000)
