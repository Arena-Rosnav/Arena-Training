"""The reward terms on recorded steps: approach pays progress and charges retreat at the same rate, the step cost charges every step, safe distance charges 0.02 per step inside 1 m."""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.reward.reward_function import RewardFunction

RECORDED = json.loads((pathlib.Path(__file__).parent / "data" / "recorded_episode.json").read_text())
TERMS = {
    "safe_distance": {"reward": -0.02},
    "step_cost": {"reward": -0.01},
    "approach_goal": {"pos_factor": 1.0, "neg_factor": 1.0, "_goal_update_threshold": 0.25, "_follow_subgoal": False, "_on_safe_dist_violation": True},
}
Pose2DType = np.dtype([("x", np.float32), ("y", np.float32), ("yaw", np.float32)])


def _relative(pose: list[float], goal: list[float]) -> np.ndarray:
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    return np.array([cos * dx + sin * dy, -sin * dx + cos * dy])


def test_recorded_steps_cost_and_pay_by_the_ruled_table():
    parameters = AgentParameters(safety_distance=1.0)
    reward_function = RewardFunction(function_dict=TERMS)
    totals: dict[str, float] = {}
    for step in RECORDED["steps"]:
        relative = _relative(step["pose"], step["goal"])
        obs = {
            "robot_pose": np.array(tuple(step["pose"]), dtype=Pose2DType),
            "goal_in_robot_frame": relative,
            "subgoal_in_robot_frame": relative,
            "laser_safety_violation": step["scan_min"] <= parameters.safety_distance,
            "simulation_state_container": parameters,
        }
        reward_function.get_reward(obs_dict=obs, simulation_state_container=parameters)
        for term, value in reward_function.state.reward_overview.items():
            totals[term] = totals.get(term, 0.0) + value

    steps = RECORDED["steps"]
    distances = [math.dist(step["pose"][:2], step["goal"]) for step in steps]
    retreats = sum(1 for before, after in zip(distances, distances[1:]) if after > before)
    violations = sum(step["scan_min"] <= 1.0 for step in steps)
    assert retreats > 0 and 0 < violations < len(steps)
    assert totals["RewardApproachGoal"] == pytest.approx(distances[0] - distances[-1], abs=1e-4)
    assert totals["RewardStepCost"] == pytest.approx(-0.01 * len(steps))
    assert totals["RewardSafeDistance"] == pytest.approx(-0.02 * violations)
