"""Collisions in the grace window neither end the episode nor cost the collision penalty, on steps recorded from a robot spawned against an obstacle in the flatsim training stack."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations import DONE_REASONS
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.reward.reward_units.reward_units import RewardCollision

from arena_training.arena_rosnav_rl.utils.envs import determine_termination
from arena_training.environments.base_env import suppress_grace_collision

RECORDED = json.loads((pathlib.Path(__file__).parent / "data" / "grace_steps.json").read_text())
REWARDS = {"safe_distance": {"reward": -0.1}, "factored_safe_distance": {"factor": -0.2}, "collision": {"reward": -15}}


def _step(reward_function: RewardFunction, parameters: AgentParameters, ranges: np.ndarray, step: int) -> tuple[float, bool, dict]:
    obs = {"front_laser": ranges, "laser_safety_violation": bool(ranges.min() <= parameters.safety_distance), "collision_monitor": False, "simulation_state_container": parameters}
    reward, reward_info = reward_function.get_reward(obs_dict=obs, simulation_state_container=parameters)
    info, done = determine_termination(reward_info=reward_info, curr_steps=step, max_steps=600)
    reward, done = suppress_grace_collision(step, reward, done, info, reward_function)
    return reward, done, info


def _ranges(recorded: dict) -> np.ndarray:
    ranges = np.asarray(recorded["ranges"], dtype=np.float32)
    return np.nan_to_num(ranges, nan=recorded["range_max"], posinf=recorded["range_max"])


def test_grace_steps_keep_the_episode_and_drop_only_the_collision_penalty():
    parameters = AgentParameters(robot_radius=RECORDED["robot_radius"], safety_distance=RECORDED["safety_distance"])
    reward_function = RewardFunction(function_dict=REWARDS)
    collision = next(unit for unit in reward_function.reward_units if isinstance(unit, RewardCollision))
    steps = [_ranges(recorded) for recorded in RECORDED["steps"]]
    assert all(ranges.min() <= RECORDED["robot_radius"] + 0.05 for ranges in steps)

    for step, ranges in enumerate(steps, start=1):
        reward, done, info = _step(reward_function, parameters, ranges, step)
        overview = reward_function.state.reward_overview
        assert not done
        assert info == {}
        assert "RewardCollision" not in overview
        assert reward == pytest.approx(sum(overview.values()), abs=1e-5)
        assert reward < 0.0
        assert not collision._collision_latched

    reward, done, info = _step(reward_function, parameters, steps[-1], len(steps) + 1)
    assert done
    assert str(info["done_reason"]) == str(DONE_REASONS.COLLISION)
    assert reward_function.state.reward_overview["RewardCollision"] == -15
