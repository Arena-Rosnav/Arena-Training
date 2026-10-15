"""Goal reached fires at the tolerance task_generator judges the episode with, read from a recorded episode record."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.reward.reward_function import RewardFunction
from task_generator_msgs.msg import EpisodeRecord

from arena_training.environments.base_env import goal_tolerance

RECORDED = json.loads((pathlib.Path(__file__).parent / "data" / "episode_phases.json").read_text())


def _record() -> EpisodeRecord:
    return EpisodeRecord(episode_id=1, outcome_state=EpisodeRecord.RUNNING, phases=RECORDED["phases"])


def _reached(parameters: AgentParameters, distance: float) -> tuple[float, dict]:
    reward_function = RewardFunction(function_dict={"goal_reached": {"reward": 20}})
    obs = {"dist_angle_to_goal": np.array([distance, 0.0]), "dist_angle_to_subgoal": np.array([distance, 0.0]), "simulation_state_container": parameters}
    return reward_function.get_reward(obs_dict=obs, simulation_state_container=parameters)


def test_episode_record_gives_the_judged_tolerance():
    assert goal_tolerance(_record(), RECORDED["robot"]) == pytest.approx(1.0)
    assert goal_tolerance(_record(), "another_robot") is None
    assert goal_tolerance(EpisodeRecord(), RECORDED["robot"]) is None


def test_goal_reached_fires_inside_the_judged_tolerance_and_not_beyond():
    parameters = AgentParameters(goal_radius=0.5).model_copy(update={"goal_radius": goal_tolerance(_record(), RECORDED["robot"])})

    reward, info = _reached(parameters, 0.9)
    assert reward == pytest.approx(20.0)
    assert info["is_success"]

    reward, info = _reached(parameters, 1.1)
    assert reward == 0.0
    assert not info["is_done"]
