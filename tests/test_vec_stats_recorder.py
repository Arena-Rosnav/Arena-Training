"""Tests for episode statistics over task-generator and reward-side done reasons."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rosnav_rl.observations import DONE_REASONS
from stable_baselines3.common.vec_env import DummyVecEnv

from arena_training.arena_rosnav_rl.stable_baselines3.vec_wrapper import VecStatsRecorder


class _OneStepEpisodes(gym.Env):
    """Ends every episode after one step with the given done reason."""

    def __init__(self, reason: object):
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reason = reason

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        info = {"done_reason": self.reason, "episode_length": 1}
        return np.zeros(1, dtype=np.float32), 1.0, True, False, info


def test_task_generator_outcomes_are_counted_next_to_reward_reasons():
    venv = VecStatsRecorder(
        DummyVecEnv([lambda: _OneStepEpisodes("task_generator:FAILED"), lambda: _OneStepEpisodes(DONE_REASONS.COLLISION)]),
        after_x_eps=100,
    )
    venv.reset()

    venv.step(np.zeros((2, 1), dtype=np.float32))
    venv.step(np.zeros((2, 1), dtype=np.float32))

    assert venv.done_reasons["task_generator:FAILED"] == 2
    assert venv.done_reasons["COLLISION"] == 2
    assert venv.done_reasons["SUCCESS"] == 0
