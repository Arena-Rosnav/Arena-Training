"""Tests for executing chunked actions one command per tick."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from arena_training.arena_rosnav_rl.stable_baselines3.vec_wrapper import VecActionChunk


class _CommandLog(gym.Env):
    """Rewards the commanded linear velocity, ends after `length` ticks, records every command."""

    def __init__(self, length: int):
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-0.5, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32))
        self.length = length
        self.ticks = 0
        self.commands: list[list[float]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.ticks = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action: np.ndarray):
        self.ticks += 1
        self.commands.append([float(a) for a in action])
        done = self.ticks >= self.length
        return np.array([self.ticks], dtype=np.float32), float(action[0]), done, False, {"tick": self.ticks}


def _chunked(lengths: list[int], horizon: int, gamma: float) -> tuple[VecActionChunk, list[_CommandLog]]:
    envs = [_CommandLog(n) for n in lengths]
    venv = DummyVecEnv([lambda env=env: env for env in envs])
    return VecActionChunk(venv, horizon=horizon, gamma=gamma), envs


def test_chunk_space_tiles_the_command_bounds():
    venv, _ = _chunked([10], horizon=3, gamma=0.9)

    np.testing.assert_allclose(venv.action_space.low, [-0.5, -1.0] * 3)
    np.testing.assert_allclose(venv.action_space.high, [1.0, 1.0] * 3)


def test_each_command_runs_for_one_tick_in_order_and_rewards_are_discounted_per_tick():
    venv, envs = _chunked([10], horizon=3, gamma=0.9)
    venv.reset()

    obs, rewards, dones, infos = venv.step(np.array([[0.1, 0.0, 0.2, 0.5, 0.3, -0.5]], dtype=np.float32))

    np.testing.assert_allclose(envs[0].commands, [[0.1, 0.0], [0.2, 0.5], [0.3, -0.5]], atol=1e-6)
    assert rewards[0] == pytest.approx(0.1 + 0.9 * 0.2 + 0.81 * 0.3)
    assert obs[0, 0] == 3.0
    assert not dones[0]
    assert infos[0]["tick"] == 3


def test_an_episode_ending_inside_the_chunk_idles_the_rest_and_keeps_its_terminal_info():
    venv, envs = _chunked([2, 10], horizon=3, gamma=0.5)
    venv.reset()
    chunk = np.array([0.4, 0.1, 0.4, 0.1, 0.4, 0.1], dtype=np.float32)

    obs, rewards, dones, infos = venv.step(np.stack([chunk, chunk]))

    assert dones.tolist() == [True, False]
    assert rewards[0] == pytest.approx(0.4 + 0.5 * 0.4)
    assert rewards[1] == pytest.approx(0.4 + 0.5 * 0.4 + 0.25 * 0.4)
    assert infos[0]["tick"] == 2
    np.testing.assert_allclose(infos[0]["terminal_observation"], [2.0])
    np.testing.assert_allclose(envs[0].commands[-1], [0.0, 0.0])
    assert obs[0, 0] == 1.0
