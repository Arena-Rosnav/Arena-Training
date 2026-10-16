"""DelayedSubprocVecEnv restarts a worker that dies mid-step and training goes on."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cloudpickle
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from rosnav_rl.observations import DONE_REASONS
from stable_baselines3 import PPO

from arena_training.arena_rosnav_rl.stable_baselines3.vec_wrapper import VecStatsRecorder
from arena_training.arena_rosnav_rl.stable_baselines3.vec_wrapper.delayed_subproc_vec_env import DelayedSubprocVecEnv

cloudpickle.register_pickle_by_value(sys.modules[__name__])


class _PidEnv(gym.Env):
    """Observes its worker pid, ends episodes after ten steps, exits its worker on a step while the crash file exists."""

    lockstep = False

    def __init__(self, crash: Path | None):
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._crash = crash
        self._t = 0

    def _initialize_environment(self) -> None:
        return None

    def _obs(self) -> np.ndarray:
        return np.array([float(os.getpid())])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._t = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        if self._crash is not None and self._crash.exists():
            self._crash.unlink()
            os._exit(1)
        self._t += 1
        done = self._t >= 10
        info = {"done_reason": DONE_REASONS.STEP_LIMIT, "episode_length": self._t} if done else {}
        return self._obs(), 1.0, done, False, info


class _GatheredPidEnv(_PidEnv):
    """A _PidEnv in gathered lockstep mode."""

    lockstep = True

    def apply_action(self, action: np.ndarray) -> None:
        return None

    def step_sim(self) -> None:
        return None


def _pool(crash: Path | None, env: type[_PidEnv] = _PidEnv) -> DelayedSubprocVecEnv:
    return DelayedSubprocVecEnv([lambda: env(None), lambda: env(crash)], start_method="forkserver")


def test_worker_dead_mid_step_is_restarted_in_place(tmp_path: Path):
    crash = tmp_path / "crash"
    venv = _pool(crash)
    try:
        venv.reset()
        old = venv.processes[1].pid
        crash.touch()

        obs, rewards, dones, infos = venv.step(np.zeros((2, 1), dtype=np.float32))

        assert dones.tolist() == [False, True]
        assert rewards[1] == 0.0
        assert infos[1]["worker_restarted"]
        assert infos[1]["episode_length"] == 0
        assert not infos[1]["TimeLimit.truncated"]
        assert venv.processes[1].pid != old
        assert venv.processes[1].is_alive()
        assert obs[1, 0] == venv.processes[1].pid

        obs, _, dones, infos = venv.step(np.zeros((2, 1), dtype=np.float32))
        assert dones.tolist() == [False, False]
        assert "worker_restarted" not in infos[1]
        assert obs[1, 0] == venv.processes[1].pid
    finally:
        venv.close()


def test_ppo_trains_past_a_worker_dead_mid_step(tmp_path: Path):
    crash = tmp_path / "crash"
    venv = _pool(crash)
    try:
        stats = VecStatsRecorder(venv, after_x_eps=100)
        model = PPO("MlpPolicy", stats, n_steps=16, batch_size=16, n_epochs=1, device="cpu")
        crash.touch()
        model.learn(total_timesteps=64)

        assert not crash.exists()
        assert model.num_timesteps >= 64
        assert all(p.is_alive() for p in venv.processes)
        assert stats.done_reasons["worker_restarted"] == 1
    finally:
        venv.close()


@pytest.mark.parametrize(("env", "dead"), [(_PidEnv, 1), (_GatheredPidEnv, 0), (_GatheredPidEnv, 1)])
def test_worker_dead_between_steps_is_restarted_in_place(env: type[_PidEnv], dead: int):
    venv = _pool(None, env)
    try:
        venv.reset()
        old = venv.processes[dead]
        old.kill()
        old.join()

        obs, rewards, dones, infos = venv.step(np.zeros((2, 1), dtype=np.float32))

        assert dones.tolist() == [i == dead for i in range(2)]
        assert rewards[dead] == 0.0
        assert infos[dead]["worker_restarted"]
        assert infos[dead]["done_reason"] == "worker_restarted"
        assert infos[dead]["episode_length"] == 0
        assert venv.processes[dead].pid != old.pid
        assert obs[dead, 0] == venv.processes[dead].pid
        assert obs[1 - dead, 0] == venv.processes[1 - dead].pid

        obs, _, dones, infos = venv.step(np.zeros((2, 1), dtype=np.float32))
        assert dones.tolist() == [False, False]
        assert "worker_restarted" not in infos[dead]
        assert obs[dead, 0] == venv.processes[dead].pid
    finally:
        venv.close()
