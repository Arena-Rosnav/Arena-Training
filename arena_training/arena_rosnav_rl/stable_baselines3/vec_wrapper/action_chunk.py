from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


class VecActionChunk(VecEnvWrapper):
    """Runs a flattened chunk of `horizon` commands one per tick, per-tick discounted, idling envs whose episode ended."""

    def __init__(self, venv: VecEnv, horizon: int, gamma: float):
        if not isinstance(venv.action_space, spaces.Box):
            raise ValueError("action chunking needs a continuous (Box) action space")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        base = venv.action_space
        super().__init__(
            venv,
            action_space=spaces.Box(
                low=np.tile(base.low, horizon),
                high=np.tile(base.high, horizon),
                dtype=base.dtype,
            ),
        )
        self.horizon = horizon
        self.gamma = gamma
        self._step_shape = base.shape
        self._chunks: np.ndarray | None = None

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self._chunks = np.asarray(actions).reshape(self.num_envs, self.horizon, *self._step_shape)

    def step_wait(self) -> VecEnvStepReturn:
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        idle = np.zeros(self._step_shape, dtype=self._chunks.dtype)
        observations = None
        for tick in range(self.horizon):
            commands = np.stack([idle if dones[i] else self._chunks[i, tick] for i in range(self.num_envs)])
            observations, tick_rewards, tick_dones, tick_infos = self.venv.step(commands)
            for i in range(self.num_envs):
                if dones[i]:
                    continue
                rewards[i] += self.gamma**tick * tick_rewards[i]
                infos[i] = tick_infos[i]
                dones[i] = tick_dones[i]
        self._chunks = None
        return observations, rewards, dones, infos
