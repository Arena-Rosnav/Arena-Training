from copy import deepcopy

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn


class GatheredDummyVecEnv(DummyVecEnv):
    """DummyVecEnv whose lockstep envs share one sim step per vec step."""

    def step_wait(self) -> VecEnvStepReturn:
        if not all(self.get_attr("lockstep")):
            return super().step_wait()
        for index, action in enumerate(self.actions):
            self.env_method("apply_action", action, indices=index)
        self.env_method("step_sim", indices=0)
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].get_wrapper_attr("observe")()
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
