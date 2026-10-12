from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn


class GatheredDummyVecEnv(DummyVecEnv):
    """DummyVecEnv whose lockstep envs share one sim step per vec step."""

    def step_wait(self) -> VecEnvStepReturn:
        if all(self.get_attr("lockstep")):
            for index, action in enumerate(self.actions):
                self.env_method("apply_action", action, indices=index)
            self.env_method("step_sim", indices=0)
        return super().step_wait()
