import pyinstrument
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from ...node import SupervisorNode


class ProfilingVecEnv(VecEnvWrapper):
    """
    A vectorized environment wrapper that adds profiling capabilities.

    Args:
        env (VecEnv): The vectorized environment to wrap.
        profile_step (bool): Whether to profile the `step` method. Default is False.
        profile_reset (bool): Whether to profile the `reset` method. Default is False.
        per_call (bool): Whether to reset the profiler after each call. Default is False.
        print_stats (bool): Whether to print the profiling stats. Default is True.
        log_file (str): Path to the file where profiling stats should be logged. Default is None (no logging).
    """

    def __init__(
        self,
        node: SupervisorNode,
        env: VecEnv,
        profile_step: bool = True,
        profile_reset: bool = True,
        per_call: bool = True,
        print_stats: bool = True,
        log_file: str = None,
        enable_subscribers: bool = True,
    ):
        super().__init__(env)
        self._node = node
        self._step_profiler = pyinstrument.Profiler()
        self._reset_profiler = pyinstrument.Profiler()

        self._profile_method_step = profile_step
        self._profile_method_reset = profile_reset

        self._per_call = per_call
        self._print_stats = print_stats
        self._log_file = log_file

    def _output_stats(self, profiler: pyinstrument.Profiler, method_name: str):
        if self._print_stats:
            self._node._logger.info(f"Profiling stats for {method_name}:")
            self._node._logger.info(profiler.output_text(unicode=True, color=True))

        if self._log_file:
            with open(self._log_file, "a") as f:
                f.write(f"\nProfiling stats for {method_name}:\n")
                f.write(profiler.output_text(unicode=True, color=False))

    def step_wait(self) -> VecEnvStepReturn:
        """
        Perform a step in the wrapped environment.

        Returns:
            tuple: A tuple containing the observations, rewards, dones, and infos.
        """
        if self._profile_method_step:
            self._step_profiler.start()

        obs, rewards, dones, infos = self.venv.step_wait()

        if self._profile_method_step and self._step_profiler.is_running:
            self._step_profiler.stop()
            self._output_stats(self._step_profiler, "step_wait")

        if self._profile_method_step and self._per_call and self._step_profiler.is_running:
            self._step_profiler.reset()

        return obs, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        """
        Reset the wrapped environment.

        Returns:
            object: The initial observations.
        """
        if self._profile_method_reset:
            self._reset_profiler.start()

        observations = self.venv.reset()

        if self._profile_method_reset and self._reset_profiler.is_running:
            self._reset_profiler.stop()
            self._output_stats(self._reset_profiler, "reset")

        if self._profile_method_reset and self._per_call and self._reset_profiler.is_running:
            self._reset_profiler.reset()

        return observations
