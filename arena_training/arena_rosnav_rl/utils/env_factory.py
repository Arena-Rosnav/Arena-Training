"""Environment factory functions, create and wrap gym environments for RL training."""

from typing import Any, Callable

import gymnasium as gym
import rosnav_rl
from rosnav_rl.utils.rostopic import Namespace
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from ... import environments as arena_envs
from ..cfg import (
    GeneralCfg,
    MonitoringCfg,
    ProfilingCfg,
)
from ..node import SupervisorNode
from ..stable_baselines3.vec_wrapper import (
    DelayedSubprocVecEnv,
    ProfilingVecEnv,
    VecStatsRecorder,
)
from .constants import Simulator


def load_vec_framestack(stack_size: int, env: VecEnv) -> VecEnv:
    """Load a vectorized environment with frame stacking."""
    return VecFrameStack(env, n_stack=stack_size, channels_order="first")


def determine_env_class(simulator: Simulator) -> gym.Env | gym.Wrapper:
    """Determines the environment class based on the specified simulator."""
    return arena_envs.GazeboEnv


def _init_env_fnc(
    node: SupervisorNode,
    env_class: gym.Env,
    ns: str | Namespace,
    space_manager: rosnav_rl.BaseSpaceManager,
    reward_function: rosnav_rl.RewardFunction,
    simulation_state_container: rosnav_rl.AgentParameters,
    max_steps_per_episode: int,
    init_by_call: bool = False,
    obs_unit_kwargs: dict = None,
    seed: int = 0,
    wrappers: list[Callable[[tuple[type[gym.Wrapper], Any]], gym.Wrapper]] = None,
) -> callable:

    def _init_env() -> gym.Env | gym.Wrapper:
        env = env_class(
            node=node,
            ns=ns,
            space_manager=space_manager,
            reward_function=reward_function,
            simulation_state_container=simulation_state_container,
            max_steps_per_episode=max_steps_per_episode,
            init_by_call=init_by_call,
            obs_unit_kwargs=obs_unit_kwargs,
            wait_for_obs=True,
        )
        for wrapper in wrappers or []:
            env = wrapper(env)
        return env

    set_random_seed(seed)
    return _init_env


def _test_init_env_fnc(
    env_class: gym.Env,
    ns: str | Namespace,
    space_manager: rosnav_rl.BaseSpaceManager,
    reward_function: rosnav_rl.RewardFunction,
    simulation_state_container: rosnav_rl.AgentParameters,
    max_steps_per_episode: int,
    node: SupervisorNode = None,
    init_by_call: bool = False,
    obs_unit_kwargs: dict = None,
    seed: int = 0,
    wrappers: list[Callable[[tuple[type[gym.Wrapper], Any]], gym.Wrapper]] = None,
    observations_config: str | None = None,
) -> callable:

    def _init_env() -> gym.Env | gym.Wrapper:
        env = env_class(
            node=node,
            ns=ns,
            space_manager=space_manager,
            reward_function=reward_function,
            simulation_state_container=simulation_state_container,
            max_steps_per_episode=max_steps_per_episode,
            init_by_call=init_by_call,
            obs_unit_kwargs=obs_unit_kwargs,
            observations_config=observations_config,
            wait_for_obs=True,
        )
        for wrapper in wrappers or []:
            env = wrapper(env)
        return env

    set_random_seed(seed)
    return _init_env


def sb3_wrap_env(
    node: SupervisorNode,
    env_fncs: list[callable],
    general_cfg: GeneralCfg,
    monitoring_cfg: MonitoringCfg,
    profiling_cfg: ProfilingCfg,
) -> VecEnv:
    """
    Creates and wraps a single vectorized environment used for both training and
    evaluation (shared-env mode).
    """

    def create_env(fncs: list[Callable]) -> VecEnv:
        return DelayedSubprocVecEnv(fncs, start_method="forkserver") if not general_cfg.debug_mode else DummyVecEnv(fncs)

    def apply_vec_stats_recorder(env: VecEnv) -> VecEnv:
        return (
            VecStatsRecorder(
                env,
                after_x_eps=monitoring_cfg.episode_logging.last_n_episodes,
                record_actions=monitoring_cfg.episode_logging.record_actions,
            )
            if monitoring_cfg is not None and monitoring_cfg.episode_logging is not None
            else env
        )

    def apply_profiling(env: VecEnv, enable_subscribers: bool = True) -> VecEnv:
        return (
            ProfilingVecEnv(
                node=node,
                env=env,
                profile_step=profiling_cfg.do_profile_step,
                profile_reset=profiling_cfg.do_profile_reset,
                per_call=profiling_cfg.per_call,
                log_file=profiling_cfg.log_file,
                print_stats=profiling_cfg.print_stats,
                enable_subscribers=enable_subscribers,
            )
            if profiling_cfg is not None and node is not None
            else env
        )

    if node is not None:
        for name, value in (
            ("lockstep", general_cfg.lockstep),
            ("lockstep_step_seconds", general_cfg.lockstep_step_seconds),
        ):
            if not node.has_parameter(name):
                node.declare_parameter(name, value)

    env = create_env(env_fncs)
    env = apply_vec_stats_recorder(env)
    env = apply_profiling(env)
    return env


def make_envs(
    rl_agent: rosnav_rl.RL_Agent,
    simulation_state_container: rosnav_rl.AgentParameters,
    n_envs: int,
    max_steps: int,
    init_env_by_call: bool,
    namespace_fn: Callable,
    node: SupervisorNode = None,
    wrappers: list[Callable[[tuple[type[gym.Wrapper], Any]], gym.Wrapper]] = None,
    observations_config: str | None = None,
) -> list[Callable]:
    """
    Creates a list of environment initialization functions.

    Args:
        rl_agent: The RL agent containing space manager and reward function.
        simulation_state_container: AgentParameters shared across environments.
        n_envs: Number of environments to create.
        max_steps: Maximum steps per episode.
        init_env_by_call: If True, environments initialize when their function is called.
        namespace_fn: Function mapping index → environment namespace string.
        wrappers: Optional gym wrappers to apply to each environment.
        observations_config: Path to a custom observations YAML config file.

    Returns:
        List of callables, each initializing a gym environment when called.
    """

    def create_env_fnc(ns: str | Namespace) -> callable:
        return _test_init_env_fnc(
            node=node,
            env_class=determine_env_class(None),
            ns=ns,
            space_manager=rl_agent.space_manager,
            reward_function=rl_agent.reward_function.copy(),
            simulation_state_container=simulation_state_container,
            max_steps_per_episode=max_steps,
            init_by_call=init_env_by_call,
            wrappers=wrappers,
            observations_config=observations_config,
        )

    return [create_env_fnc(ns=namespace_fn(idx)) for idx in range(n_envs)]
