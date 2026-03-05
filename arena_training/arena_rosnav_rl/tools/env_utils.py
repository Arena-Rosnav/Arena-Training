from typing import Any, Callable, List, Optional, Tuple, Type, Union

import gymnasium as gym
import rclpy
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
from ...environments.wrappers import TimeSyncWrapper
from ..node import SupervisorNode

# from environments.unity import UnityEnv
from ..stable_baselines3.vec_wrapper import (
    DelayedSubprocVecEnv,
    ProfilingVecEnv,
    VecStatsRecorder,
)
from ..utils.constants import Simulator


def load_vec_framestack(stack_size: int, env: VecEnv) -> VecEnv:
    """
    Load a vectorized environment with frame stacking.

    Args:
        stack_size (int): The number of frames to stack.
        env (VecEnv): The vectorized environment to wrap.

    Returns:
        VecEnv: The wrapped vectorized environment with frame stacking applied.
    """
    return VecFrameStack(env, n_stack=stack_size, channels_order="first")


def determine_env_class(simulator: Simulator) -> Union[gym.Env, gym.Wrapper]:
    """
    Determines the environment class based on the specified simulator.

    Args:
        simulator (Simulator): The simulator to use.

    Returns:
        Union[gym.Env, gym.Wrapper]: The environment class.
    """
    return arena_envs.GazeboEnv
    if simulator == Simulator.FLATLAND:
        return arena_envs.FlatlandEnv
    elif simulator == Simulator.GAZEBO:
        return arena_envs.GazeboEnv
    # elif simulator == Simulator.UNITY:
    #     return UnityEnv
    else:
        raise RuntimeError(f"Simulator {simulator} is not supported.")


def _init_env_fnc(
    node: SupervisorNode,
    env_class: gym.Env,
    ns: Union[str, Namespace],
    space_manager: rosnav_rl.BaseSpaceManager,
    reward_function: rosnav_rl.RewardFunction,
    simulation_state_container: rosnav_rl.SimulationStateContainer,
    max_steps_per_episode: int,
    init_by_call: bool = False,
    obs_unit_kwargs: dict = None,
    seed: int = 0,
    wrappers: List[Callable[[Tuple[Type[gym.Wrapper], Any]], gym.Wrapper]] = None,
) -> callable:

    def _init_env() -> Union[gym.Env, gym.Wrapper]:
        env = env_class(
            node=node,
            ns=ns,
            space_manager=space_manager,
            reward_function=reward_function,
            simulation_state_container=simulation_state_container,
            max_steps_per_episode=max_steps_per_episode,
            init_by_call=init_by_call,
            obs_unit_kwargs=obs_unit_kwargs,
        )
        for wrapper in wrappers or []:
            env = wrapper(env)
        return env

    set_random_seed(seed)
    return _init_env


def _test_init_env_fnc(
    env_class: gym.Env,
    ns: Union[str, Namespace],
    space_manager: rosnav_rl.BaseSpaceManager,
    reward_function: rosnav_rl.RewardFunction,
    simulation_state_container: rosnav_rl.SimulationStateContainer,
    max_steps_per_episode: int,
    node: SupervisorNode = None,
    init_by_call: bool = False,
    obs_unit_kwargs: dict = None,
    seed: int = 0,
    wrappers: List[Callable[[Tuple[Type[gym.Wrapper], Any]], gym.Wrapper]] = None,
    observations_config: Optional[str] = None,
) -> callable:

    def _init_env() -> Union[gym.Env, gym.Wrapper]:
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
        )
        for wrapper in wrappers or []:
            env = wrapper(env)
        return env

    set_random_seed(seed)
    return _init_env


def sb3_wrap_env(
    node: SupervisorNode,
    env_fncs: List[callable],
    general_cfg: GeneralCfg,
    monitoring_cfg: MonitoringCfg,
    profiling_cfg: ProfilingCfg,
) -> VecEnv:
    """
    Creates and wraps a single vectorized environment used for both training and
    evaluation (shared-env mode).

    Args:
        node: ROS2 supervisor node used for profiling.
        env_fncs: List of callables that each return an initialized gym environment.
        general_cfg: General training configuration (debug_mode, etc.).
        monitoring_cfg: Configuration for episode-stats recording.
        profiling_cfg: Configuration for step/reset profiling.

    Returns:
        A wrapped VecEnv ready for both rollout collection and periodic evaluation.
    """

    def create_env(fncs):
        return (
            DelayedSubprocVecEnv(fncs, start_method="forkserver")
            if not general_cfg.debug_mode
            else DummyVecEnv(fncs)
        )

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

    env = create_env(env_fncs)
    env = apply_vec_stats_recorder(env)
    env = apply_profiling(env)
    return env


def make_envs(
    rl_agent: rosnav_rl.RL_Agent,
    simulation_state_container: rosnav_rl.SimulationStateContainer,
    n_envs: int,
    max_steps: int,
    init_env_by_call: bool,
    namespace_fn: Callable,  # Changed from callable
    node: SupervisorNode = None,
    wrappers: List[Callable[[Tuple[Type[gym.Wrapper], Any]], gym.Wrapper]] = None,
    observations_config: Optional[str] = None,
) -> List[Callable]:
    """
    Creates a list of environment initialization functions.

    This function generates callable functions that each initialize a gym environment
    for reinforcement learning training. The environments are configured with the
    specified RL agent's space manager and reward function.

    Args:
        rl_agent: The reinforcement learning agent containing space manager and reward function
        simulation_state_container: Container holding the state of the simulation
        n_envs: Number of environments to create
        max_steps: Maximum number of steps per episode for each environment
        init_env_by_call: If True, environments will be initialized when their function is called
        namespace_fn: Function that takes an index and returns a namespace for the environment
        wrappers: Optional list of gym wrappers to apply to each environment
        observations_config: Path to a custom observations YAML config file

    Returns:
        List of callables, each initializing a gym environment when called
    """

    def create_env_fnc(
        ns: Union[str, Namespace],
    ) -> callable:
        return _test_init_env_fnc(
            node=node,
            env_class=determine_env_class(
                None
            ),  # Replace None with the desired simulator
            ns=ns,
            space_manager=rl_agent.space_manager,
            reward_function=rl_agent.reward_function.copy(),
            simulation_state_container=simulation_state_container,
            max_steps_per_episode=max_steps,
            init_by_call=init_env_by_call,
            wrappers=wrappers,
            observations_config=observations_config,
        )

    def create_env_fncs(
        n_envs: int,
        ns_fn: Callable,
    ) -> List[Callable]:
        return [
            create_env_fnc(
                ns=ns_fn(idx),
            )
            for idx in range(n_envs)
        ]

    return create_env_fncs(
        n_envs=n_envs,
        ns_fn=namespace_fn,
    )
