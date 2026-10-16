import logging
import multiprocessing as mp
import time
import warnings
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _stack_obs

_init_log = logging.getLogger("arena_training.init")
_JOIN_TIMEOUT_S = 5.0


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    worker_idx: int = -1,
) -> None:
    import os

    import rclpy
    from stable_baselines3.common.env_util import is_wrapped

    # Worker inherits parent stdio but not logging config, so configure here.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _tag = f"[train_worker{worker_idx} pid={os.getpid()}]"
    _init_log.info(f"{_tag} started")

    parent_remote.close()

    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass
    rclpy.init()

    env = _patch_env(env_fn_wrapper.var())
    _init_log.info(f"{_tag} env constructed, awaiting commands")
    reset_info: dict[str, Any] | None = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            elif cmd == "init":
                t0 = time.monotonic()
                result = env._initialize_environment()
                _init_log.info(f"{_tag} init done in {time.monotonic() - t0:.1f}s")
                remote.send(result)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except Exception as _e:
            # Surface worker-side exceptions to stderr instead of dying
            # silently; without this the parent sees EOFError with no clue.
            import os as _os
            import sys as _sys
            import traceback as _tb

            _sys.stderr.write(f"[DelayedSubprocVecEnv._worker pid={_os.getpid()}] cmd={cmd!r} raised {_e!r}\n{_tb.format_exc()}")
            _sys.stderr.flush()
            # Try to inform parent, then exit so the parent's recv() returns.
            try:
                remote.send(("__worker_error__", repr(_e)))
            except Exception:
                pass
            break


def _is_worker_error(result: object) -> bool:
    return isinstance(result, tuple) and len(result) == 2 and result[0] == "__worker_error__"


class DelayedSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: str | None = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        self._ctx = mp.get_context(start_method)
        self._env_fns = env_fns

        self.remotes: list[mp.connection.Connection] = []
        self.processes: list[mp.process.BaseProcess] = []
        self._restarted: dict[int, tuple[Any, float, bool, dict[str, Any], dict[str, Any]]] = {}
        self._episode_steps = [0] * n_envs
        for idx in range(n_envs):
            self._start_worker(idx)
            time.sleep(1)

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        self.parent_init(len(env_fns), observation_space, action_space)
        self._gathered = all(self.get_attr("lockstep"))

    def _start_worker(self, idx: int) -> None:
        """Start the worker for env slot idx and wait until its env is initialized."""
        remote, work_remote = self._ctx.Pipe()
        args = (work_remote, remote, CloudpickleWrapper(self._env_fns[idx]), idx)
        t_spawn = time.monotonic()
        process = self._ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
        process.start()
        work_remote.close()
        if idx < len(self.remotes):
            self.remotes[idx] = remote
            self.processes[idx] = process
        else:
            self.remotes.append(remote)
            self.processes.append(process)

        t_init = time.monotonic()
        remote.send(("init", None))
        remote.recv()
        _init_log.info(
            "worker %d ready (spawn=%.1fs, init=%.1fs)",
            idx,
            t_init - t_spawn,
            time.monotonic() - t_init,
        )

    def _restart_worker(self, idx: int) -> tuple[Any, float, bool, dict[str, Any], dict[str, Any]]:
        """Replace a dead worker with a fresh env and report its slot as an ended episode."""
        old = self.processes[idx]
        old.join(timeout=_JOIN_TIMEOUT_S)
        if old.is_alive():
            old.kill()
            old.join()
        self.remotes[idx].close()
        _init_log.warning("worker %d (pid %s) died with exit code %s, restarting it", idx, old.pid, old.exitcode)
        self._start_worker(idx)
        self.remotes[idx].send(("reset", (self._seeds[idx], self._options[idx])))
        observation, reset_info = self.remotes[idx].recv()
        info = {
            "TimeLimit.truncated": False,
            "terminal_observation": observation,
            "worker_restarted": True,
            "done_reason": "worker_restarted",
            "is_success": 0,
            "episode_length": self._episode_steps[idx],
        }
        return observation, 0.0, True, info, reset_info

    def step_wait(self) -> VecEnvStepReturn:
        results = []
        for idx, remote in enumerate(self.remotes):
            if idx in self._restarted:
                results.append(self._restarted.pop(idx))
                continue
            try:
                result = remote.recv()
            except (EOFError, ConnectionResetError):
                result = self._restart_worker(idx)
            if _is_worker_error(result):
                result = self._restart_worker(idx)
            results.append(result)
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results, strict=True)
        self._episode_steps = [0 if done else n + 1 for n, done in zip(self._episode_steps, dones, strict=True)]
        return _stack_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def _call_or_restart(self, idx: int, method: str, *args: Any) -> bool:
        """Call an env method on one worker, False when the worker was dead and got restarted instead."""
        try:
            self.remotes[idx].send(("env_method", (method, args, {})))
            if not _is_worker_error(self.remotes[idx].recv()):
                return True
        except (BrokenPipeError, EOFError, ConnectionResetError):
            pass
        self._restarted[idx] = self._restart_worker(idx)
        return False

    def step_async(self, actions: np.ndarray) -> None:
        if self._gathered:
            for index, action in enumerate(actions):
                self._call_or_restart(index, "apply_action", action)
            if not self._call_or_restart(0, "step_sim"):
                self._call_or_restart(0, "step_sim")
        for idx, action in enumerate(actions):
            if idx in self._restarted:
                continue
            try:
                self.remotes[idx].send(("step", action))
            except (BrokenPipeError, ConnectionResetError):
                self._restarted[idx] = self._restart_worker(idx)
        self.waiting = True

    def parent_init(
        self,
        num_envs: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        # store info returned by the reset method
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        # seeds to be used in the next call to env.reset()
        self._seeds: list[int | None] = [None for _ in range(num_envs)]
        # options to be used in the next call to env.reset()
        self._options: list[dict[str, Any]] = [{} for _ in range(num_envs)]

        try:
            render_modes = self.get_attr("render_mode")
        except AttributeError:
            warnings.warn("The `render_mode` attribute is not defined in your environment. It will be set to None.", stacklevel=2)
            render_modes = [None for _ in range(num_envs)]

        assert all(render_mode == render_modes[0] for render_mode in render_modes), "render_mode mode should be the same for all environments"
        self.render_mode = render_modes[0]

        render_modes = []
        if self.render_mode is not None:
            if self.render_mode == "rgb_array":
                # SB3 uses OpenCV for the "human" mode
                render_modes = ["human", "rgb_array"]
            else:
                render_modes = [self.render_mode]

        self.metadata = {"render_modes": render_modes}
