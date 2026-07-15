"""Unit tests for ArenaBaseEnv's lazy-pause race fix (P8).

base_env.py pulls in rclpy/arena_robots at import time, so this whole file
is skipped when rclpy isn't importable, same as any other ROS-dependent
test in this suite (see test_tune_agent.py).

These tests construct a bare ArenaBaseEnv via __new__ (skipping the heavy
ROS-dependent __init__) and only set the attributes pause()/_maybe_fire_pause()
touch, so no ROS node, service, or timer machinery is actually needed.
"""

import threading
from unittest.mock import MagicMock

import pytest

pytest.importorskip("rclpy")

from task_generator_msgs.msg import EpisodeRecord

from arena_training.environments.base_env import ArenaBaseEnv


class _FakeNode:
    def get_fully_qualified_name(self):
        return "/fake_env"


class _FakePauseSrv:
    """Records ACQUIRE/RELEASE calls and whether the lock was held."""

    def __init__(self, env):
        self._env = env
        self.calls = []  # list of bool: True=ACQUIRE(paused), False=RELEASE

    def call_async(self, req):
        # Mirrors the real service call; the important invariant under test
        # is that this always happens while _pause_lock is held.
        self.calls.append(req.action == req.ACQUIRE)


def _make_env():
    env = object.__new__(ArenaBaseEnv)
    env.node = _FakeNode()
    env._pause_lock = threading.Lock()
    env._pause_pending_timer = None
    env._pause_pending_token = None
    env._pause_lazy_threshold = 5.0
    env._pause_srv = _FakePauseSrv(env)
    return env


def test_release_fires_fire_pause_request_holding_lock():
    env = _make_env()
    lock_held_during_call = []

    real_fire = env._fire_pause_request

    def spy(paused):
        lock_held_during_call.append(env._pause_lock.locked())
        real_fire(paused)

    env._fire_pause_request = spy
    env.pause(False)

    assert lock_held_during_call == [True]
    assert env._pause_srv.calls == [False]


def test_acquire_fires_fire_pause_request_holding_lock():
    env = _make_env()
    lock_held_during_call = []

    real_fire = env._fire_pause_request

    def spy(paused):
        lock_held_during_call.append(env._pause_lock.locked())
        real_fire(paused)

    env._fire_pause_request = spy

    # Arm a pending ACQUIRE token directly (bypassing the real timer) and
    # fire it as the timer callback would.
    token = object()
    env._pause_pending_token = token
    env._maybe_fire_pause(token)

    assert lock_held_during_call == [True]
    assert env._pause_srv.calls == [True]


def test_stale_token_never_fires_acquire():
    env = _make_env()
    token = object()
    env._pause_pending_token = token

    # A RELEASE (pause(False)) nulls the token before the stale timer
    # callback runs — the callback must then no-op, not leak an ACQUIRE.
    env.pause(False)
    env._maybe_fire_pause(token)

    assert env._pause_srv.calls == [False]


def test_concurrent_acquire_release_race_nets_to_balanced_calls():
    """Regression for the pause-hold-leak deadlock (P8): racing the
    timer-fired ACQUIRE against a synchronous pause(False) RELEASE must
    never let a RELEASE land ahead of its matching in-flight ACQUIRE.
    Under the lock-scoped fix, every observed call sequence is either
    empty, [RELEASE], or [ACQUIRE, RELEASE] — RELEASE always comes last
    when both fire, so the two never net to a leaked hold.
    """
    for _ in range(200):
        env = _make_env()
        token = object()
        env._pause_pending_token = token

        barrier = threading.Barrier(2)

        def timer_thread():
            barrier.wait()
            env._maybe_fire_pause(token)

        t = threading.Thread(target=timer_thread)
        t.start()
        barrier.wait()
        env.pause(False)
        t.join()

        calls = env._pause_srv.calls
        assert calls in ([], [False], [True, False]), calls


def _make_reset_env():
    """Bare env wired only for the mid-episode-reset branch of reset()."""
    env = object.__new__(ArenaBaseEnv)
    env._initialized = True
    env._cmd_vel_pub = None
    env._steps_curr_episode = 5
    env._first_reset_done = True
    env._ready_event = threading.Event()
    env._ready_event.set()
    env._latest_episode = MagicMock(
        episode_id=3, outcome_state=EpisodeRecord.RUNNING
    )
    env.env_ns = MagicMock(to_string=lambda: "/env_1")
    env.node = MagicMock()
    env._reward_function = MagicMock()
    env._model_space_manager = MagicMock()
    env.observation_collector = MagicMock(get_observations=lambda **kw: {})
    env._encode_observation = lambda obs: obs
    env._before_task_reset = MagicMock()
    env.reset_task = MagicMock()
    env._wait_for_new_episode = MagicMock()
    env._after_task_reset = MagicMock()
    env._ArenaBaseEnv__is_first_step = False
    return env


def test_reset_releases_pause_before_waiting_for_new_episode():
    """P8 reset guard: pause(False) must run before reset_task()/
    _wait_for_new_episode() in the real mid-episode reset branch, so a
    leaked training_env_pause hold can never wedge the sim paused while
    task_generator is asked to step it to publish the next episode.
    """
    env = _make_reset_env()
    call_order = []
    env.pause = MagicMock(side_effect=lambda paused: call_order.append(("pause", paused)))
    env._before_task_reset.side_effect = lambda: call_order.append(("before_task_reset",))
    env.reset_task.side_effect = lambda: call_order.append(("reset_task",))
    env._wait_for_new_episode.side_effect = lambda prev_id: call_order.append(
        ("wait_for_new_episode", prev_id)
    )

    env.reset()

    assert call_order == [
        ("pause", False),
        ("before_task_reset",),
        ("reset_task",),
        ("wait_for_new_episode", 3),
    ]
