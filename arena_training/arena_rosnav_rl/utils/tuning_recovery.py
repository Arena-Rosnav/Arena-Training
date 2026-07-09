"""Inter-trial hygiene for the persistent-sim Optuna tuning loop.

``tune_agent.py`` spawns each env namespace once per study and reuses it
across every trial. A trial can crash mid-episode (bad hyperparameters,
a transient ROS/Gazebo hiccup) and leave the env in a bad state for the
*next* trial: a stuck reset hold, a task_generator process that stopped
responding, etc. This module provides the between-trial check that keeps
a bad trial from silently poisoning the rest of the study.

Eviction of a dead env's holds is arena_node's job (heartbeat timeout,
see ``arena_runtime/arena_node.py::_check_heartbeats``) — this module
only *waits* for that to happen and *observes* the result via the
latched ``state/holders`` / ``state/envs`` topics. It never acquires,
releases, or force-clears a hold itself.
"""

import asyncio
import gc
import logging
import threading

import rclpy
import rclpy.qos
from rclpy.node import Node

import arena_runtime_msgs.msg
import arena_runtime_msgs.srv

from arena_rclpy_mixins.Async import AsyncNode, ClientWrapper

logger = logging.getLogger(__name__)

_LATCHED = rclpy.qos.QoSProfile(
    depth=1,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
)


def _read_latched(node: Node, msg_type: type, topic: str, timeout: float):
    """Block until one message arrives on a latched topic, or return None."""
    event = threading.Event()
    box: list = []

    def _cb(msg):
        box.append(msg)
        event.set()

    sub = node.create_subscription(msg_type, topic, _cb, _LATCHED)
    try:
        if not event.wait(timeout=timeout):
            return None
        return box[0]
    finally:
        node.destroy_subscription(sub)


async def _wait_for_holds_clear(node: Node, timeout: float) -> bool:
    """Poll state/holders until empty or timeout. Never clears holds itself."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        msg = await loop.run_in_executor(
            None, _read_latched, node, arena_runtime_msgs.msg.HoldRegistry, "/arena/state/holders", 2.0
        )
        if msg is not None and not msg.holds:
            return True
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(1.0)


async def _get_env_registry(node: Node, timeout: float):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _read_latched, node, arena_runtime_msgs.msg.EnvRegistry, "/arena/state/envs", timeout
    )


async def _respawn_one(
    node: AsyncNode,
    despawn_cli: ClientWrapper,
    spawn_cli: ClientWrapper,
    env_id: int,
    ns: str,
    launch_args: list[str],
) -> str:
    logger.warning("respawning unhealthy env %s (env_id=%d)", ns, env_id)

    despawn_req = arena_runtime_msgs.srv.DespawnEnv.Request()
    despawn_req.env_id = env_id
    resp = await despawn_cli.call_timeout(despawn_req)
    if resp is None or not resp.success:
        err = resp.error_msg if resp is not None else "timed out"
        logger.warning("despawn_env(%d) for %s failed (continuing anyway): %s", env_id, ns, err)

    spawn_req = arena_runtime_msgs.srv.SpawnEnv.Request()
    spawn_req.ns = ns
    spawn_req.headless = True
    spawn_req.launch_args = list(launch_args)
    resp = await spawn_cli.call_timeout(spawn_req)
    if resp is None:
        raise RuntimeError(f"respawn of {ns} timed out")
    if not resp.success:
        raise RuntimeError(f"respawn of {ns} failed: {resp.error_msg}")
    return resp.ns


async def _ensure_envs_healthy(
    env_map: dict[int, str],
    per_env_launch_args: list[list[str]],
    holds_timeout: float,
    registry_timeout: float,
) -> None:
    node = AsyncNode("_tuning_recovery_node")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if not await _wait_for_holds_clear(node, holds_timeout):
            logger.warning(
                "state/holders did not clear within %.0fs — proceeding anyway; "
                "arena_node's own heartbeat eviction should have released a dead trial's hold "
                "well before this timeout",
                holds_timeout,
            )

        registry = await _get_env_registry(node, registry_timeout)
        ready_by_fqn: dict[str, bool] = {}
        env_id_by_fqn: dict[str, int] = {}
        if registry is not None:
            for record in registry.envs:
                ready_by_fqn[record.fqn] = record.ready
                env_id_by_fqn[record.fqn] = record.env_id

        unhealthy = [
            idx for idx, ns in env_map.items() if not ready_by_fqn.get(ns, False)
        ]
        if not unhealthy:
            return

        despawn_cli: ClientWrapper = node.create_client_wrapper(
            arena_runtime_msgs.srv.DespawnEnv, "/arena/despawn_env", timeout=60.0
        )
        spawn_cli: ClientWrapper = node.create_client_wrapper(
            arena_runtime_msgs.srv.SpawnEnv, "/arena/spawn_env", timeout=300.0
        )
        await despawn_cli.ensure()
        await spawn_cli.ensure()

        for idx in unhealthy:
            ns = env_map[idx]
            env_id = env_id_by_fqn.get(ns, -1)
            new_ns = await _respawn_one(
                node, despawn_cli, spawn_cli, env_id, ns, per_env_launch_args[idx]
            )
            env_map[idx] = new_ns
    finally:
        executor.shutdown()
        node.destroy_node()


def ensure_envs_healthy(
    env_map: dict[int, str],
    per_env_launch_args: list[list[str]],
    holds_timeout: float = 45.0,
    registry_timeout: float = 10.0,
) -> None:
    """Between-trial check: wait out any lingering hold, respawn unhealthy envs.

    Mutates ``env_map`` in place if an env had to be respawned. A no-op in
    the common case where every env is still ready from the previous trial.
    """
    asyncio.run(
        _ensure_envs_healthy(env_map, per_env_launch_args, holds_timeout, registry_timeout)
    )


def release_trial_resources() -> None:
    """Best-effort GPU/host memory reclaim between trials."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
