"""Simulation bootstrap shared by the training and tuning entry points."""

import asyncio
import logging
import threading
import time

import rclpy
import rclpy.qos
from arena_rclpy_mixins.Async import AsyncNode, ClientWrapper
from arena_runtime_msgs.srv import SpawnEnv
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

logger = logging.getLogger(__name__)


async def _spawn_envs(
    n_envs: int,
    per_env_launch_args: list[list[str]],
) -> dict[int, str]:
    """Call /arena/spawn_env n_envs times in parallel; returns idx -> ns map."""
    node = AsyncNode("_spawn_envs_node")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        cli: ClientWrapper = node.create_client_wrapper(SpawnEnv, "/arena/spawn_env", timeout=300.0)
        node.get_logger().info("waiting for /arena/spawn_env")
        await cli.ensure()
        node.get_logger().info("/arena/spawn_env is ready")

        async def _one(idx: int) -> tuple[int, str]:
            req = SpawnEnv.Request()
            req.headless = True
            req.launch_args = list(per_env_launch_args[idx])
            t_call = time.monotonic()
            resp = await cli.call_timeout(req)
            if resp is None:
                raise RuntimeError(f"SpawnEnv {idx} timed out")
            log_hint = f" (log: {resp.log_path})" if resp.log_path else ""
            if not resp.success:
                raise RuntimeError(f"SpawnEnv {idx} failed: {resp.error_msg}{log_hint}")
            node.get_logger().info(f"env {idx} spawned at {resp.ns} in {time.monotonic() - t_call:.1f}s{log_hint}")
            return idx, resp.ns

        results = await asyncio.gather(*[_one(i) for i in range(n_envs)], return_exceptions=True)
        env_map: dict[int, str] = {}
        errors: list[BaseException] = []
        for r in results:
            if isinstance(r, BaseException):
                errors.append(r)
                node.get_logger().error(f"spawn failed: {r!r}")
            else:
                idx, ns = r
                env_map[idx] = ns
        if errors:
            raise RuntimeError(f"{len(errors)}/{n_envs} envs failed to spawn") from errors[0]
        return env_map
    finally:
        executor.shutdown()
        node.destroy_node()


def spawn_envs(n_envs: int, per_env_launch_args: list[list[str]]) -> dict[int, str]:
    """Synchronous entry point: spawn N envs and return idx -> ns map."""
    return asyncio.run(_spawn_envs(n_envs, per_env_launch_args))


def wait_for_simulation(timeout: float = 120.0) -> bool:
    """Block until the simulation is fully loaded.

    Waits for the first message on ``/clock`` which Gazebo (and other
    simulators) only starts publishing once the physics engine is ready.

    Args:
        timeout: Maximum seconds to wait before giving up.

    Returns:
        ``True`` if the clock was received, ``False`` on timeout.
    """
    logger.info("Waiting for simulation (listening for /clock)...")
    event = threading.Event()

    node = Node("_wait_for_sim")

    def _cb(msg: Clock):
        event.set()

    # Gazebo publishes /clock BEST_EFFORT, a RELIABLE subscription receives nothing
    clock_qos = rclpy.qos.QoSProfile(
        depth=1,
        reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
        history=rclpy.qos.HistoryPolicy.KEEP_LAST,
    )
    node.create_subscription(Clock, "/clock", _cb, clock_qos)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    received = event.wait(timeout=timeout)

    executor.shutdown()
    node.destroy_node()

    if received:
        logger.info("Simulation is ready (/clock received).")
    else:
        logger.warning(f"Timed out after {timeout}s waiting for /clock. Proceeding anyway, the simulation may not be fully loaded.")
    return received
