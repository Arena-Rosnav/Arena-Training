"""A training hold whose acquire response never reaches the env is re-acquired and still counted once by the arena hold registry."""

from __future__ import annotations

import threading
import time

import pytest
import rclpy
from arena_runtime.holds import HoldRegistry
from arena_runtime_msgs.srv import LifecycleHold
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from arena_training.environments.base_env import acquire_hold

SERVICE = "/hold_acquire_test/sim_lifecycle/hold"
TIMEOUT = 0.5


@pytest.fixture
def hold_server():
    if not rclpy.ok():
        rclpy.init()
    registry = HoldRegistry()
    delayed = {"acquires": 0}
    server = Node("hold_acquire_test_server")

    def handle(request: LifecycleHold.Request, response: LifecycleHold.Response) -> LifecycleHold.Response:
        if request.action == LifecycleHold.Request.ACQUIRE:
            count = registry.acquire(request.caller_id, request.reason)
            delayed["acquires"] += 1
            if delayed["acquires"] == 1:
                time.sleep(1.6 * TIMEOUT)
        else:
            count = registry.release(request.caller_id, request.reason)
        response.paused = not registry.is_empty()
        response.hold_count = count
        return response

    server.create_service(LifecycleHold, SERVICE, handle, callback_group=ReentrantCallbackGroup())
    client_node = Node("hold_acquire_test_client")
    client = client_node.create_client(LifecycleHold, SERVICE)
    executor = MultiThreadedExecutor()
    executor.add_node(server)
    executor.add_node(client_node)
    threading.Thread(target=executor.spin, daemon=True).start()
    assert client.wait_for_service(timeout_sec=10.0)
    yield registry, client, delayed
    executor.shutdown()
    client_node.destroy_node()
    server.destroy_node()


def test_late_acquire_response_leaves_exactly_one_hold(hold_server):
    registry, client, delayed = hold_server
    acquire_hold(client, "/arena/env_0/env", "training", TIMEOUT)
    time.sleep(4 * TIMEOUT)
    assert delayed["acquires"] == 2
    assert registry.has("/arena/env_0/env", "training")
    assert registry.total_count() == 1
