"""A driven lockstep start whose response never reaches the env is re-sent, and a refused start fails the env."""

from __future__ import annotations

import threading
import time

import pytest
import rclpy
from arena_runtime_msgs.srv import LockstepStart
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from arena_training.environments.base_env import start_driven_lockstep

SERVICE = "/lockstep_start_test/sim_lifecycle/lockstep/start"
TIMEOUT = 2.0


@pytest.fixture
def start_server():
    if not rclpy.ok():
        rclpy.init()
    seen: list[LockstepStart.Request] = []
    behavior = {"refuse": False}
    server = Node("lockstep_start_test_server")

    def handle(request: LockstepStart.Request, response: LockstepStart.Response) -> LockstepStart.Response:
        seen.append(request)
        if len(seen) == 1:
            time.sleep(1.6 * TIMEOUT)
        response.success = not behavior["refuse"]
        response.error_msg = "target_rtf must be >= 0" if behavior["refuse"] else ""
        return response

    server.create_service(LockstepStart, SERVICE, handle, callback_group=ReentrantCallbackGroup())
    client_node = Node("lockstep_start_test_client")
    client = client_node.create_client(LockstepStart, SERVICE)
    executor = MultiThreadedExecutor()
    executor.add_node(server)
    executor.add_node(client_node)
    threading.Thread(target=executor.spin, daemon=True).start()
    assert client.wait_for_service(timeout_sec=10.0)
    yield client, seen, behavior
    executor.shutdown()
    client_node.destroy_node()
    server.destroy_node()


def test_a_lost_start_response_is_resent_with_the_same_driven_config(start_server):
    client, seen, _ = start_server
    start_driven_lockstep(client, TIMEOUT)
    assert len(seen) == 2
    assert all(request.driven and not request.ungated and request.target_rtf == 0.0 for request in seen)


def test_a_refused_start_fails_the_env(start_server):
    client, seen, behavior = start_server
    seen.append(LockstepStart.Request())
    behavior["refuse"] = True
    with pytest.raises(RuntimeError, match="refused: target_rtf must be >= 0"):
        start_driven_lockstep(client, TIMEOUT)
