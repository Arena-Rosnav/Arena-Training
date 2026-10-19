from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("rosnav_rl")

from arena_mcp.core import ToolError

from arena_training.arena_rosnav_rl.utils.config import load_training_config
from arena_training.arena_rosnav_rl.utils.paths import ConfigComponent
from arena_training.mcp import training_agents, training_config, training_configs, training_schema, training_start, training_status, training_validate

CONFIGS = Path(ConfigComponent().path)


def _first_with_relative_observations() -> Path:
    for path in sorted(CONFIGS.glob("*.yaml")):
        if "observations_config: observations/" in path.read_text():
            return path
    pytest.skip("no shipped config with a relative observations_config")


def test_configs_lists_every_shipped_config_with_its_loaded_fields():
    listed = training_configs()
    assert [entry["name"] for entry in listed] == sorted(path.name for path in CONFIGS.glob("*.yaml"))
    for entry in listed:
        assert entry["valid"], entry
        loaded = load_training_config(entry["path"])
        assert entry["framework"] == loaded.agent_config.framework.name
        assert entry["n_envs"] == loaded.arena_cfg.general.n_envs
        assert entry["robot"] == loaded.arena_cfg.robot.robot_model


def test_config_returns_the_parsed_model_and_its_schema():
    out = training_config("sb_training_config")
    loaded = load_training_config(str(CONFIGS / "sb_training_config.yaml")).model_dump(mode="json")
    assert out["path"] == str(CONFIGS / "sb_training_config.yaml")
    assert out["config"]["arena_cfg"] == loaded["arena_cfg"]
    assert out["config"]["agent_config"]["framework"] == loaded["agent_config"]["framework"]
    assert out["summary"]["algorithm"] == loaded["agent_config"]["framework"]["algorithm"]["type"]
    assert set(out["schema"]["properties"]) >= {"arena_cfg", "agent_config", "resume", "agents_dir"}
    assert training_schema() == out["schema"]
    assert "schema" not in training_config("sb_training_config.yaml", schema=False)


def test_config_serializes_robot_parts_of_a_manipulator_config():
    parted = [path for path in sorted(CONFIGS.glob("*.yaml")) if load_training_config(str(path)).arena_cfg.robot.parts]
    if not parted:
        pytest.skip("no shipped config requests robot parts")
    attrs = pytest.importorskip("attrs")
    loaded = load_training_config(str(parted[0])).arena_cfg.robot.parts
    out = training_config(parted[0].name, schema=False)["config"]["arena_cfg"]["robot"]["parts"]
    assert out == {kind: [attrs.asdict(part) for part in parts] for kind, parts in loaded.items()}
    json.dumps(out)


def test_config_names_the_available_configs_for_an_unknown_name():
    with pytest.raises(ToolError, match="sb_training_config.yaml"):
        training_config("no_such_config")


def test_validate_accepts_a_shipped_config_by_path_and_by_text():
    path = _first_with_relative_observations()
    assert training_validate(path=str(path))["valid"]
    by_text = training_validate(yaml_text=path.read_text())
    assert by_text["valid"], by_text
    assert by_text["source"] == "yaml_text"


def test_validate_reports_an_unknown_key_in_a_broken_copy(tmp_path):
    broken = tmp_path / "broken.yaml"
    broken.write_text((CONFIGS / "sb_training_config.yaml").read_text() + "\nno_such_section: 1\n")
    out = training_validate(path=str(broken))
    assert not out["valid"]
    assert any(error["loc"] == "no_such_section" for error in out["errors"])


def test_validate_reports_a_bad_value_and_bad_yaml():
    text = (CONFIGS / "sb_training_config.yaml").read_text().replace("n_envs: 2", "n_envs: 0", 1)
    out = training_validate(yaml_text=text)
    assert not out["valid"]
    assert any(error["loc"].endswith("general.n_envs") for error in out["errors"])
    garbled = training_validate(yaml_text="agent_config: [unclosed")
    assert not garbled["valid"] and garbled["errors"][0]["type"].endswith("Error")
    assert not training_validate(yaml_text="- a list\n")["valid"]


def test_validate_needs_exactly_one_source():
    with pytest.raises(ToolError, match="exactly one"):
        training_validate()
    with pytest.raises(ToolError, match="exactly one"):
        training_validate(yaml_text="a: 1", path="sb_training_config.yaml")


def _agent(root: Path, name: str) -> Path:
    agent = root / name
    (agent / "training_logs").mkdir(parents=True)
    (agent / "best_model.zip").write_bytes(b"best")
    (agent / "last_model.zip").write_bytes(b"last-model")
    (agent / "vec_normalize_best_model.pkl").write_bytes(b"norm")
    (agent / "training_config.yaml").write_text((CONFIGS / "sb_training_config.yaml").read_text())
    return agent


def test_agents_lists_checkpoints_newest_agent_first(tmp_path):
    old = _agent(tmp_path, "AGENT_OLD")
    _agent(tmp_path, "AGENT_NEW")
    for path in old.rglob("*"):
        os.utime(path, (1_000_000_000, 1_000_000_000))
    out = training_agents(agents_dir=str(tmp_path))
    assert out["agents_dir"] == str(tmp_path.resolve())
    assert [agent["name"] for agent in out["agents"]] == ["AGENT_NEW", "AGENT_OLD"]
    newest = out["agents"][0]
    assert sorted(checkpoint["name"] for checkpoint in newest["checkpoints"]) == ["best_model.zip", "last_model.zip"]
    assert newest["training_config"] and not newest["tensorboard"]


def test_agents_of_a_missing_dir_is_empty(tmp_path):
    assert training_agents(agents_dir=str(tmp_path / "absent"))["agents"] == []


def test_status_reads_tensorboard_scalars(tmp_path):
    tensorboard_x = pytest.importorskip("tensorboardX")
    pytest.importorskip("tensorboard")
    agent = _agent(tmp_path, "AGENT_TB")
    writer = tensorboard_x.SummaryWriter(str(agent / "training_logs" / "PPO_1"))
    for step in range(5):
        writer.add_scalar("rollout/ep_rew_mean", float(step) * 2.0, step)
    writer.close()
    out = training_status(agents_dir=str(tmp_path))
    assert out["name"] == "AGENT_TB"
    scalars = out["tensorboard"]["runs"]["training_logs/PPO_1"]["rollout/ep_rew_mean"]
    assert scalars["step"] == 4 and scalars["value"] == 8.0 and scalars["points"] == 5
    assert out["log"] is None


def test_status_names_the_agents_for_an_unknown_one(tmp_path):
    _agent(tmp_path, "AGENT_A")
    with pytest.raises(ToolError, match="AGENT_A"):
        training_status(agent="AGENT_B", agents_dir=str(tmp_path))
    with pytest.raises(ToolError, match="no agents"):
        training_status(agents_dir=str(tmp_path / "absent"))


def test_start_rejects_bad_launch_args_and_configs_before_touching_ros():
    with pytest.raises(ToolError, match="plain key"):
        asyncio.run(training_start("sb_training_config.yaml", {"sim:=gazebo": "x"}))
    with pytest.raises(ToolError, match="train_config"):
        asyncio.run(training_start("sb_training_config.yaml", {"train_config": "other.yaml"}))
    with pytest.raises(ToolError, match="no training config"):
        asyncio.run(training_start("no_such_config.yaml"))


REFUSAL = """
import asyncio, json, threading, time
import rclpy
from rclpy.executors import SingleThreadedExecutor
from arena_runtime_msgs.srv import RegisterEnv
from arena_mcp.core import ROS, ToolError
from arena_training.mcp import training_start

async def main():
    await ROS.node()
    node = rclpy.create_node("arena")
    node.create_service(RegisterEnv, "/arena/register_env", lambda request, response: response)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 10
        while "/arena/register_env" not in (await ROS.graph())["services"] and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        try:
            await training_start("sb_training_config.yaml")
        except ToolError as exc:
            print(json.dumps({"message": exc.message, "hint": exc.hint}))
    finally:
        executor.shutdown()
        node.destroy_node()
        await ROS.shutdown()

asyncio.run(main())
"""


def test_start_refuses_while_a_runtime_serves_register_env():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_runtime_msgs")
    env = {key: value for key, value in os.environ.items() if key != "ARENA_DIR"}
    env.update(ROS_DOMAIN_ID=str(200 + os.getpid() % 30), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
    done = subprocess.run([sys.executable, "-c", REFUSAL], env=env, capture_output=True, text=True, timeout=120, check=False)
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    assert lines, done.stdout + done.stderr
    out = json.loads(lines[-1])
    assert "/arena/register_env" in out["message"]
    assert "jobs_cancel" in out["hint"]


def test_domain_serves_every_tool_through_the_server():
    from mcp.server.mcpserver import MCPServer

    from arena_training.mcp import domain

    assert domain.unavailable() is None
    server = MCPServer(name="training_test")
    domain.install(server)

    async def scenario():
        tools = await server.list_tools()
        listed = await server.call_tool("training_configs", {})
        schema = await server.read_resource("arena://training/schema")
        return tools, listed, list(schema)

    tools, listed, schema = asyncio.run(scenario())
    assert sorted(tool.name for tool in tools) == sorted(spec.name for spec in domain.tools)
    assert not listed.is_error
    assert [entry["name"] for entry in json.loads(listed.content[0].text)] == sorted(path.name for path in CONFIGS.glob("*.yaml"))
    assert "arena_cfg" in json.loads(schema[0].content)["properties"]
