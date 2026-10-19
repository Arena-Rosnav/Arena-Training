"""The mobile manipulator training config drives the base and the arm and loads the arm datasources."""

from __future__ import annotations

import pathlib
import types

import pytest
import rclpy
import yaml
from rclpy.node import Node
from rosnav_rl.cfg.action_spaces import CompositeActionSpace, DiscretizationCfg, ManipulatorActionSpace
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.data_sources.collectors import ImageColorCollector, JointStateCollector
from rosnav_rl.observations.data_sources.generators import EndEffectorPoseTFGenerator
from rosnav_rl.observations.factory.factory import ObservationFactory, set_end_effector_frames, set_robot_pose_frames

from arena_training.arena_rosnav_rl.trainer.arena_trainer import ArenaTrainer
from arena_training.arena_rosnav_rl.utils.config import load_training_config

CONFIGS = pathlib.Path(__file__).parents[1] / "configs"


def _action_space(config_name: str, **agent_update: object) -> object:
    config = load_training_config(str(CONFIGS / config_name))
    config.agent_config = config.agent_config.model_copy(update=agent_update)
    ArenaTrainer._populate_agent_spec(types.SimpleNamespace(config=config))
    return config.agent_config.action_space


@pytest.fixture
def node():
    if not rclpy.ok():
        rclpy.init()
    node = Node("mobile_manipulator_config_test")
    yield node
    node.destroy_node()


def test_the_manipulator_config_drives_the_base_and_the_arm():
    space = _action_space("sb_mobile_manipulator_config.yaml")
    assert isinstance(space, CompositeActionSpace)
    base, arm = space.action_spaces
    assert base.type in ("differential_drive", "omnidirectional")
    assert isinstance(arm, ManipulatorActionSpace)
    assert (arm.controller, arm.tip_link, len(arm.joint_names)) == ("arm_controller", "robot_arm_tool0", 6)
    assert arm.stow_positions is not None
    assert space.num_dof == base.num_dof + 6


def test_a_robot_without_an_arm_keeps_its_drive_space():
    assert not isinstance(_action_space("sb_training_config.yaml"), CompositeActionSpace)


def test_discrete_drive_actions_cannot_drive_an_arm():
    with pytest.raises(ValueError, match="continuous sub-spaces only"):
        _action_space("sb_mobile_manipulator_config.yaml", discretization=DiscretizationCfg(strategy="navigational"))


def test_the_manipulator_observations_build_the_arm_datasources(node):
    config = yaml.safe_load((CONFIGS / "observations" / "manipulator.yaml").read_text())
    set_robot_pose_frames(config, "env_0/rbkairos", "base_link")
    set_end_effector_frames(config, "env_0/rbkairos", "robot_arm_tool0")

    sources = ObservationFactory().create_data_sources(config, node=node, simulation_state_container=AgentParameters())

    assert isinstance(sources["end_effector_pose"], EndEffectorPoseTFGenerator)
    assert isinstance(sources["joint_states"], JointStateCollector)
    assert isinstance(sources["wrist_color"], ImageColorCollector)
    assert {"arm_goal_in_ee_frame", "arm_dist_angle_to_goal"} <= sources.keys()
