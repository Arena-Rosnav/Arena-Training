"""A training robot's parts reach both its resolved caps and the env's robot launch argument."""

from __future__ import annotations

import json

import pytest
from arena_robots.assembly import RequestPart
from arena_robots.SetupFile import Config
from arena_runtime.env.robot_manager.robots_manager import desugar_robot_entry

from arena_training.arena_rosnav_rl.cfg.arena_cfg.robot import RobotCfg
from arena_training.arena_rosnav_rl.trainer.arena_trainer import action_ranges, command_ranges, mobile_action_space


def test_parts_survive_a_json_dump_and_reload():
    robot = RobotCfg(robot_model="rbkairos", parts={"arm": [{"variant": "ur5e", "mount": "arm_mount"}], "lidar": [{"variant": "sick_s300"}]})
    dumped = json.loads(json.dumps(robot.model_dump(mode="json")))
    assert RobotCfg(**dumped).parts == robot.parts


def test_a_robot_without_parts_requests_the_bare_model():
    assert RobotCfg(robot_model="jackal").launch_arg == "robot:=jackal"


def test_mounted_and_type_only_parts_are_both_requested():
    robot = RobotCfg(robot_model="rbkairos", parts={"arm": [{"variant": "ur5e", "mount": "arm_mount"}], "lidar": [{"variant": "sick_s300"}]})
    assert robot.launch_arg == "robot:=rbkairos[arm_mount=arm/ur5e,lidar=sick_s300]"


def test_yaml_parts_resolve_into_the_arm_caps():
    robot = RobotCfg(robot_model="rbkairos", parts={"arm": [{"variant": "ur5e", "mount": "arm_mount"}]})
    assert robot.parts == {"arm": [RequestPart(variant="ur5e", mount="arm_mount")]}
    arm = robot.effective_caps.arm["arm_mount"]
    assert arm.controller == "arm_controller"
    assert set(arm.named_poses["stow"]) == set(arm.chain)


def test_an_overriding_robot_carries_none_of_the_previous_parts():
    assert RobotCfg(robot_model="jackal").effective_caps.arm is None


def test_a_velocity_override_sets_the_instance_velocity_limits():
    robot = RobotCfg(robot_model="jackal", caps={"mobile.velocity_limits.linear": [-0.5, 1.0]})
    limits = robot.robot_description.velocity_limits
    assert (limits.linear.min, limits.linear.max) == (-0.5, 1.0)
    assert (limits.angular.min, limits.angular.max) == (-4.0, 4.0)


def test_a_scale_override_scales_the_instance_velocity_limits():
    robot = RobotCfg(robot_model="jackal", caps={"mobile.velocity_limits.linear": {"scale": 0.5}})
    limits = robot.robot_description.velocity_limits
    assert (limits.linear.min, limits.linear.max) == (-1.0, 1.0)


def test_overrides_leave_the_model_caps_of_other_instances_alone():
    RobotCfg(robot_model="jackal", caps={"mobile.velocity_limits.linear": [-0.5, 1.0]})
    limits = RobotCfg(robot_model="jackal").robot_description.velocity_limits
    assert (limits.linear.min, limits.linear.max) == (-2.0, 2.0)


def test_the_mobile_action_space_follows_the_overridden_velocity_limits():
    robot = RobotCfg(robot_model="jackal", caps={"mobile.velocity_limits.linear": [-0.5, 1.0]})
    space = mobile_action_space(robot.robot_description, None)
    linear, _lateral, angular = action_ranges(robot.robot_description)
    assert linear == (-0.5, 1.0)
    assert angular == (-4.0, 4.0)
    assert command_ranges(space)[0] == (-0.5, 1.0)


def test_an_unknown_override_path_names_the_robot():
    with pytest.raises(ValueError, match="jackal"):
        RobotCfg(robot_model="jackal", caps={"mobile.velocity_limits.linear": "fast"})


def test_an_override_of_a_cap_the_robot_lacks_names_the_robot():
    with pytest.raises(ValueError, match="jackal"):
        RobotCfg(robot_model="jackal", caps={"arm.stall_time": 2.0})


def test_caps_survive_a_json_dump_and_reload():
    robot = RobotCfg(robot_model="jackal", caps={"mobile.velocity_limits.linear": [-0.5, 1.0], "mobile.radius": {"scale": 1.2}})
    dumped = json.loads(json.dumps(robot.model_dump(mode="json")))
    assert RobotCfg(**dumped).caps == robot.caps


def test_a_robot_with_cap_overrides_requests_them_inline():
    robot = RobotCfg(robot_model="jackal", caps={"mobile.velocity_limits.linear": [-0.5, 1.0], "mobile.radius": {"scale": 1.2}})
    assert robot.launch_arg == "robot:=jackal[mobile.velocity_limits.linear=[-0.5,1.0],mobile.radius=*1.2]"


def test_the_launch_arg_of_parts_frames_and_caps_parses_back_to_the_same_request():
    robot = RobotCfg(
        robot_model="rbkairos",
        parts={"arm": [{"variant": "ur5e", "mount": "arm_mount"}], "lidar": [{"variant": "sick_s300"}]},
        frames={"arm_mount": "custom_mount"},
        caps={"mobile.velocity_limits.linear": [-0.5, 1.0], "mobile.radius": {"scale": 1.2}},
    )
    entry = robot.launch_arg.removeprefix("robot:=")
    (config,) = Config.parse(desugar_robot_entry(entry))
    assert config.robot == "rbkairos"
    assert config.parts == {"arm_mount": ["arm/ur5e"], "lidar": ["sick_s300"]}
    assert config.frames == robot.frames
    assert config.caps == robot.caps


def test_part_caps_reach_the_instance_and_the_launch_arg():
    robot = RobotCfg(robot_model="rbkairos", parts={"arm": [{"variant": "ur5e", "mount": "arm_mount", "caps": {"stall_time": 2.0}}]})
    assert robot.effective_caps.arm is not None
    assert robot.effective_caps.arm["arm_mount"].stall_time == 2.0
    entry = robot.launch_arg.removeprefix("robot:=")
    (config,) = Config.parse(desugar_robot_entry(entry))
    assert config.parts == {"arm_mount": ["arm/ur5e[stall_time=2.0]"]}


def test_a_robot_level_override_of_a_placed_part_names_the_robot():
    with pytest.raises(ValueError, match="rbkairos"):
        RobotCfg(robot_model="rbkairos", parts={"arm": [{"variant": "ur5e", "mount": "arm_mount"}]}, caps={"arm.arm_mount.stall_time": 2.0})
