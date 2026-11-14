"""A training robot's parts reach both its resolved caps and the env's robot launch argument."""

from __future__ import annotations

import json

from arena_robots.assembly import RequestPart

from arena_training.arena_rosnav_rl.cfg.arena_cfg.robot import RobotCfg


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
