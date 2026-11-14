from rosnav_rl.cfg.action_spaces import ManipulatorActionSpace

from arena_training.arena_rosnav_rl.cfg.arena_cfg.robot import RobotCfg
from arena_training.arena_rosnav_rl.utils.envs import stow_duration


def _tiago_arm():
    arm = RobotCfg(robot_model="tiago").effective_caps.arm["arm"]
    return arm, ManipulatorActionSpace.from_config(arm)


def test_stow_takes_the_slowest_joint_over_its_range():
    arm, space = _tiago_arm()
    duration = stow_duration(space.joint_names, space.stow_positions, space.position_limits, arm.joint_limits)
    first = arm.joint_limits[space.joint_names[0]]
    low, high = space.position_limits[0]
    stow = space.stow_positions[0]
    assert duration >= max(abs(stow - low), abs(stow - high)) / first["max_velocity"]
    assert duration > 0.1


def test_stow_without_velocity_limits_is_the_minimum():
    assert stow_duration(["j"], [0.0], [(-3.0, 3.0)], {}) == 0.1


def test_faster_joints_stow_sooner():
    slow = stow_duration(["j"], [0.0], [(-3.0, 3.0)], {"j": {"has_velocity_limits": True, "max_velocity": 1.0}})
    fast = stow_duration(["j"], [0.0], [(-3.0, 3.0)], {"j": {"has_velocity_limits": True, "max_velocity": 2.0}})
    assert slow == 3.0
    assert fast == 1.5
