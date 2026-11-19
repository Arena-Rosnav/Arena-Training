import numpy as np
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Twist
from rosnav_rl.observations import DONE_REASONS
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def determine_termination(
    reward_info: dict,
    curr_steps: int,
    max_steps: int,
    info: dict | None = None,
) -> tuple[dict, bool]:
    """
    Determine if the episode should terminate.

    Args:
        reward_info (dict): The reward information.
        curr_steps (int): The current number of steps in the episode.
        max_steps (int): The maximum number of steps per episode.
        info (dict): Additional information.

    Returns:
        tuple: A tuple containing the info dictionary and a boolean flag indicating if the episode should terminate.

    """

    if info is None:
        info = {}

    terminated = reward_info.get("is_done", False)

    if terminated:
        info["done_reason"] = reward_info.get("done_reason", None)
        info["is_success"] = reward_info.get("is_success", 0)
        info["episode_length"] = curr_steps

    if curr_steps >= max_steps:
        terminated = True
        info["done_reason"] = DONE_REASONS.STEP_LIMIT
        info["is_success"] = 0
        info["episode_length"] = curr_steps

    return info, terminated


def get_twist_from_action(action: np.ndarray) -> Twist:
    """
    Converts an action array to a Twist message.

    Args:
        action (np.ndarray): The action array containing linear and angular velocities.

    Returns:
        Twist: A Twist message with the linear and angular velocities set.
    """
    twist = Twist()
    twist.linear.x = float(action[0])
    twist.linear.y = float(action[1])
    twist.angular.z = float(action[2])
    return twist


def get_joint_trajectory_from_action(action: np.ndarray, joint_names: list[str], step_time: float = 0.1) -> JointTrajectory:
    """One-point trajectory reaching the target joint positions after step_time seconds."""
    point = JointTrajectoryPoint()
    point.positions = action.tolist()
    sec = int(step_time)
    point.time_from_start = Duration(sec=sec, nanosec=int((step_time - sec) * 1e9))
    trajectory = JointTrajectory()
    trajectory.joint_names = joint_names
    trajectory.points = [point]
    return trajectory


def get_gripper_command(position: float, max_effort: float) -> GripperCommand.Goal:
    """GripperCommand goal driving the gripper joint to *position* with at most *max_effort*."""
    goal = GripperCommand.Goal()
    goal.command.position = position
    goal.command.max_effort = max_effort
    return goal


def set_laser_topic(config: dict, topic: str | None) -> None:
    """Point every LaserScan collector of *config* that names no topic at the robot's laser topic."""
    for source in config.get("datasources", {}).values():
        if source.get("type") != "sensor_msgs/LaserScan":
            continue
        params = source.setdefault("params", {})
        if "topic" in params:
            continue
        if topic is None:
            raise ValueError("observations config leaves a LaserScan topic to the robot, but the robot declares no laserscan sensor")
        params["topic"] = topic


def stow_duration(
    joint_names: list[str],
    stow_positions: list[float],
    position_limits: list[tuple[float, float]],
    joint_limits: dict[str, dict],
    minimum: float = 0.1,
) -> float:
    """Seconds the slowest joint needs to reach its stow position from the far end of its range."""
    duration = minimum
    for name, stow, (low, high) in zip(joint_names, stow_positions, position_limits, strict=True):
        limits = joint_limits.get(name, next((v for k, v in joint_limits.items() if name.endswith(k)), {}))
        if limits.get("has_velocity_limits", False) and limits["max_velocity"] > 0:
            duration = max(duration, max(abs(stow - low), abs(stow - high)) / limits["max_velocity"])
    return duration
