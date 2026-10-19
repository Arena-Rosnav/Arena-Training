import numpy as np
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rosnav_rl.observations import DONE_REASONS
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def determine_termination(
    reward_info: dict,
    curr_steps: int,
    max_steps: int,
    info: dict = None,
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
    point.time_from_start = Duration(seconds=sec, nanoseconds=int((step_time - sec) * 1e9)).to_msg()
    trajectory = JointTrajectory()
    trajectory.joint_names = joint_names
    trajectory.points.append(point)
    return trajectory
