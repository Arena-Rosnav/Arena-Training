import posixpath
from typing import Self

import attrs
from arena_rclpy_mxins.shared import FrameNamespace
from arena_runtime_msgs.msg import RobotDescriptor


@attrs.frozen
class RobotFrames:
    env: str
    robot: str
    base: str
    odom: str

    @classmethod
    def from_descriptor(cls, descriptor: RobotDescriptor, odom_frame: str) -> Self:
        if not descriptor.frame:
            raise ValueError(f"robot {descriptor.name!r} has no frame in its fleet descriptor")
        robot = FrameNamespace(descriptor.frame)
        assert posixpath.basename(descriptor.frame) == descriptor.name, "robot frame is not <env>/<robot>; env frame cannot be derived by dirname"
        return cls(
            env=str(FrameNamespace(posixpath.dirname(descriptor.frame))),
            robot=str(robot),
            base=descriptor.base_frame,
            odom=str(robot(odom_frame)),
        )
