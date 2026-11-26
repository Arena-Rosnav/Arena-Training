import posixpath
from typing import Self

import attrs
from arena_runtime_msgs.msg import RobotDescriptor


@attrs.frozen
class RobotFrames:
    env: str
    robot: str
    base: str
    odom: str

    @classmethod
    def from_descriptor(cls, descriptor: RobotDescriptor) -> Self:
        if not descriptor.frame:
            raise ValueError(f"robot {descriptor.name!r} has no frame in its fleet descriptor")
        robot = descriptor.frame
        assert posixpath.basename(descriptor.frame) == descriptor.name, "robot frame is not <env>/<robot>; env frame cannot be derived by dirname"
        return cls(
            env=str(posixpath.dirname(descriptor.frame)),
            robot=str(robot),
            base=descriptor.base_frame,
            odom=posixpath.join(robot, "odom"),
        )
