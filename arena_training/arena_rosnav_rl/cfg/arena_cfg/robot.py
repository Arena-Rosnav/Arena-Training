from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

import arena_robots.Robot
from arena_robots.caps import MobileSpec


class RobotCfg(BaseModel):
    """Arena robot handle: the canonical ``robot_model`` identifier plus a
    lazily-resolved :class:`~arena_robots.caps.MobileSpec` that both training
    code and the rosnav_rl stack read for kinematics/sensor geometry.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    robot_model: str = Field("jackal", description="Robot identifier (e.g. 'jackal', 'burger')")
    robot_description: Optional[MobileSpec] = Field(
        None,
        alias="Robot Yaml Description",
        exclude=True,
    )

    def model_post_init(self, __context) -> None:
        if self.robot_description is None:
            mobile = arena_robots.Robot.RobotIdentifier(self.robot_model).resolve_sync().mobile
            if mobile is None:
                raise ValueError(
                    f"robot '{self.robot_model}' does not advertise a 'mobile' cap — "
                    f"RobotCfg requires caps/mobile.yaml"
                )
            object.__setattr__(self, "robot_description", mobile)
