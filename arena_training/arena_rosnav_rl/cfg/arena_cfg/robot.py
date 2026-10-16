import arena_robots.Robot
from arena_robots.caps import MobileSpec, RobotCaps
from arena_robots.Robot import RobotView
from arena_robots.assembly import RequestPart
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List


class RobotCfg(BaseModel):
    """Arena robot handle: the canonical ``robot_model`` identifier plus a
    lazily-resolved :class:`~arena_robots.caps.MobileSpec` that both training
    code and the rosnav_rl stack read for kinematics/sensor geometry.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    robot_model: str = Field("jackal", description="Robot identifier (e.g. 'jackal', 'burger')")
    parts: Dict[str, List[Any]] = Field(default_factory=dict, description="Slot-to-part assembly mapping")
    frames: Dict[str, str] = Field(default_factory=dict, description="frame overrides")

    robot_description: MobileSpec | None = Field(
        None,
        alias="Robot Yaml Description",
        exclude=True,
    )

    view: RobotView | None = Field(None, description="The fully resolved robot capabilities view", exclude=True)
    effective_caps: RobotCaps | None = Field(None, description="Resolved effective capabilities", exclude=True)

    def model_post_init(self, context: object, /) -> None:

        if self.view is None:
            resolved_view = arena_robots.Robot.RobotIdentifier(self.robot_model).resolve_sync()

            parsed_parts = {}
            if self.parts:
                for category, items in self.parts.items():
                    parsed_parts[category] = [
                        RequestPart(**item) if isinstance(item, dict) else item 
                        for item in items
                    ]

            caps = resolved_view.effective_caps(parts=parsed_parts, frames=self.frames or None)

            object.__setattr__(self, "effective_caps", caps)
            object.__setattr__(self, "view", resolved_view)

        if self.robot_description is None:
            mobile = self.view.mobile
            if mobile is None:
                raise ValueError(f"robot '{self.robot_model}' does not advertise a 'mobile' cap, RobotCfg requires caps/mobile.yaml")
            object.__setattr__(self, "robot_description", mobile)
