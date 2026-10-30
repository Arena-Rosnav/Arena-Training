from typing import Any

import arena_robots.Robot
import attrs
from arena_robots.assembly import RequestPart
from arena_robots.caps import MobileSpec, RobotCaps
from arena_robots.Robot import RobotView
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class RobotCfg(BaseModel):
    """Arena robot handle: the canonical ``robot_model`` identifier plus a
    lazily-resolved :class:`~arena_robots.caps.MobileSpec` that both training
    code and the rosnav_rl stack read for kinematics/sensor geometry.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    robot_model: str = Field(default="jackal", description="Robot identifier (e.g. 'jackal', 'burger')")
    parts: dict[str, list[RequestPart]] = Field(default_factory=dict, description="Slot-to-part assembly mapping")
    frames: dict[str, str] = Field(default_factory=dict, description="frame overrides")

    robot_description: MobileSpec | None = Field(
        default=None,
        alias="Robot Yaml Description",
        exclude=True,
    )

    view: RobotView | None = Field(default=None, description="The fully resolved robot capabilities view", exclude=True)
    effective_caps: RobotCaps | None = Field(default=None, description="Resolved effective capabilities", exclude=True)

    @field_validator("parts", mode="before")
    @classmethod
    def _parse_parts(cls, value: dict[str, list[dict | RequestPart]]) -> dict[str, list[RequestPart]]:
        return {category: [RequestPart(**item) if isinstance(item, dict) else item for item in items] for category, items in value.items()}

    @field_serializer("parts")
    def _dump_parts(self, parts: dict[str, list[RequestPart]]) -> dict[str, list[dict[str, Any]]]:
        return {slot: [attrs.asdict(p) for p in part_list] for slot, part_list in parts.items()}

    @property
    def launch_arg(self) -> str:
        """The ``robot:=`` launch argument requesting this model with its parts."""
        requests = [f"{part.mount}={category}/{part.variant}" if part.mount else f"{category}={part.variant}" for category, parts in self.parts.items() for part in parts]
        return f"robot:={self.robot_model}[{','.join(requests)}]" if requests else f"robot:={self.robot_model}"

    def model_post_init(self, context: object, /) -> None:
        if self.view is None:
            resolved_view = arena_robots.Robot.RobotIdentifier(self.robot_model).resolve_sync()
            caps = resolved_view.effective_caps(parts=self.parts, frames=self.frames or None)

            object.__setattr__(self, "effective_caps", caps)
            object.__setattr__(self, "view", resolved_view)

        if self.robot_description is None:
            assert self.view is not None
            mobile = self.view.mobile
            if mobile is None:
                raise ValueError(f"robot '{self.robot_model}' does not advertise a 'mobile' cap, RobotCfg requires caps/mobile.yaml")
            object.__setattr__(self, "robot_description", mobile)
