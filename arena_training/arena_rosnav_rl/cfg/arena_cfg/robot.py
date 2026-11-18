import logging
from typing import Any

import arena_robots.Robot
import attrs
from arena_robots.assembly import RequestPart
from arena_robots.caps import MobileSpec, RobotCaps
from arena_robots.inline import format_inline_override, format_part
from arena_robots.Robot import RobotView
from arena_robots.Sensor import SensorType
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
    caps: dict[str, Any] = Field(default_factory=dict, description="Cap overrides: '<cap>.<path>' to a value or an operation ({scale, offset}, {append}, {delete})")

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
        """The ``robot:=`` launch argument requesting this model with its parts, frames and cap overrides."""
        requests = [f"{part.mount}={format_part(f'{category}/{part.variant}', part.caps, part.params)}" if part.mount else f"{category}={format_part(part.variant, part.caps, part.params)}" for category, parts in self.parts.items() for part in parts]
        requests += [f"frames.{mount}={stem}" for mount, stem in self.frames.items()]
        requests += [f"{path}={format_inline_override(value)}" for path, value in self.caps.items()]
        return f"robot:={self.robot_model}[{','.join(requests)}]" if requests else f"robot:={self.robot_model}"

    @property
    def laser_topic(self) -> str | None:
        """Namespace-relative topic of the first laserscan sensor of the resolved robot."""
        assert self.view is not None
        sensors = self.view.effective_sensors(self.parts, frames=self.frames or None)
        topic = next((sensor.topic for sensor in sensors if sensor.type == SensorType.LASERSCAN), None)
        return None if topic is None else topic.removeprefix("${namespace}/")

    def model_post_init(self, context: object, /) -> None:
        if self.view is None:
            resolved_view = arena_robots.Robot.RobotIdentifier(self.robot_model).resolve_sync()
            caps = resolved_view.effective_caps(parts=self.parts, frames=self.frames or None, caps=self.caps or None)
            try:
                caps.validate_overrides()
            except ValueError as error:
                raise ValueError(f"robot '{self.robot_model}': {error}") from error
            for warning in caps.warnings():
                logging.getLogger(__name__).warning("robot %r: %s", self.robot_model, warning)

            object.__setattr__(self, "effective_caps", caps)
            object.__setattr__(self, "view", resolved_view)

        if self.robot_description is None:
            assert self.effective_caps is not None
            mobile = self.effective_caps.mobile
            if mobile is None:
                raise ValueError(f"robot '{self.robot_model}' does not advertise a 'mobile' cap, RobotCfg requires caps/mobile.yaml")
            object.__setattr__(self, "robot_description", mobile)
