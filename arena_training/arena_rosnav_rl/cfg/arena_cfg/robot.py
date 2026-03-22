from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# from ...tools.general import get_robot_yaml_path, load_yaml
import arena_robots.Robot


class DiscreteAction(BaseModel):
    name: str
    linear: float
    angular: float


class ContinuousActionCfg(BaseModel):
    linear_range: List[float]
    angular_range: List[float]


class ActionsCfg(BaseModel):
    discrete: List[DiscreteAction]
    continuous: ContinuousActionCfg


class LaserCfg(BaseModel):
    angle: Dict[str, float]
    num_beams: int
    range: float
    update_rate: int


class RobotYamlCfg(BaseModel):
    robot_model: str
    robot_radius: float
    robot_base_frame: str
    robot_sensor_frame: str
    is_holonomic: bool
    actions: ActionsCfg
    laser: LaserCfg


class RobotCfg(BaseModel):
    robot_model: str = Field("jackal", description="Robot identifier (e.g. 'jackal', 'burger')")
    robot_description: Optional[RobotYamlCfg] = Field(
        None, alias="Robot Yaml Description"
    )

    def model_post_init(self, __context) -> None:
        if self.robot_description is None:
            object.__setattr__(
                self,
                "robot_description",
                RobotYamlCfg.model_validate(
                    arena_robots.Robot.RobotIdentifier(self.robot_model).resolve_sync().model_params
                ),
            )
