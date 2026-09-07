from pydantic import BaseModel, ValidationInfo, field_validator
from rosnav_rl.cfg.logging import LoggingCfg

from .general import GeneralCfg
from .monitor import MonitoringCfg
from .profile import ProfilingCfg
from .robot import RobotCfg
from .task import TaskCfg


class ArenaBaseCfg(BaseModel):
    general: GeneralCfg | None = GeneralCfg()
    logging: LoggingCfg | None = LoggingCfg()
    monitoring: MonitoringCfg | None = MonitoringCfg()
    task: TaskCfg | None = TaskCfg()
    profiling: ProfilingCfg | None = None
    robot: RobotCfg | None = RobotCfg()

    @field_validator(
        "general",
        "monitoring",
        "task",
        "robot",
        mode="after",
    )
    @classmethod
    def check_attr_none(
        cls,
        v: GeneralCfg | MonitoringCfg | TaskCfg | RobotCfg | None,
        values: ValidationInfo,
    ) -> GeneralCfg | MonitoringCfg | TaskCfg | RobotCfg:
        if v is None:
            raise ValueError(f"{v} cannot be None")
        return v
