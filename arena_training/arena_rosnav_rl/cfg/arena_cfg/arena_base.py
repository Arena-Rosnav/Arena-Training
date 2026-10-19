from pydantic import BaseModel
from rosnav_rl.cfg.logging import LoggingCfg

from .general import GeneralCfg
from .monitor import MonitoringCfg
from .profile import ProfilingCfg
from .robot import RobotCfg
from .task import TaskCfg


class ArenaBaseCfg(BaseModel):
    general: GeneralCfg = GeneralCfg()
    logging: LoggingCfg | None = LoggingCfg()
    monitoring: MonitoringCfg = MonitoringCfg()
    task: TaskCfg = TaskCfg()
    profiling: ProfilingCfg | None = None
    robot: RobotCfg = RobotCfg()
