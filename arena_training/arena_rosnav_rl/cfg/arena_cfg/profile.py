from pydantic import BaseModel
from typing import Optional


class ProfilingCfg(BaseModel):
    log_file: Optional[str] = None
    do_profile_step: bool = False
    do_profile_reset: bool = False
    per_call: bool = False
    print_stats: bool = True  # Print profiling stats to the ROS logger
