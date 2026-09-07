from pydantic import BaseModel


class ProfilingCfg(BaseModel):
    log_file: str | None = None
    do_profile_step: bool = False
    do_profile_reset: bool = False
    per_call: bool = False
    print_stats: bool = True  # Print profiling stats to the ROS logger
