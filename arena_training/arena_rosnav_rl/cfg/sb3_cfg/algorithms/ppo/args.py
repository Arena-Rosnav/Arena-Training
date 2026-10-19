from collections.abc import Callable
from typing import NotRequired, TypedDict


class TrainArguments(TypedDict):
    total_timesteps: int
    callback: Callable
    progress_bar: NotRequired[bool]
    log_interval: NotRequired[int]
    tb_log_name: NotRequired[str]
    reset_num_timesteps: NotRequired[bool]
