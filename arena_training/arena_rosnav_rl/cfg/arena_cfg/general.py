from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union


class GeneralCfg(BaseModel):
    """
    General configuration settings for reinforcement learning environments.

    Attributes:
        debug_mode (bool): Flag to enable or disable debug mode. Default is False.
        n_envs (int): Number of environments to run in parallel. Default is 1.
        max_num_moves_per_eps (int): Maximum number of moves allowed per episode. Default is 150.
        goal_radius (float): Radius around the goal within which the agent is considered to have reached the goal. Default is 0.4.
        safety_distance (float): Minimum safety distance to be maintained by the agent (considers robot radius). Default is 1.0.
        control_hz (float): Control frequency in Hz for the TimeSyncWrapper. Default is 10.0.
        observations_config (str): Path to the observations YAML config file.
    """

    debug_mode: bool = False
    no_gpu: bool = False
    n_envs: int = Field(1, ge=1)
    max_num_moves_per_eps: int = Field(150, ge=1)
    goal_radius: float = Field(0.4, title="Goal Radius", gt=0)
    safety_distance: float = Field(1.0, gt=0)
    verbose: Union[int, bool] = Field(False, title="Verbose Mode")
    control_hz: float = Field(10.0, gt=0, description="Control frequency in Hz for TimeSyncWrapper")
