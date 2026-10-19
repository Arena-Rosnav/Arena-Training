from pydantic import BaseModel, Field


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
        lockstep (bool): Gym-exact stepping via sim_lifecycle/step instead of lazy pause. Default is False.
        lockstep_step_seconds (float): Sim seconds advanced per gym step under lockstep. Default is 0.0999.
        env_seed (int | None): Root of the episode seed stream, env i runs on run_seed "<env_seed>-<i>". None leaves it random.
        env_launch_args (list[list[str]]): Extra launch args per env, env i takes entry i modulo the list length (e.g. world or humansim.local_planner).
    """

    debug_mode: bool = False
    no_gpu: bool = False
    n_envs: int = Field(default=1, ge=1)
    max_num_moves_per_eps: int = Field(default=150, ge=1)
    goal_radius: float = Field(default=0.4, title="Goal Radius", gt=0)
    safety_distance: float = Field(default=1.0, gt=0)
    verbose: int | bool = Field(default=False, title="Verbose Mode")
    control_hz: float = Field(default=10.0, gt=0, description="Control frequency in Hz for TimeSyncWrapper")
    lockstep: bool = False
    # Quantized to physics ticks server-side by sim_lifecycle/step.
    lockstep_step_seconds: float = Field(default=0.0999, gt=0)
    env_seed: int | None = None
    env_launch_args: list[list[str]] = Field(default_factory=list)
