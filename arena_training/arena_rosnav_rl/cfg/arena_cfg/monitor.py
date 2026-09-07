from pydantic import BaseModel, Field


class EpisodeLoggingCfg(BaseModel):
    last_n_episodes: int = 20
    record_actions: bool = True


class WandbCfg(BaseModel):
    enabled: bool = False  # Set to true to enable Weights & Biases logging
    project_name: str = Field(default="Arena-RL", title="Project Name", description="Name of the Weights & Biases project.", exclude=None)
    run_name: str | None = None
    group: str | None = None
    tags: list[str] | None = None


class MonitoringCfg(BaseModel):
    wandb: WandbCfg = WandbCfg()
    training_metrics: bool | None = True
    episode_logging: EpisodeLoggingCfg | None = EpisodeLoggingCfg()
    eval_metrics: bool | None = False  # eval_log
