from pydantic import Field
from rosnav_rl.model.stable_baselines3.cfg import CallbacksCfg


# TrainingCurriculumCfg moved to task.staged configuration
# This maintains the legacy model validator behavior for backwards compatibility
class TrainingCurriculumCfg:
    """Legacy class - curriculum configuration moved to task.staged section."""

    pass


class ArenaCallbacksCfg(CallbacksCfg):
    checkpoint_ticks: list[int] = Field(default_factory=list, description="Control ticks at which the model is saved as tick_<T>.zip.")
