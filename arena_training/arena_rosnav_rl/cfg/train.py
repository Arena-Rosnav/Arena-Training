import os
from pathlib import Path
from typing import Optional, Union

import rosnav_rl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .arena_cfg import ArenaBaseCfg
from .sb3_cfg import ArenaSB3Cfg
from ..utils.training import build_training_commented_map


class TrainingCfg(BaseModel):
    __version__ = "0.1.0"

    arena_cfg: Union[ArenaSB3Cfg, ArenaBaseCfg]
    agent_config: rosnav_rl.AgentConfig
    resume: bool = False

    agents_dir: Optional[Path] = Field(
        None,
        description=(
            "Custom base directory for agent artifacts. "
            "Resolution order: this field → ROSNAV_AGENTS_DIR env var → default."
        ),
    )

    model_config = ConfigDict(
        extra="forbid",  # Reject extra fields
        arbitrary_types_allowed=True,  # Allow custom types
    )

    @property
    def resolved_agents_dir(self) -> Optional[Path]:
        """Resolve the agents directory with a 3-level fallback chain.

        Priority:
            1. Explicit ``agents_dir`` in this config (highest).
            2. ``ROSNAV_AGENTS_DIR`` environment variable.
            3. ``None`` → PathFactory uses its built-in default.
        """
        if self.agents_dir is not None:
            return Path(self.agents_dir).expanduser().resolve()

        env_dir = os.environ.get("ROSNAV_AGENTS_DIR")
        if env_dir:
            return Path(env_dir).expanduser().resolve()

        return None  # let PathFactory use its default

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save the full training config as a structured, human-readable YAML."""
        from ruamel.yaml import YAML

        ry = YAML()
        ry.default_flow_style = False
        ry.width = 120

        with open(path, "w") as f:
            ry.dump(build_training_commented_map(self), f)

    @model_validator(mode="after")
    def validate_resume(self):
        if self.resume:
            assert (
                self.agent_config.name is not None
            ), "Agent name must be provided for resume!"
            assert (
                self.agent_config.framework.algorithm.checkpoint is not None
            ), "Checkpoint must be provided for resume!"
        return self
