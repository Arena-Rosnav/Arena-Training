import os
from pathlib import Path
from typing import Optional, Union

import rosnav_rl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..cfg.arena_cfg.robot import DiscreteAction
from ..utils.type_alias.observation import CustomDiscreteActionList
from .arena_cfg import ArenaBaseCfg
from .sb3_cfg import ArenaSB3Cfg


class TrainingCfg(BaseModel):
    __version__ = "0.1.0"

    arena_cfg: Union[ArenaSB3Cfg, ArenaBaseCfg]
    agent_cfg: rosnav_rl.AgentCfg
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

    # TODO: Maybe move this to a more general place - closer to the RobotCfg
    @model_validator(mode="after")
    def generate_custom_discrete_actions(self):
        """
        Generates custom discrete actions if specified in the agent configuration.

        This method checks if a custom discretization for the action space is defined
        in the agent's configuration (`self.agent_cfg.action_space.custom_discretization`)
        and if robot configuration is available (`self.arena_cfg.robot`).

        If both conditions are met, it generates a list of discrete actions using
        the `generate_discrete_from_box_dict` method. This method takes the
        linear range for continuous actions from the robot's description
        (`self.arena_cfg.robot.robot_description.actions.continuous.linear_range`)
        as input for both x and y linear velocity discretization.

        The generated list of discrete actions (which are dictionaries) is then
        used to update the robot's discrete action space
        (`self.arena_cfg.robot.robot_description.actions.discrete`). Each action
        dictionary is converted into a `DiscreteAction` object.

        Returns:
            self: The instance of the class, allowing for method chaining.
        """
        if (
            self.agent_cfg.action_space.custom_discretization is not None
            and self.arena_cfg.robot is not None
        ):
            custom_discrete_actions_list: CustomDiscreteActionList = (
                self.agent_cfg.action_space.custom_discretization.generate_discrete_from_box_dict(
                    self.arena_cfg.robot.robot_description.actions.continuous.linear_range,
                    self.arena_cfg.robot.robot_description.actions.continuous.linear_range,
                )
            )
            self.arena_cfg.robot.robot_description.actions.discrete = [
                DiscreteAction(**action) for action in custom_discrete_actions_list
            ]
        return self

    @model_validator(mode="after")
    def validate_resume(self):
        if self.resume:
            assert (
                self.agent_cfg.name is not None
            ), "Agent name must be provided for resume!"
            assert (
                self.agent_cfg.framework.algorithm.checkpoint is not None
            ), "Checkpoint must be provided for resume!"
        return self
