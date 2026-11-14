import logging
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class CurriculumStage(BaseModel):
    """Configuration for a single curriculum stage matching actual ROS2 parameter structure."""

    model_config = {"populate_by_name": True}

    task_random_static_n: list[int] = Field(
        default_factory=lambda: [0, 0],
        description="Min/max static obstacles",
        alias="task.random.static.n",
    )
    task_random_interactive_n: list[int] = Field(
        default_factory=lambda: [0, 0],
        description="Min/max interactive obstacles",
        alias="task.random.interactive.n",
    )
    task_random_dynamic_n: list[int] = Field(
        default_factory=lambda: [0, 0],
        description="Min/max dynamic obstacles",
        alias="task.random.dynamic.n",
    )
    task_random_static_models: list[str] = Field(
        default_factory=list,
        description="Static obstacle model names",
        alias="task.random.static.models",
    )
    task_random_interactive_models: list[str] = Field(
        default_factory=list,
        description="Interactive obstacle model names",
        alias="task.random.interactive.models",
    )
    task_random_dynamic_models: list[str] = Field(
        default_factory=list,
        description="Dynamic obstacle model names",
        alias="task.random.dynamic.models",
    )
    task_episode_goto_pose_tolerance_radius: float | None = Field(
        default=None,
        description="Goal tolerance radius",
        alias="task.episode.goto_pose.tolerance.radius",
    )

    @model_validator(mode="before")
    @classmethod
    def _rename_deprecated_keys(cls, data: object) -> object:
        if isinstance(data, dict) and "goal_tolerance_radius" in data:
            logging.getLogger(__name__).warning("curriculum key 'goal_tolerance_radius' is deprecated, use 'task.episode.goto_pose.tolerance.radius'")
            data = dict(data)
            data.setdefault("task.episode.goto_pose.tolerance.radius", data.pop("goal_tolerance_radius"))
        return data


class StagedCfg(BaseModel):
    starting_stage: int = Field(default=0, description="Initial curriculum stage")
    curriculum_definition: list[CurriculumStage] = Field(default_factory=list, description="Curriculum stages defined under staged")

    # Curriculum progression parameters (moved from callbacks.training_curriculum)
    threshold_type: str = Field(
        default="succ",
        description="Metric to use: 'succ' (success rate) or 'rew' (reward)",
    )
    upper_threshold: float = Field(default=0.9, description="Performance threshold to advance to next stage")
    lower_threshold: float = Field(default=0.3, description="Performance threshold to stay at current stage")
    stage_ticks: list[int] | None = Field(
        default=None,
        description="Control tick at which each stage starts (first 0). Set, it replaces the performance thresholds and starting_stage.",
    )

    # ROS2 parameter communication settings
    timeout: float = Field(default=5.0, description="Timeout for ROS2 parameter service calls (seconds)")

    @field_validator("threshold_type")
    @classmethod
    def validate_threshold_type(cls, v: str) -> str:
        """Validate threshold_type is either 'rew' or 'succ'."""
        if v not in ["rew", "succ"]:
            raise ValueError(f"threshold_type must be 'rew' or 'succ', got '{v}'")
        return v

    @field_validator("upper_threshold", "lower_threshold")
    @classmethod
    def validate_thresholds(cls, v: float) -> float:
        """Validate threshold values are reasonable."""
        if v < 0:
            raise ValueError("Thresholds must be non-negative")
        return v

    @field_validator("starting_stage")
    @classmethod
    def validate_starting_stage(cls, v: int) -> int:
        """Validate starting_stage is non-negative."""
        if v < 0:
            raise ValueError("starting_stage must be non-negative")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v

    @model_validator(mode="after")
    def validate_threshold_relationship(self) -> Self:
        """Validate that upper_threshold > lower_threshold."""
        if self.upper_threshold <= self.lower_threshold:
            raise ValueError(f"upper_threshold ({self.upper_threshold}) must be greater than lower_threshold ({self.lower_threshold})")
        return self

    @model_validator(mode="after")
    def validate_starting_stage_bounds(self) -> Self:
        """Validate starting_stage is within curriculum_definition bounds."""
        if self.curriculum_definition and self.starting_stage >= len(self.curriculum_definition):
            raise ValueError(f"starting_stage ({self.starting_stage}) must be less than number of curriculum stages ({len(self.curriculum_definition)})")
        return self

    @model_validator(mode="after")
    def validate_curriculum_definition(self) -> Self:
        """Validate curriculum_definition has at least one stage if curriculum is used."""
        return self


DEFAULT_MODES = {"robots": "random", "obstacles": "random", "modules": "staged"}
TRAINING_MODULES = ("staged",)


class TaskCfg(BaseModel):
    modes: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODES))

    # The configuration now only supports the nested `staged` block
    staged: StagedCfg | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_modes(cls, data: object) -> object:
        """Maps the legacy tm_robots, tm_obstacles and tm_modules keys onto modes."""
        from arena_runtime.env.task_layer import TM_AXES

        if not isinstance(data, dict) or not any(f"tm_{axis}" in data for axis in TM_AXES):
            return data
        data = dict(data)
        modes = {**DEFAULT_MODES, **data.pop("modes", {})}
        for axis in TM_AXES:
            if f"tm_{axis}" in data:
                logging.getLogger(__name__).warning("task key 'tm_%s' is deprecated, use 'modes.%s'", axis, axis)
                modes[axis] = data.pop(f"tm_{axis}")
        data["modes"] = modes
        return data

    def mode_names(self) -> dict[str, list[str]]:
        return {axis: [name for name in names.split(",") if name] for axis, names in self.modes.items()}

    def env_mode_names(self) -> dict[str, list[str]]:
        """The mode names the env runs, without the modules the trainer runs itself."""
        return {axis: [name for name in names if axis != "modules" or name not in TRAINING_MODULES] for axis, names in self.mode_names().items()}

    def runs_module(self, name: str) -> bool:
        return name in self.mode_names().get("modules", [])

    def get_curriculum_stages(self) -> list[CurriculumStage] | None:
        """Get curriculum stages from configuration.

        Returns:
            List of curriculum stages or None if no curriculum configured
        """
        return self.staged.curriculum_definition if self.staged and self.staged.curriculum_definition else None

    def has_curriculum(self) -> bool:
        """Check if curriculum configuration is present."""
        return bool(self.staged and self.staged.curriculum_definition)

    def get_starting_stage(self) -> int:
        """Get the starting stage index."""
        return self.staged.starting_stage if self.staged else 0
