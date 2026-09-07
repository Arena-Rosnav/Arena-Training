from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import arena_robots.Robot
from ament_index_python.packages import get_package_share_directory

# Resolve path back through symlinks to the actual source tree so agent
# artifacts are always written to Arena/arena_training/agents, not the
# install tree or a site-packages location.
_ARENA_TRAINING_ROOT = Path(__file__).resolve().parents[3]  # .../Arena/arena_training


__all__ = [
    "PathComponent",
    "Agent",
    "AgentTensorboard",
    "AgentEval",
    "ConfigComponent",
    "TrainingCurriculum",
    "RewardFunction",
    "RobotSetting",
    "PathFactory",
    "RosPackages",
]


@dataclass(frozen=True)
class RosPackages:
    """Centralized ROS package paths"""

    ARENA_TRAINING: Path = _ARENA_TRAINING_ROOT  # source tree root, not install


class PathComponent(ABC):
    """Base class for all path components"""

    @cached_property
    @abstractmethod
    def path(self) -> Path:
        """Returns the complete path for this component"""
        pass

    def exists(self) -> bool:
        """Check if the path exists"""
        return self.path.exists()

    def create(self) -> Path:
        """Create directories if they don't exist"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.path


class AgentComponent(PathComponent):
    """Base class for agent-related paths"""

    def __init__(self, agent_name: str, agents_dir: Path | None = None):
        self.agent_name = agent_name
        self._base = (agents_dir or _ARENA_TRAINING_ROOT / "agents") / agent_name


class Agent(AgentComponent):
    """Main agent path"""

    @cached_property
    def path(self) -> Path:
        return self._base


class AgentLogs(AgentComponent):
    """Base class for agent log paths"""

    def __init__(self, agent_name: str, log_type: str, agents_dir: Path | None = None):
        super().__init__(agent_name, agents_dir=agents_dir)
        self.log_type = log_type

    @cached_property
    def path(self) -> Path:
        return self._base / f"{self.log_type}_logs"


class AgentTensorboard(AgentLogs):
    """Agent tensorboard logs"""

    def __init__(self, agent_name: str, agents_dir: Path | None = None):
        super().__init__(agent_name, "training", agents_dir=agents_dir)


class AgentEval(AgentLogs):
    """Agent evaluation logs"""

    def __init__(self, agent_name: str, agents_dir: Path | None = None):
        super().__init__(agent_name, "eval", agents_dir=agents_dir)


class ConfigComponent(PathComponent):
    """Base class for configuration paths"""

    def __init__(self, file_name: str = ""):
        self.file_name = file_name
        self._base = Path(get_package_share_directory("arena_training")) / "configs"

    @cached_property
    def path(self) -> Path:
        return self._base


class TrainingCurriculum(ConfigComponent):
    """Training curriculum paths"""

    @cached_property
    def path(self) -> Path:
        return self._base / "training_curriculums" / self.file_name


class RewardFunction(ConfigComponent):
    """Reward function paths"""

    @cached_property
    def path(self) -> Path:
        file_name = f"{self.file_name}.yaml" if not self.file_name.endswith(".yaml") else self.file_name
        return self._base / "reward_functions" / file_name


class RobotSetting(PathComponent):
    """Robot setting paths"""

    def __init__(self, robot_model: str):
        self.robot_model = robot_model

    @cached_property
    def path(self) -> Path:
        robot = arena_robots.Robot.RobotIdentifier(self.robot_model).resolve_sync()
        return robot.path / "model_params.yaml"


class PathDictionary(dict):
    def create_all(self):
        for path in self.values():
            path.create()


class PathFactory:
    """Factory class to create path instances"""

    DEFAULT_AGENTS_DIR: Path = _ARENA_TRAINING_ROOT / "agents"

    @staticmethod
    def get_paths(
        agent_name: str,
        robot_model: str,
        agents_dir: Path | None = None,
    ) -> dict[type[PathComponent], PathComponent]:
        """Generate all required paths for the agent.

        Args:
            agent_name: Name of the agent.
            robot_model: Robot identifier resolved through arena_robots.
            agents_dir: Custom base directory for agent artifacts.
                        If *None*, falls back to ``_ARENA_TRAINING_ROOT / "agents"``.
        """
        return PathDictionary(
            {
                Agent: Agent(agent_name, agents_dir=agents_dir),
                AgentTensorboard: AgentTensorboard(agent_name, agents_dir=agents_dir),
                AgentEval: AgentEval(agent_name, agents_dir=agents_dir),
                RobotSetting: RobotSetting(robot_model),
                ConfigComponent: ConfigComponent(),
            }
        )
