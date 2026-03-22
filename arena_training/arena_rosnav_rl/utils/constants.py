from enum import Enum


class Simulator(Enum):
    FLATLAND = "flatland"
    GAZEBO = "gazebo"
    UNITY = "unity"


class ArenaType(Enum):
    TRAINING = "training"
    DEPLOYMENT = "deployment"


class EntityManager(Enum):
    PEDSIM = "pedsim"
    FLATLAND = "flatland"
    CROWDSIM = "crowdsim"


class MapGenerator:
    NODE_NAME = "map_generator"
    MAP_FOLDER_NAME = "dynamic_map"


from pathlib import Path as _Path

_ARENA_TRAINING_ROOT = _Path(__file__).resolve().parents[3]  # …/Arena/arena_training


class TRAINING_CONSTANTS:
    class PATHS:
        TRAINING_CONFIGS = (
            lambda config_name: str(_ARENA_TRAINING_ROOT / "configs" / f"{config_name}.yaml")
        )
