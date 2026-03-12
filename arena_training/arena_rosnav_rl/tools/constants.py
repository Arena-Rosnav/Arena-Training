from pathlib import Path

_ARENA_TRAINING_ROOT = Path(__file__).resolve().parents[3]  # …/Arena/arena_training


class TRAINING_CONSTANTS:
    class PATHS:
        TRAINING_CONFIGS = (
            lambda config_name: str(_ARENA_TRAINING_ROOT / "configs" / f"{config_name}.yaml")
        )
