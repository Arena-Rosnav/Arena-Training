"""Training configuration loader."""

import json
import logging
from pathlib import Path

from .. import cfg
from .training import load_config

logger = logging.getLogger(__name__)


def get_config_path(config_file: str) -> Path:
    """The training config file: an absolute path, else relative to the current directory, else in the arena_training share configs."""
    config_path = Path(config_file)

    if config_path.is_absolute():
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.info(f"Using config file: {config_path}")
        return config_path

    if config_path.exists():
        logger.info(f"Using config file from current directory: {config_path.absolute()}")
        return config_path.absolute()

    from ament_index_python.packages import get_package_share_directory

    try:
        arena_training_dir = get_package_share_directory("arena_training")
        package_config_path = Path(arena_training_dir) / "configs" / config_file

        if package_config_path.exists():
            logger.info(f"Using config file from arena_training: {package_config_path}")
            return package_config_path
        else:
            logger.warning(f"Config file not found at {package_config_path}")
    except Exception as e:
        logger.warning(f"Could not locate arena_training package: {e}")

    raise FileNotFoundError(f"Config file '{config_file}' not found in:\n  - Current directory\n  - arena_training package share configs\n  - arena_bringup package configs\nPlease provide an absolute path or ensure the file exists in one of these locations.")


def load_training_config(path: str) -> cfg.TrainingCfg:
    raw = load_config(path)

    # Resolve relative observations_config path against the config file's directory
    obs_cfg = raw.get("agent_config", {}).get("observations_config")
    if obs_cfg and not Path(obs_cfg).is_absolute():
        raw["agent_config"]["observations_config"] = str(Path(path).parent / obs_cfg)

    return cfg.TrainingCfg.model_validate_json(json.dumps(raw), strict=True)
