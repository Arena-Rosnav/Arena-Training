"""Training configuration loader."""

from pathlib import Path

from .. import cfg
from .training import load_config


def load_training_config(path: str) -> cfg.TrainingCfg:
    raw = load_config(path)

    # Resolve relative observations_config path against the config file's directory
    obs_cfg = raw.get("agent_config", {}).get("observations_config")
    if obs_cfg and not Path(obs_cfg).is_absolute():
        raw["agent_config"]["observations_config"] = str(
            Path(path).parent / obs_cfg
        )

    return cfg.TrainingCfg.model_validate(
        raw, strict=True, from_attributes=True
    )
