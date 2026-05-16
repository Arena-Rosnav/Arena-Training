from collections.abc import Callable
from typing import Optional

from .arena_trainer import ArenaTrainer
from .dreamerv3_trainer import DreamerV3Trainer
from .sb3_trainer import StableBaselines3Trainer
from rosnav_rl.utils.type_aliases import SupportedRLFrameworks

_REGISTRY: dict[SupportedRLFrameworks, type[ArenaTrainer]] = {
    SupportedRLFrameworks.STABLE_BASELINES3: StableBaselines3Trainer,
    SupportedRLFrameworks.DREAMER_V3: DreamerV3Trainer,
}


def get_trainer(
    config,
    namespace_fn: Optional[Callable[[int], str]] = None,
) -> ArenaTrainer:
    """Instantiate the correct trainer for the framework named in *config*."""
    framework = SupportedRLFrameworks(config.agent_config.framework.name)
    cls = _REGISTRY.get(framework)
    if cls is None:
        raise ValueError(
            f"Unsupported framework: {framework!r}. "
            f"Supported: {[f.value for f in _REGISTRY]}"
        )
    if namespace_fn is not None:
        return cls(config, namespace_fn=namespace_fn)
    return cls(config)