from typing import (
    Any,
    Callable,
    TypeVar,
)

import gymnasium

from ...utils.paths import PathComponent

# Gym Env
EnvironmentType = TypeVar("EnvironmentType", bound=gymnasium.Env | gymnasium.Wrapper)
InformationDict = dict[str, Any]

PathsDict = dict[type[PathComponent], PathComponent]

CustomDiscreteAction = dict[str, str | float]
CustomDiscreteActionList = list[CustomDiscreteAction]

ObservationCollectorDataClass = TypeVar("ObservationCollectorDataClass")

T = TypeVar("T")
ProcessingFunction = Callable[[T], ObservationCollectorDataClass]
