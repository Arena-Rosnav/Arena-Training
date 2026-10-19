from typing import (
    Any,
    Callable,
    Protocol,
    TypeVar,
)

from ...utils.paths import PathComponent


class ClosableEnvironment(Protocol):
    def close(self) -> None: ...


InformationDict = dict[str, Any]

PathsDict = dict[type[PathComponent], PathComponent]

CustomDiscreteAction = dict[str, str | float]
CustomDiscreteActionList = list[CustomDiscreteAction]

ObservationCollectorDataClass = TypeVar("ObservationCollectorDataClass")

T = TypeVar("T")
ProcessingFunction = Callable[[T], ObservationCollectorDataClass]
