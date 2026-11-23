from __future__ import annotations

import configparser
from collections.abc import Callable
from pathlib import Path

import pytest

_ROBOTS_REPO = Path(__file__).resolve().parents[2] / "arena_robots"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_installed(*names): skip unless every submodule of each named robot or component is checked out")


def _submodules() -> dict[str, list[Path]]:
    cfg = configparser.ConfigParser()
    cfg.read(_ROBOTS_REPO / ".gitmodules")
    out: dict[str, list[Path]] = {}
    for section in cfg.sections():
        path = cfg[section]["path"]
        names = cfg[section].get("robot", "").split() + cfg[section].get("component", "").split()
        if not names and path.startswith("arena_robots/robots/"):
            names = [path.split("/")[2]]
        for name in names:
            out.setdefault(name, []).append(_ROBOTS_REPO / path)
    return out


def _missing(names: tuple[str, ...] | list[str]) -> str | None:
    submodules = _submodules()
    for name in names:
        if any(not path.is_dir() or not any(path.iterdir()) for path in submodules.get(name, [])):
            return name
    return None


@pytest.fixture(scope="session")
def require_installed() -> Callable[..., None]:
    """Skips the test unless every submodule tagged for each robot or component is checked out."""

    def require(*names: str) -> None:
        if (name := _missing(names)) is not None:
            pytest.skip(f"{name} is not installed, run `arena feature robots add {name}`")

    return require


@pytest.fixture(autouse=True)
def _requires_installed(request: pytest.FixtureRequest, require_installed: Callable[..., None]) -> None:
    for marker in request.node.iter_markers("requires_installed"):
        require_installed(*marker.args)


@pytest.fixture(scope="session")
def config_requirements() -> Callable[[Path], list[str]]:
    """Return the robot and component names a training config's robot needs checked out."""
    from arena_robots.catalog import Catalog
    from rosnav_rl.utils.yaml_io import load_yaml

    catalog = Catalog()

    def requirements(config: Path) -> list[str]:
        robot = load_yaml(str(config))["arena_cfg"].get("robot")
        if robot is None:
            return []
        names = [robot["robot_model"]]
        for kind, parts in (robot.get("parts") or {}).items():
            names.extend(f"{kind}/{catalog.owner(kind, part['variant'])}" for part in parts)
        return names

    return requirements


@pytest.fixture
def installed_configs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_requirements: Callable[[Path], list[str]]) -> list[Path]:
    """Point the arena_training share dir at the shipped configs whose robots are checked out, return those."""
    from ament_index_python.packages import get_package_share_directory
    from arena_training.arena_rosnav_rl.utils.paths import ConfigComponent

    shipped = Path(ConfigComponent().path)
    configs = tmp_path / "configs"
    configs.mkdir()
    kept = []
    for entry in sorted(shipped.iterdir()):
        if entry.suffix == ".yaml":
            if _missing(config_requirements(entry)) is not None:
                continue
            kept.append(entry)
        (configs / entry.name).symlink_to(entry)

    def share(package: str) -> str:
        return str(tmp_path) if package == "arena_training" else get_package_share_directory(package)

    monkeypatch.setattr("arena_training.arena_rosnav_rl.utils.paths.get_package_share_directory", share)
    monkeypatch.setattr("ament_index_python.packages.get_package_share_directory", share)
    return kept
