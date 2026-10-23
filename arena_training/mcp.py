from __future__ import annotations

import asyncio
import datetime
import importlib.util
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import Field

from arena_mcp.core import JOBS, ROS, Domain, Job, ToolError, cli
from arena_mcp.core.envs import RUNTIME

if TYPE_CHECKING:
    from arena_training.arena_rosnav_rl.cfg import TrainingCfg

domain = Domain(
    "training",
    """DRL training with arena_training and rosnav_rl: configs, validation, training runs, agents and their checkpoints.
    Configs are TrainingCfg YAML files, named from the arena_training share configs dir or given by path.
    training_start runs `arena train` as a job that brings up its own runtime, so no other Arena runtime may be up. Stop it with jobs_cancel.
    Agents live in ROSNAV_AGENTS_DIR, else the arena_training default agents dir, unless a config sets agents_dir.""",
    requires=("rosnav_rl", "arena_training.arena_rosnav_rl"),
)

CHECKPOINT_SUFFIXES = (".zip", ".pt")
EVENTS_GLOB = "events.out.tfevents.*"
REGISTER_ENV = f"{RUNTIME}/register_env"


def _configs_dir() -> Path:
    from arena_training.arena_rosnav_rl.utils.paths import ConfigComponent

    return Path(ConfigComponent().path)


def _config_names() -> list[str]:
    root = _configs_dir()
    return sorted(path.name for path in root.glob("*.yaml")) if root.is_dir() else []


def _resolve_config(name_or_path: str) -> Path:
    from arena_training.arena_rosnav_rl.utils.config import get_config_path

    try:
        return get_config_path(name_or_path)
    except FileNotFoundError as exc:
        reason = str(exc).splitlines()[0]
        raise ToolError(f"no training config {name_or_path!r}: {reason}, available: {', '.join(_config_names())}", hint="pass a config name from training_configs or an absolute path") from exc


def _raw(path: Path) -> dict[str, object]:
    from arena_training.arena_rosnav_rl.utils.training import load_config

    raw = load_config(str(path))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} holds a {type(raw).__name__}, a training config is a mapping with arena_cfg and agent_config")
    return raw


def _errors(exc: Exception) -> list[dict[str, object]]:
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        return [{"loc": ".".join(str(part) for part in error["loc"]), "msg": error["msg"], "type": error["type"]} for error in exc.errors(include_url=False, include_context=False, include_input=False)]
    return [{"loc": "", "msg": str(exc), "type": type(exc).__name__}]


def _load(path: Path) -> tuple[TrainingCfg, dict[str, object]]:
    from arena_training.arena_rosnav_rl.utils.config import load_training_config

    raw = _raw(path)
    return load_training_config(str(path)), raw


def _jsonable(value: object) -> object:
    import attrs

    return attrs.asdict(value) if attrs.has(type(value)) else str(value)


def _dump(config: TrainingCfg) -> dict[str, object]:
    return config.model_dump(mode="json", fallback=_jsonable)


def _summary(config: TrainingCfg, raw: dict[str, object]) -> dict[str, object]:
    dumped = _dump(config)
    agent = cast("dict[str, Any]", dumped["agent_config"])
    arena = cast("dict[str, Any]", dumped["arena_cfg"])
    raw_agent = raw.get("agent_config")
    algorithm = agent["framework"].get("algorithm")
    return {
        "agent_name": raw_agent.get("name") if isinstance(raw_agent, dict) else None,
        "framework": agent["framework"]["name"],
        "algorithm": algorithm["type"] if isinstance(algorithm, dict) else None,
        "n_envs": (arena.get("general") or {}).get("n_envs"),
        "robot": (arena.get("robot") or {}).get("robot_model"),
        "resume": dumped["resume"],
    }


def _validated(path: Path) -> tuple[TrainingCfg, dict[str, object]]:
    from pydantic import ValidationError
    from yaml import YAMLError

    try:
        return _load(path)
    except (ValidationError, YAMLError, OSError, ValueError, TypeError) as exc:
        first = _errors(exc)[0]
        raise ToolError(f"{path} does not validate: {first['loc']}: {first['msg']}", hint="training_validate lists every error") from exc


def _checked(path: Path) -> dict[str, object]:
    from pydantic import ValidationError
    from yaml import YAMLError

    try:
        config, raw = _load(path)
    except (ValidationError, YAMLError, OSError, ValueError, TypeError) as exc:
        return {"valid": False, "errors": _errors(exc)}
    return {"valid": True, **_summary(config, raw)}


def _agents_root(agents_dir: str | None) -> Path:
    from arena_training.arena_rosnav_rl.cfg.train import resolve_agents_dir
    from arena_training.arena_rosnav_rl.utils.paths import PathFactory

    return resolve_agents_dir(Path(agents_dir) if agents_dir else None) or PathFactory.DEFAULT_AGENTS_DIR


def _stamp(mtime: float) -> str:
    return datetime.datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds")


def _file(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "modified": _stamp(stat.st_mtime)}


def _checkpoints(agent_dir: Path) -> list[dict[str, object]]:
    files = [path for path in agent_dir.iterdir() if path.is_file() and path.suffix in CHECKPOINT_SUFFIXES]
    return [_file(path) for path in sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)]


def _last_modified(agent_dir: Path) -> float:
    return max((path.stat().st_mtime for path in agent_dir.rglob("*") if path.is_file()), default=agent_dir.stat().st_mtime)


def _agent(agent_dir: Path) -> dict[str, object]:
    return {
        "name": agent_dir.name,
        "path": str(agent_dir),
        "modified": _stamp(_last_modified(agent_dir)),
        "checkpoints": _checkpoints(agent_dir),
        "training_config": (agent_dir / "training_config.yaml").is_file(),
        "tensorboard": any(agent_dir.rglob(EVENTS_GLOB)),
    }


def _agent_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")), key=_last_modified, reverse=True)


def _scalars(agent_dir: Path) -> dict[str, object]:
    runs = sorted({path.parent for path in agent_dir.rglob(EVENTS_GLOB)})
    if not runs:
        return {"present": False}
    if importlib.util.find_spec("tensorboard") is None:
        return {"present": True, "read": False, "reason": "tensorboard is not installed in this environment", "runs": [str(run.relative_to(agent_dir)) for run in runs]}
    from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator

    out: dict[str, object] = {}
    for run in runs:
        accumulator = EventAccumulator(str(run), size_guidance={SCALARS: 2000})
        accumulator.Reload()
        tags: dict[str, object] = {}
        for tag in accumulator.Tags()[SCALARS]:
            events = accumulator.Scalars(tag)
            last = events[-1]
            tags[tag] = {"step": last.step, "value": last.value, "wall_time": _stamp(last.wall_time), "points": len(events)}
        out[str(run.relative_to(agent_dir))] = tags
    return {"present": True, "read": True, "runs": out}


def _log(agent: str, lines: int) -> dict[str, object] | None:
    for job in JOBS.list(kind="training", limit=200):
        text = JOBS.tail(job.id, 5000)
        if agent in text:
            return {"job": job.id, "state": job.state, "tail": "\n".join(text.splitlines()[-lines:])}
    return None


def _launch_args(extra_args: dict[str, str | int | float | bool] | None) -> list[str]:
    out: list[str] = []
    for key, value in (extra_args or {}).items():
        if not key or ":=" in key or any(char.isspace() for char in key):
            raise ToolError(f"launch arg name {key!r} is not a plain key", hint="pass extra_args as {'sim': 'gazebo', 'robot.mobile': 'rosnav_rl'}")
        if key == "train_config":
            raise ToolError("train_config comes from the config argument", hint="drop it from extra_args")
        text = str(value).lower() if isinstance(value, bool) else str(value)
        out.append(f"{key}:={text}")
    return out


@domain.tool(read_only=True)
def training_configs() -> list[dict[str, object]]:
    """Shipped training configs with agent name (null means generated at start), framework, algorithm, n_envs, robot, and validation verdict."""
    return [{"name": path.name, "path": str(path), **_checked(path)} for path in (_configs_dir() / name for name in _config_names())]


@domain.tool(read_only=True)
def training_config(
    name_or_path: Annotated[str, Field(description="config file name in the share configs dir (training_configs), or an absolute path")],
    schema: Annotated[bool, Field(description="include the TrainingCfg JSON schema")] = True,
) -> dict[str, object]:
    """One training config parsed through load_training_config, as TrainingCfg JSON, with the TrainingCfg schema."""
    from arena_training.arena_rosnav_rl.cfg import TrainingCfg

    path = _resolve_config(name_or_path)
    config, raw = _validated(path)
    out: dict[str, object] = {"path": str(path), "summary": _summary(config, raw), "config": _dump(config)}
    if schema:
        out["schema"] = TrainingCfg.model_json_schema()
    return out


@domain.tool(read_only=True)
def training_validate(
    yaml_text: Annotated[str | None, Field(description="training config YAML to check")] = None,
    path: Annotated[str | None, Field(description="config name or absolute path to check instead of yaml_text")] = None,
    base_dir: Annotated[str | None, Field(description="directory a relative observations_config in yaml_text resolves against, default the share configs dir")] = None,
) -> dict[str, object]:
    """Load a training config through load_training_config and return valid with a summary, or every validation error."""
    if path is not None and yaml_text is None:
        resolved = _resolve_config(path)
        return {"source": str(resolved), **_checked(resolved)}
    if yaml_text is None or path is not None:
        raise ToolError("pass exactly one of yaml_text or path")
    import yaml
    from yaml import YAMLError

    with tempfile.TemporaryDirectory(prefix="training_validate_") as tmp:
        target = Path(tmp) / "config.yaml"
        target.write_text(yaml_text)
        try:
            raw = _raw(target)
        except (YAMLError, ValueError) as exc:
            return {"source": "yaml_text", "valid": False, "errors": _errors(exc)}
        agent = raw.get("agent_config")
        if isinstance(agent, dict):
            observations = agent.get("observations_config")
            if isinstance(observations, str) and not Path(observations).is_absolute():
                agent["observations_config"] = str(Path(base_dir).expanduser() / observations if base_dir else _configs_dir() / observations)
                target.write_text(yaml.safe_dump(raw, sort_keys=False))
        return {"source": "yaml_text", **_checked(target)}


@domain.tool(destructive=True)
async def training_start(
    config: Annotated[str, Field(description="config name from training_configs or an absolute path")],
    extra_args: Annotated[dict[str, str | int | float | bool] | None, Field(description="launch args passed as key:=value, e.g. {'sim': 'gazebo', 'world': 'map_empty'}")] = None,
) -> dict[str, object]:
    """Start `arena train train_config:=<config>` as a job, refused while another Arena runtime is up; stop it with jobs_cancel."""
    from arena_runtime_msgs.srv import RegisterEnv

    path = _resolve_config(config)
    args = _launch_args(extra_args)
    loaded, _ = await asyncio.to_thread(_validated, path)
    agents_root = loaded.resolved_agents_dir or _agents_root(None)
    if await ROS.ready(RegisterEnv, REGISTER_ENV, timeout=2.0):
        raise ToolError(
            f"an Arena runtime is already up ({REGISTER_ENV} answers), and training brings up its own",
            hint="stop that stack first: jobs_cancel the job that launched it, or Ctrl-C the terminal running arena launch, then start training again",
        )
    started_config = str(path)

    def result(job: Job) -> dict[str, object]:
        fresh = [agent.name for agent in _agent_dirs(agents_root) if _last_modified(agent) >= job.started]
        return {"config": started_config, "agents_dir": str(agents_root), "agents": fresh}

    job = await cli.job("training", f"train {path.name}", "train", f"train_config:={path}", *args, result=result)
    return job.as_dict()


@domain.tool(read_only=True)
def training_agents(
    agents_dir: Annotated[str | None, Field(description="agents directory, default ROSNAV_AGENTS_DIR or the arena_training default")] = None,
) -> dict[str, object]:
    """Trained agents, newest first, with their checkpoints, last modification and whether tensorboard logs exist."""
    root = _agents_root(agents_dir)
    return {"agents_dir": str(root), "agents": [_agent(path) for path in _agent_dirs(root)]}


@domain.tool(read_only=True)
def training_status(
    agent: Annotated[str | None, Field(description="agent name from training_agents, default the most recently modified agent")] = None,
    agents_dir: Annotated[str | None, Field(description="agents directory, default ROSNAV_AGENTS_DIR or the arena_training default")] = None,
    log_lines: int = 40,
) -> dict[str, object]:
    """One agent's checkpoints, latest tensorboard scalars and the log tail of the training job that produced it."""
    root = _agents_root(agents_dir)
    agents = _agent_dirs(root)
    if not agents:
        raise ToolError(f"no agents in {root}", hint="training_start trains one, or pass agents_dir")
    if agent is None:
        agent_dir = agents[0]
    else:
        agent_dir = root / agent
        if agent_dir not in agents:
            raise ToolError(f"no agent {agent!r} in {root}, available: {', '.join(path.name for path in agents)}", hint="training_agents lists them")
    return {**_agent(agent_dir), "tensorboard": _scalars(agent_dir), "log": _log(agent_dir.name, log_lines)}


@domain.resource("arena://training/schema", mime_type="application/json", description="The TrainingCfg JSON schema every training config validates against")
def training_schema() -> dict[str, object]:
    from arena_training.arena_rosnav_rl.cfg import TrainingCfg

    return TrainingCfg.model_json_schema()
