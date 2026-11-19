from arena_training.arena_rosnav_rl.cfg.arena_cfg.general import GeneralCfg
from arena_training.arena_rosnav_rl.cfg.arena_cfg.robot import RobotCfg


def env_launch_args(general: GeneralCfg, robot: RobotCfg) -> list[list[str]]:
    """Per-env `arena launch` arguments of a training run, one list per env."""
    extra_args = general.env_launch_args
    seed = general.env_seed
    return [
        [
            "robot.train:=true",
            "task.episode.auto_reset:=false",
            robot.launch_arg,
            f"task.episode.spawn.robot_clearance:={general.safety_distance}",
            *([f"run_seed:={seed}-{i}"] if seed is not None else []),
            *(extra_args[i % len(extra_args)] if extra_args else []),
        ]
        for i in range(general.n_envs)
    ]
