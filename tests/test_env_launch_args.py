from arena_training.arena_rosnav_rl.cfg.arena_cfg.general import GeneralCfg
from arena_training.arena_rosnav_rl.cfg.arena_cfg.robot import RobotCfg
from arena_training.arena_rosnav_rl.utils.launch_args import env_launch_args


def test_every_env_requests_the_configured_robot():
    args = env_launch_args(GeneralCfg(n_envs=3), RobotCfg(robot_model="boxer"))
    assert len(args) == 3
    assert all("robot:=boxer" in env_args for env_args in args)


def test_seed_and_extra_args_are_per_env():
    general = GeneralCfg(n_envs=3, env_seed=7, env_launch_args=[["world:=a"], ["world:=b"]])
    args = env_launch_args(general, RobotCfg(robot_model="jackal"))
    assert [a for env_args in args for a in env_args if a.startswith(("run_seed:=", "world:="))] == [
        "run_seed:=7-0",
        "world:=a",
        "run_seed:=7-1",
        "world:=b",
        "run_seed:=7-2",
        "world:=a",
    ]


def test_clearance_follows_the_safety_distance():
    args = env_launch_args(GeneralCfg(n_envs=1, safety_distance=0.4), RobotCfg(robot_model="jackal"))
    assert "task.episode.spawn.robot_clearance:=0.4" in args[0]
