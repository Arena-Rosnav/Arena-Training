from __future__ import annotations

import pytest

from arena_training.arena_rosnav_rl.cfg.arena_cfg.task import TaskCfg


def test_default_modes_are_the_staged_curriculum_selection():
    cfg = TaskCfg()
    assert cfg.mode_names() == {"robots": ["random"], "obstacles": ["random"], "modules": ["staged"]}
    assert cfg.runs_module("staged")


def test_modes_mapping_lists_comma_separated_names_per_axis():
    cfg = TaskCfg(modes={"robots": "explore", "modules": "staged,sounds"})
    assert cfg.mode_names() == {"robots": ["explore"], "modules": ["staged", "sounds"]}
    assert cfg.runs_module("sounds")
    assert not TaskCfg(modes={"robots": "explore"}).runs_module("staged")


def test_the_env_runs_every_mode_but_the_staged_curriculum_the_trainer_runs_itself():
    cfg = TaskCfg(modes={"robots": "explore", "obstacles": "random", "modules": "staged,sounds"})
    assert cfg.env_mode_names() == {"robots": ["explore"], "obstacles": ["random"], "modules": ["sounds"]}
    assert cfg.runs_module("staged")
    assert TaskCfg().env_mode_names()["modules"] == []


def test_legacy_tm_keys_map_onto_modes_over_the_old_defaults():
    pytest.importorskip("arena_runtime")
    cfg = TaskCfg.model_validate({"tm_robots": "explore", "tm_modules": "staged,sounds"})
    assert cfg.modes == {"robots": "explore", "obstacles": "random", "modules": "staged,sounds"}
