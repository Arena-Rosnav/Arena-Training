from __future__ import annotations

from arena_training.arena_rosnav_rl.cfg.arena_cfg.task import StagedCfg


def test_default_parameter_node_template_targets_env_zero_registry_node():
    staged = StagedCfg()
    assert staged.parameter_node_template.format(i=0) == "/arena/env_0/arena_env_node"
