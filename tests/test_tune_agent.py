"""Unit tests for scripts/tune_agent.py (P7.5).

tune_agent.py is a script, not an importable package module, and its
top-level imports pull in rclpy (via sim_bootstrap/tuning_recovery) — so
this whole file is skipped when rclpy isn't importable, same as any other
ROS-dependent test in this suite. Loaded via importlib.spec_from_file_location
directly from its script path (not run as __main__, so no argparse/ROS-init
side effects fire).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("rclpy")

import importlib.util

import yaml

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tune_agent.py"
_spec = importlib.util.spec_from_file_location("tune_agent", _SCRIPT)
tune_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tune_agent)

_SMOKE_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "social_csrssm_smoke.yaml"
)


def _smoke_config_dict() -> dict:
    with open(_SMOKE_CONFIG) as f:
        return yaml.safe_load(f)


# ── _pin_wandb_group ─────────────────────────────────────────────────────────


def test_pin_wandb_group_sets_group():
    config = {"arena_cfg": {"monitoring": {"wandb": {"group": "old"}}}}
    tune_agent._pin_wandb_group(config, "my_study")
    assert config["arena_cfg"]["monitoring"]["wandb"]["group"] == "my_study"


def test_pin_wandb_group_noop_on_missing_keys():
    config = {"arena_cfg": {}}
    tune_agent._pin_wandb_group(config, "my_study")  # must not raise
    assert config == {"arena_cfg": {}}


# ── _social_context_enabled ──────────────────────────────────────────────────


def test_social_context_enabled_true():
    config = {
        "agent_config": {
            "framework": {"model": {"social": {"context": {"enabled": True}}}}
        }
    }
    assert tune_agent._social_context_enabled(config) is True


def test_social_context_enabled_false():
    config = {
        "agent_config": {
            "framework": {"model": {"social": {"context": {"enabled": False}}}}
        }
    }
    assert tune_agent._social_context_enabled(config) is False


def test_social_context_enabled_missing_keys_defaults_false():
    assert tune_agent._social_context_enabled({}) is False
    assert tune_agent._social_context_enabled({"agent_config": {}}) is False


# ── _resolve_storage ──────────────────────────────────────────────────────────


def test_resolve_storage_none_when_unset():
    from rosnav_rl.tuning.cfg import TuningCfg

    tuning_cfg = TuningCfg(base_config=Path("x.yaml"), search_space={})
    assert tune_agent._resolve_storage(tuning_cfg) is None


def test_resolve_storage_explicit_passthrough():
    from rosnav_rl.tuning.cfg import TuningCfg

    tuning_cfg = TuningCfg(
        base_config=Path("x.yaml"), search_space={}, storage="sqlite:///explicit.db"
    )
    assert tune_agent._resolve_storage(tuning_cfg) == "sqlite:///explicit.db"


def test_resolve_storage_defaults_to_sqlite_under_agents_dir(tmp_path):
    from rosnav_rl.tuning.cfg import TuningCfg

    tuning_cfg = TuningCfg(
        base_config=Path("x.yaml"),
        search_space={},
        study_name="my_study",
        agents_dir=tmp_path / "agents",
    )
    storage = tune_agent._resolve_storage(tuning_cfg)
    assert storage == f"sqlite:///{tmp_path / 'agents' / 'my_study.db'}"
    assert (tmp_path / "agents").is_dir()


# ── _build_optuna_sampler ─────────────────────────────────────────────────────


def test_build_optuna_sampler_tpe_wires_config():
    import optuna

    from rosnav_rl.tuning.cfg import SamplerCfg

    sampler_cfg = SamplerCfg(seed=42, multivariate=False, n_startup_trials=3)
    sampler = tune_agent._build_optuna_sampler(sampler_cfg)

    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert sampler._n_startup_trials == 3
    assert sampler._multivariate is False


def test_build_optuna_sampler_unknown_type_raises():
    fake_cfg = MagicMock(type="not_a_real_sampler")
    with pytest.raises(ValueError, match="Unknown sampler type"):
        tune_agent._build_optuna_sampler(fake_cfg)


# ── apply_params against a real TrainingCfg ──────────────────────────────────


def test_apply_params_overlays_real_training_cfg_dump():
    from arena_training.arena_rosnav_rl.cfg import TrainingCfg
    from rosnav_rl.tuning.sampler import apply_params

    base_cfg = TrainingCfg.model_validate(_smoke_config_dict())
    base_dump = base_cfg.model_dump()

    params = {"agent_config.framework.training.model_lr": 9.99e-4}
    patched = apply_params(base_dump, params)

    assert patched["agent_config"]["framework"]["training"]["model_lr"] == 9.99e-4
    # Original untouched.
    assert base_dump["agent_config"]["framework"]["training"]["model_lr"] != 9.99e-4
    # Result is still a valid TrainingCfg (round-trips through validation).
    reloaded = TrainingCfg.model_validate(patched)
    assert reloaded.agent_config.framework.training.model_lr == 9.99e-4


# ── Driver-level: make_objective() with a mocked trainer factory ────────────


class TestMakeObjectiveDriver:
    """Exercises make_objective() end-to-end with the trainer factory mocked
    out, so no real ROS/Gazebo/DreamerV3 training happens. Verifies the
    plumbing: namespace_fn passthrough, per-trial pruner construction, and
    that a real exception mid-train surfaces as-is (FAIL), not swallowed
    into optuna.TrialPruned.
    """

    def _make_trial(self, number: int):
        trial = MagicMock()
        trial.number = number
        return trial

    def test_two_trials_pass_namespace_fn_and_build_dreamerv3_pruner(
        self, monkeypatch
    ):
        from rosnav_rl.tuning import DreamerV3TrialPruner
        from rosnav_rl.tuning.cfg import TuningCfg

        calls = []

        def fake_make_dreamerv3_trainer(training_cfg, pruner, namespace_fn, **kw):
            calls.append(
                {
                    "training_cfg": training_cfg,
                    "pruner": pruner,
                    "namespace_fn": namespace_fn,
                }
            )
            fake_trainer = MagicMock()
            fake_trainer.train.return_value = None
            return fake_trainer

        monkeypatch.setattr(
            tune_agent, "_make_dreamerv3_trainer", fake_make_dreamerv3_trainer
        )

        tuning_cfg = TuningCfg(
            base_config=_SMOKE_CONFIG, search_space={}, study_name="unit_study"
        )
        base_config_dict = _smoke_config_dict()

        def namespace_fn(idx):
            return f"env_{idx}"

        objective = tune_agent.make_objective(
            tuning_cfg, base_config_dict, namespace_fn
        )

        objective(self._make_trial(0))
        objective(self._make_trial(1))

        assert len(calls) == 2
        for i, call in enumerate(calls):
            assert call["namespace_fn"] is namespace_fn
            assert isinstance(call["pruner"], DreamerV3TrialPruner)
            assert call["pruner"]._metric == tuning_cfg.metric
            assert (
                call["training_cfg"].agent_config.name
                == f"unit_study_trial_{i}"
            )

    def test_real_exception_propagates_not_pruned(self, monkeypatch):
        import optuna

        from rosnav_rl.tuning.cfg import TuningCfg

        closed = []

        def fake_make_dreamerv3_trainer(training_cfg, pruner, namespace_fn, **kw):
            fake_trainer = MagicMock()
            fake_trainer.train.side_effect = RuntimeError("simulated crash mid-train")
            fake_trainer.close.side_effect = lambda: closed.append(True)
            return fake_trainer

        monkeypatch.setattr(
            tune_agent, "_make_dreamerv3_trainer", fake_make_dreamerv3_trainer
        )

        tuning_cfg = TuningCfg(
            base_config=_SMOKE_CONFIG, search_space={}, study_name="unit_study"
        )
        base_config_dict = _smoke_config_dict()
        objective = tune_agent.make_objective(
            tuning_cfg, base_config_dict, lambda idx: f"env_{idx}"
        )

        with pytest.raises(RuntimeError, match="simulated crash mid-train"):
            objective(self._make_trial(0))

        # optuna.TrialPruned must NOT be what propagates here.
        assert not issubclass(RuntimeError, optuna.TrialPruned)
        # trainer.close() still runs on the failure path (finally block).
        assert closed == [True]
