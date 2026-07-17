#!/usr/bin/env python3
"""Arena Hyperparameter Tuning \u2014 Optuna-based search over any TrainingCfg field.

Usage
-----
::

    python3 tune_agent.py --config tuning_config.yaml [--n-trials N]

Each Optuna trial samples hyperparameters from the search space defined in
the YAML, patches them into a copy of the base training config, runs a
(possibly shortened) training session, and reports the result back to
the study.  Unpromising trials are pruned early.

Both SB3 and DreamerV3 frameworks are supported.  Pruning is handled by
framework-specific adapters that extend the shared ``TrialPrunerBase``\u2014
the same ``CurriculumBase`` / ``StagedTrainCallback`` pattern used for
curriculum learning.

See ``rosnav_rl/tuning/README.md`` for full documentation and YAML examples.
"""

import logging
import sys
from pathlib import Path
from typing import Callable

import yaml

from arena_training.arena_rosnav_rl.utils.sim_bootstrap import (
    build_namespace_fn,
    spawn_envs,
    wait_for_simulation,
)
from arena_training.arena_rosnav_rl.utils.tuning_recovery import (
    ensure_envs_healthy,
    release_trial_resources,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_optuna_pruner(pruner_cfg):
    """Instantiate an Optuna pruner from ``PrunerCfg``."""
    import optuna

    if pruner_cfg.type == "none":
        return optuna.pruners.NopPruner()
    if pruner_cfg.type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg.n_startup_trials,
            n_warmup_steps=pruner_cfg.n_warmup_steps,
        )
    if pruner_cfg.type == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if pruner_cfg.type == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=pruner_cfg.percentile,
            n_startup_trials=pruner_cfg.n_startup_trials,
            n_warmup_steps=pruner_cfg.n_warmup_steps,
        )
    raise ValueError(f"Unknown pruner type: {pruner_cfg.type!r}")


def _build_optuna_sampler(sampler_cfg):
    """Instantiate an Optuna sampler from ``SamplerCfg``."""
    import optuna

    if sampler_cfg.type == "tpe":
        return optuna.samplers.TPESampler(
            seed=sampler_cfg.seed,
            multivariate=sampler_cfg.multivariate,
            n_startup_trials=sampler_cfg.n_startup_trials,
        )
    raise ValueError(f"Unknown sampler type: {sampler_cfg.type!r}")


def _resolve_storage(tuning_cfg) -> str | None:
    """Default TuningCfg.storage to a sqlite file under agents_dir if unset.

    An unset storage means an in-memory study — results are lost the
    moment the process exits, and load_if_exists=True can't resume a
    crashed run. Defaulting to a file next to the trial artifacts gives
    both for free, without requiring every tuning config to spell it out.
    """
    if tuning_cfg.storage is not None:
        return tuning_cfg.storage
    if tuning_cfg.agents_dir is None:
        return None
    tuning_cfg.agents_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{tuning_cfg.agents_dir / f'{tuning_cfg.study_name}.db'}"
    logger.info("No storage configured — defaulting to %s", storage)
    return storage


def _resolve_base_config(tuning_cfg_path: Path, base_config: Path) -> Path:
    """Resolve *base_config* relative to the tuning config\u2019s directory."""
    if base_config.is_absolute() and base_config.exists():
        return base_config

    candidate = (tuning_cfg_path.parent / base_config).resolve()
    if candidate.exists():
        return candidate

    # Fallback: arena_bringup share directory
    try:
        from ament_index_python.packages import get_package_share_directory

        bringup = Path(get_package_share_directory("arena_bringup"))
        share_candidate = bringup / "configs" / "training" / str(base_config)
        if share_candidate.exists():
            return share_candidate
    except Exception:
        pass

    raise FileNotFoundError(
        f"Base config {str(base_config)!r} not found. Searched:\n"
        f"  {candidate}\n"
        f"  <arena_bringup>/configs/training/{base_config}"
    )


def _set_timesteps(config_dict: dict, timesteps: int) -> None:
    """Override total training steps in the config dict (best-effort)."""
    # SB3 path
    try:
        config_dict["agent_config"]["framework"]["algorithm"]["parameters"][
            "total_timesteps"
        ] = timesteps
        return
    except (KeyError, TypeError):
        pass

    # DreamerV3 path
    try:
        config_dict["agent_config"]["framework"]["training"]["steps"] = timesteps
        return
    except (KeyError, TypeError):
        pass

    logger.warning(
        "Could not override trial_timesteps \u2014 neither SB3 nor DreamerV3 "
        "config path matched."
    )


def _pin_fixed_difficulty(config_dict: dict) -> None:
    """Pin task difficulty for the duration of a tuning trial (best-effort).

    Trials must be comparable to each other, so task difficulty must not
    drift between trials the way it would during a real training run.
    ``_TuningDreamerV3Trainer`` separately forces ``self._curriculum = None``
    so the DreamerV3Curriculum stage-advance hook is never wired regardless
    of this config; this also disables the (currently unwired-in-tuning,
    but config-visible) Social-Dreamer model curriculum for consistency.
    """
    try:
        config_dict["agent_config"]["framework"]["model"]["social"]["curriculum"][
            "enabled"
        ] = False
    except (KeyError, TypeError):
        pass


def _pin_wandb_group(config_dict: dict, study_name: str) -> None:
    """Group every trial's W&B run under the study name (best-effort)."""
    try:
        config_dict["arena_cfg"]["monitoring"]["wandb"]["group"] = study_name
    except (KeyError, TypeError):
        pass


def _social_context_enabled(config_dict: dict) -> bool:
    """Whether this trial's config has social.context.enabled=True."""
    try:
        return bool(
            config_dict["agent_config"]["framework"]["model"]["social"]["context"][
                "enabled"
            ]
        )
    except (KeyError, TypeError):
        return False


class TrialTimeoutError(RuntimeError):
    """Raised when a trial's training loop exceeds TuningCfg.trial_timeout_s.

    A real exception (not optuna.TrialPruned) so study.optimize's
    catch=(Exception,) records the trial FAILED, distinguishing "ran out of
    time" from both a normal COMPLETE and a pruner-initiated PRUNED.
    """


def _make_sb3_trainer(training_cfg, pruner, namespace_fn, trial_timeout_s=None):
    """Return a StableBaselines3Trainer that injects *pruner* as a callback.

    The pruner is added alongside the existing ``eval_cb`` so that SB3's
    callback system drives both evaluation bookkeeping and Optuna reporting
    from the same training loop.
    """
    import time

    from arena_training.arena_rosnav_rl.trainer import StableBaselines3Trainer

    class _TuningSB3Trainer(StableBaselines3Trainer):
        def _train_impl(self, *args, **kwargs) -> None:
            from stable_baselines3.common.callbacks import BaseCallback, CallbackList

            cbs = [self.eval_cb]
            if pruner is not None:
                cbs.append(pruner)
            if trial_timeout_s is not None:
                deadline = time.monotonic() + trial_timeout_s

                class _TimeoutCallback(BaseCallback):
                    def _on_step(self) -> bool:
                        if time.monotonic() >= deadline:
                            raise TrialTimeoutError(
                                f"trial exceeded trial_timeout_s={trial_timeout_s}"
                            )
                        return True

                cbs.append(_TimeoutCallback())
            combined = CallbackList(cbs) if len(cbs) > 1 else cbs[0]

            self.agent.train(
                total_timesteps=(
                    self.config.agent_config.framework.algorithm.parameters.total_timesteps
                ),
                callback=combined,
                progress_bar=(
                    self.config.agent_config.framework.algorithm.parameters.show_progress_bar
                ),
            )

    return _TuningSB3Trainer(training_cfg, namespace_fn=namespace_fn)


def _make_dreamerv3_trainer(training_cfg, pruner, namespace_fn, trial_timeout_s=None):
    """Return a DreamerV3Trainer that chains *pruner.after_eval_hook* with the
    curriculum hook so both receive each evaluation result.
    """
    import time

    from arena_training.arena_rosnav_rl.trainer.dreamerv3_trainer import DreamerV3Trainer

    class _TuningDreamerV3Trainer(DreamerV3Trainer):
        def _setup_curriculum(self) -> None:
            # ONE-SHOT task setup, then no advancement: run the parent's curriculum
            # construction so DreamerV3Curriculum.__init__ pushes the starting-stage
            # params (obstacle counts, task.driver_set, ...) and the tm_* modes to
            # every env — WITHOUT this the base config's `starting_stage` never
            # reaches the envs at all during tuning and trials run on task_generator
            # defaults. Then drop the object so the stage can never advance: tuning
            # trials must be difficulty-comparable for the whole study.
            super()._setup_curriculum()
            self._curriculum = None

        def _train_impl(self, *args, **kwargs) -> None:
            curriculum_hook = (
                self._curriculum.after_eval_hook if self._curriculum else None
            )
            pruner_hook = pruner.after_eval_hook if pruner is not None else None
            deadline = (
                time.monotonic() + trial_timeout_s if trial_timeout_s is not None else None
            )

            def _combined(eval_return: float) -> None:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TrialTimeoutError(
                        f"trial exceeded trial_timeout_s={trial_timeout_s}"
                    )
                if curriculum_hook is not None:
                    curriculum_hook(eval_return)
                if pruner_hook is not None:
                    pruner_hook(eval_return)

            after_eval = _combined if (curriculum_hook or pruner_hook or deadline) else None

            fw_cfg = self.config.agent_config.framework
            logger.info(
                "[Train] DreamerV3 \u2014 total_steps=%d  eval_every=%d  device=%s",
                fw_cfg.training.steps,
                fw_cfg.training.eval_every,
                fw_cfg.general.device,
            )
            self.agent.model.train(
                train_envs=self.environment.train_envs,
                eval_envs=self.environment.eval_envs,
                after_eval_fn=after_eval,
            )
            logger.info("[Train] Training complete.")

    return _TuningDreamerV3Trainer(training_cfg, namespace_fn=namespace_fn)


def _bootstrap_prefill_cache(
    base_config_dict: dict,
    namespace_fn: Callable[[int], str],
    prefill_dir: Path,
) -> None:
    """One-time live-prefill collection into a directory shared by every trial.

    DreamerV3's prefill_dataset() runs config.training.prefill_steps of
    random-action rollout before real training starts. Every trial paying
    that cost independently is redundant sim time; collect it once here
    instead, and point every trial's general.offline_traindir at prefill_dir
    so prefill_dataset() short-circuits to a no-op and load_episodes() reads
    this cache directly (see rosnav_rl/model/dreamerv3/helper.py).

    prefill_dir must never be written to again after this call returns — a
    trial's own post-prefill training episodes must not land here, or later
    trials' offline_traindir reads would pick up a trained policy's rollouts
    instead of the shared random-policy prefill batch.
    """
    from copy import deepcopy

    from rosnav_rl.model.dreamerv3.helper import (
        load_episodes,
        prefill_dataset,
        prepare_directories,
        set_runtime_configuration,
    )

    from arena_training.arena_rosnav_rl.cfg import TrainingCfg

    bootstrap_dict = deepcopy(base_config_dict)
    bootstrap_dict["agent_config"]["name"] = (
        f"{bootstrap_dict['agent_config']['name']}_prefill_bootstrap"
    )
    general = bootstrap_dict["agent_config"]["framework"]["general"]
    general["logdir"] = str(prefill_dir)
    general["traindir"] = None
    general["offline_traindir"] = None
    _pin_fixed_difficulty(bootstrap_dict)

    bootstrap_cfg = TrainingCfg.model_validate(bootstrap_dict)
    trainer = _make_dreamerv3_trainer(bootstrap_cfg, None, namespace_fn)
    try:
        model = trainer.agent.model
        set_runtime_configuration(model._algorithm_cfg)
        prepare_directories(model._algorithm_cfg, model._logdir)
        train_eps, _eval_eps = load_episodes(model._algorithm_cfg)
        prefill_dataset(
            model._algorithm_cfg,
            trainer.environment.train_envs,
            train_eps,
            model._logger,
            trainer.agent.action_space,
            trainer.agent.observation_space,
        )
    finally:
        trainer.close()


def make_objective(tuning_cfg, base_config_dict: dict, namespace_fn: Callable[[int], str]):
    """Return an Optuna objective function that runs one full trial."""
    from rosnav_rl import SupportedRLFrameworks
    from rosnav_rl.tuning import apply_params, suggest_params, SB3TrialPruner, DreamerV3TrialPruner

    # Maps each supported framework to (pruner_factory, trainer_factory).
    # pruner_factory(trial, trial_config) → framework-specific TrialPruner
    # trainer_factory(training_cfg, pruner, namespace_fn) → ArenaTrainer subclass
    _tuning_registry = {
        SupportedRLFrameworks.STABLE_BASELINES3: (
            lambda trial, trial_config: SB3TrialPruner(
                trial, metric=tuning_cfg.metric, verbose=1
            ),
            _make_sb3_trainer,
        ),
        SupportedRLFrameworks.DREAMER_V3: (
            lambda trial, trial_config: DreamerV3TrialPruner(
                trial,
                metric=tuning_cfg.metric,
                health_prune=tuning_cfg.health_prune,
                social_context_enabled=_social_context_enabled(trial_config),
                verbose=1,
            ),
            _make_dreamerv3_trainer,
        ),
    }

    def objective(trial):
        # \u2500\u2500 1. Sample hyperparameters \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        params = suggest_params(trial, tuning_cfg.search_space)
        logger.info("Trial %d  params: %s", trial.number, params)

        # \u2500\u2500 2. Patch config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        trial_config = apply_params(base_config_dict, params)
        trial_config["agent_config"]["name"] = (
            f"{tuning_cfg.study_name}_trial_{trial.number}"
        )
        if tuning_cfg.trial_timesteps is not None:
            _set_timesteps(trial_config, tuning_cfg.trial_timesteps)
        _pin_fixed_difficulty(trial_config)
        _pin_wandb_group(trial_config, tuning_cfg.study_name)
        if tuning_cfg.agents_dir is not None:
            trial_config["agents_dir"] = str(tuning_cfg.agents_dir)

        # \u2500\u2500 3. Validate config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        from arena_training.arena_rosnav_rl.cfg import TrainingCfg

        training_cfg = TrainingCfg.model_validate(trial_config)

        # \u2500\u2500 4. Build pruner and framework-specific trainer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        framework = SupportedRLFrameworks(training_cfg.agent_config.framework.name)
        if framework not in _tuning_registry:
            raise ValueError(
                f"Unsupported framework for tuning: {framework!r}. "
                f"Supported: {[f.value for f in _tuning_registry]}"
            )
        make_pruner, make_trainer = _tuning_registry[framework]
        pruner = make_pruner(trial, trial_config)
        trainer = make_trainer(
            training_cfg, pruner, namespace_fn, trial_timeout_s=tuning_cfg.trial_timeout_s
        )

        # \u2500\u2500 5. Run training \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        # Real exceptions propagate to study.optimize's catch=(Exception,),
        # which records the trial as FAILED rather than PRUNED \u2014 pruning is
        # reserved for optuna.TrialPruned raised deliberately by the pruner.
        try:
            trainer.train()
        finally:
            try:
                trainer.close()
            except Exception:
                pass

        metric_value = pruner.best_metric
        if metric_value is None:
            logger.warning("Trial %d: no metric recorded, returning 0.0", trial.number)
            metric_value = 0.0

        logger.info(
            "Trial %d complete: %s = %.4f",
            trial.number, tuning_cfg.metric, metric_value,
        )
        return metric_value

    return objective


def main() -> int:
    import argparse
    import optuna

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the tuning configuration YAML.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override n_trials from the config.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        logger.error("Tuning config not found: %s", args.config)
        return 1

    from rosnav_rl.tuning import TuningCfg

    tuning_cfg = TuningCfg.model_validate(_load_yaml(args.config))
    if args.n_trials is not None:
        tuning_cfg.n_trials = args.n_trials

    base_config_path = _resolve_base_config(args.config, tuning_cfg.base_config)
    logger.info("Base training config: %s", base_config_path)
    base_config_dict = _load_yaml(base_config_path)

    wait_for_simulation(timeout=120.0)

    n_envs: int = base_config_dict["arena_cfg"]["general"]["n_envs"]
    per_env_launch_args = [["train_mode:=true", "auto_reset:=false"] for _ in range(n_envs)]
    logger.info("Spawning %d env(s), shared across all trials in this study", n_envs)
    env_map = spawn_envs(n_envs, per_env_launch_args)
    namespace_fn = build_namespace_fn(env_map)

    from rosnav_rl import SupportedRLFrameworks

    framework = SupportedRLFrameworks(base_config_dict["agent_config"]["framework"]["name"])
    if framework == SupportedRLFrameworks.DREAMER_V3:
        if tuning_cfg.agents_dir is not None:
            prefill_dir = tuning_cfg.agents_dir / f"{tuning_cfg.study_name}_prefill"
            prefill_traindir = prefill_dir / "train_eps"
            if prefill_traindir.exists() and any(prefill_traindir.iterdir()):
                logger.info("Reusing existing shared prefill cache at %s", prefill_traindir)
            else:
                logger.info("Bootstrapping shared prefill cache at %s", prefill_traindir)
                _bootstrap_prefill_cache(base_config_dict, namespace_fn, prefill_dir)
            base_config_dict["agent_config"]["framework"]["general"]["offline_traindir"] = (
                str(prefill_traindir)
            )
        else:
            logger.info(
                "tuning_cfg.agents_dir is not set — skipping shared prefill cache; "
                "each trial will prefill independently"
            )

    optuna_pruner = _build_optuna_pruner(tuning_cfg.pruner)
    optuna_sampler = _build_optuna_sampler(tuning_cfg.sampler)
    study = optuna.create_study(
        study_name=tuning_cfg.study_name,
        direction=tuning_cfg.direction,
        storage=_resolve_storage(tuning_cfg),
        pruner=optuna_pruner,
        sampler=optuna_sampler,
        load_if_exists=True,
    )

    logger.info("=" * 70)
    logger.info("  Arena Hyperparameter Tuning")
    logger.info("=" * 70)
    logger.info("  Study:     %s", tuning_cfg.study_name)
    logger.info("  Trials:    %d", tuning_cfg.n_trials)
    logger.info("  Direction: %s", tuning_cfg.direction)
    logger.info("  Metric:    %s", tuning_cfg.metric)
    logger.info("  Params:    %s", list(tuning_cfg.search_space.keys()))
    logger.info("=" * 70)

    objective = make_objective(tuning_cfg, base_config_dict, namespace_fn)

    def _after_trial(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        # Runs after every trial (COMPLETE, PRUNED, or FAILED) and before the
        # next one starts. Mutates env_map in place; namespace_fn shares the
        # same dict object, so a respawned env is picked up automatically.
        release_trial_resources()
        ensure_envs_healthy(env_map, per_env_launch_args)

    # catch=(Exception,): a trial that raises a real (non-TrialPruned) exception
    # is recorded FAILED and the study continues with the next trial, instead of
    # the whole process dying on one bad trial.
    study.optimize(
        objective,
        n_trials=tuning_cfg.n_trials,
        catch=(Exception,),
        callbacks=[_after_trial],
    )

    logger.info("\n" + "=" * 70)
    logger.info("  Tuning Complete")
    logger.info("=" * 70)
    logger.info("  Best trial: %d", study.best_trial.number)
    logger.info("  Best value: %.6f", study.best_trial.value)
    logger.info("  Best params:")
    for key, val in study.best_trial.params.items():
        logger.info("    %s: %s", key, val)
    logger.info("=" * 70)

    output_path = args.config.parent / f"{tuning_cfg.study_name}_best_params.yaml"
    with open(output_path, "w") as f:
        yaml.dump(
            {
                "study_name": tuning_cfg.study_name,
                "best_trial": study.best_trial.number,
                "best_value": study.best_trial.value,
                "best_params": study.best_trial.params,
            },
            f,
            default_flow_style=False,
        )
    logger.info("Best parameters saved to %s", output_path)
    return 0


if __name__ == "__main__":
    try:
        import rclpy

        rclpy.init()
        exit_code = main()
    except KeyboardInterrupt:
        logger.info("\nTuning interrupted by user.")
        exit_code = 130
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        exit_code = 1
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        sys.exit(exit_code)

