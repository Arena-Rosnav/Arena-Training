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

import yaml

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
        config_dict["agent_cfg"]["framework"]["algorithm"]["parameters"][
            "total_timesteps"
        ] = timesteps
        return
    except (KeyError, TypeError):
        pass

    # DreamerV3 path
    try:
        config_dict["agent_cfg"]["framework"]["training"]["steps"] = timesteps
        return
    except (KeyError, TypeError):
        pass

    logger.warning(
        "Could not override trial_timesteps \u2014 neither SB3 nor DreamerV3 "
        "config path matched."
    )


def _make_sb3_trainer(training_cfg, pruner):
    """Return a StableBaselines3Trainer that injects *pruner* as a callback.

    The pruner is added alongside the existing ``eval_cb`` so that SB3's
    callback system drives both evaluation bookkeeping and Optuna reporting
    from the same training loop.
    """
    from arena_training.arena_rosnav_rl.trainer import StableBaselines3Trainer

    class _TuningSB3Trainer(StableBaselines3Trainer):
        def _train_impl(self, *args, **kwargs) -> None:
            from stable_baselines3.common.callbacks import CallbackList

            cbs = [self.eval_cb]
            if pruner is not None:
                cbs.append(pruner)
            combined = CallbackList(cbs) if len(cbs) > 1 else cbs[0]

            self.agent.train(
                total_timesteps=(
                    self.config.agent_cfg.framework.algorithm.parameters.total_timesteps
                ),
                callback=combined,
                progress_bar=(
                    self.config.agent_cfg.framework.algorithm.parameters.show_progress_bar
                ),
            )

    return _TuningSB3Trainer(training_cfg)


def _make_dreamerv3_trainer(training_cfg, pruner):
    """Return a DreamerV3Trainer that chains *pruner.after_eval_hook* with the
    curriculum hook so both receive each evaluation result.
    """
    from arena_training.arena_rosnav_rl.trainer.dreamerv3_trainer import DreamerV3Trainer

    class _TuningDreamerV3Trainer(DreamerV3Trainer):
        def _train_impl(self, *args, **kwargs) -> None:
            curriculum_hook = (
                self._curriculum.after_eval_hook if self._curriculum else None
            )
            pruner_hook = pruner.after_eval_hook if pruner is not None else None

            def _combined(eval_return: float) -> None:
                if curriculum_hook is not None:
                    curriculum_hook(eval_return)
                if pruner_hook is not None:
                    pruner_hook(eval_return)

            after_eval = _combined if (curriculum_hook or pruner_hook) else None

            fw_cfg = self.config.agent_cfg.framework
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

    return _TuningDreamerV3Trainer(training_cfg)

def make_objective(tuning_cfg, base_config_dict: dict, tuning_cfg_path: Path):
    """Return an Optuna objective function that runs one full trial."""
    from rosnav_rl.tuning import apply_params, suggest_params, SB3TrialPruner, DreamerV3TrialPruner

    def objective(trial):
        import optuna

        # \u2500\u2500 1. Sample hyperparameters \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        params = suggest_params(trial, tuning_cfg.search_space)
        logger.info("Trial %d  params: %s", trial.number, params)

        # \u2500\u2500 2. Patch config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        trial_config = apply_params(base_config_dict, params)
        trial_config["agent_cfg"]["name"] = (
            f"{tuning_cfg.study_name}_trial_{trial.number}"
        )
        if tuning_cfg.trial_timesteps is not None:
            _set_timesteps(trial_config, tuning_cfg.trial_timesteps)
        if tuning_cfg.agents_dir is not None:
            trial_config["agents_dir"] = str(tuning_cfg.agents_dir)

        # \u2500\u2500 3. Validate config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        from arena_training.arena_rosnav_rl.cfg import TrainingCfg

        training_cfg = TrainingCfg.model_validate(trial_config)

        # \u2500\u2500 4. Build pruner and framework-specific trainer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        from rosnav_rl import SupportedRLFrameworks

        framework = SupportedRLFrameworks(training_cfg.agent_cfg.framework.name)

        if framework == SupportedRLFrameworks.STABLE_BASELINES3:
            pruner = SB3TrialPruner(trial, metric=tuning_cfg.metric, verbose=1)
            trainer = _make_sb3_trainer(training_cfg, pruner)
        elif framework == SupportedRLFrameworks.DREAMER_V3:
            pruner = DreamerV3TrialPruner(trial, verbose=1)
            trainer = _make_dreamerv3_trainer(training_cfg, pruner)
        else:
            raise ValueError(f"Unsupported framework for tuning: {framework!r}")

        # \u2500\u2500 5. Run training \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        try:
            trainer.train()
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.error("Trial %d failed: %s", trial.number, exc, exc_info=True)
            raise optuna.TrialPruned(f"Trial {trial.number} failed: {exc}") from exc
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

    optuna_pruner = _build_optuna_pruner(tuning_cfg.pruner)
    study = optuna.create_study(
        study_name=tuning_cfg.study_name,
        direction=tuning_cfg.direction,
        storage=tuning_cfg.storage,
        pruner=optuna_pruner,
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

    objective = make_objective(tuning_cfg, base_config_dict, args.config)
    study.optimize(objective, n_trials=tuning_cfg.n_trials)

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

