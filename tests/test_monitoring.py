"""Regression test for setup_wandb() login-guard (P5.6, audit 2026-07-04).

wandb.login() used to be called unconditionally, which blocks on an
interactive prompt when no API key is configured anywhere (env var or
netrc) — fatal for headless training runs. setup_wandb() now probes with
wandb.Api() (raises UsageError, no prompt, when no key is configured) and
skips wandb.login() in that case, logging a warning instead.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from arena_training.arena_rosnav_rl.utils.monitoring import setup_wandb


def _make_config():
    wandb_cfg = SimpleNamespace(
        run_name="run", group="group", project_name="proj", tags=["t"]
    )
    monitoring = SimpleNamespace(wandb=wandb_cfg)
    arena_cfg = SimpleNamespace(monitoring=monitoring)
    config = MagicMock()
    config.arena_cfg = arena_cfg
    config.model_dump.return_value = {}
    return config


def test_login_skipped_when_no_api_key_configured():
    config = _make_config()
    with patch(
        "arena_training.arena_rosnav_rl.utils.monitoring.wandb"
    ) as mock_wandb:
        mock_wandb.errors.UsageError = Exception
        mock_wandb.Api.side_effect = mock_wandb.errors.UsageError("no key")

        setup_wandb(config=config)

        mock_wandb.login.assert_not_called()
        mock_wandb.init.assert_called_once()


def test_login_called_when_api_key_configured():
    config = _make_config()
    with patch(
        "arena_training.arena_rosnav_rl.utils.monitoring.wandb"
    ) as mock_wandb:
        mock_wandb.errors.UsageError = Exception
        mock_wandb.Api.return_value = MagicMock()

        setup_wandb(config=config)

        mock_wandb.login.assert_called_once()
        mock_wandb.init.assert_called_once()
