"""CPU unit tests for DreamerV3 fixes: Fix 1 (rec_depth), Fix 5 (LR warmup), Fix 7 (clamp).

No GPU, no ROS, no sb3_contrib required.
Uses importlib to load dreamerv3.tools and dreamerv3.networks directly, with
minimal stubs for the dreamerv3's relative + absolute import dependencies.
"""

import sys
import types
import importlib.util
import importlib
import math

import torch
import pytest

# ---------------------------------------------------------------------------
# Stub out tensorboard (required by tools.py)
# ---------------------------------------------------------------------------
_fake_sw_cls = type("SummaryWriter", (), {"__init__": lambda self, *a, **kw: None})

for _m in ["tensorboard", "tensorboard.summary", "tensorboard.summary.writer"]:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
sys.modules["tensorboard.summary.writer"].SummaryWriter = _fake_sw_cls

_tsb = types.ModuleType("torch.utils.tensorboard")
_tsb.SummaryWriter = _fake_sw_cls
sys.modules["torch.utils.tensorboard"] = _tsb

# ---------------------------------------------------------------------------
# Load dreamerv3.tools and dreamerv3.networks via importlib (avoids the full
# rosnav_rl package __init__ which needs sb3_contrib / rclpy).
# ---------------------------------------------------------------------------
_BASE = (
    "/home/tuananhroman/arena_ws/src/Arena/arena_training"
    "/deps/rosnav_rl/rosnav_rl/rosnav_rl/model/dreamerv3"
)


def _load(module_name: str, filename: str):
    """Load a single .py file as a module without triggering package __init__."""
    path = f"{_BASE}/{filename}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# tools has no relative imports — load first.
_tools = _load("dreamerv3_tools_standalone", "tools.py")

# networks.py has `from ..dreamerv3 import tools` → provide as stub.
_fake_parent_pkg = types.ModuleType("dreamerv3_parent")
_fake_dreamerv3_pkg = types.ModuleType("dreamerv3_parent.dreamerv3")
_fake_dreamerv3_pkg.tools = _tools
sys.modules["dreamerv3_parent"] = _fake_parent_pkg
sys.modules["dreamerv3_parent.dreamerv3"] = _fake_dreamerv3_pkg

_net_spec = importlib.util.spec_from_file_location(
    "dreamerv3_parent.dreamerv3.networks",
    f"{_BASE}/networks.py",
    submodule_search_locations=[],
)
_net_mod = importlib.util.module_from_spec(_net_spec)
_net_mod.__package__ = "dreamerv3_parent.dreamerv3"
sys.modules["dreamerv3_parent.dreamerv3.networks"] = _net_mod
_net_spec.loader.exec_module(_net_mod)

RSSM = _net_mod.RSSM
Optimizer = _tools.Optimizer


# ===========================================================================
# Fix 1 — rec_depth correctness
# ===========================================================================

def _make_tiny_rssm(rec_depth: int) -> RSSM:
    """Build a tiny RSSM (GRU, rec_depth=<n>) that fits on CPU."""
    return RSSM(
        stoch=4,
        deter=8,
        hidden=16,
        rec_depth=rec_depth,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        cell_type="gru",
        unimix_ratio=0.0,
        initial="zeros",
        num_actions=2,
        embed=8,
        device="cpu",
    )


class TestFix1RecDepth:
    """Verify that rec_depth>1 chains GRU iterations rather than re-reading prev_state.

    Tests call img_step via torch.compiler.disable() to bypass compile (Fix 3+4A)
    which cannot trace tiny toy shapes on CPU.  The correctness fix (Fix 1) lives in
    the Python loop body — eager execution proves it.
    """

    def test_rec_depth_1_runs(self):
        """rec_depth=1 must produce a valid prior with finite tensors."""
        rssm = _make_tiny_rssm(rec_depth=1)
        rssm.eval()
        prev = rssm.initial(batch_size=2)
        action = torch.zeros(2, 2)
        with torch.no_grad(), torch.compiler.disable():
            prior = rssm.img_step(prev, action)
        assert torch.isfinite(prior["deter"]).all(), "rec_depth=1 deter has NaN/Inf"
        assert torch.isfinite(prior["mean"]).all(), "rec_depth=1 mean has NaN/Inf"

    def test_rec_depth_2_runs(self):
        """rec_depth=2 must produce valid prior (was broken before Fix 1)."""
        rssm = _make_tiny_rssm(rec_depth=2)
        rssm.eval()
        prev = rssm.initial(batch_size=2)
        action = torch.zeros(2, 2)
        with torch.no_grad(), torch.compiler.disable():
            prior = rssm.img_step(prev, action)
        assert torch.isfinite(prior["deter"]).all(), "rec_depth=2 deter has NaN/Inf"
        assert torch.isfinite(prior["mean"]).all(), "rec_depth=2 mean has NaN/Inf"

    def test_rec_depth_2_differs_from_1(self):
        """rec_depth=2 must produce a different deter than rec_depth=1 (two GRU iters vs one)."""
        rssm1 = _make_tiny_rssm(rec_depth=1)
        rssm2 = _make_tiny_rssm(rec_depth=2)
        # Force same weights so the difference comes only from the extra iteration
        rssm2._cell.load_state_dict(rssm1._cell.state_dict())
        rssm2._img_in_layers.load_state_dict(rssm1._img_in_layers.state_dict())
        rssm2._img_out_layers.load_state_dict(rssm1._img_out_layers.state_dict())
        rssm1.eval()
        rssm2.eval()
        prev1 = rssm1.initial(batch_size=2)
        prev2 = rssm2.initial(batch_size=2)
        action = torch.randn(2, 2)  # non-zero so GRU state changes
        with torch.no_grad(), torch.compiler.disable():
            prior1 = rssm1.img_step(prev1, action)
            prior2 = rssm2.img_step(prev2, action)
        assert not torch.allclose(prior1["deter"], prior2["deter"]), (
            "rec_depth=2 produced same deter as rec_depth=1 — chaining not active"
        )

    def test_rec_depth_2_is_sequential(self):
        """Verify: one img_step(rec_depth=2) == two chained img_step(rec_depth=1) calls.

        This is the golden test for Fix 1: before the fix, rec_depth=2 re-read
        prev_state["deter"] each iteration (no chaining). After Fix 1, it feeds
        the updated deter forward each iteration, matching explicit sequential calls.
        """
        rssm1 = _make_tiny_rssm(rec_depth=1)
        rssm2 = _make_tiny_rssm(rec_depth=2)
        # Share all weights so differences are purely from iteration count
        rssm2._cell.load_state_dict(rssm1._cell.state_dict())
        rssm2._img_in_layers.load_state_dict(rssm1._img_in_layers.state_dict())
        rssm2._img_out_layers.load_state_dict(rssm1._img_out_layers.state_dict())
        rssm1.eval()
        rssm2.eval()
        prev = rssm1.initial(batch_size=1)
        action = torch.randn(1, 2)

        with torch.no_grad(), torch.compiler.disable():
            # Two sequential rec_depth=1 steps: iter1 output feeds iter2 as deter
            mid = rssm1.img_step(prev, action)
            mid_state = {k: v.clone() for k, v in prev.items()}
            mid_state["deter"] = mid["deter"]
            out_chain = rssm1.img_step(mid_state, action)
            # One rec_depth=2 step — must match the chained result
            out_single = rssm2.img_step(prev, action)

        assert torch.allclose(out_chain["deter"], out_single["deter"], atol=1e-5), (
            "rec_depth=2 deter does not match two chained rec_depth=1 steps — Fix 1 may be broken"
        )


# ===========================================================================
# Fix 5 — LR warmup in Optimizer
# ===========================================================================

class TestFix5LRWarmup:
    """Verify that Optimizer with warmup_steps does a linear ramp then stays flat."""

    @staticmethod
    def _make_opt(lr: float, warmup_steps: int) -> Optimizer:
        param = torch.nn.Parameter(torch.zeros(4))
        return Optimizer(
            "test",
            [param],
            lr=lr,
            eps=1e-8,
            clip=None,
            wd=0.0,
            opt="adam",
            use_amp=False,
            warmup_steps=warmup_steps,
        )

    def test_no_warmup_no_sched(self):
        opt = self._make_opt(lr=1e-3, warmup_steps=0)
        assert opt._sched is None

    def test_warmup_sched_created(self):
        opt = self._make_opt(lr=1e-3, warmup_steps=10)
        assert opt._sched is not None

    def test_lr_ramps_linearly(self):
        """After k < warmup_steps scheduler steps, LR must be increasing linearly."""
        lr = 1e-3
        warmup = 10
        opt = self._make_opt(lr=lr, warmup_steps=warmup)
        lrs = []
        for _ in range(warmup + 2):
            lrs.append(opt._opt.param_groups[0]["lr"])
            opt._sched.step()

        # First LR should be ~1% of target (start_factor=0.01)
        assert abs(lrs[0] - lr * 0.01) < lr * 0.01 * 1e-3, f"initial LR wrong: {lrs[0]}"
        # LR must be strictly increasing across warmup
        for i in range(warmup - 1):
            assert lrs[i] < lrs[i + 1], f"LR not increasing at step {i}: {lrs[i]} >= {lrs[i+1]}"
        # After warmup, LR must equal the target
        assert abs(lrs[warmup] - lr) < lr * 1e-5, f"post-warmup LR wrong: {lrs[warmup]}"
        # After warmup + 1 more step, LR stays at target (constant)
        assert abs(lrs[warmup + 1] - lr) < lr * 1e-5, f"LR not constant after warmup: {lrs[warmup+1]}"

    def test_metrics_includes_lr(self):
        """Optimizer.__call__ must return 'model_lr' in its metrics dict."""
        param = torch.nn.Parameter(torch.randn(4))
        opt = Optimizer(
            "model",
            [param],
            lr=1e-3,
            eps=1e-8,
            clip=1000,  # clip must be >= 1 per assertion
            wd=0.0,
            opt="adam",
            use_amp=False,
            warmup_steps=5,
        )
        loss = (param ** 2).sum()
        # __call__(loss, params, retain_graph): params is the list to clip grads over
        metrics = opt(loss, [param], retain_graph=False)
        assert "model_lr" in metrics, f"model_lr not found in metrics: {metrics.keys()}"


# ===========================================================================
# Fix 7 — clip → clamp (numerical equality)
# ===========================================================================

class TestFix7ClampVsClip:
    """torch.clamp and torch.clip are semantically identical; verify Fix 7 is equivalent."""

    def test_clamp_clip_equal_positive(self):
        t = torch.tensor([0.0, 0.5, 1.0, 2.0, -0.1])
        free = 0.3
        assert torch.allclose(torch.clamp(t, min=free), torch.clip(t, min=free))

    def test_clamp_clip_equal_negative(self):
        t = torch.randn(100)
        free = 0.0
        assert torch.allclose(torch.clamp(t, min=free), torch.clip(t, min=free))

    def test_clamp_in_kl_loss_path(self):
        """Smoke: call RSSM.kl_loss and assert outputs are finite (uses clamp now).
        discrete=False so state uses mean/std keys for ContDist / Normal.
        """
        rssm = _make_tiny_rssm(rec_depth=1)
        rssm.eval()
        B, T = 2, 4
        # Build minimal post/prior with keys expected by get_dist when discrete=False
        def _fake_state():
            return {
                "mean": torch.randn(B, T, 4),
                "std": torch.ones(B, T, 4) * 0.5,
                "stoch": torch.randn(B, T, 4),
                "deter": torch.randn(B, T, 8),
            }

        post = _fake_state()
        prior = _fake_state()
        with torch.no_grad():
            loss, value, dyn_loss, rep_loss = rssm.kl_loss(
                post, prior, free=1.0, dyn_scale=0.5, rep_scale=0.1
            )
        assert torch.isfinite(loss).all()
        assert torch.isfinite(dyn_loss).all()
        assert torch.isfinite(rep_loss).all()
        # After clamp(min=free=1.0), all values must be >= 1.0
        assert (dyn_loss >= 1.0 - 1e-5).all(), "dyn_loss below free bits threshold"
        assert (rep_loss >= 1.0 - 1e-5).all(), "rep_loss below free bits threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
