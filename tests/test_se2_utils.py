"""CPU unit tests for se2_utils.py — SE(2) primitives.

No GPU, no ROS, no torch.compile required.
"""

import math
import sys
import types

import torch

# ---------------------------------------------------------------------------
# Stub tensorboard so se2_utils imports cleanly without optional deps
# ---------------------------------------------------------------------------
_fake_sw = type("SummaryWriter", (), {"__init__": lambda self, *a, **kw: None})
for _m in ["tensorboard", "tensorboard.summary", "tensorboard.summary.writer"]:
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["tensorboard.summary.writer"].SummaryWriter = _fake_sw
_tsb = types.ModuleType("torch.utils.tensorboard")
_tsb.SummaryWriter = _fake_sw
sys.modules["torch.utils.tensorboard"] = _tsb

import importlib.util, pathlib

_DREAMER = pathlib.Path(__file__).parent.parent / "deps/rosnav_rl/rosnav_rl/rosnav_rl/model/dreamerv3"

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = "rosnav_rl.model.dreamerv3"
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

se2 = _load("rosnav_rl.model.dreamerv3.se2_utils", _DREAMER / "se2_utils.py")

TOL = 1e-5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rand_pose(B=4):
    return torch.stack([
        torch.randn(B) * 3,
        torch.randn(B) * 3,
        (torch.rand(B) * 2 - 1) * math.pi,
    ], dim=-1)


def identity(B=4):
    return torch.zeros(B, 3)


# ---------------------------------------------------------------------------
# Test se2_compose
# ---------------------------------------------------------------------------
class TestSe2Compose:
    def test_identity_left(self):
        T = rand_pose()
        I = identity(len(T))
        result = se2.se2_compose(I, T)
        assert torch.allclose(result, T, atol=TOL), f"identity∘T != T: {result} vs {T}"

    def test_identity_right(self):
        T = rand_pose()
        I = identity(len(T))
        result = se2.se2_compose(T, I)
        assert torch.allclose(result, T, atol=TOL), f"T∘identity != T: {result} vs {T}"

    def test_angle_wraps(self):
        # Composing two pi/2 rotations should give pi (mod 2pi), not -pi issue
        T1 = torch.tensor([[0., 0., math.pi * 0.9]])
        T2 = torch.tensor([[0., 0., math.pi * 0.9]])
        result = se2.se2_compose(T1, T2)
        # Should wrap to -~0.28 instead of ~5.65
        assert result[0, 2].abs() <= math.pi + 1e-4

    def test_translation_only(self):
        T1 = torch.tensor([[1., 0., 0.]])
        T2 = torch.tensor([[2., 0., 0.]])
        result = se2.se2_compose(T1, T2)
        expected = torch.tensor([[3., 0., 0.]])
        assert torch.allclose(result, expected, atol=TOL)

    def test_rotation_then_translation(self):
        # T1 rotates 90°, T2 translates world-frame x+1.
        # T2's translation is in T2's frame (parent/world here since θ2=0).
        # Composed: R(90°) rotate, then translate [1,0] in world frame.
        T1 = torch.tensor([[0., 0., math.pi / 2]])
        T2 = torch.tensor([[1., 0., 0.]])
        result = se2.se2_compose(T1, T2)
        # xc = cos(0)*0 - sin(0)*0 + 1 = 1
        # yc = sin(0)*0 + cos(0)*0 + 0 = 0
        expected = torch.tensor([[1., 0., math.pi / 2]])
        assert torch.allclose(result, expected, atol=TOL)


# ---------------------------------------------------------------------------
# Test se2_inverse
# ---------------------------------------------------------------------------
class TestSe2Inverse:
    def test_identity_inverse_is_identity(self):
        I = identity()
        result = se2.se2_inverse(I)
        assert torch.allclose(result, I, atol=TOL)

    def test_compose_with_inverse_is_identity(self):
        T = rand_pose()
        I_approx = se2.se2_compose(T, se2.se2_inverse(T))
        expected = identity(len(T))
        assert torch.allclose(I_approx, expected, atol=TOL), \
            f"T∘T^{{-1}} != I: max err {(I_approx - expected).abs().max()}"

    def test_inverse_compose_is_identity(self):
        T = rand_pose()
        I_approx = se2.se2_compose(se2.se2_inverse(T), T)
        expected = identity(len(T))
        assert torch.allclose(I_approx, expected, atol=TOL), \
            f"T^{{-1}}∘T != I: max err {(I_approx - expected).abs().max()}"

    def test_pure_rotation_inverse(self):
        T = torch.tensor([[0., 0., math.pi / 3]])
        Ti = se2.se2_inverse(T)
        expected = torch.tensor([[0., 0., -math.pi / 3]])
        assert torch.allclose(Ti, expected, atol=TOL)


# ---------------------------------------------------------------------------
# Test se2_between
# ---------------------------------------------------------------------------
class TestSe2Between:
    def test_between_same_pose_is_identity(self):
        T = rand_pose()
        result = se2.se2_between(T, T)
        expected = identity(len(T))
        assert torch.allclose(result, expected, atol=TOL)

    def test_between_anchor_and_identity(self):
        Ta = rand_pose()
        Tb = identity(len(Ta))
        result = se2.se2_between(Ta, Tb)
        expected = se2.se2_inverse(Ta)
        assert torch.allclose(result, expected, atol=TOL)

    def test_roundtrip(self):
        Ta = rand_pose()
        Tb = rand_pose()
        P = se2.se2_between(Ta, Tb)
        Tb_recovered = se2.se2_compose(Ta, P)
        assert torch.allclose(Tb_recovered, Tb, atol=TOL)


# ---------------------------------------------------------------------------
# Test se2_transform_points
# ---------------------------------------------------------------------------
class TestSe2TransformPoints:
    def test_identity_noop(self):
        T = identity(3)
        pts = torch.randn(3, 5, 2)
        result = se2.se2_transform_points(T, pts)
        assert torch.allclose(result, pts, atol=TOL)

    def test_translation_only(self):
        T = torch.tensor([[2., 3., 0.]])
        pts = torch.zeros(1, 4, 2)
        result = se2.se2_transform_points(T, pts)
        expected = torch.tensor([2., 3.]).view(1, 1, 2).expand(1, 4, 2)
        assert torch.allclose(result, expected, atol=TOL)

    def test_rotation_90_degrees(self):
        T = torch.tensor([[0., 0., math.pi / 2]])
        pts = torch.tensor([[[1., 0.]]])  # (1, 1, 2)
        result = se2.se2_transform_points(T, pts)
        expected = torch.tensor([[[0., 1.]]])
        assert torch.allclose(result, expected, atol=TOL)

    def test_inverse_roundtrip(self):
        T = rand_pose(2)
        pts = torch.randn(2, 6, 2)
        pts_forward = se2.se2_transform_points(T, pts)
        pts_back = se2.se2_transform_points(se2.se2_inverse(T), pts_forward)
        assert torch.allclose(pts_back, pts, atol=TOL)

    def test_anchor_to_current_frame(self):
        # P_k = (1, 0, pi/2): robot moved 1m right and rotated 90°
        P_k = torch.tensor([[1., 0., math.pi / 2]])
        # Ped at (0, 1) in anchor frame (forward of initial robot position)
        peds_anchor = torch.tensor([[[0., 1.]]])
        # In current frame: robot has moved to (1,0), θ=90°.
        # Inverse of P_k: xi = -(cos(90°)*1 + sin(90°)*0) = -0, yi = sin(90°)*1 - cos(90°)*0 = 1
        # Actually: T^{-1} = (-cos(θ)*x - sin(θ)*y, sin(θ)*x - cos(θ)*y, -θ)
        # = (-0*1 - 1*0, 1*1 - 0*0, -90°) = (0, 1, -90°)
        # Apply T^{-1} to (0,1): R(-90°)@[0,1] + [0,1]
        # R(-90°) = [[0,1],[-1,0]]
        # R(-90°)@[0,1] = [1, 0]
        # [1,0] + [0,1] = [1,1]
        P_k_inv = se2.se2_inverse(P_k)
        peds_current = se2.se2_transform_points(P_k_inv, peds_anchor)
        expected = torch.tensor([[[1., 1.]]])
        assert torch.allclose(peds_current, expected, atol=TOL), \
            f"frame transform wrong: {peds_current} != {expected}"


# ---------------------------------------------------------------------------
# Test se2_rotate_vectors (no translation)
# ---------------------------------------------------------------------------
class TestSe2RotateVectors:
    def test_identity(self):
        T = identity(2)
        vecs = torch.randn(2, 4, 2)
        result = se2.se2_rotate_vectors(T, vecs)
        assert torch.allclose(result, vecs, atol=TOL)

    def test_90_degree_rotation(self):
        T = torch.tensor([[5., -3., math.pi / 2]])  # translation irrelevant
        vecs = torch.tensor([[[1., 0.]]])
        result = se2.se2_rotate_vectors(T, vecs)
        expected = torch.tensor([[[0., 1.]]])
        assert torch.allclose(result, expected, atol=TOL)


# ---------------------------------------------------------------------------
# Test integrate_se2
# ---------------------------------------------------------------------------
class TestIntegrateSe2:
    def test_zero_action_no_motion(self):
        pose = rand_pose(3)
        action = torch.zeros(3, 2)
        scale = torch.tensor([1.0, 1.0])
        result = se2.integrate_se2(pose, action, scale, dt=0.1, holonomic=False)
        assert torch.allclose(result, pose, atol=TOL)

    def test_pure_forward_motion(self):
        pose = torch.zeros(1, 3)
        action = torch.tensor([[1.0, 0.0]])  # v=1, omega=0
        scale = torch.tensor([1.0, 1.0])
        result = se2.integrate_se2(pose, action, scale, dt=0.1, holonomic=False)
        expected = torch.tensor([[0.1, 0., 0.]])
        assert torch.allclose(result, expected, atol=TOL)

    def test_pure_rotation(self):
        pose = torch.zeros(1, 3)
        action = torch.tensor([[0.0, 1.0]])  # v=0, omega=1 rad/s
        scale = torch.tensor([1.0, 1.0])
        result = se2.integrate_se2(pose, action, scale, dt=0.1, holonomic=False)
        expected = torch.tensor([[0., 0., 0.1]])
        assert torch.allclose(result, expected, atol=TOL)

    def test_action_scale_denorm(self):
        pose = torch.zeros(1, 3)
        # Normalized action of 1.0 with scale=0.5 means v=0.5 m/s
        action = torch.tensor([[1.0, 0.0]])
        scale = torch.tensor([0.5, 1.0])
        result = se2.integrate_se2(pose, action, scale, dt=0.1, holonomic=False)
        expected = torch.tensor([[0.05, 0., 0.]])
        assert torch.allclose(result, expected, atol=TOL)

    def test_holonomic_vy(self):
        pose = torch.zeros(1, 3)
        action = torch.tensor([[0.0, 1.0, 0.0]])  # vx=0, vy=1, omega=0
        scale = torch.tensor([1.0, 1.0, 1.0])
        result = se2.integrate_se2(pose, action, scale, dt=0.1, holonomic=True)
        expected = torch.tensor([[0., 0.1, 0.]])
        assert torch.allclose(result, expected, atol=TOL)

    def test_accumulated_pose_roundtrip(self):
        # After N steps forward then N steps back, should return to origin (approx)
        pose = torch.zeros(1, 3)
        act_fwd = torch.tensor([[1.0, 0.0]])
        act_bwd = torch.tensor([[-1.0, 0.0]])
        scale = torch.tensor([1.0, 1.0])
        for _ in range(10):
            pose = se2.integrate_se2(pose, act_fwd, scale, dt=0.1)
        for _ in range(10):
            pose = se2.integrate_se2(pose, act_bwd, scale, dt=0.1)
        assert torch.allclose(pose, torch.zeros(1, 3), atol=TOL * 10)

    def test_angle_wrap_stays_bounded(self):
        pose = torch.zeros(1, 3)
        action = torch.tensor([[0.0, 1.0]])  # max angular
        scale = torch.tensor([1.0, 2.0])     # omega_max=2 rad/s
        for _ in range(100):
            pose = se2.integrate_se2(pose, action, scale, dt=0.1)
        assert pose[0, 2].abs() <= math.pi + 1e-4, f"angle out of range: {pose[0, 2]}"


# ---------------------------------------------------------------------------
# Test apply_se2_to_peds_flat
# ---------------------------------------------------------------------------
class TestApplySe2ToPedsFlat:
    def test_identity_noop(self):
        B, N, F = 2, 4, 5
        T = torch.zeros(B, 3)
        peds = torch.randn(B, N * (F + 1))
        result = se2.apply_se2_to_peds_flat(T, peds, N, F)
        assert torch.allclose(result, peds, atol=TOL)

    def test_translation_shifts_positions(self):
        B, N, F = 1, 2, 5
        T = torch.tensor([[1., 2., 0.]])
        peds = torch.zeros(B, N * (F + 1))
        result = se2.apply_se2_to_peds_flat(T, peds, N, F)
        # All x positions shift by 1, y by 2
        for i in range(N):
            base = i * (F + 1)
            assert abs(result[0, base].item() - 1.0) < TOL
            assert abs(result[0, base + 1].item() - 2.0) < TOL

    def test_rotation_rotates_velocity(self):
        B, N, F = 1, 1, 5
        T = torch.tensor([[0., 0., math.pi / 2]])
        peds = torch.zeros(B, N * (F + 1))
        peds[0, 2] = 1.0  # vx = 1
        result = se2.apply_se2_to_peds_flat(T, peds, N, F)
        # After 90° rotation: vx -> vy
        assert abs(result[0, 2].item()) < TOL       # new vx ~ 0
        assert abs(result[0, 3].item() - 1.0) < TOL # new vy ~ 1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
