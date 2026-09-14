"""Tests for MiniMax H3's RMSNorm+split-half-RoPE port.

`test_matches_real_comfy_kitchen_reference` is the load-bearing test: the
call site (`comfy/ldm/minimax/model.py::Attention.forward`) uses a compiled
kernel with no plain-Python source in `comfy/` to read, so this module was
ported from `comfy_kitchen`'s pure-PyTorch "eager" backend instead (see
rope.py's module docstring) and is verified here against that same real
backend, not just against a formula derived from the call site's comments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

rope = load_native_module("minimax_h3.rope")

_COMFYUI_VENV_SITE_PACKAGES = Path(
    "/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/.venv/lib/python3.13/site-packages"
)


def _load_reference_rms_rope1():
    if not _COMFYUI_VENV_SITE_PACKAGES.exists():
        pytest.skip("comfy_kitchen (ComfyUI venv) not present on this machine")
    sys.path.insert(0, str(_COMFYUI_VENV_SITE_PACKAGES))
    try:
        import torch
        from comfy_kitchen.backends.eager.rope import _rms_rope1
    except ImportError:
        pytest.skip("comfy_kitchen not importable from the ComfyUI venv")
    return torch, _rms_rope1


def test_matches_real_comfy_kitchen_reference():
    torch, rms_rope1 = _load_reference_rms_rope1()
    torch.manual_seed(0)
    seq_len, heads, head_dim, rot_dim = 7, 3, 128, 96
    half = rot_dim // 2

    x = torch.randn(1, seq_len, heads, head_dim)
    scale = torch.randn(head_dim)
    ang = torch.rand(seq_len, half) * 3.14159
    cos_t, sin_t = torch.cos(ang), torch.sin(ang)
    table = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=-1).reshape(1, seq_len, 1, half, 2, 2)

    expected = rms_rope1(x, table, scale, epsilon=1e-5, split_half=True, rot_dim=rot_dim)
    expected_np = expected.numpy()[0]  # [S, heads, head_dim]

    x_mx = mx.array(x.numpy()[0])  # [S, heads, head_dim]
    scale_mx = mx.array(scale.numpy())
    cos_mx = mx.array(cos_t.numpy())[:, None, :]  # [S, 1, half] broadcasts over heads
    sin_mx = mx.array(sin_t.numpy())[:, None, :]

    out = rope.rms_norm_rope_split_half(x_mx, scale_mx, 1e-5, rot_dim, cos_mx, sin_mx)
    out_np = np.array(out)

    assert np.abs(out_np - expected_np).max() < 1e-5


def test_full_rotation_when_rot_dim_equals_head_dim():
    torch, rms_rope1 = _load_reference_rms_rope1()
    torch.manual_seed(1)
    seq_len, head_dim = 4, 8
    half = head_dim // 2

    x = torch.randn(1, seq_len, 1, head_dim)
    scale = torch.randn(head_dim)
    ang = torch.rand(seq_len, half)
    cos_t, sin_t = torch.cos(ang), torch.sin(ang)
    table = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=-1).reshape(1, seq_len, 1, half, 2, 2)

    expected = rms_rope1(x, table, scale, epsilon=1e-5, split_half=True, rot_dim=head_dim)
    expected_np = expected.numpy()[0, :, 0]

    x_mx = mx.array(x.numpy()[0, :, 0])
    scale_mx = mx.array(scale.numpy())
    cos_mx = mx.array(cos_t.numpy())
    sin_mx = mx.array(sin_t.numpy())

    out = rope.rms_norm_rope_split_half(x_mx, scale_mx, 1e-5, head_dim, cos_mx, sin_mx)
    assert np.abs(np.array(out) - expected_np).max() < 1e-5


def test_rope_freqs_shape_and_duplication():
    position_ids = mx.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]])  # [S=2, 3]
    inv_freq = mx.array([0.1, 0.2])  # F=2
    angles = rope.rope_freqs(position_ids, inv_freq)
    assert angles.shape == (2, 12)  # 6*F
    # second half duplicates the first half exactly
    assert np.allclose(np.array(angles[:, :6]), np.array(angles[:, 6:]))


def test_rope_cos_sin_uses_only_first_half():
    angles = mx.array([[0.0, 1.5707963, 0.0, 1.5707963]])  # [S=1, 2*half=4], half=2
    cos, sin = rope.rope_cos_sin(angles)
    assert cos.shape == (1, 2)
    assert np.allclose(np.array(cos), [[1.0, 0.0]], atol=1e-6)
    assert np.allclose(np.array(sin), [[0.0, 1.0]], atol=1e-6)


def test_rms_norm_matches_manual_formula():
    x = mx.array([[1.0, 2.0, 3.0, 4.0]])
    weight = mx.array([1.0, 1.0, 1.0, 1.0])
    out = rope.rms_norm(x, weight, eps=1e-6)
    expected = np.array([1.0, 2.0, 3.0, 4.0]) / np.sqrt(np.mean(np.array([1.0, 4.0, 9.0, 16.0])) + 1e-6)
    assert np.allclose(np.array(out)[0], expected, atol=1e-5)
