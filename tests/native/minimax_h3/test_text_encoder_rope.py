"""Tests for the Qwen3 text-only RoPE, verified against the real
comfy.text_encoders.llama reference."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

rope_mod = load_native_module("minimax_h3.rope")
text_rope_mod = load_native_module("minimax_h3.text_encoder_rope")

_COMFYUI_ROOT = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")
_COMFYUI_VENV_SITE_PACKAGES = _COMFYUI_ROOT / ".venv" / "lib" / "python3.13" / "site-packages"


def _load_reference_llama_module():
    if not _COMFYUI_ROOT.exists() or not _COMFYUI_VENV_SITE_PACKAGES.exists():
        pytest.skip("ComfyUI install not present on this machine")
    import sys as _sys

    for name in list(_sys.modules):
        if name == "comfy" or name.startswith("comfy."):
            del _sys.modules[name]
    _sys.path.insert(0, str(_COMFYUI_ROOT))
    _sys.path.insert(0, str(_COMFYUI_VENV_SITE_PACKAGES))
    try:
        import comfy.text_encoders.llama as llama_mod
    except ImportError as e:
        pytest.skip(f"comfy.text_encoders.llama not importable: {e}")
    return llama_mod


def test_cos_sin_match_reference():
    llama_mod = _load_reference_llama_module()
    import torch

    seq_len, head_dim, theta = 6, 16, 5000000.0
    pos = torch.arange(0, seq_len).unsqueeze(0)
    cos_ref, sin_ref, nsin_ref = llama_mod.precompute_freqs_cis(head_dim, pos, theta)
    # reference cos/sin have shape [1, 1, S, head_dim] (duplicated across the
    # two halves); ours is [S, head_dim//2] (the non-duplicated half only).
    half = head_dim // 2
    cos_ref_half = cos_ref[0, 0, :, :half].numpy()
    sin_ref_half = sin_ref[0, 0, :, :half].numpy()

    cos, sin = text_rope_mod.qwen3_rope_cos_sin(seq_len, head_dim, theta)
    assert np.allclose(np.array(cos), cos_ref_half, atol=1e-5)
    assert np.allclose(np.array(sin), sin_ref_half, atol=1e-5)


def test_full_rotation_matches_reference_apply_rope():
    llama_mod = _load_reference_llama_module()
    import torch

    seq_len, heads, head_dim, theta = 5, 2, 16, 5000000.0
    torch.manual_seed(0)
    xq = torch.randn(1, heads, seq_len, head_dim)
    pos = torch.arange(0, seq_len).unsqueeze(0)
    freqs_cis = llama_mod.precompute_freqs_cis(head_dim, pos, theta)
    xk = torch.randn(1, heads, seq_len, head_dim)
    q_ref, _ = llama_mod.apply_rope(xq, xk, freqs_cis)

    cos, sin = text_rope_mod.qwen3_rope_cos_sin(seq_len, head_dim, theta)
    x_mx = mx.array(xq.numpy()[0]).transpose(1, 0, 2)  # [S, heads, head_dim]
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    out = rope_mod.apply_rope_split_half(x_mx, cos_b, sin_b)
    out_np = np.array(out).transpose(1, 0, 2)  # back to [heads, S, head_dim]

    assert np.allclose(out_np, q_ref.numpy()[0], atol=1e-5)


def test_rms_norm_rope_full_rotation_via_shared_helper():
    # rot_dim == head_dim exercises rope.py's "full rotation" branch --
    # confirms it's reusable as-is for Qwen3, not just MiniMax H3's partial case.
    seq_len, head_dim = 4, 8
    x = mx.random.normal((seq_len, 1, head_dim))
    weight = mx.ones(head_dim)
    cos, sin = text_rope_mod.qwen3_rope_cos_sin(seq_len, head_dim, theta=10000.0)
    out = rope_mod.rms_norm_rope_split_half(x, weight, 1e-5, head_dim, cos[:, None, :], sin[:, None, :])
    assert out.shape == x.shape
    assert bool(mx.all(mx.isfinite(out)).item())
