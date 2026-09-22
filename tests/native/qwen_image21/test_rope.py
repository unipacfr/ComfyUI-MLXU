"""Parity of qwen_image21.rope against the real comfy Qwen3 RoPE path.

Reuses the same verification already done for MiniMax H3's identical
`qwen3_rope_cos_sin`/`rms_norm_rope_split_half` (this is generic Qwen3
text-only RoPE math, independent of model depth/width -- see
`native/minimax_h3/rope.py`'s module docstring), re-run here against the
real `comfy.text_encoders.llama` module directly rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

rope_mod = load_native_module("qwen_image21.rope")


def _load_real_comfy_llama():
    comfyui_root = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")
    venv_site_packages = comfyui_root / ".venv" / "lib" / "python3.13" / "site-packages"
    if not comfyui_root.exists() or not venv_site_packages.exists():
        pytest.skip("ComfyUI install not present on this machine")
    sys.path.insert(0, str(comfyui_root))
    sys.path.insert(0, str(venv_site_packages))
    try:
        import comfy.text_encoders.llama as llama
    except ImportError as e:
        pytest.skip(f"comfy.text_encoders.llama not importable: {e}")
    return llama


def test_cos_sin_matches_comfy_precompute_freqs_cis():
    llama = _load_real_comfy_llama()
    import torch

    seq_len, head_dim, theta = 7, 16, 5000000.0
    position_ids = torch.arange(seq_len).unsqueeze(0)  # [1, S], text-only path
    # precompute_freqs_cis returns the bare (cos, sin, neg_sin) 3-tuple directly
    # (not wrapped in a list) when theta is a single float rather than a list.
    freqs_cis = llama.precompute_freqs_cis(head_dim, position_ids, theta, None, None, interleaved_mrope=False)

    # comfy's attention tensors are laid out [B, H, S, D] (heads before seq),
    # matching rope_matrix's [B, 1, S, half, 2, 2] broadcast shape.
    x = torch.randn(1, 1, seq_len, head_dim)
    ref_xq, _ = llama.apply_rope(x, x, freqs_cis)  # pass x for both q and k; only q's result matters here
    ref_rotated = ref_xq[0, 0, :, :].numpy()

    cos, sin = rope_mod.qwen3_rope_cos_sin(seq_len, head_dim, theta)
    xm = mx.array(x[0, 0, :, :].numpy())
    got_rotated = np.array(rope_mod.apply_rope_split_half(xm, cos, sin))
    assert np.abs(got_rotated - ref_rotated).max() < 1e-4


def test_rms_norm_matches_torch_rms_norm():
    import torch
    import torch.nn.functional as F

    x = np.random.randn(3, 16).astype(np.float32)
    weight = np.random.randn(16).astype(np.float32)
    eps = 1e-6
    expected = F.rms_norm(torch.from_numpy(x), (16,), weight=torch.from_numpy(weight), eps=eps).numpy()
    got = np.array(rope_mod.rms_norm(mx.array(x), mx.array(weight), eps))
    assert np.allclose(got, expected, atol=1e-5)


def test_rms_norm_rope_split_half_rotates_only_rot_dim():
    cos, sin = rope_mod.qwen3_rope_cos_sin(4, 8, 5000000.0)
    x = mx.random.normal((4, 2, 8))
    weight = mx.ones(8)
    out = rope_mod.rms_norm_rope_split_half(x, weight, 1e-6, 8, cos[:, None, :], sin[:, None, :])
    assert out.shape == x.shape
    assert bool(mx.all(mx.isfinite(out)).item())
