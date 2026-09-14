"""Tests for MiniMax H3's patchify/unpatchify and audio pack/unpack, verified
against the real comfy/ldm/minimax/model.py reference (pure torch reshape/
permute, no compiled kernel needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_minimax_model as _load_reference_minimax_model
from support.minimax_h3_module_loader import load_native_module

patchify_mod = load_native_module("minimax_h3.patchify")


def test_patchify_video_matches_reference():
    mm = _load_reference_minimax_model()
    import torch

    rng = np.random.default_rng(0)
    latent_np = rng.normal(size=(1, 24, 2, 4, 6)).astype(np.float32)

    ref = mm.patchify_video(torch.tensor(latent_np), patch_size=(1, 2, 2)).numpy()
    mine = np.array(patchify_mod.patchify_video(mx.array(latent_np), patch_size=(1, 2, 2)))

    assert mine.shape == ref.shape
    assert np.allclose(mine, ref, atol=1e-5)


def test_unpatchify_video_matches_reference():
    mm = _load_reference_minimax_model()
    import torch

    rng = np.random.default_rng(1)
    t, h, w, c = 2, 2, 3, 24
    rows_np = rng.normal(size=(t * h * w, c * 1 * 2 * 2)).astype(np.float32)

    ref = mm.unpatchify_video(torch.tensor(rows_np), t, h, w, c=c, patch_size=(1, 2, 2)).numpy()
    mine = np.array(patchify_mod.unpatchify_video(mx.array(rows_np), t, h, w, c=c, patch_size=(1, 2, 2)))

    assert mine.shape == ref.shape
    assert np.allclose(mine, ref, atol=1e-5)


def test_patchify_unpatchify_round_trip():
    latent = mx.random.normal((1, 24, 2, 4, 6))
    rows = patchify_mod.patchify_video(latent, patch_size=(1, 2, 2))
    back = patchify_mod.unpatchify_video(rows, t=2, h=2, w=3, c=24, patch_size=(1, 2, 2))
    assert bool(mx.allclose(back, latent, atol=1e-5).item())


def test_pack_audio_matches_reference():
    mm = _load_reference_minimax_model()
    import torch

    rng = np.random.default_rng(2)
    latent_np = rng.normal(size=(1, 32, 2, 5)).astype(np.float32)

    ref = mm.pack_audio(torch.tensor(latent_np)).numpy()
    mine = np.array(patchify_mod.pack_audio(mx.array(latent_np)))

    assert mine.shape == ref.shape
    assert np.allclose(mine, ref, atol=1e-5)


def test_unpack_audio_matches_reference():
    mm = _load_reference_minimax_model()
    import torch

    rng = np.random.default_rng(3)
    rows_np = rng.normal(size=(10, 32)).astype(np.float32)  # ch=2, t=5

    ref = mm.unpack_audio(torch.tensor(rows_np), ch=2).numpy()
    mine = np.array(patchify_mod.unpack_audio(mx.array(rows_np), ch=2))

    assert mine.shape == ref.shape
    assert np.allclose(mine, ref, atol=1e-5)


def test_pack_unpack_audio_round_trip():
    latent = mx.random.normal((1, 32, 2, 5))
    rows = patchify_mod.pack_audio(latent)
    back = patchify_mod.unpack_audio(rows, ch=2)
    assert bool(mx.allclose(back, latent, atol=1e-5).item())
