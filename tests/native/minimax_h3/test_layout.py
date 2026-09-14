"""Tests for MiniMax H3's packed-sequence layout (position ids + segments).

The load-bearing tests import `comfy/ldm/minimax/model.py` directly (pure
torch/math, no GPU/compiled kernel needed for these specific functions) and
compare against it numerically -- these grid-building functions are dense
enough (meshgrid, cumulative sums, tiling) that a subtle index-order bug
would not show up as a shape mismatch or a NaN, only as wrong numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_minimax_model as _load_reference_minimax_model
from support.minimax_h3_module_loader import load_native_module

layout_mod = load_native_module("minimax_h3.layout")


def test_frame_grid_matches_reference():
    mm = _load_reference_minimax_model()
    for h, w in [(4, 4), (6, 8), (2, 2)]:
        ref_frame, ref_w = mm._frame_grid(h, w)
        frame, w_axis = layout_mod.frame_grid(h, w)
        assert np.allclose(np.array(frame), ref_frame.numpy())
        assert np.allclose(np.array(w_axis), ref_w.numpy())


def test_video_t_grid_matches_reference():
    mm = _load_reference_minimax_model()
    for n, origin in [(1, 0.0), (5, 3.0), (17, 2.5)]:
        ref = mm._video_t_grid(n, origin)
        mine = layout_mod.video_t_grid(n, origin)
        assert np.allclose(np.array(mine), ref.numpy())


def test_video_grid_matches_reference():
    mm = _load_reference_minimax_model()
    import torch

    frame_t, _ = mm._frame_grid(4, 4)
    frame_mx, _ = layout_mod.frame_grid(4, 4)
    for vt, cursor in [(1, 0.0), (3, 2.0)]:
        ref = mm._video_grid(vt, frame_t, cursor)
        mine = layout_mod.video_grid(vt, frame_mx, cursor)
        assert np.allclose(np.array(mine), ref.numpy())


def test_audio_grid_matches_reference():
    mm = _load_reference_minimax_model()
    for cursor, t, w_low, w_high in [(0.0, 3, 0.0, 16.0), (5.0, 7, -8.0, 8.0)]:
        ref = mm._audio_grid(cursor, t, w_low, w_high)
        mine = layout_mod.audio_grid(cursor, t, w_low, w_high)
        assert np.allclose(np.array(mine), ref.numpy())


def test_packed_layout_segments_and_seq_len():
    layout = layout_mod.PackedLayout(text_len=4, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    # frame_rows for 4x4 = 2x2 = 4; n_video = latent_t(2)*4 = 8; audio rows = 3*2=6
    assert layout.segments == [(0, 4, "text"), (4, 10, "audio"), (10, 18, "video")]
    assert layout.seq_len == 18
    assert layout.position_ids.shape == (18, 3)


def test_packed_layout_matches_reference_position_ids():
    mm = _load_reference_minimax_model()
    text_len, latent_t, latent_h, latent_w, audio_t = 5, 2, 4, 4, 3

    ref_layout = mm.PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t)
    mine = layout_mod.PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t)

    assert mine.seq_len == ref_layout.seq_len
    assert mine.segments == ref_layout.segments
    assert np.allclose(np.array(mine.position_ids), ref_layout.position_ids.numpy())
