"""Tests for MiniMax H3's FinalLayer and build_mod_segments."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

model_mod = load_native_module("minimax_h3.model")
FinalLayer = model_mod.FinalLayer
build_mod_segments = model_mod.build_mod_segments


def test_build_mod_segments_two_distinct_timesteps():
    segments = [(0, 3, "text"), (3, 9, "audio"), (9, 20, "video")]
    mod_segments, unique_t = build_mod_segments(segments, t_video=0.3, t_audio=0.7)

    assert unique_t == [0.3, 0.7]
    # t_row: 0.3 -> 0, 0.7 -> 1. tags: video=0, text=1, audio=2.
    assert mod_segments == [
        (0, 3, 0 * 3 + 1),  # text at t_video's row
        (3, 9, 1 * 3 + 2),  # audio at t_audio's row
        (9, 20, 0 * 3 + 0),  # video at t_video's row
    ]


def test_build_mod_segments_same_timestep_for_both_streams():
    segments = [(0, 2, "text"), (2, 4, "audio"), (4, 6, "video")]
    mod_segments, unique_t = build_mod_segments(segments, t_video=0.5, t_audio=0.5)

    assert unique_t == [0.5]
    assert mod_segments == [(0, 2, 1), (2, 4, 2), (4, 6, 0)]


def test_final_layer_output_shapes():
    hidden, t_dim, video_dim, audio_dim = 16, 8, 96, 32
    layer = FinalLayer(hidden, t_dim, video_dim, audio_dim, eps=1e-5)
    seq_len = 10
    x = mx.random.normal((seq_len, hidden))
    t_emb = mx.random.normal((2, t_dim))
    video_seg = (0, 6, 0)
    audio_seg = (6, 10, 1)

    video_out, audio_out = layer(x, t_emb, video_seg, audio_seg)
    assert video_out.shape == (6, video_dim)
    assert audio_out.shape == (4, audio_dim)
    assert bool(mx.all(mx.isfinite(video_out)).item())
    assert bool(mx.all(mx.isfinite(audio_out)).item())


def test_final_layer_matches_manual_modulation():
    hidden, t_dim, video_dim, audio_dim = 8, 4, 12, 6
    layer = FinalLayer(hidden, t_dim, video_dim, audio_dim, eps=1e-5)
    seq_len = 5
    x = mx.random.normal((seq_len, hidden))
    t_emb = mx.random.normal((2, t_dim))
    video_seg = (0, 3, 1)
    audio_seg = (3, 5, 0)

    video_out, audio_out = layer(x, t_emb, video_seg, audio_seg)

    shift, scale = layer.adaln_proj(t_emb)
    expected_video_in = layer.norm(x[0:3]) * (1.0 + scale[1]) + shift[1]
    expected_video = layer.video_out(expected_video_in)
    assert np.allclose(np.array(video_out), np.array(expected_video), atol=1e-5)
