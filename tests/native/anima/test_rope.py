from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.anima_module_loader import load_native_module

rope = load_native_module("anima.rope")


def _ref_cosmos(head_dim, T, H, W, hr, wr, tr):
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h
    rs = np.arange(0, dim_h, 2)[: dim_h // 2].astype(np.float64) / dim_h
    rt = np.arange(0, dim_t, 2)[: dim_t // 2].astype(np.float64) / dim_t
    fh = 1.0 / ((10000.0 * hr ** (dim_h / (dim_h - 2))) ** rs)
    fw = 1.0 / ((10000.0 * wr ** (dim_h / (dim_h - 2))) ** rs)
    ft = 1.0 / ((10000.0 * tr ** (dim_t / (dim_t - 2))) ** rt)
    et, eh, ew = np.outer(np.arange(T), ft), np.outer(np.arange(H), fh), np.outer(np.arange(W), fw)
    ang = np.concatenate([
        np.broadcast_to(et[:, None, None, :], (T, H, W, et.shape[1])),
        np.broadcast_to(eh[None, :, None, :], (T, H, W, eh.shape[1])),
        np.broadcast_to(ew[None, None, :, :], (T, H, W, ew.shape[1])),
    ], axis=-1).reshape(T * H * W, -1)
    return np.cos(ang), np.sin(ang)


def test_cosmos_rope_matches_reference():
    cos, sin = rope.cosmos_rope_3d(128, 1, 6, 5, h_ratio=4.0, w_ratio=4.0, t_ratio=1.0)
    rc, rs = _ref_cosmos(128, 1, 6, 5, 4.0, 4.0, 1.0)
    assert cos.shape == (30, 64)
    np.testing.assert_allclose(np.array(cos), rc, atol=1e-5)
    np.testing.assert_allclose(np.array(sin), rs, atol=1e-5)


def test_split_half_rotation_matches_rotate_half():
    x = mx.random.normal((1, 2, 7, 8))
    cos, sin = rope.adapter_rope(7, 8)
    got = np.array(rope.apply_rope_split_half(x, cos, sin))
    xn = np.array(x)
    c = np.concatenate([np.array(cos)] * 2, -1)
    s = np.concatenate([np.array(sin)] * 2, -1)
    rot = np.concatenate([-xn[..., 4:], xn[..., :4]], -1)
    np.testing.assert_allclose(got, xn * c + rot * s, atol=1e-5)


def test_zero_position_is_identity():
    x = mx.random.normal((1, 1, 1, 128))
    cos, sin = rope.cosmos_rope_3d(128, 1, 1, 1, 4.0, 4.0, 1.0)
    np.testing.assert_allclose(np.array(rope.apply_rope_split_half(x, cos, sin)), np.array(x), atol=1e-6)
