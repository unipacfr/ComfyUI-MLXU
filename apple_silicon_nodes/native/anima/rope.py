"""RoPE for Anima: Cosmos 3-axis NTK RoPE (DiT self-attn) and the LLM
adapter's 1-D RoPE. Both rotate split halves (x[:D/2], x[D/2:]).

Ported from `comfy/ldm/cosmos/position_embedding.py::VideoRopePosition3DEmb`
and `comfy/ldm/anima/model.py::RotaryEmbedding`."""

from __future__ import annotations

import mlx.core as mx


def cosmos_rope_3d(
    head_dim: int, t: int, h: int, w: int, h_ratio: float, w_ratio: float, t_ratio: float
) -> tuple[mx.array, mx.array]:
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h
    spatial = mx.arange(0, dim_h, 2, dtype=mx.float32)[: dim_h // 2] / dim_h
    temporal = mx.arange(0, dim_t, 2, dtype=mx.float32)[: dim_t // 2] / dim_t
    h_freqs = 1.0 / ((10000.0 * h_ratio ** (dim_h / (dim_h - 2))) ** spatial)
    w_freqs = 1.0 / ((10000.0 * w_ratio ** (dim_h / (dim_h - 2))) ** spatial)
    t_freqs = 1.0 / ((10000.0 * t_ratio ** (dim_t / (dim_t - 2))) ** temporal)
    et = mx.arange(t, dtype=mx.float32)[:, None] * t_freqs[None, :]
    eh = mx.arange(h, dtype=mx.float32)[:, None] * h_freqs[None, :]
    ew = mx.arange(w, dtype=mx.float32)[:, None] * w_freqs[None, :]
    ang = mx.concatenate(
        [
            mx.broadcast_to(et[:, None, None, :], (t, h, w, et.shape[1])),
            mx.broadcast_to(eh[None, :, None, :], (t, h, w, eh.shape[1])),
            mx.broadcast_to(ew[None, None, :, :], (t, h, w, ew.shape[1])),
        ],
        axis=-1,
    ).reshape(t * h * w, -1)
    return mx.cos(ang), mx.sin(ang)


def adapter_rope(length: int, head_dim: int, theta: float = 10000.0) -> tuple[mx.array, mx.array]:
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    ang = mx.arange(length, dtype=mx.float32)[:, None] * inv_freq[None, :]
    return mx.cos(ang), mx.sin(ang)


def apply_rope_split_half(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: [B, H, L, D]; cos/sin: [L, D/2]. Computes in cos.dtype, returns x.dtype
    (the DiT passes float32 tables like comfy_kitchen; the adapter passes tables
    already cast to the activation dtype like RotaryEmbedding.forward)."""
    half = x.shape[-1] // 2
    t = x.astype(cos.dtype)
    x1, x2 = t[..., :half], t[..., half:]
    out = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return out.astype(x.dtype)
