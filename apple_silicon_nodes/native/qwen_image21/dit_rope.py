"""FLUX-style N-axis RoPE, ported verbatim from `native/flux2/model.py`'s `rope_freqs`/
`embed_nd`/`apply_rope`/`timestep_embedding` -- the real ComfyUI reference for Qwen Image
2.1's DiT (`comfy/ldm/qwen_image21/model.py`) imports `comfy.ldm.flux.layers.EmbedND` and
`comfy.ldm.flux.math.apply_rope1`/`rope` directly, unchanged, so this is the same math,
just with axes_dims_rope=(16, 56, 56) instead of Flux2's own axis split. Duplicated per
this project's per-family file convention (see Krea2's own `rope.py`) rather than
cross-imported from `flux2`.
"""

from __future__ import annotations

import math

import mlx.core as mx


def rope_freqs(pos: mx.array, dim: int, theta: float) -> mx.array:
    """[N, dim/2, 2, 2] rotation-matrix RoPE table for one axis."""
    assert dim % 2 == 0
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta**scale)
    out = pos.astype(mx.float32)[:, None] * omega[None, :]
    cos, sin = mx.cos(out), mx.sin(out)
    return mx.stack([cos, -sin, sin, cos], axis=-1).reshape(*out.shape, 2, 2)


def embed_nd(ids: mx.array, axes_dim: tuple[int, ...], theta: float) -> mx.array:
    """N-axis RoPE embedding table. ids: [N, len(axes_dim)]."""
    parts = [rope_freqs(ids[:, i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return mx.concatenate(parts, axis=-3)


def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply the [...,2,2] rotation-matrix RoPE to Q or K. x: [B,H,N,D].

    Restores `x`'s own dtype on return: `freqs` is float32 (from `rope_freqs`), so the
    multiply upconverts an fp16/bf16 `x` to float32 unless cast back explicitly --
    matching the reference's `_apply_rope1` (`comfy/ldm/flux/math.py:40`), which ends
    with `.type_as(x)`. Without this, the model silently runs its attention path in
    float32 regardless of its configured dtype (2x activation memory, and
    `QwenImage21TransformerBlock`'s fp16 overflow clip guard never fires since nothing
    downstream is fp16 anymore)."""
    B, H, N, D = x.shape
    x_pairs = x.reshape(B, H, N, D // 2, 1, 2)
    f = freqs[None, None]
    out = (f[..., 0] * x_pairs[..., 0]) + (f[..., 1] * x_pairs[..., 1])
    return out.reshape(B, H, N, D).astype(x.dtype)


def timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0,
                        time_factor: float = 1000.0) -> mx.array:
    """Sinusoidal timestep embedding, matching comfy.ldm.flux.layers.timestep_embedding."""
    t = time_factor * t
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb
