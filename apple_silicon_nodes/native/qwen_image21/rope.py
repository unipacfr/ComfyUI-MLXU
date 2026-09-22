"""Standard (non-multimodal) RoPE + fused per-head RMSNorm for Qwen3-VL-8B's
text-only path (Qwen Image 2.1's T2I encoder -- no image reference
conditioning in this brick).

For text-only input, `comfy/text_encoders/llama.py`'s `precompute_freqs_cis`
builds `position_ids` as `[1, S]` (single sequential axis), which fails its
own `position_ids.shape[0] > 1` gate for Qwen3-VL's interleaved M-RoPE -- so
it always takes the plain single-axis branch:

    inv_freq[i] = 1 / theta^(2i/head_dim), i in [0, head_dim/2)
    angle[s, i] = s * inv_freq[i]
    cos/sin = cat(angle, angle).cos()/.sin()

and `apply_rope`'s rotation is the "split in half, not interleaved pairs"
convention. Ported and re-verified here against real `comfy.text_encoders.
llama.precompute_freqs_cis`/`apply_rope` (see `test_rope.py`) -- this is the
same math already verified for MiniMax H3's Qwen3-VL-32B (`native/minimax_h3/
rope.py`, `text_encoder_rope.py`), which shares the same `Qwen3_8BConfig`
base class and text-only RoPE branch; duplicated here per this project's
per-family file convention (see Krea2's own `rope.py`) rather than
cross-imported from `minimax_h3`.
"""

from __future__ import annotations

import mlx.core as mx


def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Standard RMSNorm over the last axis, matching
    `torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)`."""
    x32 = x.astype(mx.float32)
    variance = mx.mean(x32 * x32, axis=-1, keepdims=True)
    normed = x32 * mx.rsqrt(variance + eps)
    return (normed.astype(x.dtype)) * weight


def apply_rope_split_half(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """`x`: `[..., S, 2*half]`. `cos`/`sin`: `[S, half]` (or broadcastable,
    e.g. `[S, 1, half]` against `x` shaped `[S, heads, head_dim]`). Splits
    `x` into two contiguous halves (not interleaved pairs) and rotates pair
    `i = (x1[i], x2[i])` by angle `i`."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return mx.concatenate([out1, out2], axis=-1)


def rms_norm_rope_split_half(
    x: mx.array, weight: mx.array, eps: float, rot_dim: int, cos: mx.array, sin: mx.array
) -> mx.array:
    """Fused RMSNorm + partial split-half RoPE. `x`: `[..., S, head_dim]`.
    `cos`/`sin`: `[S, rot_dim//2]`, broadcastable against `x`'s leading axes."""
    x_norm = rms_norm(x, weight, eps)
    if rot_dim == 0 or rot_dim == x.shape[-1]:
        return apply_rope_split_half(x_norm, cos, sin) if rot_dim else x_norm
    rotated = apply_rope_split_half(x_norm[..., :rot_dim], cos, sin)
    return mx.concatenate([rotated, x_norm[..., rot_dim:]], axis=-1)


def qwen3_rope_cos_sin(seq_len: int, head_dim: int, theta: float) -> tuple[mx.array, mx.array]:
    """`[S, head_dim // 2]` cos/sin for sequential positions `0..seq_len-1`,
    single axis, full `head_dim` rotated -- matches
    `precompute_freqs_cis(head_dim, torch.arange(seq_len).unsqueeze(0), theta)`'s
    plain (non-interleaved-MRoPE) branch."""
    exponent = mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim
    inv_freq = 1.0 / (theta**exponent)  # [half]
    positions = mx.arange(seq_len, dtype=mx.float32)  # [S]
    angles = positions[:, None] * inv_freq[None, :]  # [S, half]
    return mx.cos(angles), mx.sin(angles)
