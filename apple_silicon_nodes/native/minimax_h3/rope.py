"""MiniMax H3's fused per-head RMSNorm + partial split-half RoPE.

`comfy/ldm/minimax/model.py::Attention.forward` calls a compiled kernel
(`comfy.quant_ops.ck.rms_rope_split_half_`, from the `comfy_kitchen` PyPI
package) for this step -- there is no plain-Python version of that kernel in
`comfy/`'s own source tree to read. `comfy_kitchen` does ship a pure-PyTorch
"eager" backend for it though (`comfy_kitchen/backends/eager/rope.py`,
installed in the ComfyUI venv on this machine at
`/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/.venv/.../comfy_kitchen/`), which is
the actual real reference this module is ported from and verified against
(see `tests/native/minimax_h3/test_rope.py`) -- not a formula reverse-engineered
from the call site's comments alone, per this project's rule for any quant/math
divergence.

The math (`comfy_kitchen.backends.eager.rope._rms_rope1` with
`split_half=True`), for one head's `head_dim`-wide vector `x`:

    x_norm = rms_norm(x, weight=scale, eps)             # over head_dim, standard
    x1, x2 = x_norm[:rot_dim//2], x_norm[rot_dim//2:rot_dim]   # split in half, not interleaved
    out1 = x1 * cos - x2 * sin                            # per-pair rotation, pair i = (x1[i], x2[i])
    out2 = x1 * sin + x2 * cos
    return concat(out1, out2, x_norm[rot_dim:])            # dims >= rot_dim pass through unrotated

MiniMax H3 uses `rot_dim=96` of `head_dim=128` (`Attention.forward`:
`rot_dim = rope_freqs.shape[-3] * 2`, and `rope_rotation_table`'s `half=48`
comes from 3 RoPE axes (t, h, w) x 16 frequencies each = 48 unique angles) --
the last 32 channels of every head are never rotated.
"""

from __future__ import annotations

import mlx.core as mx


def rope_freqs(position_ids: mx.array, inv_freq: mx.array) -> mx.array:
    """Port of `MiniMaxH3Model.rope_freqs`: `[S, 3]` (t, h, w) position ids and
    `[F]` per-axis inverse frequencies -> `[S, 6*F]` angles (3 axes, each
    duplicated: `cat(t,h,w,t,h,w)`, matching `cat(half, half)` in the
    reference -- the duplication is exactly what `rope_rotation_table` below
    then discards half of, so producing it is redundant work kept only to
    mirror the reference 1:1 for anyone diffing against it). `position_ids`
    is cast to float32 here even if given as float64 (`PackedLayout` builds
    it at float64 for precision, matching the reference), mirroring
    `MiniMaxH3Model.rope_freqs`'s own `pos = position_ids.to(torch.float32)`.
    The cast itself must run on the CPU stream: MLX rejects float64 (even
    just reading it to cast away) on the GPU/Metal stream entirely (see
    layout.py's module docstring for where this was first found)."""
    if position_ids.dtype == mx.float64:
        with mx.stream(mx.cpu):
            position_ids = position_ids.astype(mx.float32)
    else:
        position_ids = position_ids.astype(mx.float32)
    per_axis = position_ids[:, :, None] * inv_freq[None, None, :]  # [S, 3, F]
    half = mx.concatenate([per_axis[:, 0], per_axis[:, 1], per_axis[:, 2]], axis=-1)  # [S, 3F]
    return mx.concatenate([half, half], axis=-1)  # [S, 6F]


def rope_cos_sin(angles: mx.array) -> tuple[mx.array, mx.array]:
    """`[S, 2*half]` duplicated angles (as `rope_freqs` produces) -> `cos`,
    `sin`, each `[S, half]` -- the first half of the duplicated angles is the
    only part `rope_rotation_table` actually uses (`ang = angles[:, :half]`),
    so slice rather than build the PyTorch reference's redundant `[..., 2, 2]`
    rotation-matrix tensor; the numbers this produces are identical."""
    half = angles.shape[-1] // 2
    ang = angles[:, :half]
    return mx.cos(ang), mx.sin(ang)


def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Standard RMSNorm over the last axis, matching
    `torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)`."""
    x32 = x.astype(mx.float32)
    variance = mx.mean(x32 * x32, axis=-1, keepdims=True)
    normed = x32 * mx.rsqrt(variance + eps)
    return (normed.astype(x.dtype)) * weight


def apply_rope_split_half(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """`x`: `[..., S, 2*half]` (already restricted to the rotated dims).
    `cos`/`sin`: `[S, half]`, broadcast over every leading axis (batch, heads).
    Splits `x` into two contiguous halves (not interleaved pairs) and rotates
    pair `i = (x1[i], x2[i])` by angle `i`, matching `apply_rope_split_half1`
    in the real `comfy_kitchen` eager reference exactly."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return mx.concatenate([out1, out2], axis=-1)


def rms_norm_rope_split_half(
    x: mx.array, weight: mx.array, eps: float, rot_dim: int, cos: mx.array, sin: mx.array
) -> mx.array:
    """Fused RMSNorm + partial split-half RoPE, matching
    `comfy_kitchen.backends.eager.rope._rms_rope1(..., split_half=True,
    rot_dim=rot_dim)`. `x`: `[..., S, head_dim]`. `cos`/`sin`: `[S, rot_dim//2]`,
    broadcastable against `x`'s leading axes (e.g. `x` shaped `[S, heads,
    head_dim]` broadcasts against `cos`/`sin` shaped `[S, 1, rot_dim//2]`)."""
    x_norm = rms_norm(x, weight, eps)
    if rot_dim == 0 or rot_dim == x.shape[-1]:
        rotated = apply_rope_split_half(x_norm, cos, sin) if rot_dim else x_norm
        return rotated
    rotated = apply_rope_split_half(x_norm[..., :rot_dim], cos, sin)
    return mx.concatenate([rotated, x_norm[..., rot_dim:]], axis=-1)
