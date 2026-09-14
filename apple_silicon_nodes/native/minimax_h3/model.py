"""
MiniMax H3 audio-video DiT -- MLX port (in progress, minimal t2va/fl2va path).

Ports `comfy/ldm/minimax/model.py` block by block. Scope for this first pass,
per plan §5 Phase 6: no reference conditioning (`MiniMaxH3ReferenceToVideo`),
no extra guide frames (`MiniMaxH3AddGuide`), no VSA sparse-attention gate
(`gate_compress`), no PDD head bank (`FinalLayer`'s multi-head-per-timestep
variant -- not present on the real checkpoints this project targets, see
`config.py`'s `video_out.weight.shape[0] // out_features == 1` check in the
reference), batch size 1 only. Curve-form adaln only (`adaln_curve_grid` set)
-- the only variant seen on a real checkpoint so far; see config.py's module
docstring.

Weight names match the checkpoint directly (`blocks.N.attn.qkv_proj.weight`,
etc.) so `mlx.utils.tree_flatten`/`tree_unflatten` round-trips against the
real state dict without a translation table, the same convention every other
native/ family here uses.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import MiniMaxH3Config
from .rope import rms_norm as _rope_rms_norm
from .rope import rms_norm_rope_split_half


class RMSNorm(nn.Module):
    """Plain RMSNorm (`x / sqrt(mean(x^2) + eps) * weight`) -- MiniMax H3's
    checkpoint weights center around 1.0 (e.g. a real `norm1.weight` sample:
    `[1.03, 1.19, 1.0, 1.01, 1.07]`), confirming the direct-multiply
    convention `torch.nn.functional.rms_norm` uses, NOT the "(1 + scale)"
    zero-centered convention some other families in this project (Krea2) use
    for a different checkpoint's training setup."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return _rope_rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    """Fused qkv self-attention with per-head RMSNorm and, when `cos`/`sin`
    are given, MiniMax H3's partial split-half RoPE (see `rope.py`).

    `cos`/`sin` are omitted entirely (plain per-head RMSNorm, no rotation)
    for `RefinerBlock`'s text-only pre-refinement -- matching
    `comfy/ldm/minimax/model.py::RefinerBlock.forward`, which calls
    `self.attn(...)` without `rope_freqs` (defaults to `None` there). Every
    main `DiTBlock`, by contrast, always passes rope info -- there is no real
    checkpoint or call path that skips rope for the main stack, so this port
    does not special-case that combination as `RefinerBlock` does not need it.
    """

    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float, rot_dim: int = 0):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.eps = eps
        self.rot_dim = rot_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(
        self, x: mx.array, cos: mx.array | None = None, sin: mx.array | None = None
    ) -> mx.array:
        """`x`: `[S, hidden]` (batch size 1, squeezed -- matches the packed
        single-sequence layout every caller here uses). `cos`/`sin`, when
        given: `[S, rot_dim // 2]`, broadcastable against the head axis."""
        s = x.shape[0]
        qkv = self.qkv_proj(x)
        inner = self.heads * self.head_dim
        q, k, v = qkv[:, :inner], qkv[:, inner : 2 * inner], qkv[:, 2 * inner :]
        q = q.reshape(s, self.heads, self.head_dim)
        k = k.reshape(s, self.heads, self.head_dim)
        v = v.reshape(s, self.heads, self.head_dim)

        if cos is not None:
            cos_b = cos[:, None, :]
            sin_b = sin[:, None, :]
            q = rms_norm_rope_split_half(q, self.q_norm.weight, self.eps, self.rot_dim, cos_b, sin_b)
            k = rms_norm_rope_split_half(k, self.k_norm.weight, self.eps, self.rot_dim, cos_b, sin_b)
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [S, heads, head_dim] -> [1, heads, S, head_dim] for mx.fast.sdpa
        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        scale = 1.0 / (self.head_dim**0.5)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out[0].transpose(1, 0, 2).reshape(s, inner)
        return self.out_proj(out)


class MLP(nn.Module):
    """SwiGLU MLP: `fc1` projects to `2*ffn` (gate + value, first half is the
    gate), `fc2` projects the gated `ffn`-wide result back to `hidden`.
    Matches `comfy.ops.linear_input_act(fc2, fc1(x), "swiglu")` via its
    `_swiglu_eager` implementation (`comfy/ops.py:947`):
    `gate, up = x.chunk(2, dim=-1); return silu(gate) * up`."""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.fc1(x)
        ffn = h.shape[-1] // 2
        gate, up = h[..., :ffn], h[..., ffn:]
        return self.fc2(nn.silu(gate) * up)


class AdalnProj(nn.Module):
    """Projects a per-unique-timestep embedding `t_emb` (`[M, t_dim]`) into
    `expand` modulation tensors, each `[M * modalities, hidden]` -- one row
    per (unique timestep, modality) pair. `modalities` is 3 for `DiTBlock`
    (video=0, text=1, audio=2 -- fixed tags every caller here uses) and 1 for
    `FinalLayer`. `apply_silu` is False for the curve-form adaln this project
    has a real checkpoint for (`curve["apply_silu"] = not use_adaln_curves`
    in `comfy/ldm/minimax/model.py::MiniMaxH3Model.__init__` -- the plain
    `TimeEmbedder` variant would set it True, out of scope here, see
    config.py's module docstring)."""

    def __init__(self, t_dim: int, hidden: int, expand: int, modalities: int, apply_silu: bool = False):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True)

    def __call__(self, t_emb: mx.array) -> tuple[mx.array, ...]:
        x = self.linear(nn.silu(t_emb) if self.apply_silu else t_emb)
        m = x.shape[0]
        x = x.reshape(m * self.modalities, self.expand * self.hidden)
        return tuple(mx.split(x, self.expand, axis=-1))


def mod_scale_shift(h: mx.array, shift: mx.array, scale: mx.array, segments: list[tuple[int, int, int]]) -> mx.array:
    """`h[a:b] = h[a:b] * (1 + scale[row]) + shift[row]` per segment, ported
    from `_mod_scale_shift`. MLX arrays are immutable, so this rebuilds `h`
    from per-segment slices instead of the reference's in-place `.mul_`/
    `.add_` -- same math, different (functional) execution style. `segments`
    covers `h` contiguously and in order (guaranteed by `PackedLayout`), so
    concatenating the rebuilt slices reproduces `h`'s original row order."""
    pieces = []
    for a, b, row in segments:
        pieces.append(h[a:b] * (1.0 + scale[row]) + shift[row])
    return mx.concatenate(pieces, axis=0)


def mod_gate(x: mx.array, gate: mx.array, other: mx.array, segments: list[tuple[int, int, int]]) -> mx.array:
    """`x[a:b] += other[a:b] * gate[row]` per segment, ported from
    `_mod_gate` (same functional-rebuild adaptation as `mod_scale_shift`)."""
    pieces = []
    for a, b, row in segments:
        pieces.append(x[a:b] + other[a:b] * gate[row])
    return mx.concatenate(pieces, axis=0)
