"""
Native MLX Transformer for FLUX.1
==================================
A FLUX.1 transformer implementation in pure MLX, matching the reference
architecture in ComfyUI's `comfy/ldm/flux/{model,layers,math}.py`.

This module provides:
  - FluxConfig: model configuration (from config submodule)
  - FluxTransformer: the full transformer with double_blocks + single_blocks
  - load_transformer: loads safetensors checkpoints into MLX arrays

Architecture (FLUX.1 dev):
  - 19 double transformer blocks (img + txt joint self-attention)
  - 38 single transformer blocks (joint img+txt attention + MLP, fused linear1/linear2)
  - Hidden dim: 3072, MLP dim: 12288 (mlp_ratio=4)
  - 24 attention heads (head_dim = 128)
  - 3-axis RoPE (frame, height, width), axes_dim=[16,56,56], theta=10000
  - Per-head QK RMSNorm
  - adaLN-style modulation driven by timestep+guidance+pooled conditioning

Design decisions:
  - Use mx.Linear for all linear layers (built-in, efficient on Metal)
  - Lazy evaluation with strategic mx.eval() for bridge points
  - float16 default, bfloat16 supported
"""

from __future__ import annotations

__all__ = ["FluxConfig", "FluxTransformer", "load_transformer"]

# ── Krea2 re-exports ───────────────────────────────────────────────────
# Krea2 components are in native/krea2/; re-export for convenience.

from .krea2 import (  # noqa: E402
    Krea2Config,
    SingleStreamDiT,
    DoubleSharedModulation,
    SimpleModulation,
    RMSNorm as Krea2RMSNorm,
    QKNorm as Krea2QKNorm,
    SingleStreamBlock as Krea2SingleStreamBlock,
    Attention as Krea2Attention,
    SwiGLU,
    EmbedND as Krea2EmbedND,
    apply_rope_3d,
    compute_rope_3d,
    load_krea2_transformer,
    normalize_krea2_keys,
    map_krea2_to_native,
)

__all__ += [
    "Krea2Config",
    "SingleStreamDiT",
    "DoubleSharedModulation",
    "SimpleModulation",
    "Krea2RMSNorm",
    "Krea2QKNorm",
    "Krea2SingleStreamBlock",
    "Krea2Attention",
    "SwiGLU",
    "Krea2EmbedND",
    "apply_rope_3d",
    "compute_rope_3d",
    "load_krea2_transformer",
    "normalize_krea2_keys",
    "map_krea2_to_native",
]

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Re-export config from submodule
from .config import FluxConfig  # noqa: E402
from .weight_map import normalize_flux_keys, map_flux_to_native, detect_flux_in_channels  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────

HIDDEN_DIM = 3072
MLP_DIM = 12288
NUM_HEADS = 24
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 128
AXES_DIM = (16, 56, 56)  # frame, height, width — sums to HEAD_DIM
ROPE_THETA = 10000.0
CONTEXT_IN_DIM = 4096  # T5-XXL
VEC_IN_DIM = 768  # pooled CLIP-L


# ── RoPE (3-axis, paired-interleave convention) ─────────────────────────

def rope_freqs(pos: mx.array, dim: int, theta: float) -> mx.array:
    """Compute a [N, dim/2, 2, 2] rotation-matrix RoPE table for one axis.

    Matches comfy.ldm.flux.math.rope: paired-interleave rotation (adjacent
    dim pairs), NOT the rotate-half/LLaMA convention.

    Args:
        pos: [N] position indices for this axis.
        dim: axis dimension (must be even).
        theta: frequency base for this axis.

    Returns:
        [N, dim/2, 2, 2] rotation matrices — out[n, i] = [[cos, -sin], [sin, cos]].
    """
    assert dim % 2 == 0
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta ** scale)  # [dim/2]
    out = pos.astype(mx.float32)[:, None] * omega[None, :]  # [N, dim/2]
    cos, sin = mx.cos(out), mx.sin(out)
    # [N, dim/2, 2, 2]: [[cos, -sin], [sin, cos]]
    return mx.stack([cos, -sin, sin, cos], axis=-1).reshape(*out.shape, 2, 2)


def embed_nd(ids: mx.array, axes_dim: tuple[int, ...], theta: float) -> mx.array:
    """3-axis RoPE embedding table.

    Args:
        ids: [N, num_axes] position indices, one column per axis.
        axes_dim: per-axis dimension (sums to head_dim).
        theta: frequency base.

    Returns:
        [N, head_dim/2, 2, 2] rotation matrices, concatenated across axes.
    """
    parts = [rope_freqs(ids[:, i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return mx.concatenate(parts, axis=-3)


def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply paired-interleave RoPE rotation to Q or K.

    Args:
        x: [B, H, N, D] query or key tensor.
        freqs: [N, D/2, 2, 2] rotation matrices from embed_nd.

    Returns:
        [B, H, N, D] rotated tensor.
    """
    B, H, N, D = x.shape
    x_pairs = x.reshape(B, H, N, D // 2, 1, 2)  # [B,H,N,D/2,1,2]
    f = freqs[None, None]  # [1,1,N,D/2,2,2]
    out = (f[..., 0] * x_pairs[..., 0]) + (f[..., 1] * x_pairs[..., 1])  # [B,H,N,D/2,2]
    return out.reshape(B, H, N, D)


class QKNorm(nn.Module):
    """Per-head RMSNorm applied to Q and K after the head split."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

    def __call__(self, q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
        return self.query_norm(q), self.key_norm(k)


# ── Modulation (adaLN-style, driven by conditioning vector) ────────────

class Modulation(nn.Module):
    """adaLN-style modulation: SiLU(vec) -> Linear -> chunk into ModulationOut(s).

    double=True:  returns (mod1, mod2), each (shift, scale, gate) — 6 total params.
    double=False: returns (mod, None), (shift, scale, gate) — 3 total params.
    """

    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim)

    def __call__(self, vec: mx.array) -> tuple[tuple[mx.array, mx.array, mx.array],
                                                tuple[mx.array, mx.array, mx.array] | None]:
        """Args: vec [B, dim] conditioning vector.

        Returns: (mod1, mod2) where each is (shift, scale, gate) of shape [B, 1, dim],
        or (mod1, None) when not double.
        """
        out = self.lin(nn.silu(vec))  # [B, multiplier*dim]
        parts = mx.split(out, self.multiplier, axis=-1)
        parts = [p[:, None, :] for p in parts]  # [B, 1, dim] for broadcast over [B, N, dim]
        mod1 = (parts[0], parts[1], parts[2])
        mod2 = (parts[3], parts[4], parts[5]) if self.is_double else None
        return mod1, mod2


def apply_mod(x: mx.array, scale: mx.array, shift: mx.array | None) -> mx.array:
    """x * (1 + scale) [+ shift]. scale/shift are [B, 1, D], x is [B, N, D]."""
    out = x * (1 + scale)
    if shift is not None:
        out = out + shift
    return out


# ── Joint attention (shared by DoubleStreamBlock and SingleStreamBlock) ─

def joint_attention(
    q: mx.array, k: mx.array, v: mx.array, freqs: mx.array,
) -> mx.array:
    """Scaled dot-product attention with RoPE.

    Args:
        q, k, v: [B, H, N, head_dim]
        freqs: [N, head_dim/2, 2, 2] RoPE table

    Returns:
        [B, N, H*head_dim] attention output (heads merged back).
    """
    B, H, N, Dh = q.shape
    q = apply_rope(q, freqs)
    k = apply_rope(k, freqs)

    scale = 1.0 / math.sqrt(Dh)
    attn = (q * scale) @ k.transpose(0, 1, 3, 2)  # [B, H, N, N+]
    attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
    out = attn @ v  # [B, H, N, Dh]
    return out.transpose(0, 2, 1, 3).reshape(B, N, H * Dh)


# ── Double Block (joint img+txt self-attention) ─────────────────────────

class SelfAttentionProj(nn.Module):
    """QKV + QKNorm + output projection, without running attention itself.

    Matches comfy.ldm.flux.layers.SelfAttention: the block computes q/k/v via
    this module's `qkv`/`norm`, runs joint attention externally, then calls
    `proj` on the merged result.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)


class DoubleBlock(nn.Module):
    """FLUX.1 double transformer block.

    Image and text tokens each get their own modulation + QKV, but attention
    is JOINT: img and txt Q/K/V are concatenated (txt first) into one
    attention call, then split back for the separate MLP branches.
    """

    def __init__(self, dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.img_mod = Modulation(dim, double=True)
        self.img_norm1 = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.img_attn = SelfAttentionProj(dim, num_heads, qkv_bias)
        self.img_norm2 = nn.LayerNorm(dim, affine=False, eps=1e-6)

        self.txt_mod = Modulation(dim, double=True)
        self.txt_norm1 = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.txt_attn = SelfAttentionProj(dim, num_heads, qkv_bias)
        self.txt_norm2 = nn.LayerNorm(dim, affine=False, eps=1e-6)

        hidden_mlp = int(dim * mlp_ratio)
        self.img_mlp_0 = nn.Linear(dim, hidden_mlp)
        self.img_mlp_2 = nn.Linear(hidden_mlp, dim)
        self.txt_mlp_0 = nn.Linear(dim, hidden_mlp)
        self.txt_mlp_2 = nn.Linear(hidden_mlp, dim)

    def _qkv_heads(self, attn: SelfAttentionProj, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        B, N, D = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = attn.norm(q, k)
        return q, k, v

    def __call__(
        self,
        img: mx.array,
        txt: mx.array,
        vec: mx.array,
        freqs: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Args:
            img: [B, N_img, D] image tokens
            txt: [B, N_txt, D] text tokens
            vec: [B, D] conditioning vector (timestep + guidance + pooled)
            freqs: [N_txt+N_img, head_dim/2, 2, 2] RoPE table (txt positions first)

        Returns:
            (img_out, txt_out)
        """
        (img_shift1, img_scale1, img_gate1), (img_shift2, img_scale2, img_gate2) = self.img_mod(vec)
        (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = self.txt_mod(vec)

        img_modulated = apply_mod(self.img_norm1(img), img_scale1, img_shift1)
        img_q, img_k, img_v = self._qkv_heads(self.img_attn, img_modulated)

        txt_modulated = apply_mod(self.txt_norm1(txt), txt_scale1, txt_shift1)
        txt_q, txt_k, txt_v = self._qkv_heads(self.txt_attn, txt_modulated)

        # Joint attention: txt tokens first, then img tokens (matches RoPE id order)
        q = mx.concatenate([txt_q, img_q], axis=2)
        k = mx.concatenate([txt_k, img_k], axis=2)
        v = mx.concatenate([txt_v, img_v], axis=2)

        attn = joint_attention(q, k, v, freqs)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        # Attention residual (gate1), no shift — matches apply_mod(x, gate, None)
        img = img + self.img_attn.proj(img_attn) * img_gate1
        txt = txt + self.txt_attn.proj(txt_attn) * txt_gate1

        # MLP branch: fresh norm2 + modulation2 on the POST-attention residual
        img_mlp_in = apply_mod(self.img_norm2(img), img_scale2, img_shift2)
        img_mlp = self.img_mlp_2(nn.gelu_approx(self.img_mlp_0(img_mlp_in)))
        img = img + img_mlp * img_gate2

        txt_mlp_in = apply_mod(self.txt_norm2(txt), txt_scale2, txt_shift2)
        txt_mlp = self.txt_mlp_2(nn.gelu_approx(self.txt_mlp_0(txt_mlp_in)))
        txt = txt + txt_mlp * txt_gate2

        return img, txt


# ── Single Block (fused joint attention + MLP) ──────────────────────────

class SingleBlock(nn.Module):
    """FLUX.1 single transformer block.

    Operates on the concatenated [txt, img] sequence. QKV and the MLP's
    first projection are fused into one `linear1`; the attention output and
    MLP activation are fused back together through one `linear2`. Modulation
    has only 3 params (shift, scale, gate) — no separate second modulation.
    """

    def __init__(self, dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.hidden_mlp = int(dim * mlp_ratio)

        self.modulation = Modulation(dim, double=False)
        self.pre_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.norm = QKNorm(self.head_dim)

        # qkv (3*dim) + mlp_in (hidden_mlp) fused
        self.linear1 = nn.Linear(dim, dim * 3 + self.hidden_mlp)
        # attn_out (dim) + mlp_out (hidden_mlp) fused back to dim
        self.linear2 = nn.Linear(dim + self.hidden_mlp, dim)

    def __call__(
        self,
        x: mx.array,
        vec: mx.array,
        freqs: mx.array,
    ) -> mx.array:
        """
        Args:
            x: [B, N, D] concatenated [txt, img] tokens
            vec: [B, D] conditioning vector
            freqs: [N, head_dim/2, 2, 2] RoPE table

        Returns:
            [B, N, D] output
        """
        (shift, scale, gate), _ = self.modulation(vec)

        B, N, D = x.shape
        modulated = apply_mod(self.pre_norm(x), scale, shift)
        fused = self.linear1(modulated)
        qkv, mlp_in = mx.split(fused, [3 * self.dim], axis=-1)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.norm(q, k)

        attn = joint_attention(q, k, v, freqs)
        mlp_act = nn.gelu_approx(mlp_in)
        out = self.linear2(mx.concatenate([attn, mlp_act], axis=-1))
        x = x + out * gate

        return x


# ── LastLayer ────────────────────────────────────────────────────────────

class LastLayer(nn.Module):
    """Final output layer: adaLN modulation -> LayerNorm -> Linear."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )

    def __call__(self, x: mx.array, vec: mx.array) -> mx.array:
        shift, scale = mx.split(self.adaLN_modulation(vec), 2, axis=-1)
        x = apply_mod(self.norm_final(x), scale[:, None, :], shift[:, None, :])
        return self.linear(x)


# ── MLPEmbedder (time_in / vector_in / guidance_in) ─────────────────────

class MLPEmbedder(nn.Module):
    """in_dim -> hidden_dim -> SiLU -> hidden_dim, matching FLUX's time/vector/guidance embedders."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.silu(self.in_layer(x)))


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


# ── Main Transformer ──────────────────────────────────────────────────

class FluxTransformer(nn.Module):
    """Complete FLUX.1 transformer.

    Architecture:
      img_in / txt_in  ->  19x DoubleBlock  ->  38x SingleBlock  ->  final_layer
    """

    def __init__(self, config: FluxConfig | None = None):
        super().__init__()
        config = config or FluxConfig()
        self.config = config
        self.dtype = config.mlx_dtype

        # Input projections
        self.img_in = nn.Linear(config.in_channels, HIDDEN_DIM)
        self.txt_in = nn.Linear(CONTEXT_IN_DIM, HIDDEN_DIM)

        # Time / vector / guidance embedding (MLPEmbedder: in -> hidden -> SiLU -> hidden)
        self.time_in = MLPEmbedder(256, HIDDEN_DIM)
        self.vector_in = MLPEmbedder(VEC_IN_DIM, HIDDEN_DIM)
        self.guidance_in = MLPEmbedder(256, HIDDEN_DIM) if config.guidance_embed else None

        # Transformer blocks
        self.double_blocks = [
            DoubleBlock() for _ in range(config.num_double_blocks)
        ]
        self.single_blocks = [
            SingleBlock() for _ in range(config.num_single_blocks)
        ]

        # Output
        self.final_layer = LastLayer(HIDDEN_DIM, 64)

    def get_rope(
        self,
        img_h: int,
        img_w: int,
        txt_len: int,
        ref_grids: list[tuple[int, int]] | None = None,
    ) -> mx.array:
        """Compute the 3-axis RoPE table for a [txt, img, ref...] sequence.

        Matches comfy's `process_img`/`img_ids`: each image token gets its own
        (row, col) grid coordinate on axes 1/2, not a flat sequential index
        reused on both axes. Text tokens sit at position 0 on every axis
        (comfy's `txt_ids` are zeros for standard FLUX.1, since `txt_ids_dims`
        is empty).

        Kontext reference tokens (`ref_grids`) are appended after the target
        image tokens, matching `comfy/ldm/flux/model.py::_forward`'s default
        "offset" `ref_latents_method`: each reference gets its own row/col
        grid (same coordinate space as the target) but is distinguished by an
        increasing index on axis 0 (1, 2, 3, ...) — the target image and text
        both sit at axis-0 index 0.

        Args:
            img_h: image token grid height (in patches).
            img_w: image token grid width (in patches).
            txt_len: number of text tokens.
            ref_grids: optional list of (h, w) grid sizes, one per Kontext
                reference image, in the same order they'll be concatenated
                to `img` by the caller.

        Returns:
            [txt_len + img_h*img_w + sum(ref_h*ref_w), head_dim/2, 2, 2] table.
        """
        ref_grids = ref_grids or []
        img_len = img_h * img_w
        ref_lens = [rh * rw for rh, rw in ref_grids]
        total = txt_len + img_len + sum(ref_lens)
        ids = mx.zeros((total, 3), dtype=mx.float32)
        if img_len > 0:
            rows = mx.arange(img_h, dtype=mx.float32)[:, None]
            cols = mx.arange(img_w, dtype=mx.float32)[None, :]
            rows = mx.broadcast_to(rows, (img_h, img_w)).reshape(-1)
            cols = mx.broadcast_to(cols, (img_h, img_w)).reshape(-1)
            ids[txt_len:txt_len + img_len, 1] = rows
            ids[txt_len:txt_len + img_len, 2] = cols

        offset = txt_len + img_len
        for ref_idx, (rh, rw) in enumerate(ref_grids, start=1):
            rlen = rh * rw
            if rlen > 0:
                rrows = mx.arange(rh, dtype=mx.float32)[:, None]
                rcols = mx.arange(rw, dtype=mx.float32)[None, :]
                rrows = mx.broadcast_to(rrows, (rh, rw)).reshape(-1)
                rcols = mx.broadcast_to(rcols, (rh, rw)).reshape(-1)
                ids[offset:offset + rlen, 0] = float(ref_idx)
                ids[offset:offset + rlen, 1] = rrows
                ids[offset:offset + rlen, 2] = rcols
            offset += rlen
        return embed_nd(ids, AXES_DIM, ROPE_THETA)

    def time_embed(self, t: mx.array, guidance: mx.array | None = None,
                   pooled: mx.array | None = None) -> mx.array:
        """Compute the [B, hidden_dim] conditioning vector: time + guidance + pooled."""
        vec = self.time_in(timestep_embedding(t, 256).astype(self.dtype))

        if guidance is not None and self.guidance_in is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).astype(self.dtype))

        if pooled is not None:
            vec = vec + self.vector_in(pooled.astype(self.dtype))

        return vec

    def __call__(
        self,
        img: mx.array,       # [B, N_img, 64] packed image patches
        txt: mx.array,       # [B, N_txt, 4096] T5 embeddings
        t: mx.array,         # [B] timestep
        guidance: mx.array | None = None,  # [B] guidance scale
        pooled: mx.array | None = None,    # [B, 768] pooled CLIP
        rope: mx.array | None = None,      # precomputed 3-axis RoPE table
        control: dict[str, list[mx.array | None]] | None = None,
        ref_img: mx.array | None = None,   # packed Kontext reference tokens [B, N_ref, 64]
    ) -> mx.array:
        """Forward pass.

        Args:
            img: packed image tokens [B, N_img, 64]
            txt: text embeddings [B, N_txt, 4096]
            t: timesteps [B]
            guidance: guidance scale [B] (FLUX dev)
            pooled: pooled CLIP output [B, 768]
            rope: precomputed RoPE table [N_txt+N_img+N_ref, head_dim/2, 2, 2]
            control: optional ControlNet residuals, {"input": [...], "output": [...]},
                     one entry per double/single block (None entries are skipped).
                     Matches comfy.ldm.flux.model.Flux's `control` dict convention.
            ref_img: optional packed Kontext reference tokens [B, N_ref, 64],
                     same raw patch space as `img` (pre-`img_in`). Concatenated
                     onto `img` before the input projection, matching comfy's
                     `_forward` (`img = torch.cat([img, kontext], dim=1)`), and
                     dropped again from the output before `final_layer`
                     (`out = out[:, :img_tokens]`).

        Returns:
            [B, N_img, 64] noise prediction
        """
        img_tokens = img.shape[1]
        if ref_img is not None:
            img = mx.concatenate([img, ref_img], axis=1)

        # Input projections
        img = self.img_in(img.astype(self.dtype))
        txt = self.txt_in(txt.astype(self.dtype))

        # Conditioning vector drives modulation in every block
        vec = self.time_embed(t, guidance=guidance, pooled=pooled)

        if rope is None:
            raise ValueError(
                "ASDX: FluxTransformer.__call__ requires a precomputed `rope` table. "
                "get_rope(img_h, img_w, txt_len) needs the actual image token grid "
                "shape, which a flat token count (img.shape[1]) cannot provide; "
                "callers must precompute it via self.get_rope(...) with the real "
                "height/width before calling predict()/__call__."
            )

        control_input = control.get("input") if control is not None else None
        for i, block in enumerate(self.double_blocks):
            img, txt = block(img, txt, vec, rope)
            if control_input is not None and i < len(control_input):
                add = control_input[i]
                if add is not None:
                    img = img.at[:, :add.shape[1]].add(add)

        # Concatenate for single blocks: txt first, matching RoPE id order
        x = mx.concatenate([txt, img], axis=1)

        control_output = control.get("output") if control is not None else None
        for i, block in enumerate(self.single_blocks):
            x = block(x, vec, rope)
            if control_output is not None and i < len(control_output):
                add = control_output[i]
                if add is not None:
                    start = txt.shape[1]
                    x = x.at[:, start:start + add.shape[1]].add(add)

        # Split back: image tokens are after the text tokens. Kontext reference
        # tokens (if any) trail the target image tokens — drop them here,
        # matching comfy's `out = out[:, :img_tokens]`.
        img_out = x[:, txt.shape[1]:txt.shape[1] + img_tokens, :]

        return self.final_layer(img_out, vec)

    def predict(
        self,
        img: mx.array,
        txt: mx.array,
        timestep: float,
        guidance: float = 3.5,
        pooled: mx.array | None = None,
        rope: mx.array | None = None,
        control: dict[str, list[mx.array | None]] | None = None,
        ref_img: mx.array | None = None,
    ) -> mx.array:
        """Convenience method for one denoising step.

        Args:
            img: [B, N, 64] current latent
            txt: [B, T, 4096] text embedding
            timestep: float sigma/timestep value
            guidance: guidance scale
            pooled: [B, 768] pooled CLIP
            rope: optional precomputed rope
            control: optional ControlNet residuals (see __call__)
            ref_img: optional packed Kontext reference tokens (see __call__)

        Returns:
            [B, N, 64] predicted noise
        """
        t = mx.array([timestep], dtype=mx.float32)
        g = mx.array([guidance], dtype=mx.float32) if guidance is not None else None
        return self(img, txt, t, guidance=g, pooled=pooled, rope=rope, control=control,
                    ref_img=ref_img)


# ── Loader ────────────────────────────────────────────────────────────

_HADAMARD_CACHE: dict[int, "torch.Tensor"] = {}


def _build_hadamard(size: int) -> "torch.Tensor":
    """Normalized regular (power-of-4) Hadamard matrix, ported 1:1 from
    comfy_kitchen.tensor.int8_utils._build_hadamard -- required to undo
    ComfyUI's offline 'ConvRot' weight rotation bit-for-bit rather than
    approximating it."""
    import math
    import torch

    if size in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[size]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"ASDX: ConvRot Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    h_normalized = h / (size ** 0.5)
    _HADAMARD_CACHE[size] = h_normalized
    return h_normalized


def _rotate_weight_groups(weight: "torch.Tensor", h: "torch.Tensor", group_size: int) -> "torch.Tensor":
    """W_rot = W @ H_block^T, applied per group along the input-feature axis
    (comfy_kitchen.tensor.int8_utils._rotate_weight). H is symmetric and
    involutory (H @ H == I), so calling this a second time with the same H
    undoes the original quantization-time rotation."""
    import torch

    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(
            f"ASDX: ConvRot group_size {group_size} does not divide "
            f"in_features {in_f}"
        )
    n_groups = in_f // group_size
    weight_grouped = weight.reshape(out_f, n_groups, group_size)
    rotated = torch.matmul(weight_grouped, h.T)
    return rotated.reshape(out_f, in_f)


def _dequantize_comfy_quant_int8(
    state: dict[str, "torch.Tensor"], filename: str
) -> dict[str, "torch.Tensor"]:
    """Replace every ComfyUI int8 '.weight'/'.weight_scale'/'.comfy_quant'
    triplet with a dense float32 '.weight', undoing per-row INT8
    quantization and (when present) the offline Hadamard 'ConvRot' rotation
    -- ported from comfy_kitchen.backends.eager.quantization.
    dequantize_int8_convrot_weight / dequantize_int8_simple. This project
    runs no int8 GEMM kernels, so materializing the plain dense weight once
    at load time lets every existing architecture file stay untouched.

    Verified against a real checkpoint (darkBeast30BF16INT8_darkBeast330.
    safetensors: 224 uniform format="int8_tensorwise"/convrot=true/
    group_size=256 markers). Any marker deviating from that verified shape
    raises rather than guessing, per this project's fail-closed doctrine.
    """
    import json
    import torch

    marker_keys = [k for k in state if k.endswith(".comfy_quant")]
    if not marker_keys:
        return state

    new_state = dict(state)
    for marker_key in marker_keys:
        prefix = marker_key[: -len(".comfy_quant")]
        weight_key = f"{prefix}.weight"
        scale_key = f"{prefix}.weight_scale"
        if weight_key not in state or scale_key not in state:
            raise ValueError(
                f"ASDX: '{filename}': '.comfy_quant' marker at '{marker_key}' "
                f"has no matching '.weight'/'.weight_scale' pair -- refusing "
                f"to guess the quantization layout."
            )
        blob = json.loads(state[marker_key].numpy().tobytes())
        if blob.get("format") != "int8_tensorwise":
            raise NotImplementedError(
                f"ASDX: '{filename}': '{marker_key}' declares comfy_quant "
                f"format {blob.get('format')!r} -- only 'int8_tensorwise' is "
                f"implemented."
            )
        q = state[weight_key]
        scale = state[scale_key]
        if blob.get("convrot"):
            group_size = blob.get("convrot_groupsize")
            if not isinstance(group_size, int) or group_size <= 0:
                raise ValueError(
                    f"ASDX: '{filename}': '{marker_key}' has convrot=true "
                    f"with an invalid convrot_groupsize ({group_size!r})."
                )
            h = _build_hadamard(group_size)
            dense = q.to(torch.float32) * scale.to(torch.float32)
            dense = _rotate_weight_groups(dense, h, group_size)
        else:
            dense = q.to(torch.float32) * scale.to(torch.float32)
        new_state[weight_key] = dense
        del new_state[marker_key]
        del new_state[scale_key]
    return new_state


def _dequantize_fp8_scaled(
    state: dict[str, "torch.Tensor"], filename: str
) -> dict[str, "torch.Tensor"]:
    """Replace every per-tensor-scaled FP8 '.weight' with a dense float32
    '.weight', undoing ComfyUI's "scaled fp8" convention -- ported from
    `comfy_kitchen.backends.eager.quantization.dequantize_per_tensor_fp8`
    (`dq = qdata.to(dtype) * scale.to(dtype)`, the same math the real
    `TensorCoreFP8Layout.dequantize()` calls). `.input_scale` (activation-side
    scale for a fused FP8xFP8 GEMM) is irrelevant once we materialize a dense
    weight for a plain float matmul, so it's dropped along with the weight
    scale rather than applied.

    Verified against a real checkpoint (`flux2_dev_fp8mixed.safetensors`:
    128 F8_E4M3 weights, each with exactly one scalar `.weight_scale` +
    `.input_scale` pair, no `.comfy_quant` marker at all -- see
    `weight_format.py` module docstring for the convention this predates).

    `classify_quant_format` verdicts at whole-file granularity: one FP8
    weight anywhere with a scale companion routes the entire file through
    this function. A bundled "AIO" checkpoint can legitimately mix
    conventions PER SUBMODULE though (confirmed on
    `trueRealVisionFlux_v276AIO.safetensors`: the T5 text encoder's 169 FP8
    weights each carry a `.scale_weight`, the FLUX transformer's 780 FP8
    weights carry none at all) -- so a weight with no scale companion here
    is left untouched rather than treated as an error: that is exactly the
    FP8_NAIVE convention for that one weight, and `_load_safetensors`'s
    final upcast loop handles it the same way a purely FP8_NAIVE file would.
    A scale that isn't a 0-d/1-element tensor still raises rather than
    guessing (see `_dequantize_comfy_quant_int8` above for the same pattern).
    """
    import torch

    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    fp8_weight_keys = [
        k for k, v in state.items()
        if v.dtype in fp8_dtypes and k.endswith(".weight")
    ]
    if not fp8_weight_keys:
        return state

    new_state = dict(state)
    for weight_key in fp8_weight_keys:
        prefix = weight_key[: -len(".weight")]
        scale_key = None
        for candidate in (f"{prefix}.weight_scale", f"{prefix}.scale_weight"):
            if candidate in state:
                scale_key = candidate
                break
        if scale_key is None:
            # No companion for THIS weight specifically -- FP8_NAIVE for this
            # tensor, not a guess (see docstring). Leave it as raw FP8; the
            # upcast loop in `_load_safetensors` converts it to float32.
            continue
        scale = state[scale_key]
        if scale.numel() != 1:
            raise ValueError(
                f"ASDX: '{filename}': '{scale_key}' has {scale.numel()} elements "
                f"-- only per-tensor (scalar) FP8 scaling is implemented, not "
                f"per-channel/per-block."
            )
        q = state[weight_key]
        dense = q.to(torch.float32) * scale.to(torch.float32)
        new_state[weight_key] = dense
        del new_state[scale_key]
        input_scale_key = f"{prefix}.input_scale"
        if input_scale_key in new_state:
            del new_state[input_scale_key]
    return new_state


def _check_weight_match(matched: int, total: int, label: str, path: str | Path) -> None:
    """Raise if a checkpoint load matched zero of the native model's params.

    A legitimate checkpoint variant can still miss a real chunk of keys (a
    bias-free-trained Krea2 Turbo checkpoint matches only 430/686 -- see
    `krea2/model.py::load_krea2_transformer`), so this only guards the
    unambiguous failure: zero matches means every key was a structural
    mismatch, e.g. a checkpoint routed to the wrong model family by
    `loader.py::_detect_model_type`. Left unchecked, the model runs on
    randomly-initialized weights and silently produces garbage with no
    error -- only a printed line most users won't notice.
    """
    if matched == 0:
        raise RuntimeError(
            f"ASDX: {label} matched 0/{total} params from checkpoint "
            f"'{Path(path).name}' -- its keys don't match the expected "
            f"architecture at all. It was likely misdetected as the wrong "
            f"model type; check loader.py::_detect_model_type."
        )


def _load_safetensors(path: str | Path) -> dict[str, mx.array]:
    """Load a safetensors file into MLX arrays.

    Uses safetensors.torch to support BF16 (numpy backend doesn't).
    BF16 and "naive" FP8 (F8_E4M3/F8_E5M2, e.g. ComfyUI's own
    UNETLoader/ModelSave weight_dtype="fp8_e4m3fn" -- a straight per-tensor
    cast with no companion scale, unlike ComfyUI's separate "scaled fp8"
    format which stores a scale_weight/input_scale per layer and needs
    dequantizing by multiplication) are upcast to float32 before conversion
    to MLX; the caller casts back down to the requested precision afterward.
    INT8_TENSORWISE checkpoints (ComfyUI's per-row int8, optionally with an
    offline Hadamard 'ConvRot' rotation) are dequantized to dense float32
    weights before that same upcast path, via
    `_dequantize_comfy_quant_int8`. FP8_SCALED checkpoints (per-tensor
    `.weight_scale`/`.scale_weight`) are likewise dequantized first, via
    `_dequantize_fp8_scaled`.

    Single entry point for all 5 model families (flux1 directly, krea2/
    sdxl/zimage/flux2 via `from .. import _load_safetensors`), so the
    integrity check and quant-format gate below cover every family from
    one place instead of being duplicated per family.
    """
    from .safetensors_header import read_safetensors_header, verify_safetensors_integrity
    from .weight_format import QuantFormat, Unrecognized, classify_quant_format

    header = read_safetensors_header(path)
    verify_safetensors_integrity(path, header)

    quant_format = classify_quant_format(header)
    if isinstance(quant_format, Unrecognized):
        raise ValueError(
            f"ASDX: '{Path(path).name}' uses an unrecognized weight format "
            f"({quant_format.reason}). Refusing to load rather than guess -- "
            f"this project has hit silent-corruption bugs from optimistic "
            f"fallbacks before."
        )
    if quant_format == QuantFormat.FP4_PACKED:
        raise NotImplementedError(
            f"ASDX: '{Path(path).name}' uses the '{quant_format.value}' "
            f"ComfyUI quantization convention -- unpacking/dequantization for "
            f"this format is not implemented yet."
        )

    import torch
    import safetensors.torch
    state = safetensors.torch.load_file(path, device="cpu")
    if quant_format == QuantFormat.INT8_TENSORWISE:
        state = _dequantize_comfy_quant_int8(state, Path(path).name)
    elif quant_format == QuantFormat.FP8_SCALED:
        state = _dequantize_fp8_scaled(state, Path(path).name)
    result = {}
    for k, v in state.items():
        if v.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
            v = v.float()
        result[k] = mx.array(v.cpu().numpy())
    return result


def load_transformer(
    path: str | Path,
    dtype: str = "float16",
) -> FluxTransformer:
    """Load a FLUX.1 checkpoint into a FluxTransformer.

    Args:
        path: path to safetensors checkpoint
        dtype: "float16" or "bfloat16"

    Returns:
        Loaded FluxTransformer with weights assigned.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    path = Path(path)
    state = _load_safetensors(path)

    # Normalize keys: strip prefixes, map to native naming (matches ComfyUI's
    # process_unet_state_dict: only "*_norm.scale" -> "*_norm.weight" renaming;
    # everything else keeps the checkpoint's own double_blocks/single_blocks layout).
    normalized = normalize_flux_keys(state)
    normalized = map_flux_to_native(normalized)
    detected_in_channels = detect_flux_in_channels(normalized)

    config = FluxConfig(dtype=dtype, in_channels=detected_in_channels)
    model = FluxTransformer(config)

    # Assign weights via tree_unflatten: navigating attribute-by-attribute and
    # reassigning only works down to the parent module (mx.array leaves have
    # no .weight/.value to update in place), so build the full parameter tree
    # from flat checkpoint keys and hand it to model.update() in one shot.
    # tree_flatten returns (dotted_string_key, array) pairs — matching
    # directly against the checkpoint's own dotted string keys (both use the
    # same "." separator and plain integer block indices).
    model_flat = tree_flatten(model.parameters())

    new_flat = []
    matched = 0
    for flat_key, value in model_flat:
        if flat_key in normalized:
            new_flat.append((flat_key, normalized[flat_key].astype(config.mlx_dtype)))
            matched += 1
        else:
            new_flat.append((flat_key, value))
    new_nested = tree_unflatten(new_flat)
    model.update(new_nested)
    mx.eval(model.parameters())

    print(f"[ASDX] FLUX transformer: matched {matched}/{len(model_flat)} params from checkpoint")
    _check_weight_match(matched, len(model_flat), "FLUX transformer", path)
    return model
