"""Qwen Image 2.1 DiT (`QwenImage21Transformer2DModel`) -- ported from
`comfy/ldm/qwen_image21/model.py`. T2I only: no `ref_latents`/`image_slots` (no multi-image
editing/reference conditioning in this brick), no prefix KV cache across sampling steps
(recompute every step -- numerically identical to the cached path, just slower; no other
family in this project has that optimization either). Ports the reference's `in_training`
branch throughout (plain PyTorch-equivalent math), never the fused `comfy.quant_ops.ck.*`
kernel branch -- mathematically identical, ComfyUI itself falls back to this branch outside
its compiled-kernel fast path.

This file is built across three plan tasks (this one: base layers; next: TransformerBlock/
LastLayer/block_causal_attention; then: full model assembly) -- see the plan for the split.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .dit_rope import timestep_embedding


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    x32 = x.astype(mx.float32)
    variance = mx.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * mx.rsqrt(variance + eps)).astype(x.dtype) * weight


class ZeroCenteredRMSNorm(nn.Module):
    """Stored weight is scale - 1 (checkpoint convention); applied as (weight + 1) in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x.astype(mx.float32), self.weight.astype(mx.float32) + 1.0, self.eps).astype(x.dtype)


class RMSNorm(nn.Module):
    """Plain (non-zero-centered) RMSNorm, matching operations.RMSNorm -- used for
    Attention's per-head norm_q/norm_k, which the checkpoint stores as a plain scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x, self.weight, self.eps)


class TextProjection(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps)
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.gelu_approx(self.in_layer(self.text_norm(x))))


class TimestepProjEmbeddings(nn.Module):
    """`comfy.ldm.lightricks.model.TimestepEmbedding`: Linear -> SiLU -> Linear, bias-free
    (`sample_proj_bias=False`), matching the real checkpoint's `linear_1`/`linear_2` keys."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, embedding_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def __call__(self, timestep: mx.array) -> mx.array:
        emb = timestep_embedding(timestep.astype(mx.float32), 256)
        return self.linear_2(nn.silu(self.linear_1(emb.astype(timestep.dtype))))


class SwiGLUFeedForward(nn.Module):
    """Fused gate_up (checkpoint has `img_mlp.gate_up.weight [2*hidden_dim, dim]`,
    `img_mlp.out.weight [dim, hidden_dim]` -- always `fused=True` for the real checkpoint,
    the `fused=False` branch in the reference is dead for this model and not ported)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up(x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.out(nn.silu(gate) * up)


class Attention(nn.Module):
    """`(B, N, H, D)` throughout -- no transposes before RoPE, matching the reference's
    fused-kernel-friendly layout. Ports the `in_training` branch: RMSNorm on Q/K, then RoPE,
    then scaled-dot-product attention. `pe`: `[N, head_dim//2, 2, 2]` -- the same
    (un-batched) freqs table `dit_rope.apply_rope` already consumes for `flux2`'s own
    `Attention`/`joint_attention` (see `dit_rope.apply_rope`'s `f[None,None]` against
    `x_pairs` `[B,H,N,D/2,1,2]`, which broadcasts a single freqs table across both the
    batch and head axes)."""

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = [nn.Linear(inner_dim, dim, bias=False)]
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def __call__(self, x: mx.array, pe: mx.array, attn_fn=None) -> mx.array:
        """`attn_fn(q, k, v, heads) -> [B, N, inner_dim]` implements the block-causal
        attention pattern (Task 4); when omitted, plain full self-attention (used by this
        task's own unit tests only, never by the assembled model)."""
        B, N, _ = x.shape
        q = self.to_q(x).reshape(B, N, self.heads, self.dim_head)
        k = self.to_k(x).reshape(B, N, self.heads, self.dim_head)
        v = self.to_v(x).reshape(B, N, self.heads, self.dim_head)

        q, k = self.norm_q(q), self.norm_k(k)
        # apply_rope expects [B,H,N,D]; Attention's own layout is [B,N,H,D] (see docstring)
        from .dit_rope import apply_rope
        q = apply_rope(q.transpose(0, 2, 1, 3), pe).transpose(0, 2, 1, 3)
        k = apply_rope(k.transpose(0, 2, 1, 3), pe).transpose(0, 2, 1, 3)

        if attn_fn is None:
            qh, kh, vh = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
            scale = 1.0 / (self.dim_head**0.5)
            out = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=scale)
            out = out.transpose(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head)
        else:
            out = attn_fn(q, k, v, self.heads)
        return self.to_out[0](out)


def _split_rows(p: mx.array) -> tuple[mx.array, mx.array]:
    """Shared modulation rows: (t=0 row for the text prefix, sampled-t rows for the
    target). `p`: [seq, dim] -> (prefix [1, 1, dim], target [1, seq-1, dim])."""
    return p[-1:][None], p[:-1][None]


def _modulated_norm(norm: nn.Module, x: mx.array, scale: tuple[mx.array, mx.array], prefix_len: int) -> mx.array:
    """LayerNorm (no affine) * (1 + scale), the target scale for every row, the prefix
    rows (if any) redone with the t=0 scale. Plain-math equivalent of the reference's
    `comfy.quant_ops.ck.adaln` fused-kernel branch."""
    s_prefix, s_target = scale
    out = norm(x)
    if prefix_len:
        prefix_out = out[:, :prefix_len] * (1 + s_prefix)
        target_out = out[:, prefix_len:] * (1 + s_target)
        return mx.concatenate([prefix_out, target_out], axis=1)
    return out * (1 + s_target)


def _gated_residual(x: mx.array, y: mx.array, gate: tuple[mx.array, mx.array], prefix_len: int) -> mx.array:
    g_prefix, g_target = gate
    if prefix_len:
        prefix_out = x[:, :prefix_len] + y[:, :prefix_len] * g_prefix
        target_out = x[:, prefix_len:] + y[:, prefix_len:] * g_target
        return mx.concatenate([prefix_out, target_out], axis=1)
    return x + y * g_target


class LayerNormNoAffine(nn.Module):
    """LayerNorm with elementwise_affine=False: normalizes, no learned scale/bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        mean = mx.mean(x32, axis=-1, keepdims=True)
        var = mx.var(x32, axis=-1, keepdims=True)
        return ((x32 - mean) * mx.rsqrt(var + self.eps)).astype(x.dtype)


def block_causal_attention(segments: list[tuple[int, int, mx.array | None]]):
    """`segments`: list of `(start, end, mask)`. Each segment's query rows `[start:end]`
    attend over key/value rows `[0:end]` (everything up to and including this segment),
    with `mask` applied if given (a `[n, end]` boolean mask, `True` = attend) or full
    attention over `[0:end]` if `mask` is `None`. Matches the reference's
    `block_causal_attention` exactly, minus the prefix-cache `cache.put` call (cache
    disabled in this brick -- see the brick 2 design's Risques section)."""

    def attn(q: mx.array, k: mx.array, v: mx.array, heads: int) -> mx.array:
        B, N, H, D = q.shape
        outs = []
        for start, end, mask in segments:
            qs = q[:, start:end].reshape(B, end - start, H, D).transpose(0, 2, 1, 3)
            ks = k[:, :end].reshape(B, end, H, D).transpose(0, 2, 1, 3)
            vs = v[:, :end].reshape(B, end, H, D).transpose(0, 2, 1, 3)
            scale = 1.0 / (D**0.5)
            attn_mask = None
            if mask is not None:
                attn_mask = mx.where(mask, mx.array(0.0), mx.array(-mx.inf)).astype(qs.dtype)
            out = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=scale, mask=attn_mask)
            outs.append(out.transpose(0, 2, 1, 3).reshape(B, end - start, H * D))
        return mx.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]

    return attn


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 mlp_ratio: int = 3, eps: float = 1e-6):
        super().__init__()
        self.img_norm1 = LayerNormNoAffine(dim, eps=eps)
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, eps=eps)
        self.img_norm2 = LayerNormNoAffine(dim, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio)

    def __call__(self, x: mx.array, mod, pe: mx.array, attn_fn, prefix_len: int) -> mx.array:
        scale1, gate1, scale2, gate2, _zero = mod
        x = _gated_residual(
            x, self.attn(_modulated_norm(self.img_norm1, x, scale1, prefix_len), pe, attn_fn), gate1, prefix_len
        )
        x = _gated_residual(x, self.img_mlp(_modulated_norm(self.img_norm2, x, scale2, prefix_len)), gate2, prefix_len)
        if x.dtype == mx.float16:
            x = mx.clip(x, -65504, 65504)
        return x


class LastLayer(nn.Module):
    """Scale only, no shift."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = LayerNormNoAffine(dim, eps=eps)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        scale = self.linear(nn.silu(temb))[:, None]
        return self.norm(x) * (1 + scale)
