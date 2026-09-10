"""Normalized Attention Guidance (NAG) for Krea2 / Krea2 Identity Edit.

Port of https://arxiv.org/abs/2505.21179 (eqs. 7-10), following the
reference ComfyUI implementation at
/Volumes/X10Pro/Images/Projet/krea2-nag (krea2_nag.py, nag_math.py).
Gives Krea2 a negative prompt without a second sampler eps pass: both a
positive and a negative forward share the same image query, only the
text K/V differs, and the two attention outputs are combined here,
inside attention, before the block's own gate/wo/mlp.

`_raw_attention` duplicates `Attention.__call__` (native/krea2/model.py) up
to but not including gate/wo, so its math must be kept in sync by hand if
that method ever changes.
"""
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx


def normalized_attention_guidance(
    positive: mx.array,
    negative: mx.array,
    phi: float,
    tau: float,
    alpha: float,
) -> mx.array:
    """NAG eqs. 7-10, norms evaluated in float32 regardless of input dtype.

    ``positive``/``negative`` are raw attention outputs (post softmax@V,
    pre gate/wo) produced from the same image query and image K/V — only
    their text context differs.
    """
    if positive.shape != negative.shape:
        raise ValueError(
            f"NAG attention shapes must match, got {tuple(positive.shape)} "
            f"and {tuple(negative.shape)}"
        )

    dtype = positive.dtype
    z_pos = positive.astype(mx.float32)
    z_neg = negative.astype(mx.float32)

    guided = z_pos + float(phi) * (z_pos - z_neg)
    eps = float(mx.finfo(mx.float32).eps)
    pos_norm = mx.clip(mx.abs(z_pos).sum(axis=-1, keepdims=True), eps, None)
    guided_norm = mx.clip(mx.abs(guided).sum(axis=-1, keepdims=True), eps, None)
    ratio = guided_norm / pos_norm
    normalized = guided * (mx.clip(ratio, None, float(tau)) / ratio)
    refined = float(alpha) * normalized + (1.0 - float(alpha)) * z_pos
    return refined.astype(dtype)


def guide_attention_tail(
    positive: mx.array,
    negative: mx.array,
    positive_start: int,
    negative_start: int,
    phi: float,
    tau: float,
    alpha: float,
) -> mx.array:
    """Keep the positive prefix (text + source-reference tokens) untouched,
    guide only the matching tail (target-image tokens). Used by Identity
    Edit so NAG never touches source-reference attention.
    """
    positive_tail = positive[:, positive_start:]
    negative_tail = negative[:, negative_start:]
    guided_tail = normalized_attention_guidance(
        positive_tail, negative_tail, phi=phi, tau=tau, alpha=alpha
    )
    return mx.concatenate([positive[:, :positive_start], guided_tail], axis=1)


def _raw_attention(
    attn: Any,
    x: mx.array,
    freqs: mx.array | None,
    ref_boost: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Duplicate of Attention.__call__ (native/krea2/model.py) stopping
    before ``out = out * gate; return self.wo(out)``. Returns (raw_out,
    gate) so the caller can slice rows, apply NAG to only some of them,
    then gate/wo the reassembled tensor itself.

    NOTE: this must be kept in sync with Attention.__call__ by hand if
    that method's math ever changes -- there is no way to share the
    implementation without also changing Attention.__call__ itself.
    """
    assert not attn.cpu_attention, (
        "_raw_attention only duplicates Attention.__call__'s fused-kernel "
        "branch (cpu_attention=False); it must not be called on a "
        "cpu_attention=True instance (e.g. TextFusionBlock) or it silently "
        "diverges from Attention.__call__'s CPU-stream branch."
    )
    from .model import apply_rope  # local import: keeps nag.py importable
    # without model.py in scope until this is actually called.

    B, L, D = x.shape
    q = attn.wq(x).reshape(B, L, attn.heads, attn.headdim).transpose(0, 2, 1, 3)
    k = attn.wk(x).reshape(B, L, attn.kvheads, attn.headdim).transpose(0, 2, 1, 3)
    v = attn.wv(x).reshape(B, L, attn.kvheads, attn.headdim).transpose(0, 2, 1, 3)

    q, k = attn.qknorm(q, k)
    if freqs is not None:
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

    scale = 1.0 / math.sqrt(attn.headdim)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=ref_boost)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
    gate = mx.sigmoid(attn.gate_proj(x))
    return out, gate
