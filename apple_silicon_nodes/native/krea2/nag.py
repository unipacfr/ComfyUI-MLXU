"""Normalized Attention Guidance (NAG) for Krea2 / Krea2 Identity Edit.

Port of https://arxiv.org/abs/2505.21179 (eqs. 7-10), following the
reference ComfyUI implementation at
/Volumes/X10Pro/Images/Projet/krea2-nag (krea2_nag.py, nag_math.py).
Gives Krea2 a negative prompt without a second sampler eps pass: both a
positive and a negative forward share the same image query, only the
text K/V differs, and the two attention outputs are combined here,
inside attention, before the block's own gate/wo/mlp.
"""
from __future__ import annotations

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
