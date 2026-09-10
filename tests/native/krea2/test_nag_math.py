from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tests.support.krea2_module_loader import load_krea2_module

nag = load_krea2_module("nag")


def _np_reference_nag(pos: np.ndarray, neg: np.ndarray, phi: float, tau: float, alpha: float) -> np.ndarray:
    pos = pos.astype(np.float32)
    neg = neg.astype(np.float32)
    guided = pos + phi * (pos - neg)
    eps = np.finfo(np.float32).eps
    pos_norm = np.clip(np.abs(pos).sum(axis=-1, keepdims=True), eps, None)
    guided_norm = np.clip(np.abs(guided).sum(axis=-1, keepdims=True), eps, None)
    ratio = guided_norm / pos_norm
    normalized = guided * (np.minimum(ratio, tau) / ratio)
    return alpha * normalized + (1.0 - alpha) * pos


def test_normalized_attention_guidance_matches_reference_formula():
    rng = np.random.default_rng(0)
    pos_np = rng.normal(size=(2, 4, 8, 16)).astype(np.float32)
    neg_np = rng.normal(size=(2, 4, 8, 16)).astype(np.float32)
    expected = _np_reference_nag(pos_np, neg_np, phi=4.0, tau=2.5, alpha=0.25)

    result = nag.normalized_attention_guidance(
        mx.array(pos_np), mx.array(neg_np), phi=4.0, tau=2.5, alpha=0.25
    )
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-3, atol=1e-3)


def test_normalized_attention_guidance_null_case_identical_inputs():
    """pos == neg must return pos unchanged: guided=pos, ratio=1, no tau
    clamp, refined = alpha*pos + (1-alpha)*pos = pos. Sanity-checks the
    harness can actually see 'identical' before trusting a divergent case."""
    rng = np.random.default_rng(1)
    x_np = rng.normal(size=(1, 2, 4, 8)).astype(np.float32)
    x = mx.array(x_np)

    result = nag.normalized_attention_guidance(x, x, phi=4.0, tau=2.5, alpha=0.25)

    np.testing.assert_allclose(np.array(result), x_np, rtol=1e-5, atol=1e-5)


def test_normalized_attention_guidance_shape_mismatch_raises():
    a = mx.zeros((1, 2, 4))
    b = mx.zeros((1, 2, 5))
    with pytest.raises(ValueError, match="shapes must match"):
        nag.normalized_attention_guidance(a, b, phi=1.0, tau=1.0, alpha=1.0)


def test_guide_attention_tail_preserves_prefix_and_guides_tail():
    rng = np.random.default_rng(2)
    pos_np = rng.normal(size=(1, 6, 4)).astype(np.float32)
    neg_np = rng.normal(size=(1, 5, 4)).astype(np.float32)
    pos, neg = mx.array(pos_np), mx.array(neg_np)

    result = nag.guide_attention_tail(
        pos, neg, positive_start=2, negative_start=1, phi=4.0, tau=2.5, alpha=0.25
    )

    np.testing.assert_array_equal(np.array(result[:, :2]), pos_np[:, :2])
    expected_tail = nag.normalized_attention_guidance(
        pos[:, 2:], neg[:, 1:], phi=4.0, tau=2.5, alpha=0.25
    )
    np.testing.assert_allclose(np.array(result[:, 2:]), np.array(expected_tail), rtol=1e-6)
