from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from tests.support.krea2_module_loader import load_krea2_module

model = load_krea2_module("model")
config_module = load_krea2_module("config")


def _make_tiny_dit():
    config = config_module.Krea2Config(
        hidden_dim=32, num_blocks=2, num_heads=4, num_kv_heads=2,
        text_dim=20, text_layers=1, rope_axes_dim=(2, 2, 4), dtype="float32",
    )
    dit = model.SingleStreamDiT(config)
    mx.eval(dit.parameters())
    return dit


def test_predict_nag_null_case_matches_predict():
    dit = _make_tiny_dit()
    img_h, img_w = 2, 3
    img = mx.random.normal((1, img_h * img_w, dit.channels * dit.patch ** 2))
    context = dit.encode_text(mx.random.normal((1, 4, dit.txtlayers * dit.txtdim)))
    t = mx.array([0.5])
    freqs = dit.get_rope_grid(img_h, img_w, context.shape[1])

    baseline = dit.predict(img=img, context=context, timestep=t, img_h=img_h, img_w=img_w, freqs=freqs)
    nag_out = dit.predict_nag(
        img=img, context=context, neg_context=context, timestep=t,
        img_h=img_h, img_w=img_w, freqs=freqs, neg_freqs=freqs,
        phi=4.0, tau=2.5, alpha=0.25,
    )

    np.testing.assert_allclose(np.array(nag_out), np.array(baseline), rtol=1e-5, atol=1e-5)


def test_predict_nag_edit_null_case_with_ref_boost_matches_predict():
    """Identity Edit + ref_boost, null case (neg_context == context) must
    reproduce plain predict(..., ref_boost=...) bit-for-bit -- same as
    every other null-case test in this plan. Guards the negative pass's
    ref_boost mask (Finding 1 of the final whole-branch review): before
    the fix, the negative pass got `None` instead of a same-shaped
    ref_boost mask sized to the negative text length, so NAG's internal
    negative-vs-positive attention comparison was skewed and this null
    case diverged from predict() by ~0.0174 max abs diff instead of 0.0.
    """
    dit = _make_tiny_dit()
    img_h, img_w = 2, 3
    src_h, src_w = 2, 2
    img = mx.random.normal((1, img_h * img_w, dit.channels * dit.patch ** 2))
    src = mx.random.normal((1, src_h * src_w, dit.channels * dit.patch ** 2))
    context = dit.encode_text(mx.random.normal((1, 4, dit.txtlayers * dit.txtdim)))
    t = mx.array([0.5])
    freqs = dit.get_rope_grid(img_h, img_w, context.shape[1], [(src_h, src_w)], [(0, 0)])

    txt_len = context.shape[1]
    src_len = src_h * src_w
    tgt_len = img_h * img_w
    total = txt_len + src_len + tgt_len
    ref_boost = mx.zeros((1, 1, total, total), dtype=mx.float32)
    ref_boost[:, :, txt_len + src_len:, txt_len:txt_len + src_len] = math.log(4.0)

    baseline = dit.predict(
        img=img, context=context, timestep=t, img_h=img_h, img_w=img_w,
        freqs=freqs, ref_boost=ref_boost, src=src, src_h=src_h, src_w=src_w,
    )
    nag_out = dit.predict_nag(
        img=img, context=context, neg_context=context, timestep=t,
        img_h=img_h, img_w=img_w, freqs=freqs, neg_freqs=freqs,
        ref_boost=ref_boost, neg_ref_boost=ref_boost, src=src, src_h=src_h, src_w=src_w,
        phi=4.0, tau=2.5, alpha=0.25,
    )

    np.testing.assert_allclose(np.array(nag_out), np.array(baseline), rtol=1e-5, atol=1e-5)


def test_predict_nag_different_negative_changes_output():
    dit = _make_tiny_dit()
    img_h, img_w = 2, 3
    img = mx.random.normal((1, img_h * img_w, dit.channels * dit.patch ** 2))
    pos_context = dit.encode_text(mx.random.normal((1, 4, dit.txtlayers * dit.txtdim)))
    neg_context = dit.encode_text(mx.random.normal((1, 4, dit.txtlayers * dit.txtdim)))
    t = mx.array([0.5])
    freqs = dit.get_rope_grid(img_h, img_w, pos_context.shape[1])

    out_neg = dit.predict_nag(
        img=img, context=pos_context, neg_context=neg_context, timestep=t,
        img_h=img_h, img_w=img_w, freqs=freqs, neg_freqs=freqs,
        phi=4.0, tau=2.5, alpha=0.25,
    )
    out_null = dit.predict_nag(
        img=img, context=pos_context, neg_context=pos_context, timestep=t,
        img_h=img_h, img_w=img_w, freqs=freqs, neg_freqs=freqs,
        phi=4.0, tau=2.5, alpha=0.25,
    )

    assert not np.allclose(np.array(out_neg), np.array(out_null))
