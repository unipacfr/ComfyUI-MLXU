from __future__ import annotations

import mlx.core as mx
import numpy as np

from tests.support.krea2_module_loader import load_krea2_module

model = load_krea2_module("model")
nag = load_krea2_module("nag")


def _make_block(dim=32, heads=4, kvheads=2):
    block = model.SingleStreamBlock(dim, heads, multiplier=2, kvheads=kvheads)
    mx.eval(block.parameters())
    return block


def test_nag_block_null_case_matches_plain_block_forward():
    block = _make_block()
    text = mx.random.normal((1, 3, 32))
    image = mx.random.normal((1, 5, 32))
    vec = mx.random.normal((1, 6 * 32))
    freqs = None

    plain_combined = block(mx.concatenate([text, image], axis=1), vec, freqs)
    plain_image_out = plain_combined[:, 3:]
    plain_text_out = plain_combined[:, :3]

    pos_text_out, neg_text_out, image_out = nag._nag_block(
        block, text, text, image, vec, freqs, freqs, phi=4.0, tau=2.5, alpha=0.25
    )

    np.testing.assert_allclose(np.array(image_out), np.array(plain_image_out), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(pos_text_out), np.array(plain_text_out), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(neg_text_out), np.array(plain_text_out), rtol=1e-5, atol=1e-5)


def test_nag_block_different_negative_changes_image_output():
    block = _make_block()
    pos_text = mx.random.normal((1, 3, 32))
    neg_text = mx.random.normal((1, 3, 32))
    image = mx.random.normal((1, 5, 32))
    vec = mx.random.normal((1, 6 * 32))

    _, _, image_out = nag._nag_block(
        block, pos_text, neg_text, image, vec, None, None, phi=4.0, tau=2.5, alpha=0.25
    )
    _, _, image_out_null = nag._nag_block(
        block, pos_text, pos_text, image, vec, None, None, phi=4.0, tau=2.5, alpha=0.25
    )

    assert not np.allclose(np.array(image_out), np.array(image_out_null))
