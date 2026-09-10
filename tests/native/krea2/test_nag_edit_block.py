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


def test_nag_edit_block_null_case_matches_plain_block_forward():
    block = _make_block()
    text = mx.random.normal((1, 3, 32))
    source = mx.random.normal((1, 4, 32))
    target = mx.random.normal((1, 5, 32))
    image = mx.concatenate([source, target], axis=1)
    vec = mx.random.normal((1, 6 * 32))

    plain_combined = block(mx.concatenate([text, image], axis=1), vec, None)
    plain_target_out = plain_combined[:, 3 + 4:]

    _, _, image_out = nag._nag_edit_block(
        block, text, text, image, target_offset=4, vec=vec,
        positive_freqs=None, negative_freqs=None, ref_boost=None,
        phi=4.0, tau=2.5, alpha=0.25,
    )

    np.testing.assert_allclose(
        np.array(image_out[:, 4:]), np.array(plain_target_out), rtol=1e-5, atol=1e-5
    )


def test_nag_edit_block_preserves_source_tokens_exactly():
    """NAG must never touch source-reference tokens, even with a
    different negative prompt -- guide_attention_tail's whole point."""
    block = _make_block()
    pos_text = mx.random.normal((1, 3, 32))
    neg_text = mx.random.normal((1, 3, 32))
    source = mx.random.normal((1, 4, 32))
    target = mx.random.normal((1, 5, 32))
    image = mx.concatenate([source, target], axis=1)
    vec = mx.random.normal((1, 6 * 32))

    _, _, image_out = nag._nag_edit_block(
        block, pos_text, neg_text, image, target_offset=4, vec=vec,
        positive_freqs=None, negative_freqs=None, ref_boost=None,
        phi=4.0, tau=2.5, alpha=0.25,
    )
    plain_combined = block(mx.concatenate([pos_text, image], axis=1), vec, None)
    plain_source_out = plain_combined[:, 3:3 + 4]

    np.testing.assert_allclose(
        np.array(image_out[:, :4]), np.array(plain_source_out), rtol=1e-5, atol=1e-5
    )
