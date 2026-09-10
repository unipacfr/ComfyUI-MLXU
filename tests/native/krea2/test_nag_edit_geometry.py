# tests/native/krea2/test_nag_edit_geometry.py
from __future__ import annotations

import mlx.core as mx
import numpy as np

from tests.support.krea2_module_loader import load_krea2_module

model = load_krea2_module("model")
config_module = load_krea2_module("config")


def _make_tiny_dit():
    # Krea2Config's real field names (verified against config.py during
    # Task 6, not the names a naive guess would use): num_heads/num_kv_heads
    # (not heads/kvheads), text_layers/text_dim (not txtlayers/txtdim), no
    # num_txt_layers field at all. rope_axes_dim must sum to head_dim
    # (hidden_dim/num_heads) with every axis even; text_dim must be
    # divisible by 20 (TextFusionTransformer hardcodes heads=20 internally).
    config = config_module.Krea2Config(
        hidden_dim=32, num_blocks=2, num_heads=4, num_kv_heads=2,
        text_layers=1, text_dim=20, rope_axes_dim=(2, 2, 4), dtype="float32",
    )
    return model.SingleStreamDiT(config)


def test_predict_nag_target_offset_matches_source_token_count():
    """target_offset must equal src_h*src_w -- the same 'content-only ref
    grid' length the non-NAG Identity Edit path already computes as
    self._identity_edit_src_grid (see sampler/core.py and the 2026-09-07
    pixel-space-fit canon record). A wrong offset would apply NAG to part
    of the source tokens, exactly the class of bug guide_attention_tail
    exists to prevent.
    """
    dit = _make_tiny_dit()
    mx.eval(dit.parameters())
    img_h, img_w = 2, 3
    src_h, src_w = 2, 2
    img = mx.random.normal((1, img_h * img_w, dit.channels * dit.patch ** 2))
    src = mx.random.normal((1, src_h * src_w, dit.channels * dit.patch ** 2))
    context = dit.encode_text(mx.random.normal((1, 4, dit.txtlayers * dit.txtdim)))
    t = mx.array([0.5])
    freqs = dit.get_rope_grid(img_h, img_w, context.shape[1], [(src_h, src_w)], [(0, 0)])

    # A source token identical between the two calls, with a target that
    # differs, must leave the source-tail output identical -- proving NAG
    # never touched it (this is what target_offset gates).
    out_a = dit.predict_nag(
        img=img, context=context, neg_context=context, timestep=t,
        img_h=img_h, img_w=img_w, freqs=freqs, neg_freqs=freqs,
        src=src, src_h=src_h, src_w=src_w, phi=4.0, tau=2.5, alpha=0.25,
    )
    out_b = dit.predict_nag(
        img=img + 1.0, context=context, neg_context=context, timestep=t,
        img_h=img_h, img_w=img_w, freqs=freqs, neg_freqs=freqs,
        src=src, src_h=src_h, src_w=src_w, phi=4.0, tau=2.5, alpha=0.25,
    )
    # Both outputs are target-only ([B, img_h*img_w, ...]) by construction
    # (predict_nag/predict_edit_block already slice off the source tail),
    # so this only confirms predict_nag runs end-to-end on a src+img
    # input without shape errors and produces a differing target output
    # when the target image differs.
    assert out_a.shape == (1, img_h * img_w, dit.channels * dit.patch ** 2)
    assert not np.allclose(np.array(out_a), np.array(out_b))


def test_predict_nag_passes_src_len_as_target_offset(monkeypatch):
    """Strengthens the invariant above: this file's other test would still
    pass even with a completely wrong `target_offset`, since it only
    checks output shape and that a different TARGET image changes the
    output -- it never inspects `target_offset` itself.

    `_nag_edit_block` (tested directly in test_nag_edit_block.py::
    test_nag_edit_block_preserves_source_tokens_exactly) already proves
    that a CORRECT target_offset keeps source tokens untouched. What is
    NOT covered anywhere is that `predict_nag` (native/krea2/model.py)
    actually calls it with `target_offset=src_len` (the source token
    COUNT) rather than some other value (0, img_h*img_w, a stale
    variable, ...). Guard that wiring directly, at the level named in
    the finding, by capturing the `target_offset` every `_nag_edit_block`
    call receives and asserting it equals `src_h*src_w`.
    """
    from tests.support.krea2_module_loader import load_krea2_module

    nag_module = load_krea2_module("nag")

    dit = _make_tiny_dit()
    mx.eval(dit.parameters())
    img_h, img_w = 2, 3
    src_h, src_w = 2, 2
    src_len = src_h * src_w
    img = mx.random.normal((1, img_h * img_w, dit.channels * dit.patch ** 2))
    src = mx.random.normal((1, src_h * src_w, dit.channels * dit.patch ** 2))
    context = dit.encode_text(mx.random.normal((1, 4, dit.txtlayers * dit.txtdim)))
    t = mx.array([0.5])
    freqs = dit.get_rope_grid(img_h, img_w, context.shape[1], [(src_h, src_w)], [(0, 0)])

    seen_offsets: list[int] = []
    real_nag_edit_block = nag_module._nag_edit_block

    def _spy(*args, **kwargs):
        seen_offsets.append(kwargs["target_offset"])
        return real_nag_edit_block(*args, **kwargs)

    monkeypatch.setattr(nag_module, "_nag_edit_block", _spy)

    dit.predict_nag(
        img=img, context=context, neg_context=context, timestep=t,
        img_h=img_h, img_w=img_w, freqs=freqs, neg_freqs=freqs,
        src=src, src_h=src_h, src_w=src_w, phi=4.0, tau=2.5, alpha=0.25,
    )

    assert len(seen_offsets) == len(dit.blocks)
    assert all(offset == src_len for offset in seen_offsets)
