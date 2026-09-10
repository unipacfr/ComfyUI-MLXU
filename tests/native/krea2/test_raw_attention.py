from __future__ import annotations

import mlx.core as mx
import numpy as np

from tests.support.krea2_module_loader import load_krea2_module

model = load_krea2_module("model")
nag = load_krea2_module("nag")


def _make_attention(dim=32, heads=4, kvheads=2):
    attn = model.Attention(dim, heads, kvheads, cpu_attention=False)
    mx.eval(attn.parameters())
    return attn


def test_raw_attention_reassembly_matches_full_forward():
    attn = _make_attention()
    x = mx.random.normal((1, 6, 32))
    freqs = None  # no RoPE needed to check this invariant

    full_out = attn(x, freqs)
    raw_out, gate = nag._raw_attention(attn, x, freqs)
    reassembled = attn.wo(raw_out * gate)

    np.testing.assert_allclose(np.array(reassembled), np.array(full_out), rtol=1e-6, atol=1e-6)


def test_raw_attention_respects_ref_boost():
    attn = _make_attention()
    x = mx.random.normal((1, 6, 32))
    ref_boost = mx.zeros((1, attn.heads, 6, 6))
    ref_boost = ref_boost.at[:, :, :, :3].add(-1e9)  # mask out first 3 keys

    raw_masked, _ = nag._raw_attention(attn, x, None, ref_boost)
    raw_unmasked, _ = nag._raw_attention(attn, x, None, None)

    assert not np.allclose(np.array(raw_masked), np.array(raw_unmasked))
