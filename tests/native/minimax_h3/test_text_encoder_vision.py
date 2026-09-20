"""Qwen3TextEncoder with vision rows: parity vs the real ComfyUI Llama2_ (interleaved
M-RoPE + DeepStack) on a tiny random-weight config, and a GPU-vs-CPU self-consistency test."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_text_encoders
from support.minimax_h3_module_loader import load_native_module

config_mod = load_native_module("minimax_h3.text_encoder_config")
enc_mod = load_native_module("minimax_h3.text_encoder")
rope_mod = load_native_module("minimax_h3.vision_rope")

HIDDEN, HEAD_DIM, ROPE_DIMS = 32, 8, (2, 1, 1)  # sum == HEAD_DIM // 2
PARITY_HEAD_DIM, PARITY_ROPE_DIMS = 128, (24, 20, 20)  # real H3 split: a tiny split cannot detect a broken M-RoPE
SEQ = 16
INFOS = [dict(index=3, size=4, grid=(1, 4, 4)), dict(index=10, size=2, grid=(1, 2, 4))]


def _cfg(head_dim: int = HEAD_DIM):
    return config_mod.Qwen3TextEncoderConfig(
        vocab_size=100, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
        rms_norm_eps=1e-6, rope_theta=5000000.0, dtype="float32",
    )


def _vision(seed: int = 0) -> "enc_mod.VisionInputs":
    rng = np.random.default_rng(seed)
    idx = np.concatenate([np.arange(i["index"], i["index"] + i["size"]) for i in INFOS])
    n = len(idx)
    return enc_mod.VisionInputs(
        rows=mx.array(rng.standard_normal((n, HIDDEN)).astype(np.float32)),
        row_indices=idx,
        deepstack=[mx.array(rng.standard_normal((n, HIDDEN)).astype(np.float32)) for _ in range(3)],
        position_ids=rope_mod.mrope_position_ids(INFOS, SEQ),
        rope_dims=ROPE_DIMS,
    )


def _seeded_encoder(seed: int = 0):
    enc = enc_mod.Qwen3TextEncoder(_cfg())
    rng = np.random.default_rng(seed)
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(enc.parameters()))
    enc.update(tree_unflatten([(k, mx.array((rng.standard_normal(v.shape) * 0.1).astype(np.float32))) for k, v in flat.items()]))
    return enc


IDS = mx.array(np.random.default_rng(5).integers(0, 100, SEQ).astype(np.int32))


def test_text_only_path_is_unchanged():
    enc = _seeded_encoder()
    a = np.array(enc(IDS))
    b = np.array(enc(IDS, vision=None))
    assert np.array_equal(a, b)


def test_vision_rows_change_the_output():
    enc = _seeded_encoder()
    text = np.array(enc(IDS))
    with_vision = np.array(enc(IDS, vision=_vision()))
    assert np.abs(text - with_vision).max() > 1e-2  # null case: the vision input must matter
    other = np.array(enc(IDS, vision=_vision(seed=9)))
    assert np.abs(other - with_vision).max() > 1e-2


def test_matches_comfyui_llama_with_deepstack_and_mrope():
    import torch

    _, _, llama, ops = load_real_comfy_text_encoders()
    cfg = llama.Qwen3VL_32BConfig(
        vocab_size=100, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2,
    )
    cfg.head_dim = PARITY_HEAD_DIM  # plain class attribute in the reference, not a dataclass field
    cfg.rope_dims = list(PARITY_ROPE_DIMS)
    ref = llama.Llama2_(cfg, device="cpu", dtype=torch.float32, ops=ops.disable_weight_init)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in ref.parameters():  # disable_weight_init leaves weights uninitialized
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)

    vision = dataclasses.replace(_vision(), rope_dims=PARITY_ROPE_DIMS)
    ids = torch.from_numpy(np.array(IDS)).long()[None]
    embeds = ref.embed_tokens(ids).clone()
    rows = torch.from_numpy(np.array(vision.rows))
    mask = torch.zeros((1, SEQ), dtype=torch.bool)
    mask[0, torch.from_numpy(vision.row_indices)] = True
    embeds[0, torch.from_numpy(vision.row_indices)] = rows
    deepstack = [torch.from_numpy(np.array(d)) for d in vision.deepstack]
    with torch.no_grad():
        out = ref(ids, embeds=embeds, position_ids=torch.from_numpy(vision.position_ids),
                  visual_pos_masks=mask, deepstack_embeds=deepstack)
    ref_hidden = (out[0] if isinstance(out, tuple) else out)[0].detach().numpy()

    with mx.stream(mx.cpu):
        enc = enc_mod.Qwen3TextEncoder(_cfg(PARITY_HEAD_DIM))
        enc.update(tree_unflatten([("model." + k, mx.array(v.detach().numpy())) for k, v in ref.state_dict().items()]))
        got = enc(IDS, vision=vision)
        mx.eval(got)
    assert got.shape == ref_hidden.shape
    assert np.abs(np.array(got) - ref_hidden).max() < 1e-5


def test_gpu_stream_agrees_with_cpu_stream():
    enc = _seeded_encoder()
    vision = _vision()
    gpu = np.array(enc(IDS, vision=vision))
    with mx.stream(mx.cpu):
        cpu = np.array(enc(IDS, vision=vision))
    bound = 2e-3 * np.abs(cpu).max()  # measured GPU-vs-CPU noise 2.3e-4 vs bound 1.1e-2 (48x margin)
    assert np.abs(gpu - cpu).max() < bound
    other = _seeded_encoder(seed=7)
    with mx.stream(mx.cpu):
        far = np.array(other(IDS, vision=vision))
        no_deepstack = np.array(enc(IDS, vision=dataclasses.replace(vision, deepstack=[])))
    assert np.abs(gpu - far).max() > bound  # null case: different weights (measured 0.40)
    assert np.abs(gpu - no_deepstack).max() > bound  # subtler null: same weights, no DeepStack (measured 6.4)


def _zero_embedding_encoder():
    enc = _seeded_encoder()
    enc.model.embed_tokens.weight = mx.zeros_like(enc.model.embed_tokens.weight)
    return enc


def test_zero_embedding_table_is_refused_even_with_vision_rows():
    # A failed lazy read yields an all-zero table; vision rows written over the
    # pad positions used to mask it. The screen must run before the splice.
    with pytest.raises(RuntimeError, match="token embedding"):
        _zero_embedding_encoder()(IDS, vision=_vision())


def test_zero_embedding_table_is_refused_text_only():
    with pytest.raises(RuntimeError, match="token embedding"):
        _zero_embedding_encoder()(IDS, vision=None)


def test_screen_skips_when_every_position_is_a_vision_row():
    enc = _zero_embedding_encoder()
    v = _vision()
    all_idx = np.arange(SEQ)
    n = SEQ
    rng = np.random.default_rng(1)
    full = enc_mod.VisionInputs(
        rows=mx.array(rng.standard_normal((n, HIDDEN)).astype(np.float32)),
        row_indices=all_idx,
        deepstack=[],
        position_ids=v.position_ids,
        rope_dims=ROPE_DIMS,
    )
    assert enc(IDS, vision=full).shape == (SEQ, HIDDEN)


def test_normal_encoder_passes_and_text_only_output_is_reference_identical():
    enc = _seeded_encoder()
    backbone = enc.model
    x = backbone.embed_tokens(IDS)
    cos, sin = enc_mod.qwen3_rope_cos_sin(SEQ, HEAD_DIM, 5000000.0)
    for layer in backbone.layers:
        x = layer(x, cos, sin)
    assert np.array_equal(np.array(enc(IDS, vision=None)), np.array(x))


def test_refuse_if_degenerate_rejects_zeros_and_non_finite_but_not_normal():
    with pytest.raises(RuntimeError, match=r"^ASDX MiniMax H3: degenerate conditioning from probe:.*shape"):
        enc_mod.refuse_if_degenerate(mx.zeros((2, 3)), "probe")
    for bad in (float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="non-finite"):
            enc_mod.refuse_if_degenerate(mx.array([[1.0, bad]]), "probe")
    enc_mod.refuse_if_degenerate(mx.array([[0.0, 1e-3], [2.0, -3.0]]), "probe")
