"""Parity of vision_rope against comfy.text_encoders.qwen_vl.qwen2vl_mrope_position_ids
and llama.precompute_freqs_cis (Qwen3-VL interleaved M-RoPE)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_text_encoders
from support.minimax_h3_module_loader import load_native_module

rope = load_native_module("minimax_h3.vision_rope")

# (index, size, grid): sizes = t*h*w/4 for the merged tokens
CASES = [
    [dict(index=5, size=6, grid=(1, 4, 6))],
    [dict(index=3, size=6, grid=(1, 4, 6)), dict(index=20, size=2, grid=(1, 2, 4))],
    [dict(index=0, size=4, grid=(1, 4, 4)), dict(index=9, size=4, grid=(1, 4, 4)), dict(index=30, size=6, grid=(1, 4, 6))],
]


def _ref_infos(cases, torch):
    return [
        {"type": "image", "index": c["index"], "size": c["size"], "extra": {"grid": torch.tensor([list(c["grid"])])}}
        for c in cases
    ]


@pytest.mark.parametrize("infos", CASES)
def test_position_ids_match_comfyui(infos):
    import torch

    _, qwen_vl, _, _ = load_real_comfy_text_encoders()
    seq_len = infos[-1]["index"] + infos[-1]["size"] + 7
    ref = qwen_vl.qwen2vl_mrope_position_ids(_ref_infos(infos, torch), seq_len, "cpu")
    got = rope.mrope_position_ids(infos, seq_len)
    assert got.shape == tuple(ref.shape) == (3, seq_len)
    assert np.array_equal(got, ref.numpy())


def test_position_ids_empty_is_none():
    assert rope.mrope_position_ids([], 10) is None


@pytest.mark.parametrize("infos", CASES)
def test_interleaved_cos_sin_match_comfyui(infos):
    import torch

    _, qwen_vl, llama, _ = load_real_comfy_text_encoders()
    seq_len = infos[-1]["index"] + infos[-1]["size"] + 7
    pos = qwen_vl.qwen2vl_mrope_position_ids(_ref_infos(infos, torch), seq_len, "cpu")
    ref_cos, ref_sin, _ = llama.precompute_freqs_cis(128, pos, 5000000.0, rope_dims=[24, 20, 20], interleaved_mrope=True)
    with mx.stream(mx.cpu):
        cos, sin = rope.interleaved_mrope_cos_sin(pos.numpy(), 128, 5000000.0, (24, 20, 20))
        mx.eval(cos, sin)
    assert cos.shape == (seq_len, 64)
    assert np.abs(np.array(cos) - ref_cos[0, :, :64].numpy()).max() < 1e-4
    assert np.abs(np.array(sin) - ref_sin[0].numpy()).max() < 1e-4


def test_interleaving_differs_from_plain_rope():
    """Null case: with H/W positions that differ from T, the interleaved table must
    differ from a plain single-axis table."""
    pos = np.stack([np.arange(8), np.arange(8) * 3, np.arange(8) * 5]).astype(np.float32)
    cos, _ = rope.interleaved_mrope_cos_sin(pos, 128, 5000000.0)
    plain, _ = rope.interleaved_mrope_cos_sin(np.stack([pos[0]] * 3), 128, 5000000.0)
    assert np.abs(np.array(cos) - np.array(plain)).max() > 1e-2


def test_rope_dims_must_cover_half_head_dim():
    with pytest.raises(ValueError, match="rope_dims"):
        rope.interleaved_mrope_cos_sin(np.zeros((3, 4), dtype=np.float32), 128, 1.0, (24, 20, 10))
