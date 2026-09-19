"""Tests for checkpoint_source.py: ComfyUI INT8/ConvRot safetensors streaming,
W4A8 rejection, and an end-to-end synthetic safetensors DiT load."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

source_mod = load_native_module("minimax_h3.checkpoint_source")
model_mod = load_native_module("minimax_h3.model")
weight_map_mod = load_native_module("minimax_h3.weight_map")
from test_weight_map import _tiny_config  # noqa: E402  (same directory)


def _marker(convrot_group: int | None) -> mx.array:
    blob = {"format": "int8_tensorwise", "per_row": True, "convrot": bool(convrot_group)}
    if convrot_group:
        blob["convrot_groupsize"] = convrot_group
    return mx.array(np.frombuffer(json.dumps(blob).encode(), dtype=np.uint8))


def _quantize_int8(w: np.ndarray, convrot_group: int | None) -> tuple[mx.array, mx.array]:
    """Forward (quantization-time) side, independent of the code under test:
    optional per-group Hadamard rotation then per-row symmetric int8."""
    if convrot_group:
        h = np.array(source_mod._hadamard(convrot_group))
        out_f, in_f = w.shape
        w = (w.reshape(out_f, in_f // convrot_group, convrot_group) @ h.T).reshape(out_f, in_f)
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return mx.array(q), mx.array(scale.astype(np.float32))


@pytest.mark.parametrize("group", [None, 16])
def test_int8_round_trip(tmp_path, group):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    q, scale = _quantize_int8(w, group)
    mx.save_safetensors(
        str(tmp_path / "m.safetensors"),
        {
            "a.weight": q,
            "a.weight_scale": scale,
            "a.comfy_quant": _marker(group),
            "b.weight": mx.array(w).astype(mx.bfloat16),
        },
    )
    src = source_mod.open_checkpoint(tmp_path / "m.safetensors")
    assert set(src.shapes()) == {"a.weight", "b.weight"}
    got = np.array(src.get("a.weight"))
    assert np.abs(got - w).max() < 0.05 * np.abs(w).max()  # int8 noise only
    assert src.get("b.weight").dtype == mx.bfloat16


def test_null_case_rotation_actually_matters(tmp_path):
    """Metric sanity: skipping the un-rotation must NOT reproduce w."""
    rng = np.random.default_rng(1)
    w = rng.standard_normal((16, 64)).astype(np.float32)
    q, scale = _quantize_int8(w, 16)
    no_unrotate = np.array(source_mod.dequantize_int8_convrot(q, scale, None))
    assert np.abs(no_unrotate - w).max() > 0.5


def test_w4a8_checkpoint_is_rejected(tmp_path):
    q, scale = _quantize_int8(np.ones((4, 16), np.float32), None)
    mx.save_safetensors(
        str(tmp_path / "w4.safetensors"),
        {"a.weight": q, "a.comfy_quant": _marker(None), "a.weight_codebook": mx.zeros((16,))},
    )
    with pytest.raises(NotImplementedError, match="W4A8"):
        source_mod.open_checkpoint(tmp_path / "w4.safetensors")


def test_dit_loads_from_safetensors_matching_dense(tmp_path):
    cfg = _tiny_config()
    original = model_mod.MiniMaxH3Model(cfg)
    flat = dict(tree_flatten(original.parameters()))
    mx.save_safetensors(str(tmp_path / "tiny.safetensors"), flat)

    loaded = weight_map_mod.load_minimax_h3_checkpoint(tmp_path / "tiny.safetensors", dtype="float32", group_size=64, bits=4)
    assert loaded.config == cfg
    loaded_flat = dict(tree_flatten(loaded.parameters()))
    quantized = {k[: -len(".scales")] for k in loaded_flat if k.endswith(".scales")}
    for key, value in flat.items():
        if key.endswith(".weight") and key[: -len(".weight")] in quantized:
            continue
        assert np.allclose(np.array(loaded_flat[key]), np.array(value), atol=1e-4), key


_REAL_DIT = Path("/Volumes/X10Pro/Images/models/diffusion_models/MiniMax H3/base model/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
_REAL_GGUF = Path("/Volumes/X10Pro/Images/models/unet/MiniMax H3/minimax_h3_fl2va_pruned-Q5_0.gguf")


@pytest.mark.skipif(os.environ.get("ASDX_FULL_GGUF_TEST") != "1", reason="reads real multi-GB checkpoints; set ASDX_FULL_GGUF_TEST=1")
def test_real_int8_convrot_matches_real_gguf():
    """Independent reference: the Q5_0 GGUF holds the plain unrotated weights.
    After un-rotating the INT8 ConvRot file, the two must agree closely."""
    if not (_REAL_DIT.exists() and _REAL_GGUF.exists()):
        pytest.skip("real checkpoints not present")
    name = "blocks.0.attn.qkv_proj.weight"
    a = np.array(source_mod.open_checkpoint(_REAL_DIT).get(name))
    b = np.array(source_mod.open_checkpoint(_REAL_GGUF).get(name))
    assert a.shape == b.shape
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert 0.99 < cos <= 1.0 + 1e-6, cos
