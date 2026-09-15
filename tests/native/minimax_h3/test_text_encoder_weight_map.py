"""End-to-end test for load_qwen3_text_encoder_from_gguf: build a tiny real
Qwen3TextEncoder, write its parameters as a synthetic GGUF file (F32), load
it back, and confirm dense parameters match exactly and quantized ones are
a reasonable approximation. Mirrors test_weight_map.py's DiT version.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

config_mod = load_native_module("minimax_h3.text_encoder_config")
text_encoder_mod = load_native_module("minimax_h3.text_encoder")
weight_map_mod = load_native_module("minimax_h3.text_encoder_weight_map")
quantized_linear_mod = load_native_module("minimax_h3.quantized_linear")
gguf_reader_mod = load_native_module("gguf.reader")

Qwen3TextEncoderConfig = config_mod.Qwen3TextEncoderConfig
Qwen3TextEncoder = text_encoder_mod.Qwen3TextEncoder
load_qwen3_text_encoder_from_gguf = weight_map_mod.load_qwen3_text_encoder_from_gguf
GGMLQuantizationType = gguf_reader_mod.GGMLQuantizationType
GGUF_MAGIC = gguf_reader_mod.GGUF_MAGIC


def _u32(v):
    return struct.pack("<I", v)


def _u64(v):
    return struct.pack("<Q", v)


def _gguf_string(s):
    b = s.encode()
    return _u64(len(b)) + b


def _write_gguf_from_tensors(tensors: dict[str, mx.array]) -> bytes:
    infos = []
    for name, arr in tensors.items():
        dims = tuple(reversed(arr.shape))
        infos.append((name, dims, np.array(arr, dtype=np.float32)))

    header = bytearray()
    header += _u32(GGUF_MAGIC)
    header += _u32(3)
    header += _u64(len(infos))
    header += _u64(0)

    offset = 0
    tensor_infos = bytearray()
    data_blobs = []
    for name, dims, np_arr in infos:
        tensor_infos += _gguf_string(name)
        tensor_infos += _u32(len(dims))
        for d in dims:
            tensor_infos += _u64(d)
        tensor_infos += _u32(int(GGMLQuantizationType.F32))
        tensor_infos += _u64(offset)
        blob = np_arr.tobytes()
        pad = (-len(blob)) % 4
        blob += b"\x00" * pad
        data_blobs.append(blob)
        offset += len(blob)

    prefix_len = 4 + 4 + 8 + 8
    section_end = prefix_len + len(tensor_infos)
    alignment = 32
    data_start = ((section_end + alignment - 1) // alignment) * alignment
    padding = b"\x00" * (data_start - section_end)
    return bytes(header) + bytes(tensor_infos) + padding + b"".join(data_blobs)


def _tiny_config(**overrides) -> "Qwen3TextEncoderConfig":
    base = dict(
        vocab_size=50,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        dtype="float32",
    )
    base.update(overrides)
    return Qwen3TextEncoderConfig(**base)


def test_round_trip_through_synthetic_gguf(tmp_path):
    cfg = _tiny_config()
    original = Qwen3TextEncoder(cfg)
    flat = dict(tree_flatten(original.parameters()))

    path = tmp_path / "tiny_qwen3.gguf"
    path.write_bytes(_write_gguf_from_tensors(flat))

    loaded = load_qwen3_text_encoder_from_gguf(path, dtype="float32", group_size=64, bits=4)
    assert loaded.config == cfg

    loaded_flat = dict(tree_flatten(loaded.parameters()))
    quantized_prefixes = {k[: -len(".scales")] for k in loaded_flat if k.endswith(".scales")}

    for key, original_value in flat.items():
        module_prefix = key[: -len(".weight")] if key.endswith(".weight") else None
        if module_prefix in quantized_prefixes:
            continue
        assert key in loaded_flat, f"missing dense key {key}"
        assert np.allclose(np.array(loaded_flat[key]), np.array(original_value), atol=1e-4), key


def test_embedding_is_quantized_and_approximates_original(tmp_path):
    cfg = _tiny_config()
    original = Qwen3TextEncoder(cfg)
    flat = dict(tree_flatten(original.parameters()))
    path = tmp_path / "tiny_qwen3.gguf"
    path.write_bytes(_write_gguf_from_tensors(flat))

    loaded = load_qwen3_text_encoder_from_gguf(path, dtype="float32", group_size=64, bits=4)
    loaded_flat = dict(tree_flatten(loaded.parameters()))

    assert loaded_flat["model.embed_tokens.weight"].dtype == mx.uint32
    reconstructed = quantized_linear_mod.dequantize_for_check(
        loaded_flat["model.embed_tokens.weight"],
        loaded_flat["model.embed_tokens.scales"],
        loaded_flat["model.embed_tokens.biases"],
        group_size=64,
        bits=4,
    )
    original_weight = flat["model.embed_tokens.weight"]
    rel_error = float(mx.mean(mx.abs(reconstructed - original_weight)).item()) / float(
        mx.mean(mx.abs(original_weight)).item()
    )
    assert rel_error < 0.3


def test_zero_matches_raises(tmp_path):
    bogus = {"not_a_real_key.weight": mx.zeros((4, 4))}
    path = tmp_path / "bogus.gguf"
    path.write_bytes(_write_gguf_from_tensors(bogus))
    with pytest.raises((ValueError, RuntimeError)):
        load_qwen3_text_encoder_from_gguf(path)


import os  # noqa: E402

_MINIMAX_H3_TEXT_ENCODER_GGUF = Path(
    "/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
)


@pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="loads the real ~18GB Qwen3 text encoder checkpoint; set ASDX_FULL_GGUF_TEST=1 to run",
)
def test_loads_real_text_encoder_gguf_checkpoint():
    if not _MINIMAX_H3_TEXT_ENCODER_GGUF.exists():
        pytest.skip("no local MiniMax H3 text encoder GGUF")

    model = load_qwen3_text_encoder_from_gguf(_MINIMAX_H3_TEXT_ENCODER_GGUF, dtype="float16")

    assert model.config.num_hidden_layers == 50
    assert model.config.hidden_size == 5120

    flat = dict(tree_flatten(model.parameters()))
    assert flat["model.layers.0.self_attn.q_proj.weight"].dtype == mx.uint32
    assert flat["model.embed_tokens.weight"].dtype == mx.uint32

    input_ids = mx.array([1, 2, 3, 4, 5])
    out = model(input_ids)
    assert out.shape == (5, 5120)
    assert bool(mx.all(mx.isfinite(out)).item())
