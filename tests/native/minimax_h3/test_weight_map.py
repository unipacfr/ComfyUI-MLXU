"""End-to-end test for load_minimax_h3_from_gguf: build a tiny real
MiniMaxH3Model, write its parameters out as a synthetic GGUF file (F32,
so this doesn't re-test Q5_0/Q4_K/Q6_K dequant -- those already have their
own bit-exact reference tests), load it back, and confirm the loaded
model's dense parameters match the original and the requantized (big
linear) parameters are a reasonable approximation of the originals.
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

config_mod = load_native_module("minimax_h3.config")
model_mod = load_native_module("minimax_h3.model")
weight_map_mod = load_native_module("minimax_h3.weight_map")
quantized_linear_mod = load_native_module("minimax_h3.quantized_linear")
gguf_reader_mod = load_native_module("gguf.reader")

MiniMaxH3Config = config_mod.MiniMaxH3Config
MiniMaxH3Model = model_mod.MiniMaxH3Model
load_minimax_h3_from_gguf = weight_map_mod.load_minimax_h3_from_gguf
GGUFValueType = gguf_reader_mod.GGUFValueType
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
    """Encode `tensors` (all F32, GGUF dim order = reversed torch order) as
    a minimal GGUF file: header + tensor infos + 32-byte-aligned data."""
    infos = []
    for name, arr in tensors.items():
        dims = tuple(reversed(arr.shape))  # torch/mlx order -> GGUF order
        infos.append((name, dims, np.array(arr, dtype=np.float32)))

    header = bytearray()
    header += _u32(GGUF_MAGIC)
    header += _u32(3)
    header += _u64(len(infos))
    header += _u64(0)  # no metadata

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
        # pad each tensor's data to a multiple of 4 bytes (F32 element size;
        # real GGUF aligns per-tensor to the format's block size, 4 is enough here)
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


def _tiny_config(**overrides) -> "MiniMaxH3Config":
    base = dict(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=64,
        latents_dim=4,
        audio_latents_dim=6,
        attention_head_dim=32,
        num_attention_heads=2,
        ffn_hidden_size=64,
        text_dim=10,
        patch_size=(1, 2, 2),
        rope_inv_freq_len=2,  # rot_dim = 2*3*2=12 < head_dim=32
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        final_norm_eps=1e-5,
        sigma_shift_video=12.0,
        sigma_shift_audio=3.0,
        gate_compress=False,
        adaln_curve_grid=17,
        time_embed_dim=6,
        dtype="float32",
    )
    base.update(overrides)
    return MiniMaxH3Config(**base)


def test_round_trip_through_synthetic_gguf(tmp_path):
    cfg = _tiny_config()
    original = MiniMaxH3Model(cfg)
    flat = dict(tree_flatten(original.parameters()))
    # qkv_proj is 3*heads*head_dim=192 wide, divisible by group_size 64 -- fine.
    # fc1 is ffn*2=128 wide -- also fine. Both required for a valid group_size=64 quantize.

    gguf_bytes = _write_gguf_from_tensors(flat)
    path = tmp_path / "tiny.gguf"
    path.write_bytes(gguf_bytes)

    loaded = load_minimax_h3_from_gguf(path, dtype="float32", group_size=64, bits=4)

    assert loaded.config == cfg

    loaded_flat = dict(tree_flatten(loaded.parameters()))
    quantized_prefixes = {k[: -len(".scales")] for k in loaded_flat if k.endswith(".scales")}

    for key, original_value in flat.items():
        module_prefix = key[: -len(".weight")] if key.endswith(".weight") else None
        if module_prefix in quantized_prefixes:
            continue  # checked separately below, via dequantize + tolerance
        assert key in loaded_flat, f"missing dense key {key}"
        assert np.allclose(np.array(loaded_flat[key]), np.array(original_value), atol=1e-4), key


def test_quantized_weights_are_reasonable_approximation(tmp_path):
    cfg = _tiny_config()
    original = MiniMaxH3Model(cfg)
    flat = dict(tree_flatten(original.parameters()))
    gguf_bytes = _write_gguf_from_tensors(flat)
    path = tmp_path / "tiny.gguf"
    path.write_bytes(gguf_bytes)

    loaded = load_minimax_h3_from_gguf(path, dtype="float32", group_size=64, bits=4)
    loaded_flat = dict(tree_flatten(loaded.parameters()))

    # blocks.0.attn.qkv_proj should be quantized: present as weight/scales/biases.
    assert "blocks.0.attn.qkv_proj.weight" in loaded_flat
    assert "blocks.0.attn.qkv_proj.scales" in loaded_flat
    assert "blocks.0.attn.qkv_proj.biases" in loaded_flat
    assert loaded_flat["blocks.0.attn.qkv_proj.weight"].dtype == mx.uint32

    reconstructed = quantized_linear_mod.dequantize_for_check(
        loaded_flat["blocks.0.attn.qkv_proj.weight"],
        loaded_flat["blocks.0.attn.qkv_proj.scales"],
        loaded_flat["blocks.0.attn.qkv_proj.biases"],
        group_size=64,
        bits=4,
    )
    original_weight = flat["blocks.0.attn.qkv_proj.weight"]
    rel_error = float(mx.mean(mx.abs(reconstructed - original_weight)).item()) / float(
        mx.mean(mx.abs(original_weight)).item()
    )
    assert rel_error < 0.3


_MINIMAX_H3_DIT_GGUF = Path(
    "/Volumes/X10Pro/Images/models/unet/MiniMax H3/minimax_h3_fl2va_pruned-Q5_0.gguf"
)


@pytest.mark.skipif(
    __import__("os").environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="loads the real ~14GB MiniMax H3 DiT checkpoint; set ASDX_FULL_GGUF_TEST=1 to run",
)
def test_loads_real_dit_gguf_checkpoint():
    if not _MINIMAX_H3_DIT_GGUF.exists():
        pytest.skip("no local MiniMax H3 DiT GGUF")

    model = load_minimax_h3_from_gguf(_MINIMAX_H3_DIT_GGUF, dtype="float16")

    assert model.config.num_layers == 50
    assert model.config.token_refiner_num_layers == 2
    assert model.config.hidden_size == 5376

    flat = dict(tree_flatten(model.parameters()))
    assert flat["blocks.0.attn.qkv_proj.weight"].dtype == mx.uint32
    assert bool(mx.all(mx.isfinite(flat["blocks.0.norm1.weight"])).item())

    # Exercise the loaded model on a small synthetic input -- confirms the
    # loaded weights actually run through a forward pass without shape
    # errors, not just that they were assigned.
    video = mx.random.normal((1, model.config.latents_dim, 1, 4, 4))
    audio = mx.random.normal((1, model.config.audio_latents_dim, 2, 2))
    context = mx.random.normal((3, model.config.text_dim))
    video_out, audio_out = model(video, audio, context, sigma_v=0.5)
    assert bool(mx.all(mx.isfinite(video_out)).item())
    assert bool(mx.all(mx.isfinite(audio_out)).item())


def test_zero_matches_raises():
    # A GGUF file with tensor names that share no keys with the model
    # architecture must fail loudly, not silently run on random init.
    import tempfile

    bogus = {"not_a_real_key.weight": mx.zeros((4, 4))}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bogus.gguf"
        path.write_bytes(_write_gguf_from_tensors(bogus))
        with pytest.raises((ValueError, RuntimeError)):
            load_minimax_h3_from_gguf(path)


def _record_mx_calls(monkeypatch, mod) -> list[str]:
    """Wrap `mx.eval` and `mx.clear_cache` on `mod.mx`, returning the ordered call log."""
    events: list[str] = []
    real_eval = mod.mx.eval
    monkeypatch.setattr(mod.mx, "eval", lambda *a, **k: (events.append("eval"), real_eval(*a, **k))[1])
    monkeypatch.setattr(mod.mx, "clear_cache", lambda: events.append("clear_cache"))
    return events


def test_loader_clears_mlx_cache_once_after_final_eval(tmp_path, monkeypatch):
    cfg = _tiny_config()
    flat = dict(tree_flatten(MiniMaxH3Model(cfg).parameters()))
    path = tmp_path / "tiny.gguf"
    path.write_bytes(_write_gguf_from_tensors(flat))
    events = _record_mx_calls(monkeypatch, weight_map_mod)
    load_minimax_h3_from_gguf(path, dtype="float32", group_size=64, bits=4)
    assert events.count("clear_cache") == 1
    assert events[-1] == "clear_cache" and "eval" in events[:-1]
