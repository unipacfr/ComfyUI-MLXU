"""Tests for the GGUF dequantizer.

`test_q5_0_matches_reference_implementation` is the load-bearing test: it
quantizes random data with the real, locally installed `calcuis/gguf`
reference (`gguf_connector.quant.Q5_0`) and checks our dequantizer
reproduces its dequantized values bit-for-bit -- not just "close", exact --
per this project's rule of verifying ported math against a real reference
rather than trusting a formula derived by inspection alone.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.support.gguf_module_loader import load_gguf_module

_reader = load_gguf_module("reader")
_dequant = load_gguf_module("dequant")
GGMLQuantizationType = _reader.GGMLQuantizationType
read_gguf_header = _reader.read_gguf_header
dequantize_tensor = _dequant.dequantize_tensor

_REFERENCE_GGUF_NODE = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/custom_nodes/gguf")


def _load_reference_q5_0():
    if not _REFERENCE_GGUF_NODE.exists():
        pytest.skip("calcuis/gguf reference node pack not present on this machine")
    sys.path.insert(0, str(_REFERENCE_GGUF_NODE))
    from gguf_connector.quant import Q5_0

    return Q5_0


def test_q5_0_matches_reference_implementation():
    Q5_0 = _load_reference_q5_0()
    rng = np.random.default_rng(42)
    data = rng.uniform(-2, 2, size=(8, 32)).astype(np.float32)
    blocks = Q5_0.quantize_rows(data)
    reference = Q5_0.dequantize_rows(blocks)

    mine = _dequant._dequantize_q5_0(blocks.tobytes(), data.size).reshape(8, 32)

    assert np.array_equal(mine, reference)


def test_q5_0_known_block_all_max_positive():
    # A block where every raw quant index is 31 (max): low_nibble=0xF for all,
    # high_bits=1 for all -> value 31 - 16 = 15 for every element.
    d_bytes = np.float16(2.0).tobytes()
    qh_bytes = np.uint32(0xFFFFFFFF).tobytes()
    qs_bytes = bytes([0xFF] * 16)
    block = d_bytes + qh_bytes + qs_bytes

    out = _dequant._dequantize_q5_0(block, 32)
    assert np.allclose(out, 2.0 * 15.0)


def test_q5_0_known_block_all_min():
    # Every raw quant index is 0 -> value 0 - 16 = -16.
    d_bytes = np.float16(1.5).tobytes()
    qh_bytes = np.uint32(0).tobytes()
    qs_bytes = bytes([0x00] * 16)
    block = d_bytes + qh_bytes + qs_bytes

    out = _dequant._dequantize_q5_0(block, 32)
    assert np.allclose(out, 1.5 * -16.0)


def test_bf16_matches_expected_upper_bits():
    # BF16 keeps float32's upper 16 bits: 1.0 (0x3F800000) -> bf16 0x3F80.
    raw = np.uint16(0x3F80).tobytes()
    out = _dequant._dequantize_bf16(raw)
    assert out[0] == 1.0


def test_f16_and_f32_passthrough():
    f16_raw = np.array([1.5], dtype=np.float16).tobytes()
    assert _dequant._dequantize_f16(f16_raw)[0] == 1.5

    f32_raw = np.array([2.5], dtype=np.float32).tobytes()
    assert _dequant._dequantize_f32(f32_raw)[0] == 2.5


def test_unsupported_dtype_raises(tmp_path):
    import struct

    def u32(v):
        return struct.pack("<I", v)

    def u64(v):
        return struct.pack("<Q", v)

    def gguf_string(s):
        b = s.encode()
        return u64(len(b)) + b

    data = bytearray()
    data += u32(_reader.GGUF_MAGIC)
    data += u32(3)
    data += u64(1)  # tensor_count
    data += u64(0)  # metadata_kv_count
    data += gguf_string("w")
    data += u32(1)  # n_dims
    data += u64(256)  # dims[0], a valid Q4_K block count
    data += u32(int(GGMLQuantizationType.Q4_K))
    data += u64(0)  # offset

    path = tmp_path / "k.gguf"
    path.write_bytes(bytes(data))

    header = read_gguf_header(path)
    with pytest.raises(NotImplementedError, match="Q4_K"):
        dequantize_tensor(path, header, "w")


# ---------------------------------------------------------------------------
# Real-checkpoint cross-check, gated like test_reader.py's real-file tests.
# ---------------------------------------------------------------------------

_MINIMAX_H3_DIT_GGUF = Path(
    "/Volumes/X10Pro/Images/models/unet/MiniMax H3/minimax_h3_fl2va_pruned-Q5_0.gguf"
)

_real_gguf_gate = pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="reads a real multi-GB GGUF checkpoint; set ASDX_FULL_GGUF_TEST=1 to run",
)


@_real_gguf_gate
def test_real_dit_small_tensor_dequantizes_to_finite_values():
    if not _MINIMAX_H3_DIT_GGUF.exists():
        pytest.skip("no local MiniMax H3 DiT GGUF")
    import mlx.core as mx

    header = read_gguf_header(_MINIMAX_H3_DIT_GGUF)
    # A norm weight (BF16, small) and a real Q5_0 linear weight.
    norm = dequantize_tensor(_MINIMAX_H3_DIT_GGUF, header, "blocks.0.norm1.weight")
    assert norm.shape == (5376,)
    assert bool(mx.all(mx.isfinite(norm)).item())

    qkv = dequantize_tensor(_MINIMAX_H3_DIT_GGUF, header, "blocks.0.attn.qkv_proj.weight")
    assert qkv.shape == (21504, 5376)
    assert bool(mx.all(mx.isfinite(qkv)).item())
    # Q5_0 values are always d * (an integer in [-16, 15]); a real trained
    # weight matrix should not be all-zero or absurdly large.
    max_abs = float(mx.max(mx.abs(qkv)).item())
    assert 0.0 < max_abs < 100.0
