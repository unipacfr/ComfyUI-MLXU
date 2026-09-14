"""Unit tests for the header-only GGUF reader, plus real-checkpoint
cross-checks against MiniMax H3's GGUF files on this machine (gated, like
`tests/test_krea2_edit.py::test_full_lora_apply`, behind an env var and a
path-exists check so the suite stays runnable on a machine without them)."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from tests.support.gguf_module_loader import load_gguf_module

_reader = load_gguf_module("reader")
GGMLQuantizationType = _reader.GGMLQuantizationType
GGUF_MAGIC = _reader.GGUF_MAGIC
GGUFValueType = _reader.GGUFValueType
read_gguf_header = _reader.read_gguf_header


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _gguf_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return _u64(len(encoded)) + encoded


def _build_gguf(
    *,
    metadata: list[tuple[str, GGUFValueType, object]],
    tensors: list[tuple[str, tuple[int, ...], GGMLQuantizationType, int]],
    version: int = 3,
) -> bytes:
    """Hand-assemble a minimal GGUF byte stream (header + tensor infos only,
    no tensor data) covering the value types this reader must handle."""
    out = bytearray()
    out += _u32(GGUF_MAGIC)
    out += _u32(version)
    out += _u64(len(tensors))
    out += _u64(len(metadata))

    for key, value_type, value in metadata:
        out += _gguf_string(key)
        out += _u32(int(value_type))
        out += _encode_value(value_type, value)

    for name, dims, dtype, offset in tensors:
        out += _gguf_string(name)
        out += _u32(len(dims))
        for d in dims:
            out += _u64(d)
        out += _u32(int(dtype))
        out += _u64(offset)

    return bytes(out)


def _encode_value(value_type: GGUFValueType, value) -> bytes:
    if value_type == GGUFValueType.STRING:
        return _gguf_string(value)
    if value_type == GGUFValueType.BOOL:
        return bytes([1 if value else 0])
    if value_type == GGUFValueType.UINT32:
        return _u32(value)
    if value_type == GGUFValueType.UINT64:
        return _u64(value)
    if value_type == GGUFValueType.FLOAT32:
        return struct.pack("<f", value)
    if value_type == GGUFValueType.ARRAY:
        item_type, items = value
        out = _u32(int(item_type)) + _u64(len(items))
        for item in items:
            out += _encode_value(item_type, item)
        return out
    raise NotImplementedError(f"test helper does not encode {value_type!r}")


def test_reads_magic_version_and_scalar_metadata(tmp_path):
    data = _build_gguf(
        metadata=[
            ("general.name", GGUFValueType.STRING, "minimax_h3_test"),
            ("general.alignment", GGUFValueType.UINT32, 32),
            ("some.flag", GGUFValueType.BOOL, True),
        ],
        tensors=[],
    )
    path = tmp_path / "empty.gguf"
    path.write_bytes(data)

    header = read_gguf_header(path)
    assert header.metadata["general.name"] == "minimax_h3_test"
    assert header.metadata["general.alignment"] == 32
    assert header.metadata["some.flag"] is True
    assert header.tensors == {}


def test_reads_array_metadata():
    # covered indirectly through the full round trip in the tensor test below;
    # exercised directly here since arrays recurse through `_Cursor.value`.
    payload = _encode_value(GGUFValueType.ARRAY, (GGUFValueType.UINT32, [1, 2, 3]))
    cursor = _reader._Cursor(payload)
    item_type = GGUFValueType(cursor.u32())
    count = cursor.u64()
    values = [cursor.value(item_type) for _ in range(count)]
    assert values == [1, 2, 3]


def test_tensor_dims_kept_in_raw_gguf_order_and_reversed_for_torch_shape(tmp_path):
    # A [21504, 5376] PyTorch/safetensors weight is stored in GGUF as
    # dims=(5376, 21504) -- fastest-varying dimension first.
    data = _build_gguf(
        metadata=[],
        tensors=[("blocks.0.attn.qkv_proj.weight", (5376, 21504), GGMLQuantizationType.F16, 0)],
    )
    path = tmp_path / "t.gguf"
    path.write_bytes(data)

    header = read_gguf_header(path)
    info = header.tensors["blocks.0.attn.qkv_proj.weight"]
    assert info.dims == (5376, 21504)
    assert info.torch_shape == (21504, 5376)
    assert info.n_elements == 5376 * 21504
    assert info.nbytes == 5376 * 21504 * 2  # F16: 1-element blocks, 2 bytes each


def test_quantized_tensor_nbytes_uses_block_size(tmp_path):
    # Q5_0: 32-element blocks, 22 bytes/block (2 scale + 4 qh + 16 qs).
    n_elements = 32 * 100
    data = _build_gguf(
        metadata=[],
        tensors=[("w", (n_elements,), GGMLQuantizationType.Q5_0, 0)],
    )
    path = tmp_path / "q.gguf"
    path.write_bytes(data)

    header = read_gguf_header(path)
    info = header.tensors["w"]
    assert info.nbytes == 100 * 22


def test_misaligned_quantized_tensor_raises(tmp_path):
    # 33 elements is not a multiple of Q5_0's 32-element block -- must fail
    # closed rather than silently rounding.
    data = _build_gguf(
        metadata=[],
        tensors=[("w", (33,), GGMLQuantizationType.Q5_0, 0)],
    )
    path = tmp_path / "bad.gguf"
    path.write_bytes(data)

    with pytest.raises(ValueError, match="not a multiple"):
        read_gguf_header(path)


def test_bad_magic_rejected(tmp_path):
    path = tmp_path / "not_gguf.bin"
    path.write_bytes(b"NOPE" + b"\x00" * 20)
    with pytest.raises(ValueError, match="not a GGUF file"):
        read_gguf_header(path)


def test_unsupported_version_rejected(tmp_path):
    data = _build_gguf(metadata=[], tensors=[], version=99)
    path = tmp_path / "v.gguf"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="version 99"):
        read_gguf_header(path)


def test_data_start_offset_is_aligned(tmp_path):
    data = _build_gguf(
        metadata=[("general.alignment", GGUFValueType.UINT32, 32)],
        tensors=[("w", (1,), GGMLQuantizationType.F32, 0)],
    )
    path = tmp_path / "a.gguf"
    path.write_bytes(data)

    header = read_gguf_header(path)
    assert header.data_start_offset % header.alignment == 0
    assert header.data_start_offset >= len(data)


# ---------------------------------------------------------------------------
# Real-checkpoint cross-checks. Gated behind ASDX_FULL_GGUF_TEST=1 like
# tests/test_krea2_edit.py's real-LoRA test -- these read multi-GB files that
# only exist on this machine; the header parse itself only touches a few KB.
# ---------------------------------------------------------------------------

_MINIMAX_H3_DIT_GGUF = Path(
    "/Volumes/X10Pro/Images/models/unet/MiniMax H3/minimax_h3_fl2va_pruned-Q5_0.gguf"
)
_MINIMAX_H3_TEXT_ENCODER_GGUF = Path(
    "/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
)

_real_gguf_gate = pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="reads real multi-GB GGUF checkpoints; set ASDX_FULL_GGUF_TEST=1 to run",
)


@_real_gguf_gate
def test_real_minimax_h3_dit_gguf_matches_known_architecture():
    if not _MINIMAX_H3_DIT_GGUF.exists():
        pytest.skip("no local MiniMax H3 DiT GGUF")
    header = read_gguf_header(_MINIMAX_H3_DIT_GGUF)

    assert len(header.tensors) == 532

    main_block_ids = {
        int(name.split(".")[1]) for name in header.tensors if name.startswith("blocks.")
    }
    assert len(main_block_ids) == 50

    refiner_block_ids = {
        int(name.split(".")[2])
        for name in header.tensors
        if name.startswith("token_refiner.blocks.")
    }
    assert len(refiner_block_ids) == 2

    qkv = header.tensors["blocks.0.attn.qkv_proj.weight"]
    assert qkv.dtype == GGMLQuantizationType.Q5_0
    assert qkv.torch_shape == (21504, 5376)  # matches the safetensors INT8 checkpoint

    norm = header.tensors["blocks.0.norm1.weight"]
    assert norm.dtype in (GGMLQuantizationType.BF16, GGMLQuantizationType.F16)


@_real_gguf_gate
def test_real_minimax_h3_text_encoder_gguf_is_k_quant():
    if not _MINIMAX_H3_TEXT_ENCODER_GGUF.exists():
        pytest.skip("no local MiniMax H3 text encoder GGUF")
    header = read_gguf_header(_MINIMAX_H3_TEXT_ENCODER_GGUF)

    assert len(header.tensors) > 0
    dtypes = {info.dtype for info in header.tensors.values()}
    assert GGMLQuantizationType.Q4_K in dtypes
