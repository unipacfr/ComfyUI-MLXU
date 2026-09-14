"""GGUF tensor loading: reads a tensor's raw bytes and dequantizes them to
an `mx.array`.

Follows the same bridge convention `native/__init__.py::_load_safetensors`
already uses for the project's other checkpoint format: bit-unpacking and
dequantization math run in numpy (vectorized, and this is exactly what the
reference implementation itself does -- see below), the *result* handed to
model code is always an `mx.array`. Nothing here runs on GPU/Metal by
itself; it produces the dense float32 arrays that the native MLX model
code then runs on Metal, same as `_dequantize_comfy_quant_int8` does for
safetensors INT8 checkpoints today.

Q5_0 block layout (22 bytes for 32 values) and the dequantization formula
are ported bit-for-bit from the real, locally installed reference
`custom_nodes/gguf` (`gguf_connector/quant.py::Q5_0.dequantize_blocks`,
package `calcuis/gguf`), itself matching upstream llama.cpp's
`dequantize_row_q5_0`:

    block = [d: f16][qh: u32][qs: 16 bytes]
    for i in 0..31:
        low4  = qs[i] & 0xF        if i < 16 else (qs[i-16] >> 4) & 0xF
        high1 = (qh >> i) & 1
        x[i]  = d * ((low4 | (high1 << 4)) - 16)

Verified two ways against the real MiniMax H3 DiT GGUF (`models/unet/
MiniMax H3/minimax_h3_fl2va_pruned-Q5_0.gguf`): `tests/native/gguf/
test_dequant.py` checks Q5_0 tensors decode to finite, plausibly-scaled
values; a one-off comparison during development read the same
`blocks.N.norm1.weight` (BF16, unquantized in both files) from this GGUF
and from the safetensors checkpoint and got bit-identical values,
confirming the byte-offset/reshape plumbing is correct.

The corresponding *quantized* linear weights (e.g. `qkv_proj.weight`) do
**not** match between the two files even after dequantizing both --
this is not a dequantizer bug. The safetensors checkpoint's filename
(`..._int8_convrot`) says why: ComfyUI's INT8 "ConvRot" convention
applies an offline Hadamard rotation to the weight before quantizing
(`native/__init__.py::_dequantize_comfy_quant_int8`/`_rotate_weight_groups`
undoes it for that format), while this GGUF (`..._pruned-Q5_0`, no
"convrot" in the name) stores the weight in its plain, unrotated basis.
The two on-disk representations only agree after each format's own full
reconstruction chain runs -- a Q5_0-vs-plain-INT8 numeric comparison
would not have this caveat, only Q5_0-vs-INT8_convrot does. Relevant for
`native/minimax_h3/weight_map.py` once weight loading is unified across
both formats.

Q4_K and Q6_K (the K-quant super-block formats, needed for the Qwen3-VL-32B
text encoder GGUF -- its real checkpoint mixes both: Q4_K on most linear
weights, Q6_K on `down_proj`/`v_proj`, matching llama.cpp's standard
"Q4_K_M" scheme) are ported the same way, bit-for-bit from
`gguf_connector/quant.py::Q4_K.dequantize_blocks`/`Q6_K.dequantize_blocks`
and verified against that reference in `test_dequant.py`
(`test_q4_k_matches_reference_implementation`,
`test_q6_k_matches_reference_implementation`). Both use QK_K=256-element
super-blocks split into 32-element (Q4_K) or 16-element (Q6_K) sub-blocks,
each with its own 6-bit-packed scale (Q4_K also has a per-sub-block min);
the exact bit-packing of those scales into 12 bytes (`Q4_K.get_scale_min`)
is llama.cpp's own space-saving trick, ported as-is rather than re-derived.

F32/F16/BF16/Q5_0/Q4_K/Q6_K cover both real MiniMax H3 GGUF checkpoints on
this machine. Other K-quant/legacy types (Q4_0, Q5_K, Q2_K/Q3_K/Q8_K, the
IQ*/TQ*/NVFP4 families) are not implemented; unsupported types raise
rather than guess.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .reader import GGMLQuantizationType, GGUFHeader, GGUFTensorInfo

_Q5_0_BLOCK_ELEMENTS = 32
_Q5_0_BLOCK_BYTES = 22

_QK_K = 256
_K_SCALE_SIZE = 12


def _read_tensor_bytes(path: str | Path, header: GGUFHeader, name: str) -> bytes:
    info = header.tensors[name]
    with open(path, "rb") as f:
        f.seek(header.data_start_offset + info.offset)
        raw = f.read(info.nbytes)
    if len(raw) != info.nbytes:
        raise ValueError(
            f"ASDX: '{Path(path).name}' tensor '{name}' truncated -- expected "
            f"{info.nbytes} bytes, got {len(raw)} (corrupted or interrupted download)"
        )
    return raw


def _dequantize_q5_0(raw: bytes, n_elements: int) -> np.ndarray:
    if n_elements % _Q5_0_BLOCK_ELEMENTS != 0:
        raise ValueError(
            f"ASDX: Q5_0 tensor has {n_elements} elements, not a multiple of "
            f"the block size ({_Q5_0_BLOCK_ELEMENTS})"
        )
    n_blocks = n_elements // _Q5_0_BLOCK_ELEMENTS
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, _Q5_0_BLOCK_BYTES)

    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)  # (n_blocks, 1)
    qh = blocks[:, 2:6].copy().view(np.uint32).reshape(n_blocks, 1)  # (n_blocks, 1)
    qs = blocks[:, 6:22]  # (n_blocks, 16)

    high_bits = (qh >> np.arange(32, dtype=np.uint32).reshape(1, 32)) & np.uint32(1)
    high_bits = high_bits.astype(np.uint8)  # (n_blocks, 32), one bit per value, index order

    low_nibbles = qs.reshape(n_blocks, 1, 1, 16) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    low_nibbles = (low_nibbles & np.uint8(15)).reshape(n_blocks, 32)

    quantized = (low_nibbles | (high_bits << np.uint8(4))).astype(np.int8) - np.int8(16)
    return (d * quantized.astype(np.float32)).reshape(-1)


def _get_scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack Q4_K/Q5_K's 8 sub-block (scale, min) pairs from their 12-byte
    6-bit-packed encoding. Ported from
    `gguf_connector/quant.py::Q4_K.get_scale_min`."""
    n_blocks = scales.shape[0]
    scales = scales.reshape(n_blocks, 3, 4)
    d, m, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([d & 63, (m_d & 15) | ((d >> 2) & 48)], axis=-1)
    mn = np.concatenate([m & 63, (m_d >> 4) | ((m >> 2) & 48)], axis=-1)
    return sc.reshape(n_blocks, 8), mn.reshape(n_blocks, 8)


def _dequantize_q4_k(raw: bytes, n_elements: int) -> np.ndarray:
    if n_elements % _QK_K != 0:
        raise ValueError(f"ASDX: Q4_K tensor has {n_elements} elements, not a multiple of {_QK_K}")
    n_blocks = n_elements // _QK_K
    block_bytes = 2 + 2 + _K_SCALE_SIZE + _QK_K // 2  # 144
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, block_bytes)

    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    scales = blocks[:, 4 : 4 + _K_SCALE_SIZE]
    qs = blocks[:, 4 + _K_SCALE_SIZE :]  # (n_blocks, 128)

    sc, mn = _get_scale_min_k4(scales)
    d_sub = (d * sc.astype(np.float32)).reshape(n_blocks, 8, 1)
    dm_sub = (dmin * mn.astype(np.float32)).reshape(n_blocks, 8, 1)

    nibbles = qs.reshape(n_blocks, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    nibbles = (nibbles & np.uint8(15)).reshape(n_blocks, 8, 32).astype(np.float32)

    return (d_sub * nibbles - dm_sub).reshape(-1)


def _dequantize_q6_k(raw: bytes, n_elements: int) -> np.ndarray:
    if n_elements % _QK_K != 0:
        raise ValueError(f"ASDX: Q6_K tensor has {n_elements} elements, not a multiple of {_QK_K}")
    n_blocks = n_elements // _QK_K
    block_bytes = _QK_K // 2 + _QK_K // 4 + _QK_K // 16 + 2  # 210
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, block_bytes)

    ql = blocks[:, : _QK_K // 2]  # 128 bytes
    qh = blocks[:, _QK_K // 2 : _QK_K // 2 + _QK_K // 4]  # 64 bytes
    scales = blocks[:, _QK_K // 2 + _QK_K // 4 : _QK_K // 2 + _QK_K // 4 + _QK_K // 16]  # 16 bytes
    d = blocks[:, _QK_K // 2 + _QK_K // 4 + _QK_K // 16 :]  # 2 bytes

    scales = scales.copy().view(np.int8).astype(np.float32)
    d = d.copy().view(np.float16).astype(np.float32)
    d_sub = (d * scales).reshape(n_blocks, _QK_K // 16, 1)

    ql_bits = ql.reshape(n_blocks, 2, 1, 64) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    ql_bits = (ql_bits & np.uint8(15)).reshape(n_blocks, 8, 32)

    qh_bits = qh.reshape(n_blocks, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    qh_bits = (qh_bits & np.uint8(3)).reshape(n_blocks, 8, 32)

    q = (ql_bits | (qh_bits << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape(n_blocks, _QK_K // 16, -1).astype(np.float32)

    return (d_sub * q).reshape(-1)


def _dequantize_bf16(raw: bytes) -> np.ndarray:
    packed = np.frombuffer(raw, dtype=np.uint16)
    return (packed.astype(np.int32) << 16).view(np.float32).copy()


def _dequantize_f16(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


def _dequantize_f32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32).copy()


_DEQUANTIZERS = {
    GGMLQuantizationType.F32: lambda raw, n: _dequantize_f32(raw),
    GGMLQuantizationType.F16: lambda raw, n: _dequantize_f16(raw),
    GGMLQuantizationType.BF16: lambda raw, n: _dequantize_bf16(raw),
    GGMLQuantizationType.Q5_0: _dequantize_q5_0,
    GGMLQuantizationType.Q4_K: _dequantize_q4_k,
    GGMLQuantizationType.Q6_K: _dequantize_q6_k,
}


def dequantize_tensor(path: str | Path, header: GGUFHeader, name: str) -> mx.array:
    """Read tensor `name` from the GGUF file at `path` and return it as a
    dense float32 `mx.array`, shaped `torch_shape` (the PyTorch/safetensors
    convention this project's weight maps already use). Raises for a
    quantization type with no dequantizer registered above."""
    info: GGUFTensorInfo = header.tensors[name]
    dequant = _DEQUANTIZERS.get(info.dtype)
    if dequant is None:
        raise NotImplementedError(
            f"ASDX: GGUF tensor '{name}' uses quantization type {info.dtype.name}, "
            f"which has no dequantizer implemented yet."
        )
    raw = _read_tensor_bytes(path, header, name)
    flat = dequant(raw, info.n_elements)
    if flat.shape[0] != info.n_elements:
        raise ValueError(
            f"ASDX: dequantizing '{name}' ({info.dtype.name}) produced "
            f"{flat.shape[0]} values, expected {info.n_elements}"
        )
    return mx.array(flat.reshape(info.torch_shape))
