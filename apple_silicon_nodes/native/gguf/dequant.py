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

Only F32/F16/BF16/Q5_0 are implemented -- the four types the real DiT
checkpoint actually uses. K-quant (Q4_K/Q6_K, needed for the text encoder)
is a separate follow-up; unsupported types raise rather than guess.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .reader import GGMLQuantizationType, GGUFHeader, GGUFTensorInfo

_Q5_0_BLOCK_ELEMENTS = 32
_Q5_0_BLOCK_BYTES = 22


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
