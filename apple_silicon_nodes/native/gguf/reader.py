"""Header-only GGUF reader.

Parses the GGUF container format (llama.cpp) far enough to enumerate tensor
names, shapes, quantization types and byte lengths, plus the metadata
key-value block -- never touches tensor data. Companion to
`safetensors_header.py`: that module gives the project's other checkpoint
format a header-only entry point; this one does the same for GGUF, needed for
MiniMax H3 (both a Q5_0 diffusion_model checkpoint and a Q4_K_M text encoder
checkpoint on this machine ship only as GGUF).

Format spec and the `GGMLQuantizationType`/`GGML_QUANT_SIZES` tables are
ported from the real, locally installed reference implementation
`custom_nodes/gguf` (package `calcuis/gguf`, PyPI `gguf-node`,
`gguf_connector/const.py` + `reader.py`), itself matching the upstream
llama.cpp `gguf` package. Verified in development against two real
checkpoints on this machine: `models/unet/MiniMax H3/
minimax_h3_fl2va_pruned-Q5_0.gguf` (legacy Q5_0) and
`qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (K-quant) -- see
`tests/native/gguf/test_reader.py`.

GGUF dimension order gotcha: the container stores each tensor's dimensions
fastest-varying-first (`ne0, ne1, ...`), the reverse of the numpy/PyTorch
convention this project's safetensors code uses everywhere else. `dims` below
keeps the raw on-disk order; `GGUFTensorInfo.torch_shape` gives the reversed,
PyTorch-convention shape for comparing against a safetensors weight_map.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

GGUF_MAGIC = 0x46554747  # b"GGUF" read as a little-endian uint32
GGUF_SUPPORTED_VERSIONS = (2, 3)
GGUF_DEFAULT_ALIGNMENT = 32

# A generous upper bound on metadata+tensor-info section size. GGUF declares
# neither section's byte length up front (only counts), so the section must
# actually be parsed to find where it ends; real headers -- even for
# multi-billion-parameter checkpoints with hundreds of tensors -- are well
# under 1 MiB. 16 MiB leaves ample room without ever reading tensor data.
_HEADER_READ_WINDOW = 16 << 20


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_STRUCT: dict[GGUFValueType, str] = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
}


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41
    Q2_0 = 42


_QK_K = 256

# (block size in elements, block size in bytes) per quantization type --
# needed to compute a tensor's total byte length from its element count alone
# (GGUF stores no explicit per-tensor byte size, only each tensor's start
# offset). Ported from `gguf_connector/const.py::GGML_QUANT_SIZES`.
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32: (1, 4),
    GGMLQuantizationType.F16: (1, 2),
    GGMLQuantizationType.Q4_0: (32, 2 + 16),
    GGMLQuantizationType.Q4_1: (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0: (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1: (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0: (32, 2 + 32),
    GGMLQuantizationType.Q8_1: (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K: (256, 2 + 2 + _QK_K // 16 + _QK_K // 4),
    GGMLQuantizationType.Q3_K: (256, 2 + _QK_K // 4 + _QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K: (256, 2 + 2 + _QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K: (256, 2 + 2 + _QK_K // 2 + _QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K: (256, 2 + _QK_K // 2 + _QK_K // 4 + _QK_K // 16),
    GGMLQuantizationType.Q8_K: (256, 4 + _QK_K + _QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + _QK_K // 4),
    GGMLQuantizationType.IQ2_XS: (256, 2 + _QK_K // 4 + _QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + _QK_K // 4 + _QK_K // 8),
    GGMLQuantizationType.IQ1_S: (256, 2 + _QK_K // 8 + _QK_K // 16),
    GGMLQuantizationType.IQ4_NL: (32, 2 + 16),
    GGMLQuantizationType.IQ3_S: (256, 2 + _QK_K // 4 + _QK_K // 8 + _QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S: (256, 2 + _QK_K // 4 + _QK_K // 16),
    GGMLQuantizationType.IQ4_XS: (256, 2 + 2 + _QK_K // 2 + _QK_K // 64),
    GGMLQuantizationType.I8: (1, 1),
    GGMLQuantizationType.I16: (1, 2),
    GGMLQuantizationType.I32: (1, 4),
    GGMLQuantizationType.I64: (1, 8),
    GGMLQuantizationType.F64: (1, 8),
    GGMLQuantizationType.IQ1_M: (256, _QK_K // 8 + _QK_K // 16 + _QK_K // 32),
    GGMLQuantizationType.BF16: (1, 2),
    GGMLQuantizationType.TQ1_0: (256, 2 + 4 * 13),
    GGMLQuantizationType.TQ2_0: (256, 2 + 64),
    GGMLQuantizationType.MXFP4: (32, 1 + 16),
    GGMLQuantizationType.NVFP4: (64, 4 + 32),
    GGMLQuantizationType.Q1_0: (128, 2 + 16),
    GGMLQuantizationType.Q2_0: (64, 2 + 16),
}


@dataclass(frozen=True)
class GGUFTensorInfo:
    dtype: GGMLQuantizationType
    dims: tuple[int, ...]  # raw GGUF order: fastest-varying dimension first
    offset: int  # relative to `GGUFHeader.data_start_offset`
    nbytes: int

    @property
    def torch_shape(self) -> tuple[int, ...]:
        """`dims` reversed to the numpy/PyTorch convention (outermost
        dimension first) used by this project's safetensors weight maps."""
        return tuple(reversed(self.dims))

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass(frozen=True)
class GGUFHeader:
    tensors: dict[str, GGUFTensorInfo]
    metadata: dict[str, object]
    data_start_offset: int  # absolute file offset where tensor data begins
    alignment: int


class _Cursor:
    """Sequential little-endian reader over an in-memory buffer."""

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def read(self, n: int) -> bytes:
        chunk = self.buf[self.pos : self.pos + n]
        if len(chunk) != n:
            raise ValueError("GGUF header truncated mid-field")
        self.pos += n
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        length = self.u64()
        return self.read(length).decode("utf-8")

    def scalar(self, value_type: GGUFValueType):
        if value_type == GGUFValueType.BOOL:
            return self.read(1)[0] != 0
        if value_type == GGUFValueType.STRING:
            return self.string()
        fmt = _SCALAR_STRUCT.get(value_type)
        if fmt is None:
            raise ValueError(f"unhandled GGUF metadata value type {value_type!r}")
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def value(self, value_type: GGUFValueType):
        if value_type == GGUFValueType.ARRAY:
            item_type = GGUFValueType(self.u32())
            count = self.u64()
            return [self.value(item_type) for _ in range(count)]
        return self.scalar(value_type)


def _tensor_nbytes(dtype: GGMLQuantizationType, n_elements: int, tensor_name: str) -> int:
    sizes = GGML_QUANT_SIZES.get(dtype)
    if sizes is None:
        raise ValueError(f"no known block size for GGML quantization type {dtype!r} (tensor {tensor_name!r})")
    block_elements, block_bytes = sizes
    if n_elements % block_elements != 0:
        raise ValueError(
            f"tensor {tensor_name!r} has {n_elements} elements, not a multiple of "
            f"{dtype.name}'s block size ({block_elements}) -- malformed file or "
            f"misclassified quantization type"
        )
    return (n_elements // block_elements) * block_bytes


def read_gguf_header(path: str | Path) -> GGUFHeader:
    """Parse the GGUF container header (magic/version/metadata/tensor infos)
    at `path`. Raises on a malformed header -- never returns a partial or
    guessed result. Reads only the metadata + tensor-info section, never
    tensor data itself."""
    path = Path(path)
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        magic_bytes = f.read(4)
        if len(magic_bytes) != 4 or struct.unpack("<I", magic_bytes)[0] != GGUF_MAGIC:
            raise ValueError(f"ASDX: {path.name} is not a GGUF file (bad magic)")
        version = struct.unpack("<I", f.read(4))[0]
        if version not in GGUF_SUPPORTED_VERSIONS:
            raise ValueError(
                f"ASDX: {path.name} declares GGUF version {version}, "
                f"only {GGUF_SUPPORTED_VERSIONS} are supported"
            )
        tensor_count, metadata_kv_count = struct.unpack("<QQ", f.read(16))
        prefix_len = 4 + 4 + 8 + 8

        window_size = min(_HEADER_READ_WINDOW, max(file_size - prefix_len, 0))
        buf = f.read(window_size)

    try:
        cursor = _Cursor(buf)
        metadata: dict[str, object] = {}
        for _ in range(metadata_kv_count):
            key = cursor.string()
            value_type = GGUFValueType(cursor.u32())
            metadata[key] = cursor.value(value_type)

        raw_infos: list[tuple[str, tuple[int, ...], GGMLQuantizationType, int]] = []
        for _ in range(tensor_count):
            name = cursor.string()
            n_dims = cursor.u32()
            dims = tuple(cursor.u64() for _ in range(n_dims))
            ggml_type = GGMLQuantizationType(cursor.u32())
            offset = cursor.u64()
            raw_infos.append((name, dims, ggml_type, offset))
    except ValueError as e:
        raise ValueError(
            f"ASDX: {path.name}'s GGUF metadata/tensor-info section did not parse "
            f"cleanly ({e}) -- truncated file, or the {_HEADER_READ_WINDOW}-byte "
            f"read window was too small for this checkpoint's header"
        ) from e

    alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
    section_end = prefix_len + cursor.pos
    data_start_offset = ((section_end + alignment - 1) // alignment) * alignment

    tensors: dict[str, GGUFTensorInfo] = {}
    for name, dims, ggml_type, offset in raw_infos:
        n_elements = 1
        for d in dims:
            n_elements *= d
        nbytes = _tensor_nbytes(ggml_type, n_elements, name)
        tensors[name] = GGUFTensorInfo(dtype=ggml_type, dims=dims, offset=offset, nbytes=nbytes)

    return GGUFHeader(
        tensors=tensors,
        metadata=metadata,
        data_start_offset=data_start_offset,
        alignment=alignment,
    )
