"""Per-tensor checkpoint sources for the MiniMax H3 loaders (GGUF and
ComfyUI safetensors), so `weight_map.py` and `text_encoder_weight_map.py`
stream one dense tensor at a time from either format with the same loop.

The safetensors source never goes through torch or
`native/__init__.py::_load_safetensors` (which dequantizes the whole state
dict to dense at once -- 40GB+ for this model). `mx.load` on a safetensors
file returns lazy arrays; each tensor is popped from that dict as it is
consumed so its evaluated buffer can be freed immediately.

Supported safetensors conventions (classified from the header, never guessed):
DENSE (F16/BF16/F32) and INT8_TENSORWISE (`.comfy_quant` + per-row
`.weight_scale`, optionally with the offline Hadamard "ConvRot" rotation --
math ported from `native/__init__.py::_dequantize_comfy_quant_int8`, itself a
port of comfy_kitchen). Anything else raises. In particular the H3 "W4A8"
converter output (`.weight_codebook`/`.weight_s_rel`/`.weight_s_channel`,
e.g. h3ErosMax_beta5_fp8.safetensors) carries `.comfy_quant` markers on I8
payloads that are really packed 4-bit, so `classify_quant_format` alone would
mislabel it INT8_TENSORWISE; it is rejected here explicitly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np

from ..gguf.dequant import dequantize_tensor
from ..gguf.reader import GGUFHeader, read_gguf_header
from ..safetensors_header import SafetensorsHeader, read_safetensors_header
from ..weight_format import QuantFormat, classify_quant_format

_W4A8_SUFFIXES = (".weight_codebook", ".weight_s_rel", ".weight_s_channel")
_COMPANION_SUFFIXES = (".comfy_quant", ".weight_scale")


class TensorSource(Protocol):
    def shapes(self) -> dict[str, tuple[int, ...]]:
        """Dense shape of every real tensor (quant companions excluded)."""

    def get(self, name: str) -> mx.array:
        """Dense float (or native dtype for unquantized) tensor `name`. Each
        name may be fetched at most once (the backing buffer is released)."""


class GGUFSource:
    def __init__(self, path: Path):
        self._path = path
        self._header: GGUFHeader = read_gguf_header(path)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(info.torch_shape) for name, info in self._header.tensors.items()}

    def get(self, name: str) -> mx.array:
        return dequantize_tensor(self._path, self._header, name)


@lru_cache(maxsize=8)
def _hadamard(size: int) -> mx.array:
    """Normalized power-of-4 Hadamard matrix, same construction as
    `native/__init__.py::_build_hadamard` (comfy_kitchen), in numpy."""
    if size < 4 or (size & (size - 1)) != 0 or round(np.log(size) / np.log(4), 9) % 1 != 0:
        raise ValueError(f"ASDX: ConvRot Hadamard size must be a power of 4, got {size}")
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float32)
    h = h4
    while h.shape[0] < size:
        h = np.kron(h, h4)
    return mx.array(h / np.sqrt(size))


def dequantize_int8_convrot(q: mx.array, scale: mx.array, convrot_group_size: int | None) -> mx.array:
    """`q` int8 [out, in], `scale` [out, 1] (per-row). Returns dense float32.
    With ConvRot, undoes W_rot = W @ H_block^T per input-feature group
    (H symmetric and involutory, so the same rotation inverts it)."""
    dense = q.astype(mx.float32) * scale.astype(mx.float32)
    if not convrot_group_size:
        return dense
    out_f, in_f = dense.shape
    if in_f % convrot_group_size != 0:
        raise ValueError(f"ASDX: ConvRot group_size {convrot_group_size} does not divide in_features {in_f}")
    h = _hadamard(convrot_group_size)
    grouped = dense.reshape(out_f, in_f // convrot_group_size, convrot_group_size)
    return mx.matmul(grouped, h.T).reshape(out_f, in_f)


class SafetensorsSource:
    def __init__(self, path: Path):
        self._path = path
        self._header: SafetensorsHeader = read_safetensors_header(path)
        tensors = self._header.tensors

        w4a8 = [k for k in tensors if k.endswith(_W4A8_SUFFIXES)]
        if w4a8:
            raise NotImplementedError(
                f"ASDX: '{path.name}' is an H3 W4A8 checkpoint ({len(w4a8)} codebook/s_rel/s_channel "
                f"tensors, e.g. '{w4a8[0]}'): packed 4-bit weights with no verified reference "
                f"implementation. Use the INT8 (ConvRot) or GGUF variant of this model instead."
            )
        verdict = classify_quant_format(self._header)
        if verdict not in (QuantFormat.DENSE, QuantFormat.INT8_TENSORWISE):
            raise NotImplementedError(
                f"ASDX: '{path.name}': safetensors quantization convention {verdict!r} is not "
                f"supported by the MiniMax H3 loaders (only dense and comfy INT8_TENSORWISE)."
            )
        self._lazy = mx.load(str(path))

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            name: entry.shape
            for name, entry in self._header.tensors.items()
            if not name.endswith(_COMPANION_SUFFIXES)
        }

    def _marker(self, prefix: str) -> dict:
        raw = np.array(self._lazy.pop(f"{prefix}.comfy_quant"), dtype=np.uint8).tobytes()
        blob = json.loads(raw)
        if blob.get("format") != "int8_tensorwise":
            raise NotImplementedError(
                f"ASDX: '{self._path.name}': '{prefix}.comfy_quant' declares format "
                f"{blob.get('format')!r}; only 'int8_tensorwise' is implemented."
            )
        return blob

    def get(self, name: str) -> mx.array:
        prefix = name[: -len(".weight")] if name.endswith(".weight") else None
        if prefix is not None and f"{prefix}.comfy_quant" in self._header.tensors:
            blob = self._marker(prefix)
            group = blob.get("convrot_groupsize") if blob.get("convrot") else None
            if blob.get("convrot") and (not isinstance(group, int) or group <= 0):
                raise ValueError(f"ASDX: '{prefix}.comfy_quant' has convrot=true with invalid convrot_groupsize {group!r}")
            dense = dequantize_int8_convrot(self._lazy.pop(name), self._lazy.pop(f"{prefix}.weight_scale"), group)
            mx.eval(dense)
            return dense
        arr = self._lazy.pop(name)
        mx.eval(arr)
        return arr


def open_checkpoint(path: str | Path) -> TensorSource:
    """Route by extension: `.gguf` -> GGUF, otherwise safetensors."""
    path = Path(path)
    return GGUFSource(path) if path.suffix.lower() == ".gguf" else SafetensorsSource(path)
