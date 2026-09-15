"""Requantizing checkpoint weights into MLX's own native quantized format.

Why this exists: MiniMax H3's DiT (20.1B parameters) and text encoder
(25.8B parameters) are far larger than every other family in this project.
Fully dequantizing either to a dense array before running it -- the pattern
`native/__init__.py::_load_safetensors` already uses for every other family
here -- does not fit in 64GB unified memory: the DiT alone is ~40GB dense at
float16 (80GB at float32), the text encoder ~51.5GB at float16 (103GB at
float32). Measured directly from the real checkpoints' element counts, not
estimated from on-disk (quantized) file size, which is 2-4x smaller and was
this session's earlier (wrong) basis for a "each stage fits comfortably"
conclusion -- see the project memory `project_minimax-h3-memory-budget.md`
for the correction.

The fix: never materialize the dense weight for the model to keep. Each
linear weight is dequantized to a transient float array (using the existing
GGUF/safetensors-INT8 dequantizers), immediately re-quantized into MLX's own
native affine format (`mx.quantize`) which `mx.quantized_matmul`/
`nn.QuantizedLinear` run against directly -- no dense weight ever needs to
exist for more than one tensor's lifetime. This is the same workflow
`mlx-lm` uses for large models (`nn.quantize(model, ...)` to convert
`nn.Linear` -> `nn.QuantizedLinear` shells, then load real (weight, scales,
biases) triples into them) -- adapted here because the SOURCE weights are
GGUF/ComfyUI-INT8-quantized, not already in MLX's format, so a dequantize
step has to happen first.

Only the large per-block linear weights (`qkv_proj`, `out_proj`, `fc1`,
`fc2` in `Attention`/`MLP`) are requantized -- everything else (patch
projections, `condition_proj`, `adaln_proj`, norm weights, the fp32 output
heads) is small enough that keeping it dense costs only a few hundred MB
total across the whole model, and several of those are documented in the
reference as deliberately higher-precision ("fp32 island" output heads).
"""

from __future__ import annotations

import mlx.core as mx

DEFAULT_GROUP_SIZE = 64
DEFAULT_BITS = 4


def requantize(dense_weight: mx.array, group_size: int = DEFAULT_GROUP_SIZE, bits: int = DEFAULT_BITS) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize an already-dequantized weight into MLX's native affine
    format. `dense_weight` should be dropped by the caller immediately
    after this call -- it exists only to be re-quantized, never to be kept
    or run against directly."""
    return mx.quantize(dense_weight, group_size=group_size, bits=bits)


def dequantize_for_check(w_q: mx.array, scales: mx.array, biases: mx.array, group_size: int = DEFAULT_GROUP_SIZE, bits: int = DEFAULT_BITS) -> mx.array:
    """Inverse of `requantize`, for tests/verification only -- never on the
    hot path (defeats the entire point of keeping weights quantized)."""
    return mx.dequantize(w_q, scales, biases, group_size=group_size, bits=bits)
