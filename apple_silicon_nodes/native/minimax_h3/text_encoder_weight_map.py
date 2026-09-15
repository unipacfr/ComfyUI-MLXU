"""Loads the Qwen3-VL-32B text encoder GGUF checkpoint into a
`Qwen3TextEncoder`, streaming one tensor at a time and keeping every large
weight (all attention/MLP linears, plus the embedding table) in MLX's own
quantized format -- same rationale and mechanism as
`weight_map.py::load_minimax_h3_from_gguf`, even more necessary here: this
checkpoint is 25.8B parameters, ~51.5GB dense at float16 (~103GB at
float32) -- see `quantized_linear.py`'s module docstring and the
project-memory note this session corrected.

Unlike the DiT, every linear in this backbone (`q_proj`/`k_proj`/`v_proj`/
`o_proj`/`gate_proj`/`up_proj`/`down_proj`) is large enough to matter, plus
`embed_tokens` (151936 x 5120, ~1.5GB dense at float16) -- so the
quantization predicate here is simply "every `nn.Linear` and `nn.Embedding`
in the model", not a curated subset like the DiT's four per-block names.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from ..gguf.dequant import dequantize_tensor
from ..gguf.reader import GGUFHeader, read_gguf_header
from .quantized_linear import DEFAULT_BITS, DEFAULT_GROUP_SIZE, requantize
from .text_encoder import Qwen3TextEncoder
from .text_encoder_config import detect_qwen3_text_encoder_config


def _check_weight_match(matched: int, total: int, label: str, path: str | Path) -> None:
    """Same guard as `weight_map.py::_check_weight_match` -- reimplemented
    locally for the same reason (avoid pulling in
    `apple_silicon_nodes/native/__init__.py`'s unrelated family imports)."""
    if matched == 0:
        raise RuntimeError(
            f"ASDX: {label} matched 0/{total} params from checkpoint "
            f"'{Path(path).name}' -- its keys don't match the expected architecture at all."
        )


def _is_quantizable(path: str, module: nn.Module) -> bool:
    return isinstance(module, (nn.Linear, nn.Embedding))


def _placeholder_state_dict(header: GGUFHeader) -> dict[str, mx.array]:
    return {name: mx.zeros(info.torch_shape) for name, info in header.tensors.items()}


def load_qwen3_text_encoder_from_gguf(
    path: str | Path,
    dtype: str = "float16",
    group_size: int = DEFAULT_GROUP_SIZE,
    bits: int = DEFAULT_BITS,
) -> Qwen3TextEncoder:
    """Build a `Qwen3TextEncoder` sized from `path`'s own tensor shapes and
    load its real weights, one tensor at a time, keeping every large weight
    quantized (never materializing the whole checkpoint's dense size at
    once)."""
    path = Path(path)
    header = read_gguf_header(path)

    config = detect_qwen3_text_encoder_config(_placeholder_state_dict(header), dtype=dtype)
    model = Qwen3TextEncoder(config)
    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=_is_quantizable)

    model_flat = dict(tree_flatten(model.parameters()))
    quantized_module_prefixes = {key[: -len(".scales")] for key in model_flat if key.endswith(".scales")}

    matched = 0
    for name in header.tensors:
        module_prefix = name[: -len(".weight")] if name.endswith(".weight") else None
        is_quantized_weight = module_prefix is not None and module_prefix in quantized_module_prefixes

        if not is_quantized_weight and name not in model_flat:
            continue

        dense = dequantize_tensor(path, header, name)

        if is_quantized_weight:
            w_q, scales, biases = requantize(dense.astype(mx.float32), group_size=group_size, bits=bits)
            model_flat[f"{module_prefix}.weight"] = w_q
            model_flat[f"{module_prefix}.scales"] = scales
            model_flat[f"{module_prefix}.biases"] = biases
            matched += 3
        else:
            model_flat[name] = dense.astype(model_flat[name].dtype)
            matched += 1
        del dense

    model.update(tree_unflatten(list(model_flat.items())))
    mx.eval(model.parameters())

    print(f"[ASDX] Qwen3 text encoder (GGUF): matched {matched}/{len(model_flat)} params from checkpoint")
    _check_weight_match(matched, len(model_flat), "Qwen3 text encoder", path)
    return model
