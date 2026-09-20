"""Loads a MiniMax H3 GGUF checkpoint into a `MiniMaxH3Model`, streaming one
tensor at a time and keeping the big per-block linear weights in MLX's own
quantized format rather than dense -- see `quantized_linear.py`'s module
docstring for why dense loading (this project's convention for every other
family) does not fit in 64GB for this model.

Both GGUF and ComfyUI safetensors (dense or INT8-tensorwise/ConvRot) are
read through `checkpoint_source.py`, one dense tensor at a time (this
project's existing `_dequantize_comfy_quant_int8` dequantizes a whole state
dict at once via torch, which is exactly the dense-materialization pattern
this module exists to avoid).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .checkpoint_source import open_checkpoint
from .config import MiniMaxH3Config, detect_minimax_h3_config
from .model import MiniMaxH3Model
from .quantized_linear import DEFAULT_BITS, DEFAULT_GROUP_SIZE, requantize

# Only these four per-block linears are large enough to matter for memory
# (7168x5376-class matrices, x50 main blocks + x2 token_refiner blocks);
# everything else (patch/condition projections, adaln_proj, norms, the fp32
# output heads) stays dense -- see quantized_linear.py's module docstring.
_QUANTIZE_SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def _check_weight_match(matched: int, total: int, label: str, path: str | Path) -> None:
    """Same guard as `native/__init__.py::_check_weight_match` (zero matches
    means the checkpoint's keys don't match this architecture at all --
    likely misdetected, would otherwise run silently on random init).
    Reimplemented locally rather than imported: `apple_silicon_nodes/
    native/__init__.py` pulls in every other family's model modules at
    import time (FLUX.1, Krea2, ...), an unrelated and heavy dependency for
    this one small check."""
    if matched == 0:
        raise RuntimeError(
            f"ASDX: {label} matched 0/{total} params from checkpoint "
            f"'{Path(path).name}' -- its keys don't match the expected architecture at all."
        )


def _is_big_linear(path: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and any(path.endswith(s) for s in _QUANTIZE_SUFFIXES)


def _placeholder_state_dict(shapes: dict[str, tuple[int, ...]]) -> dict[str, mx.array]:
    """Shapes only, no real data -- enough for `detect_minimax_h3_config`,
    which reads nothing but `.shape`. Mirrors the trick
    `tests/native/minimax_h3/test_config.py`'s real-checkpoint tests use."""
    return {name: mx.zeros(shape) for name, shape in shapes.items()}


def load_minimax_h3_checkpoint(
    path: str | Path,
    dtype: str = "float16",
    group_size: int = DEFAULT_GROUP_SIZE,
    bits: int = DEFAULT_BITS,
) -> MiniMaxH3Model:
    """Build a `MiniMaxH3Model` sized from `path`'s own tensor shapes (GGUF or
    safetensors, routed by extension) and load its real weights, one tensor at a time. Peak memory during loading
    is bounded by one dequantized tensor's dense size (at most a few hundred
    MB for the largest single linear, `mlp.fc1` at 28672x5376) plus the
    model's own already-quantized-or-dense parameters -- never the whole
    checkpoint's dense size at once.
    """
    path = Path(path)
    source = open_checkpoint(path)
    shapes = source.shapes()

    config = detect_minimax_h3_config(_placeholder_state_dict(shapes), dtype=dtype)
    model = MiniMaxH3Model(config)
    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=_is_big_linear)

    model_flat = dict(tree_flatten(model.parameters()))
    quantized_module_prefixes = {key[: -len(".scales")] for key in model_flat if key.endswith(".scales")}

    matched = 0
    for name in shapes:
        module_prefix = name[: -len(".weight")] if name.endswith(".weight") else None
        is_quantized_weight = module_prefix is not None and module_prefix in quantized_module_prefixes

        if not is_quantized_weight and name not in model_flat:
            continue  # e.g. a marker/scale key from a different quant convention, or unused by this model

        dense = source.get(name)

        if is_quantized_weight:
            w_q, scales, biases = requantize(dense.astype(mx.float32), group_size=group_size, bits=bits)
            model_flat[f"{module_prefix}.weight"] = w_q
            model_flat[f"{module_prefix}.scales"] = scales
            model_flat[f"{module_prefix}.biases"] = biases
            # requantize is lazy: without evaluating here, the graph keeps every
            # dense fp32 `dense` alive until the final mx.eval (measured 79GB
            # peak vs 12.8GB resident on the real 20B DiT).
            mx.eval(w_q, scales, biases)
            matched += 3
        else:
            model_flat[name] = dense.astype(model_flat[name].dtype)
            mx.eval(model_flat[name])
            matched += 1
        del dense

    model.update(tree_unflatten(list(model_flat.items())))
    mx.eval(model.parameters())
    mx.clear_cache()  # return the freed load-time buffers MLX keeps cached (up to ~6 GB after the text encoder)

    print(f"[ASDX] MiniMax H3 DiT ({path.suffix.lstrip(".").lower()}): matched {matched}/{len(model_flat)} params from checkpoint")
    _check_weight_match(matched, len(model_flat), "MiniMax H3 DiT", path)
    return model


load_minimax_h3_from_gguf = load_minimax_h3_checkpoint  # kept for existing callers/tests
