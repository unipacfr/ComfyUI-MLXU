"""Small helpers shared by every native model family.

Leaf module (imports only mlx/stdlib) so family submodules can use it
without pulling in `native/__init__.py`'s eager family imports -- the test
loaders in `tests/support/` rely on that isolation.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

_MLX_DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}


def to_mlx_dtype(name: str, label: str) -> mx.Dtype:
    """Map a config dtype string to its mlx.core dtype."""
    if name not in _MLX_DTYPES:
        raise ValueError(
            f"ASDX: unsupported {label} dtype {name!r}. Use float16, bfloat16, or float32."
        )
    return _MLX_DTYPES[name]


def timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0,
                       time_factor: float = 1000.0) -> mx.array:
    """Sinusoidal timestep embedding, matching comfy.ldm.flux.layers.timestep_embedding.

    SDXL and Z-Image pass `time_factor=1.0`: their timesteps are already
    scaled (comfy's `openaimodel.py` / Lumina call it with no *1000).
    """
    t = time_factor * t
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Standard RMSNorm over the last axis, matching
    `torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)`."""
    x32 = x.astype(mx.float32)
    variance = mx.mean(x32 * x32, axis=-1, keepdims=True)
    normed = x32 * mx.rsqrt(variance + eps)
    return (normed.astype(x.dtype)) * weight


def _check_weight_match(matched: int, total: int, label: str, path: str | Path) -> None:
    """Raise if a checkpoint load matched zero of the native model's params.

    A legitimate checkpoint variant can still miss a real chunk of keys (a
    bias-free-trained Krea2 Turbo checkpoint matches only 430/686 -- see
    `krea2/model.py::load_krea2_transformer`), so this only guards the
    unambiguous failure: zero matches means every key was a structural
    mismatch, e.g. a checkpoint routed to the wrong model family by
    `loader.py::_detect_model_type`. Left unchecked, the model runs on
    randomly-initialized weights and silently produces garbage with no
    error -- only a printed line most users won't notice.
    """
    if matched == 0:
        raise RuntimeError(
            f"ASDX: {label} matched 0/{total} params from checkpoint "
            f"'{Path(path).name}' -- its keys don't match the expected "
            f"architecture at all. It was likely misdetected as the wrong "
            f"model type; check loader.py::_detect_model_type."
        )
