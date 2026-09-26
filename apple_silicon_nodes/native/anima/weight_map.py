"""Strict Anima checkpoint loading. Goes through `_load_safetensors` so the
int8-convrot / fp8-scaled dequantization covers Anima like every family."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from ..common import _check_weight_match
from .config import detect_anima_config
from .model import AnimaTransformer

_PREFIXES = ("model.diffusion_model.", "net.")


def strip_anima_prefix(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_anima_checkpoint(path: str | Path, dtype: str = "bfloat16") -> AnimaTransformer:
    from .. import _load_safetensors

    path = Path(path)
    state = {strip_anima_prefix(k): v for k, v in _load_safetensors(path).items()}
    config = detect_anima_config(state, dtype=dtype)
    model = AnimaTransformer(config)

    expected = dict(tree_flatten(model.parameters()))
    unexpected = sorted(set(state) - set(expected))
    missing = sorted(set(expected) - set(state))
    if unexpected or missing:
        raise ValueError(
            f"ASDX: Anima checkpoint '{path.name}' does not match the architecture: "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}), {len(missing)} missing (e.g. {missing[:3]})."
        )
    weights = []
    for key, param in expected.items():
        tensor = state.pop(key)
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(f"ASDX: Anima key {key!r} has shape {tuple(tensor.shape)}, expected {tuple(param.shape)}.")
        weights.append((key, tensor.astype(config.mlx_dtype)))

    print(f"[ASDX] Anima DiT: matched {len(weights)}/{len(expected)} params from checkpoint")
    _check_weight_match(len(weights), len(expected), "Anima DiT", path)
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    mx.clear_cache()
    return model
