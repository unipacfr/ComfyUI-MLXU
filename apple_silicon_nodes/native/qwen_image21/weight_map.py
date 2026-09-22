"""Checkpoint loading for QwenImage21Transformer2DModel: bf16 safetensors (mx.load, same
bfloat16-safe approach as the brique 1 text encoder loader) and GGUF Q8 (native/gguf/).

The real checkpoint (`qwen_image_2.1_bf16.safetensors`, 265 tensors, inspected directly on
2026-09-22) has NO `model.diffusion_model.` prefix and its keys already match this module's
own parameter names 1:1, with one exception: `time_text_embed.timestep_embedder.*`, which
needs the `_KEY_REMAP_PREFIXES` remap below (confirmed against the real checkpoint -- only
those 2 of 265 tensors diverged, not assumed)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .config import detect_qwen_image21_config
from .model import QwenImage21Transformer2DModel

# The real checkpoint (`qwen_image_2.1_bf16.safetensors`, inspected directly on 2026-09-22)
# nests the timestep embedder one level deeper than this module's own `TimestepProjEmbeddings`
# attribute (`time_text_embed.timestep_embedder.linear_{1,2}.weight` vs. this module's flat
# `time_text_embed.linear_{1,2}.weight`) -- confirmed to be the ONLY divergence (263/265
# tensors matched 1:1 with no remapping at all before this fix was added).
_KEY_REMAP_PREFIXES = {"time_text_embed.timestep_embedder.": "time_text_embed."}


def _remap_checkpoint_key(key: str) -> str:
    for old_prefix, new_prefix in _KEY_REMAP_PREFIXES.items():
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    return key


def _flatten_module_params(model: QwenImage21Transformer2DModel) -> list[tuple[str, mx.array]]:
    return list(tree_flatten(model.parameters()))


def _assign_from_state_dict(model: QwenImage21Transformer2DModel, config, state_dict: dict[str, mx.array]) -> None:
    state_dict = {_remap_checkpoint_key(k): v for k, v in state_dict.items()}
    expected_keys = set(k for k, _ in _flatten_module_params(model))
    unexpected = [k for k in state_dict if k not in expected_keys]
    if unexpected:
        raise ValueError(
            f"ASDX: {len(unexpected)} unrecognized key(s) in checkpoint, e.g. {unexpected[:5]} -- "
            "not in QwenImage21Transformer2DModel's parameters."
        )

    weights = []
    for key, module_param in _flatten_module_params(model):
        if key not in state_dict:
            raise ValueError(f"ASDX: checkpoint is missing required key {key!r} for QwenImage21Transformer2DModel.")
        checkpoint_tensor = state_dict[key]
        if tuple(checkpoint_tensor.shape) != tuple(module_param.shape):
            raise ValueError(
                f"ASDX: checkpoint key {key!r} has shape {tuple(checkpoint_tensor.shape)}, "
                f"expected {tuple(module_param.shape)} -- checkpoint does not match the "
                "detected config (a config-detection bug or a genuinely different checkpoint)."
            )
        weights.append((key, checkpoint_tensor.astype(config.mlx_dtype)))

    print(f"[ASDX] Qwen Image 2.1 DiT: matched {len(weights)}/{len(expected_keys)} weights.")
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    mx.clear_cache()


def load_qwen_image21_dit_checkpoint(path: str | Path, dtype: str = "float16") -> QwenImage21Transformer2DModel:
    path = Path(path)
    state_dict: dict[str, mx.array] = mx.load(str(path))
    config = detect_qwen_image21_config(state_dict, dtype=dtype)
    model = QwenImage21Transformer2DModel(config)
    _assign_from_state_dict(model, config, state_dict)
    return model


def load_qwen_image21_dit_from_gguf(path: str | Path, dtype: str = "float16") -> QwenImage21Transformer2DModel:
    from ..gguf.dequant import dequantize_tensor
    from ..gguf.reader import read_gguf_header

    path = Path(path)
    header = read_gguf_header(path)
    state_dict: dict[str, mx.array] = {name: dequantize_tensor(path, header, name) for name in header.tensors}
    config = detect_qwen_image21_config(state_dict, dtype=dtype)
    model = QwenImage21Transformer2DModel(config)
    _assign_from_state_dict(model, config, state_dict)
    return model
