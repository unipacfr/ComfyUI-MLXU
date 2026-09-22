"""Checkpoint loading for Qwen3VL8BTextEncoder.

The real checkpoint (`qwen3vl_8b_bf16.safetensors`, 750 tensors, inspected
directly on 2026-09-22) is the FULL Qwen3-VL-8B: it carries a complete
`model.visual.*` vision tower (496 tensors) plus `model.norm.weight` and
`lm_head.weight`, none of which `Qwen3VL8BTextEncoder` builds (see
`text_encoder_config.py`'s module docstring for why). Those keys are
explicitly recognized and skipped -- any OTHER unrecognized key still
raises, so a genuinely unexpected checkpoint shape is never silently
accepted (fail-closed, this project's convention throughout).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from .text_encoder import Qwen3VL8BTextEncoder
from .text_encoder_config import detect_qwen3vl_8b_text_encoder_config


def _is_skippable_extra_key(key: str) -> bool:
    """Keys present in the real checkpoint that this text-only, no-final-norm
    encoder deliberately does not build."""
    return key.startswith("model.visual.") or key in ("model.norm.weight", "lm_head.weight")


def load_qwen_image21_text_encoder_checkpoint(path: str | Path, dtype: str = "float16") -> Qwen3VL8BTextEncoder:
    path = Path(path)
    # `mx.load` (not `safetensors.safe_open(framework="numpy")`) -- numpy has
    # no bfloat16 dtype, and the real checkpoint is stored bf16.
    state_dict: dict[str, mx.array] = mx.load(str(path))

    config = detect_qwen3vl_8b_text_encoder_config(state_dict, dtype=dtype)
    model = Qwen3VL8BTextEncoder(config)

    expected_keys = set(k for k, _ in _flatten_module_params(model))
    skipped = [k for k in state_dict if _is_skippable_extra_key(k)]
    unexpected = [k for k in state_dict if k not in expected_keys and not _is_skippable_extra_key(k)]
    if unexpected:
        raise ValueError(
            f"ASDX: {len(unexpected)} unrecognized key(s) in {path.name}, e.g. {unexpected[:5]} -- "
            "not in Qwen3VL8BTextEncoder's parameters and not a known skippable extra "
            "(model.visual.*, model.norm.weight, lm_head.weight)."
        )
    print(f"[ASDX] Qwen Image 2.1 text encoder: skipping {len(skipped)} extra key(s) "
          "(vision tower + final norm + lm_head, not used by the T2I text path).")

    weights = []
    for key, _ in _flatten_module_params(model):
        if key not in state_dict:
            raise ValueError(f"ASDX: {path.name} is missing required key {key!r} for Qwen3VL8BTextEncoder.")
        weights.append((key, state_dict[key].astype(config.mlx_dtype)))

    # This project's `verify-checkpoint` convention: every family's loader logs how many
    # of the model's own parameter keys were found in the checkpoint. The loop above
    # already raises on any miss, so this is always matched == total on success -- logged
    # anyway for the same audit trail every other family's loader produces.
    print(f"[ASDX] Qwen Image 2.1 text encoder: matched {len(weights)}/{len(expected_keys)} weights.")

    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    mx.clear_cache()
    return model


def _flatten_module_params(model: Qwen3VL8BTextEncoder) -> list[tuple[str, mx.array]]:
    from mlx.utils import tree_flatten
    return list(tree_flatten(model.parameters()))
