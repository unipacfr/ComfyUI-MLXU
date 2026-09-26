"""Anima native MLX implementation: Cosmos-Predict2 MiniTrainDIT + LLM adapter.

The Qwen3-0.6B text encoder is ComfyUI's own `AnimaTEModel` (via
`ASDX_CLIPLoader`); see `bridge.conditioning_anima_to_mlx`."""

from __future__ import annotations

from .config import AnimaConfig, detect_anima_config
from .model import AnimaTransformer
from .weight_map import load_anima_checkpoint, strip_anima_prefix

__all__ = [
    "AnimaConfig",
    "detect_anima_config",
    "AnimaTransformer",
    "load_anima_checkpoint",
    "strip_anima_prefix",
]
