"""Anima native MLX implementation: Cosmos-Predict2 MiniTrainDIT + LLM adapter.

The Qwen3-0.6B text encoder is ComfyUI's own `AnimaTEModel` (via
`ASDX_CLIPLoader`); see `bridge.conditioning_anima_to_mlx`."""

from __future__ import annotations

from .config import AnimaConfig, detect_anima_config

__all__ = ["AnimaConfig", "detect_anima_config"]
