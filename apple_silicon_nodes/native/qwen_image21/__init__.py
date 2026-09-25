"""
Qwen Image 2.1 native MLX implementation (in progress).

So far:
- config.py / dit_rope.py / model.py / weight_map.py: the DiT
  (`QwenImage21Transformer2DModel`), T2I only -- no reference-image conditioning, no
  prefix KV cache across sampling steps (brick 2).

The Qwen3-VL-8B text encoder is ComfyUI's own (`ASDX_CLIPLoader` with
`type="qwen_image"`), not an MLX port -- see `bridge.conditioning_qwen_image21_to_mlx`.

Not yet implemented: ComfyUI node/loader/bridge integration (brick 3). See the task
plans under `docs/superpowers/plans/`.
"""

from __future__ import annotations

from .config import QwenImage21Config, detect_qwen_image21_config
from .model import QwenImage21Transformer2DModel
from .weight_map import load_qwen_image21_dit_checkpoint, load_qwen_image21_dit_from_gguf

__all__ = [
    "QwenImage21Config",
    "detect_qwen_image21_config",
    "QwenImage21Transformer2DModel",
    "load_qwen_image21_dit_checkpoint",
    "load_qwen_image21_dit_from_gguf",
]
