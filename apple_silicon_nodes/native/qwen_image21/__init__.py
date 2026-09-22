"""
Qwen Image 2.1 native MLX implementation (in progress).

So far:
- text_encoder_config.py / text_encoder.py / rope.py / text_encoder_weight_map.py: the
  Qwen3-VL-8B text-only backbone used as T2I conditioning (brick 1).
- config.py / dit_rope.py / model.py / weight_map.py: the DiT
  (`QwenImage21Transformer2DModel`), T2I only -- no reference-image conditioning, no
  prefix KV cache across sampling steps (brick 2).

Not yet implemented: ComfyUI node/loader/bridge integration (brick 3). See the task
plans under `docs/superpowers/plans/`.
"""

from __future__ import annotations

from .config import QwenImage21Config, detect_qwen_image21_config
from .model import QwenImage21Transformer2DModel
from .text_encoder import Qwen3VL8BTextEncoder
from .text_encoder_config import Qwen3VL8BTextEncoderConfig, detect_qwen3vl_8b_text_encoder_config
from .text_encoder_weight_map import load_qwen_image21_text_encoder_checkpoint
from .weight_map import load_qwen_image21_dit_checkpoint, load_qwen_image21_dit_from_gguf

__all__ = [
    "Qwen3VL8BTextEncoderConfig",
    "detect_qwen3vl_8b_text_encoder_config",
    "Qwen3VL8BTextEncoder",
    "load_qwen_image21_text_encoder_checkpoint",
    "QwenImage21Config",
    "detect_qwen_image21_config",
    "QwenImage21Transformer2DModel",
    "load_qwen_image21_dit_checkpoint",
    "load_qwen_image21_dit_from_gguf",
]
