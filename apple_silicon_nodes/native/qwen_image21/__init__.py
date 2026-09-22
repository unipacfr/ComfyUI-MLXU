"""
Qwen Image 2.1 native MLX implementation (in progress).

So far:
- Qwen3VL8BTextEncoderConfig / detect_qwen3vl_8b_text_encoder_config: text
  encoder configuration derived from checkpoint tensor shapes, for the
  Qwen3-VL-8B text-only backbone used as conditioning in T2I.

Not yet implemented: the text encoder transformer (`model.py`), weight mapping
(`text_encoder_weight_map.py`), and the full Qwen Image 2.1 conditioning
pipeline. See the task plan.
"""

from __future__ import annotations

from .text_encoder_config import Qwen3VL8BTextEncoderConfig, detect_qwen3vl_8b_text_encoder_config

__all__ = [
    "Qwen3VL8BTextEncoderConfig",
    "detect_qwen3vl_8b_text_encoder_config",
]
