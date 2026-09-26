"""Anima native MLX implementation: Cosmos-Predict2 MiniTrainDIT + LLM adapter.

The Qwen3-0.6B text encoder is ComfyUI's own `AnimaTEModel` (via
`ASDX_CLIPLoader`); see `bridge.conditioning_anima_to_mlx`.

Final status: the DiT (`model.py`), RoPE, LLM adapter and checkpoint loader
(bf16 and int8 convrot) all run natively in MLX -- verified against real
ComfyUI weights (`scripts/anima_parity.py`, fp32 cosine > 0.9999). Only the
Qwen3-0.6B text encoder stays in ComfyUI/PyTorch, loaded through
`ASDX_CLIPLoader`; the LLM adapter that consumes its hidden states runs once
per prompt in MLX (`AnimaTransformer.encode_context`), not per sampling step.
Not yet supported: LoRA (`lora.py` raises a clear error for Anima) and
img2img/inpaint (no `process_wan21_latent_in` noise-blend path)."""

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
