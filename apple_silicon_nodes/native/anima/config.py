"""Anima (Cosmos-Predict2 MiniTrainDIT + LLM adapter) config.

Values from `comfy/model_detection.py:855-900` (the `anima` branch) and
`comfy/ldm/anima/model.py::LLMAdapter` defaults, checked against the real
checkpoint headers (2048-wide, 28 blocks, 17-ch patch input)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import mlx.core as mx

from ..common import to_mlx_dtype

_HEADS_BY_WIDTH = {2048: 16, 5120: 40}


@dataclass(frozen=True)
class AnimaConfig:
    dtype: str = "bfloat16"
    in_channels: int = 16
    out_channels: int = 16
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    crossattn_emb_channels: int = 1024
    adaln_lora_dim: int = 256
    patch_spatial: int = 2
    # (t, h, w) RoPE NTK extrapolation ratios for in_channels == 16.
    rope_ratios: tuple[float, float, float] = (1.0, 4.0, 4.0)
    adapter_dim: int = 1024
    adapter_layers: int = 6
    adapter_heads: int = 16
    adapter_vocab: int = 32128
    min_context_len: int = 512

    @property
    def head_dim(self) -> int:
        return self.model_channels // self.num_heads

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return to_mlx_dtype(self.dtype, "Anima")


def detect_anima_config(state: dict[str, mx.array], dtype: str) -> AnimaConfig:
    """`state` keys are already prefix-stripped (`blocks.N...`, `x_embedder...`)."""
    w = state["x_embedder.proj.1.weight"]
    model_channels = int(w.shape[0])
    in_channels = int(w.shape[1]) // 4 - 1
    if in_channels != 16:
        raise ValueError(
            f"ASDX: Anima checkpoint has in_channels={in_channels}; only the 16-channel "
            "text-to-image variant is supported (17 is Cosmos image-to-video)."
        )
    if model_channels not in _HEADS_BY_WIDTH:
        raise ValueError(f"ASDX: unknown Anima model_channels={model_channels}.")
    num_blocks = 1 + max(int(m.group(1)) for k in state if (m := re.match(r"blocks\.(\d+)\.", k)))
    return AnimaConfig(
        dtype=dtype,
        model_channels=model_channels,
        num_blocks=num_blocks,
        num_heads=_HEADS_BY_WIDTH[model_channels],
    )
