"""
SDXL (UNetModel) architecture configuration.

Covers SDXL base and its checkpoint-compatible finetunes (Illustrious, Pony,
etc. — same UNet/CLIP architecture, different weights).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..common import to_mlx_dtype


@dataclass(frozen=True)
class SDXLConfig:
    """SDXL UNet architecture configuration.

    Matches comfy's SDXL unet_config (`comfy/supported_models.py` + auto
    detection in `comfy/model_detection.py:1116-1244` against a real
    checkpoint's tensor shapes).

    Attributes:
        model_channels: Base channel count (320 for SDXL).
        channel_mult: Channel multiplier per resolution level (1, 2, 4).
        num_res_blocks: ResBlocks per resolution level (2 for SDXL).
        transformer_depth: SpatialTransformer depth per input-block pair, one
            entry per (level, res-block-index) — (0, 0, 2, 2, 10, 10) for
            SDXL: level 0 has no attention, level 1 has depth 2, level 2 has
            depth 10. Mirrored on the output (up) path.
        transformer_depth_middle: SpatialTransformer depth in the middle
            block (10 for SDXL).
        context_dim: Cross-attention context dimension — concatenated
            CLIP-L(768) + CLIP-G(1280) = 2048.
        adm_in_channels: ADM/"y" vector dimension — pooled CLIP-G(1280) + 6
            sinusoidal(256) size/crop scalars = 2816.
        num_head_channels: Channels per attention head (64) — actual head
            count is `channels // num_head_channels`, computed per block.
        in_channels: UNet input latent channels (4).
        out_channels: UNet output latent channels (4).
        dtype: MLX dtype string ("float16" or "bfloat16").
    """
    model_channels: int = 320
    channel_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    transformer_depth: tuple[int, ...] = (0, 0, 2, 2, 10, 10)
    transformer_depth_middle: int = 10
    context_dim: int = 2048
    adm_in_channels: int = 2816
    num_head_channels: int = 64
    in_channels: int = 4
    out_channels: int = 4
    dtype: str = "float16"

    @property
    def transformer_depth_output(self) -> tuple[int, ...]:
        """SpatialTransformer depth per output-path (up) block.

        The output/up path runs `num_res_blocks + 1` iterations per
        resolution level (one extra to consume the matching skip
        connection), vs `num_res_blocks` on the input/down path — so this
        list has more entries than `transformer_depth` even though it
        encodes the same per-level depths. Matches comfy's auto-detected
        `transformer_depth_output` (e.g. SDXL: `transformer_depth=[0,0,2,2,10,10]`
        -> `transformer_depth_output=[0,0,0,2,2,2,10,10,10]`,
        `comfy/model_detection.py:1416`).
        """
        per_level = self.transformer_depth[0::self.num_res_blocks]
        return tuple(d for d in per_level for _ in range(self.num_res_blocks + 1))

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return to_mlx_dtype(self.dtype, "SDXL")

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert self.model_channels > 0, "model_channels must be positive"
        assert len(self.transformer_depth) == self.num_res_blocks * len(self.channel_mult), (
            f"transformer_depth ({len(self.transformer_depth)} entries) must have "
            f"num_res_blocks ({self.num_res_blocks}) entries per resolution level "
            f"({len(self.channel_mult)} levels)"
        )
        assert self.model_channels % self.num_head_channels == 0, (
            f"model_channels ({self.model_channels}) must be divisible by "
            f"num_head_channels ({self.num_head_channels})"
        )
        assert self.dtype in ("float16", "bfloat16", "float32"), \
            f"unsupported dtype: {self.dtype}"

    def __post_init__(self) -> None:
        self.validate()


# ── SDXL latent space constants ─────────────────────────────────────────
# KL-f8 VAE (same architecture as SD1.5, retrained weights). No shift term,
# unlike FLUX/Krea2 — comfy/latent_formats.py:32-45 confirms SDXL.scale_factor
# with no `shift_factor` attribute.

SDXL_LATENT_SCALE: float = 0.13025
"""Scale factor for SDXL latent space transformation (comfy/latent_formats.py::SDXL)."""


def process_sdxl_latent_in(latent: Any) -> Any:
    """Process latent for model input: latent * scale."""
    return latent * SDXL_LATENT_SCALE


def process_sdxl_latent_out(latent: Any) -> Any:
    """Process latent for model output: latent / scale."""
    return latent / SDXL_LATENT_SCALE
