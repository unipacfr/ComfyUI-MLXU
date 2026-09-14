"""
Configuration module for the native MLX transformer.

Centralizes hyperparameters and provides validation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class FluxConfig:
    """FLUX.1 model architecture configuration.

    Attributes:
        num_double_blocks: Number of double transformer blocks (19 for FLUX.1-dev)
        num_single_blocks: Number of single transformer blocks (38 for FLUX.1-dev)
        hidden_dim: Hidden dimension size (3072 for FLUX.1)
        mlp_dim: MLP expansion dimension (12288 for FLUX.1)
        num_heads: Number of attention heads (24 for FLUX.1)
        guidance_embed: Whether the model includes guidance embedding (dev=True, schnell=False)
        dtype: MLX dtype string ("float16" or "bfloat16")
        in_channels: img_in's input width (64 = plain FLUX.1; 128 = depth/canny
            dev variants, detected from the checkpoint's own img_in.weight shape)
    """
    num_double_blocks: int = 19
    num_single_blocks: int = 38
    hidden_dim: int = 3072
    mlp_dim: int = 12288
    num_heads: int = 24
    guidance_embed: bool = True
    dtype: str = "float16"
    in_channels: int = 64

    @property
    def mlx_dtype(self) -> mx.Dtype:
        """Convert dtype string to mlx.core dtype."""
        dtype_map = {
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
        }
        if self.dtype not in dtype_map:
            raise ValueError(f"ASDX: unsupported dtype '{self.dtype}'. Use float16, bfloat16, or float32.")
        return dtype_map[self.dtype]

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_dim // self.num_heads

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.num_double_blocks > 0, "num_double_blocks must be positive"
        assert self.num_single_blocks > 0, "num_single_blocks must be positive"
        assert self.dtype in ("float16", "bfloat16", "float32"), \
            f"unsupported dtype: {self.dtype}"

    def __post_init__(self) -> None:
        self.validate()


# ── FLUX latent space constants ──────────────────────────────────────────
# Adapted from DiffusionKit's FluxLatentFormat

FLUX_LATENT_SCALE: float = 0.3611
"""Scale factor for FLUX latent space transformation."""

FLUX_LATENT_SHIFT: float = 0.1159
"""Shift factor for FLUX latent space transformation."""


def process_flux_latent_in(latent: Any) -> Any:
    """Process latent for model input: (latent - shift) * scale.

    Args:
        latent: Input latent tensor (any array-like).

    Returns:
        Processed latent tensor.
    """
    return (latent - FLUX_LATENT_SHIFT) * FLUX_LATENT_SCALE


def process_flux_latent_out(latent: Any) -> Any:
    """Process latent for model output: latent / scale + shift.

    Args:
        latent: Output latent tensor (any array-like).

    Returns:
        Reconstructed latent tensor.
    """
    return (latent / FLUX_LATENT_SCALE) + FLUX_LATENT_SHIFT


# ── Wan21 latent space constants (Krea2's VAE latent space) ─────────────
# Krea2 registers `latent_format = latent_formats.Wan21` (comfy/
# supported_models.py). Wan21 DOES define __init__/process_in/process_out
# -- despite `scale_factor=1.0` -- with a real per-channel affine (de-)
# whitening transform using 16 per-channel mean/std constants. This was
# previously misread as an identity no-op (scale_factor=1.0 only cancels
# part of the formula); confirmed by reading comfy/latent_formats.py::
# Wan21 directly. comfy/samplers.py's CFGGuider.inner_sample() applies
# `process_latent_out` unconditionally once, after the whole Euler loop,
# converting the model's internal ("whitened") latent space back to true
# VAE latent space -- values copied verbatim from the real source.

WAN21_LATENTS_MEAN: tuple[float, ...] = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
"""Per-channel latent mean for Wan21 (Krea2's VAE) latent space, 16 channels."""

WAN21_LATENTS_STD: tuple[float, ...] = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)
"""Per-channel latent std for Wan21 (Krea2's VAE) latent space, 16 channels."""


def process_wan21_latent_in(latent: Any) -> Any:
    """Process latent for model input: (latent - mean) / std, per-channel.

    Args:
        latent: Input latent tensor [B, 16, ...] (any array-like with a
                 channel axis at index 1).

    Returns:
        Processed (whitened) latent tensor.
    """
    shape = (1, 16) + (1,) * (latent.ndim - 2)
    mean = mx.array(WAN21_LATENTS_MEAN, dtype=latent.dtype).reshape(shape)
    std = mx.array(WAN21_LATENTS_STD, dtype=latent.dtype).reshape(shape)
    return (latent - mean) / std


def process_wan21_latent_out(latent: Any) -> Any:
    """Process latent for model output: latent * std + mean, per-channel.

    Args:
        latent: Output latent tensor [B, 16, ...] (any array-like with a
                 channel axis at index 1).

    Returns:
        Reconstructed (de-whitened) latent tensor.
    """
    shape = (1, 16) + (1,) * (latent.ndim - 2)
    mean = mx.array(WAN21_LATENTS_MEAN, dtype=latent.dtype).reshape(shape)
    std = mx.array(WAN21_LATENTS_STD, dtype=latent.dtype).reshape(shape)
    return latent * std + mean
