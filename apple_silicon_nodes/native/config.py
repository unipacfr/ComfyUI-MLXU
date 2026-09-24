"""
Configuration module for the native MLX transformer.

Centralizes hyperparameters and provides validation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .common import to_mlx_dtype


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
        return to_mlx_dtype(self.dtype, "FLUX.1")

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


# ── QwenImage21 latent space constants (Qwen Image 2.1's VAE latent space) ──
# `comfy/latent_formats.py::QwenImage21` defines a real per-channel affine
# (de-)whitening transform using 64 per-channel mean/std constants, same
# shape of bug as Wan21/Krea2 above (see `process_wan21_latent_out`'s own
# docstring) -- values copied verbatim from the real source. `comfy/
# samplers.py`'s `KSamplerX0Inpaint.inner_sample()` applies `process_latent_out`
# unconditionally once, after the whole sampling loop, converting the
# model's internal ("whitened") latent space back to true VAE latent space.

QWEN_IMAGE21_LATENTS_MEAN: tuple[float, ...] = (
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
    1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
    -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
    -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
    0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
)
"""Per-channel latent mean for QwenImage21's VAE latent space, 64 channels."""

QWEN_IMAGE21_LATENTS_STD: tuple[float, ...] = (
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
    3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
    3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
    3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
    3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
    3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
)
"""Per-channel latent std for QwenImage21's VAE latent space, 64 channels."""


def process_qwen_image21_latent_in(latent: Any) -> Any:
    """Process latent for model input: (latent - mean) / std, per-channel.

    Args:
        latent: Input latent tensor [B, 64, ...] (any array-like with a
                 channel axis at index 1).

    Returns:
        Processed (whitened) latent tensor.
    """
    shape = (1, 64) + (1,) * (latent.ndim - 2)
    mean = mx.array(QWEN_IMAGE21_LATENTS_MEAN, dtype=latent.dtype).reshape(shape)
    std = mx.array(QWEN_IMAGE21_LATENTS_STD, dtype=latent.dtype).reshape(shape)
    return (latent - mean) / std


def process_qwen_image21_latent_out(latent: Any) -> Any:
    """Process latent for model output: latent * std + mean, per-channel.

    Args:
        latent: Output latent tensor [B, 64, ...] (any array-like with a
                 channel axis at index 1).

    Returns:
        Reconstructed (de-whitened) latent tensor.
    """
    shape = (1, 64) + (1,) * (latent.ndim - 2)
    mean = mx.array(QWEN_IMAGE21_LATENTS_MEAN, dtype=latent.dtype).reshape(shape)
    std = mx.array(QWEN_IMAGE21_LATENTS_STD, dtype=latent.dtype).reshape(shape)
    return latent * std + mean
