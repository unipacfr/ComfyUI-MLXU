"""
Krea2 (SingleStreamDiT) model configuration.

Defines architecture parameters for the Krea2 model family:
- Krea2 Raw: standard, 20+ steps, CFG ~3
- Krea2 Turbo: distilled, 8 steps, CFG 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..common import to_mlx_dtype


@dataclass(frozen=True)
class Krea2Config:
    """Krea2 (SingleStreamDiT) model architecture configuration.

    Attributes:
        num_blocks: Number of single-stream transformer blocks (28 for Krea2)
        hidden_dim: Hidden dimension size (6144 for Krea2)
        mlp_dim: MLP expansion dimension (6720 for Krea2, SwiGLU)
        num_heads: Number of Q attention heads (48 for Krea2)
        num_kv_heads: Number of K/V heads for GQA (12 for Krea2, ratio 4:1)
        text_dim: Text embedding dimension from Qwen3-VL (2560)
        text_layers: Number of Qwen3-VL layer taps (12)
        text_layer_indices: Indices of Qwen3-VL layers to tap
        time_dim: Timestep embedding dimension (256)
        patch_size: Patch size for image tokens (2)
        latent_channels: Latent channels (16 for FLUX-compatible VAE)
        rope_axes_dim: RoPE axis dimensions (frame, height, width) = (32, 48, 48)
        rope_theta: RoPE base frequency (1000 for Krea2, vs 10000 for FLUX)
        guidance_embed: Whether the model includes guidance embedding (False for Krea2)
        dtype: MLX dtype string ("float16" or "bfloat16")
    """
    num_blocks: int = 28
    hidden_dim: int = 6144
    mlp_dim: int = 6720
    num_heads: int = 48
    num_kv_heads: int = 12
    text_dim: int = 2560
    text_layers: int = 12
    text_layer_indices: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    time_dim: int = 256
    patch_size: int = 2
    latent_channels: int = 16
    rope_axes_dim: tuple[int, int, int] = (32, 48, 48)
    rope_theta: float = 1000.0
    guidance_embed: bool = False
    dtype: str = "float16"

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return to_mlx_dtype(self.dtype, "Krea2")

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_dim // self.num_heads  # 6144 / 48 = 128

    @property
    def mlp_dim_padded(self) -> int:
        """MLP dimension rounded up to multiple of 128."""
        raw = int(2 / 3 * self.hidden_dim * 4)
        return int((raw + 127) // 128 * 128)  # 6720

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        assert self.num_blocks > 0, "num_blocks must be positive"
        assert self.patch_size > 0, "patch_size must be positive"
        assert self.rope_axes_dim[0] + self.rope_axes_dim[1] + self.rope_axes_dim[2] == self.head_dim, \
            f"rope_axes_dim sum ({sum(self.rope_axes_dim)}) must equal head_dim ({self.head_dim})"
        assert self.dtype in ("float16", "bfloat16", "float32"), \
            f"unsupported dtype: {self.dtype}"

    def __post_init__(self) -> None:
        self.validate()


# Krea2's real latent space is `comfy.latent_formats.Wan21` (a per-channel
# affine transform, 16 mean/std constants) -- NOT FLUX's scalar scale/shift.
# See native/config.py::WAN21_LATENTS_MEAN/STD and process_wan21_latent_in/
# out, applied in bridge.py::_unpack_krea2_latents and sampler/core.py's
# Identity Edit source-latent prep. (A previous FLUX-scale/shift constant
# pair lived here under these names -- wrong for Krea2 and unused; removed.)
