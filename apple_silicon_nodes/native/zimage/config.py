"""
Z-Image (NextDiT / Lumina2 family) model configuration.

Values confirmed against a real checkpoint on this machine
(`z_image_bf16.safetensors`, unet/ZImageBase/) and against comfy's own
detection branch (`comfy/model_detection.py:568-591`, `dim==3840` case).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..common import to_mlx_dtype


@dataclass(frozen=True)
class ZImageConfig:
    """Z-Image (NextDiT) architecture configuration.

    Attributes:
        dim: Hidden dimension (3840).
        n_layers: Number of main joint transformer blocks (30, checkpoint-verified).
        n_refiner_layers: Number of context_refiner/noise_refiner blocks each (2).
        n_heads: Number of attention heads (30).
        n_kv_heads: Number of K/V heads (30 — full MHA, no GQA, unlike Krea2's 48:12).
        multiple_of: FFN hidden dim rounding granularity (256).
        ffn_dim_multiplier: FFN hidden dim multiplier (8/3 — hidden=10240 for dim=3840).
        norm_eps: RMSNorm epsilon (1e-5, matches Lumina2 default).
        qk_norm: Whether attention applies per-head Q/K RMSNorm (True).
        cap_feat_dim: Text encoder (Qwen3-4B) hidden dim (2560, checkpoint-verified).
        axes_dims: RoPE axis dimensions (frame, height, width) = (32, 48, 48).
        rope_theta: RoPE base frequency (256.0 — much lower than FLUX's 10000).
        time_scale: Timestep multiplier before sinusoidal embedding (1000.0).
        pad_tokens_multiple: Pad cap/image token sequences to this multiple
            using a learned pad token (32) — present in the real checkpoint
            (`cap_pad_token`/`x_pad_token` keys exist), so implemented here
            rather than skipped; needed for correctness whenever the text
            prompt's token count isn't already a multiple of 32 (nearly
            always, in practice).
        patch_size: Patch size for image tokens (2).
        in_channels: Latent channels (16 — same as FLUX's VAE, see
            `latent_formats.Flux` inherited by comfy's Lumina2/ZImage class).
        dtype: MLX dtype string ("float16" or "bfloat16").

    Explicitly out of scope (not modeled here — no real checkpoint on this
    machine exercises them, and comfy's own NextDiT treats them as optional):
    the "omni" multi-reference-image path (`ref_latents`/`ref_contexts`,
    `timestep_zero_index` splitting), SigLIP image conditioning, and the
    pixel-space (no-VAE) `NextDiTPixelSpace` variant.
    """
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    n_kv_heads: int = 30
    multiple_of: int = 256
    ffn_dim_multiplier: float = 8.0 / 3.0
    norm_eps: float = 1e-5
    qk_norm: bool = True
    cap_feat_dim: int = 2560
    axes_dims: tuple[int, int, int] = (32, 48, 48)
    rope_theta: float = 256.0
    time_scale: float = 1000.0
    pad_tokens_multiple: int = 32
    patch_size: int = 2
    in_channels: int = 16
    dtype: str = "float16"

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return to_mlx_dtype(self.dtype, "Z-Image")

    @property
    def head_dim(self) -> int:
        """Dimension per attention head (128 for Z-Image)."""
        return self.dim // self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        """FFN hidden dim: multiple_of * ceil(ffn_dim_multiplier * dim / multiple_of)."""
        raw = int(self.ffn_dim_multiplier * self.dim)
        return self.multiple_of * ((raw + self.multiple_of - 1) // self.multiple_of)

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert self.dim % self.n_heads == 0, \
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        assert sum(self.axes_dims) == self.head_dim, \
            f"axes_dims sum ({sum(self.axes_dims)}) must equal head_dim ({self.head_dim})"
        assert self.n_layers > 0 and self.n_refiner_layers > 0
        assert self.dtype in ("float16", "bfloat16", "float32"), \
            f"unsupported dtype: {self.dtype}"

    def __post_init__(self) -> None:
        self.validate()
