"""
Flux2 (FLUX.2/Klein) model configuration.

Defaults verified against a real checkpoint on this machine
(`diffusion_models/Flux.2 Klein 9B/base model/flux2Klein_9b.safetensors`,
201 top-level key patterns, matched 100%) and against comfy's own
config-detection branch for `image_model == "flux2"`
(`comfy/model_detection.py:237-289`) and the `Flux2` model config
(`comfy/supported_models.py:794`, `sampling_settings={"shift": 2.02}`).

"Klein" and "Flux2" (the larger "D" variant) share this exact same
`comfy.ldm.flux.model.Flux` architecture with `global_modulation=True` —
but they are NOT just a checkpoint-size relabeling of one fixed shape.
Verified against two distinct real checkpoints on this machine
(`Flux.2 Klein 9B/.../flux2Klein_9b.safetensors`: hidden_size=4096,
8 double + 24 single blocks, context_in_dim=12288, no guidance embed; vs
`Flux.2 D/.../flux2_dev_fp8mixed.safetensors`: hidden_size=6144,
8 double + 48 single blocks, context_in_dim=15360, HAS a guidance embed)
— `detect_flux2_config()` below derives these per-checkpoint, mirroring
`comfy/model_detection.py:235-289` exactly, rather than assuming Klein's
shape for every Flux2 checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..common import to_mlx_dtype


@dataclass(frozen=True)
class Flux2Config:
    """Flux2/Klein model architecture configuration.

    Attributes:
        num_double_blocks: Number of double-stream blocks (8 for the 9B Klein
            checkpoint on this machine — comfy derives this per-checkpoint via
            `count_blocks`, not a fixed constant across all Flux2 variants).
        num_single_blocks: Number of single-stream blocks (24 for the same).
        hidden_size: Hidden dimension (4096, from `txt_in.weight.shape[0]`).
        context_in_dim: Text embedding input dim (12288 = 3 tapped hidden
            layers of the text encoder concatenated — `layer=[9,18,27]` for
            Klein/Qwen3, `layer=[10,20,30]` for Flux2-D/Mistral3, each
            4096-dim — see `comfy/text_encoders/flux.py`).
        mlp_ratio: MLP expansion ratio (3.0, vs FLUX.1's 4.0).
        num_heads: Attention heads (`hidden_size // sum(axes_dim)` = 4096/128 = 32
            — comfy hardcodes 48 as an initial guess then immediately
            overwrites it with this derived value; 48 is dead code for every
            real checkpoint).
        axes_dim: RoPE axis dimensions, 4 axes (vs FLUX.1's 3) — sums to head_dim.
        theta: RoPE base frequency (2000, vs FLUX.1's 10000).
        in_channels: Latent channels (128, vs FLUX.1's 16 — Flux2's VAE
            downscales 16x spatially instead of 8x, so no 2x2 patch packing
            is needed: `patch_size=1`).
        patch_size: Always 1 for Flux2 (no token patchify, unlike FLUX.1's 2).
        qkv_bias: False — Flux2 is bias-free throughout (`ops_bias=False` in
            comfy's detected config applies to every Linear, not just qkv).
        guidance_embed: False for the checkpoint on this machine (no
            `guidance_in.in_layer.weight` key) — Flux2/Klein has no
            guidance-embedding mechanism.
        global_modulation: True — the defining structural difference from
            FLUX.1: three top-level `Modulation` layers
            (`double_stream_modulation_img/txt`, `single_stream_modulation`)
            are computed ONCE per forward pass and shared across every
            double/single block, instead of each block owning its own.
        mlp_silu_act: True — each MLP is a fused SiLU-gated GLU
            (`Linear(dim, 2*mlp_hidden, bias=False) -> silu(x1)*x2 ->
            Linear(mlp_hidden, dim, bias=False)`), not FLUX.1's GELU-tanh MLP.
        ref_index_scale: 10.0 — Kontext-style reference latents (if ever
            wired up) get RoPE axis-0 index `i * ref_index_scale` per
            reference, vs FLUX.1's `i * 1.0`. NOT used by the txt2img-only
            forward pass implemented here — recorded for future Phase E work.
        txt_ids_dim: Which RoPE axis (of the 4) carries the text tokens'
            sequential position index (3 — the axis image tokens never touch,
            since `process_img` only ever fills axes 0/1/2). FLUX.1's
            `txt_ids_dims` is empty (all axes 0 for text); Flux2 gives text
            its own dedicated axis instead.
        dtype: MLX dtype string ("float16" or "bfloat16").
    """
    num_double_blocks: int = 8
    num_single_blocks: int = 24
    hidden_size: int = 4096
    context_in_dim: int = 12288
    mlp_ratio: float = 3.0
    num_heads: int = 32
    axes_dim: tuple[int, int, int, int] = (32, 32, 32, 32)
    theta: float = 2000.0
    in_channels: int = 128
    patch_size: int = 1
    qkv_bias: bool = False
    guidance_embed: bool = False
    global_modulation: bool = True
    mlp_silu_act: bool = True
    ref_index_scale: float = 10.0
    txt_ids_dim: int = 3
    dtype: str = "float16"

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return to_mlx_dtype(self.dtype, "FLUX.2")

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden(self) -> int:
        """MLP hidden dim (down-projection input size): hidden_size * mlp_ratio."""
        return int(self.hidden_size * self.mlp_ratio)

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        assert sum(self.axes_dim) == self.head_dim, \
            f"axes_dim sum ({sum(self.axes_dim)}) must equal head_dim ({self.head_dim})"
        assert self.num_double_blocks > 0, "num_double_blocks must be positive"
        assert self.num_single_blocks > 0, "num_single_blocks must be positive"
        assert 0 <= self.txt_ids_dim < len(self.axes_dim), \
            f"txt_ids_dim ({self.txt_ids_dim}) out of range for {len(self.axes_dim)} axes"
        assert self.dtype in ("float16", "bfloat16", "float32"), \
            f"unsupported dtype: {self.dtype}"

    def __post_init__(self) -> None:
        self.validate()


# ── Flux2 latent space constants ────────────────────────────────────────
# Flux2's `latent_formats.Flux2` (comfy/latent_formats.py:192) does NOT
# override `LatentFormat.scale_factor` (base class default: 1.0) and defines
# no shift at all — unlike FLUX.1/Krea2/Z-Image, which all share
# scale=0.3611/shift=0.1159. Flux2 latents pass through RAW, unscaled. This
# was verified by reading the real source, not assumed from the other
# families' pattern — a scale/shift mismatch here would silently degrade
# every generation without erroring.

FLUX2_LATENT_SCALE: float = 1.0
"""Flux2 latent scale factor — 1.0 (no scaling), confirmed against comfy source."""

FLUX2_LATENT_SHIFT: float = 0.0
"""Flux2 latent shift — 0.0 (no shift), confirmed against comfy source."""


def process_flux2_latent_in(latent: Any) -> Any:
    """Process latent for model input: (latent - shift) * scale. No-op for Flux2."""
    return (latent - FLUX2_LATENT_SHIFT) * FLUX2_LATENT_SCALE


def process_flux2_latent_out(latent: Any) -> Any:
    """Process latent for model output: latent / scale + shift. No-op for Flux2."""
    return (latent / FLUX2_LATENT_SCALE) + FLUX2_LATENT_SHIFT


def detect_flux2_config(state_dict: dict[str, Any], dtype: str = "float16") -> Flux2Config:
    """Derive a Flux2Config from a (normalized, mapped) checkpoint state dict.

    Mirrors `comfy/model_detection.py`'s `image_model == "flux2"` branch
    (lines 235-289): `hidden_size`/`context_in_dim`/`in_channels` come from
    `img_in.weight`/`txt_in.weight` shapes, block counts from counting
    `double_blocks.N.`/`single_blocks.N.` keys, `guidance_embed` from the
    presence of `guidance_in.in_layer.weight`. `axes_dim`, `theta`,
    `mlp_ratio`, bias flags, and `global_modulation`/`mlp_silu_act` are NOT
    shape-derived — comfy hardcodes them as constants for this branch too
    (they're what distinguishes "flux2" detection from plain "flux" in the
    first place, not something read from the checkpoint).

    Required because this project has two real checkpoints with genuinely
    different shapes (Klein 9B vs the larger Flux2-D, which also has a
    guidance embedding Klein lacks) — a single hardcoded config would
    silently near-zero-match whichever checkpoint doesn't match the default.
    """
    img_in_w = state_dict.get("img_in.weight")
    txt_in_w = state_dict.get("txt_in.weight")
    if img_in_w is None or txt_in_w is None:
        raise ValueError(
            "ASDX: cannot detect Flux2 config — checkpoint is missing "
            "img_in.weight or txt_in.weight after key normalization."
        )

    hidden_size = int(img_in_w.shape[0])
    in_channels = int(img_in_w.shape[1])  # patch_size is always 1 for Flux2
    context_in_dim = int(txt_in_w.shape[1])

    double_idx: set[int] = set()
    single_idx: set[int] = set()
    for key in state_dict:
        if key.startswith("double_blocks."):
            double_idx.add(int(key.split(".")[1]))
        elif key.startswith("single_blocks."):
            single_idx.add(int(key.split(".")[1]))
    if not double_idx or not single_idx:
        raise ValueError("ASDX: cannot detect Flux2 config — no double_blocks/single_blocks keys found.")

    num_double_blocks = len(double_idx)
    num_single_blocks = len(single_idx)

    axes_dim = (32, 32, 32, 32)
    num_heads = hidden_size // sum(axes_dim)

    guidance_embed = "guidance_in.in_layer.weight" in state_dict

    return Flux2Config(
        num_double_blocks=num_double_blocks,
        num_single_blocks=num_single_blocks,
        hidden_size=hidden_size,
        context_in_dim=context_in_dim,
        mlp_ratio=3.0,
        num_heads=num_heads,
        axes_dim=axes_dim,
        theta=2000.0,
        in_channels=in_channels,
        patch_size=1,
        qkv_bias=False,
        guidance_embed=guidance_embed,
        global_modulation=True,
        mlp_silu_act=True,
        ref_index_scale=10.0,
        txt_ids_dim=3,
        dtype=dtype,
    )
