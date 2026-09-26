"""
Empty Latent
============
Creates empty latents (FLUX/Krea2/Z-Image, Flux2/Klein, or SDXL)
optimized for Apple Silicon.
"""

from __future__ import annotations

from typing import Any

import torch

import comfy.model_management
from comfy_api.latest import io


# ── Resolution presets ───────────────────────────────────────────────

# FLUX works best with resolutions divisible by 16.
# These are common aspect ratios optimized for Apple Silicon memory.
_RESOLUTION_PRESETS = {
    "1024 x 1024 (1:1)": (1024, 1024),
    "1024 x 1024 (1:1) XL": (1152, 1152),
    "832 x 1216 (5:8)": (832, 1216),
    "1216 x 832 (8:5)": (1216, 832),
    "896 x 1152 (3:4)": (896, 1152),
    "1152 x 896 (4:3)": (1152, 896),
    "768 x 1344 (9:16)": (768, 1344),
    "1344 x 768 (16:9)": (1344, 768),
    "640 x 1344 (2:5)": (640, 1344),
    "1344 x 640 (5:2)": (1344, 640),
}


class ASDX_EmptyLatent(io.ComfyNode):
    """Create an empty latent tensor for FLUX/Krea2/Z-Image, Flux2/Klein, or SDXL.

    The latent is placed on the appropriate device (MPS when available)
    for zero-copy with the MLX sampler.
    """

    # (channels, spatial downscale) per model family -- FLUX.1/Krea2/Z-Image
    # share the same 16ch/8x VAE latent; Flux2/Klein uses a distinct
    # 128ch/16x VAE (see bridge.py::FLUX2_LATENT_CHANNELS/FLUX2_VAE_DOWNSCALE);
    # SDXL uses the standard 4ch/8x VAE (see bridge.py::SDXL_LATENT_CHANNELS).
    _LATENT_FORMATS = {
        "flux": (16, 8),
        "krea2": (16, 8),
        "zimage": (16, 8),
        "flux2": (128, 16),
        "sdxl": (4, 8),
        # 64ch/16x VAE, no patchify (see bridge.py::QWEN_IMAGE21_LATENT_CHANNELS/
        # QWEN_IMAGE21_VAE_DOWNSCALE) -- confirmed via real comfy.sd.VAE.
        "qwen_image21": (64, 16),
        # Wan21 16ch/8x VAE latent; the DiT patchifies 2x2 internally.
        "anima": (16, 8),
    }

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_EmptyLatent",
            display_name="🍏 ASDX Empty Latent",
            category="ASDX/Latent",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=2048, step=16),
                io.Int.Input("height", default=1024, min=64, max=2048, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=8),
                io.Combo.Input(
                    "aspect_ratio", options=["auto"] + list(_RESOLUTION_PRESETS.keys()),
                    default="auto", optional=True,
                ),
                io.Combo.Input(
                    "latent_format", options=list(cls._LATENT_FORMATS.keys()),
                    default="flux", optional=True,
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        width: int,
        height: int,
        batch_size: int,
        aspect_ratio: str = "auto",
        latent_format: str = "flux",
    ) -> io.NodeOutput:
        # Apply aspect ratio preset if selected
        if aspect_ratio != "auto" and aspect_ratio in _RESOLUTION_PRESETS:
            width, height = _RESOLUTION_PRESETS[aspect_ratio]

        # Ensure divisible by 16
        width = (width // 16) * 16
        height = (height // 16) * 16

        channels, downscale = cls._LATENT_FORMATS.get(latent_format, cls._LATENT_FORMATS["flux"])

        # Create latent on MPS device for zero-copy with sampler
        device = cls._get_device()
        latent = torch.zeros(
            [batch_size, channels, height // downscale, width // downscale],
            device=device,
            dtype=comfy.model_management.intermediate_dtype(),
        )

        print(f"[ASDX] Empty Latent ({latent_format}): {width}x{height}, batch={batch_size}, "
              f"latent_shape=[{batch_size}, {channels}, {height//downscale}, {width//downscale}]")

        return io.NodeOutput({"samples": latent, "downscale_ratio_spacial": downscale})

    @staticmethod
    def _get_device() -> torch.device:
        """Get the best available device."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")


NODE_LIST = [
    ASDX_EmptyLatent,
]
