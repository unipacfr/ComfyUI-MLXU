"""Generation metadata for sampled latents.

Adapted from Mflux-ComfyUI's save_images_with_metadata pattern.
Builds the dict of generation parameters attached to output latents
for reproducibility and tracing.
"""

from __future__ import annotations

import time
from typing import Any


def build_generation_metadata(
    model_name: str = "unknown",
    model_type: str = "dev",
    precision: str = "float16",
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 3.5,
    lora_names: list[str] | None = None,
    lora_scales: list[float] | None = None,
    mode: str = "text2img",
    **extra: Any,
) -> dict[str, Any]:
    """Build a metadata dictionary from generation parameters.

    Args:
        model_name: Checkpoint/model filename.
        model_type: "dev" or "schnell".
        precision: "float16" or "bfloat16".
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        seed: Random seed used.
        width: Image width in pixels.
        height: Image height in pixels.
        steps: Number of denoising steps.
        cfg: CFG scale.
        lora_names: List of LoRA filenames used.
        lora_scales: List of corresponding LoRA scales.
        mode: Sampling mode (text2img, img2img, inpainting, etc.).
        **extra: Additional metadata fields to include.

    Returns:
        Complete metadata dictionary ready for JSON serialization.
    """
    meta: dict[str, Any] = {
        "generator": "ComfyUI-MLXU (ASDX)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epoch": int(time.time()),
        "model": {
            "name": model_name,
            "type": model_type,
            "precision": precision,
        },
        "generation": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": cfg,
            "mode": mode,
        },
    }

    if lora_names:
        meta["lora"] = {
            "names": lora_names,
            "scales": lora_scales or [],
        }

    if extra:
        meta["extras"] = extra

    return meta
