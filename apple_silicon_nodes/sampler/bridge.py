"""
Sampler-level bridge utilities.

Re-exports from the parent bridge module and adds Krea2-specific
conditioning helpers.
"""

from __future__ import annotations

# Re-export parent bridge functions
from ..bridge import (  # noqa: F401
    FLUX_LATENT_CHANNELS,
    FLUX2_LATENT_CHANNELS,
    FLUX2_VAE_DOWNSCALE,
    SDXL_LATENT_CHANNELS,
    _mps_allocator_gb,
    _process_rss_gb,
    collect_mlx_memory,
    conditioning_flux2_to_mlx,
    conditioning_sdxl_to_mlx,
    conditioning_to_mlx,
    conditioning_zimage_to_mlx,
    clear_mlx_cache,
    conditioning_qwen_image21_to_mlx,
    mlx_to_comfy_image,
    mlx_to_comfy_latent,
    mlx_to_comfy_latent_flux2,
    mlx_to_comfy_latent_krea2,
    mlx_to_comfy_latent_qwen_image21,
    mlx_to_comfy_latent_sdxl,
    mlx_to_comfy_latent_zimage,
    prepare_noise_from_latent,
    prepare_noise_from_latent_flux2,
    prepare_noise_from_latent_qwen_image21,
    prepare_noise_from_latent_sdxl,
    prepare_noise_from_latent_zimage,
    set_mlx_cache_limit_gb,
    tensor_to_mlx,
)

# ── Krea2 conditioning ─────────────────────────────────────────────────

from typing import Any

import mlx.core as mx
import numpy as np


def conditioning_krea2_to_mlx(
    conditioning: Any,
    precision: mx.Dtype,
) -> mx.array:
    """Extract text embeddings for Krea2 from conditioning.

    Krea2 uses Qwen3-VL-4B as text encoder, producing [B, T, 2560] per-layer.
    The full context is [B, T, 12*2560] = [B, T, 30720] (12-layer fused).

    Unlike FLUX (T5-XXL 4096 + CLIP-L 768), Krea2 has a single text stream
    that is stacked across 12 Qwen3-VL layers and fused by the txtfusion adapter.

    Args:
        conditioning: Comfy conditioning dict or list.
        precision: MLX dtype.

    Returns:
        prompt_embeds: [B, T, 30720] fused text embeddings.

    Note:
        This expects the conditioning to already contain Qwen3-VL outputs.
        In a full pipeline, the Qwen3-VL encoder would run before this step.
        If the conditioning has 2560-dim embeddings (single layer), they are
        repeated 12 times to create the fused format.
        If 4096-dim (T5), a warning is printed and truncated to 2560 then repeated.
    """
    guidance = None

    # Same fix as bridge.py::conditioning_to_mlx (the FLUX.1 KeyError: 0 bug):
    # unwrap ASDX_CLIPTextEncode's wrapper dict on the presence of the
    # "conditioning" key itself, not gated on "type" == "krea2" -- the
    # SD-style path (t5xxl left blank) tags it "type": "clip" instead, and
    # gating the unwrap on "krea2" left the wrapper dict un-unwrapped,
    # indexed like a list below (KeyError: 0, dict has no int key 0).
    if isinstance(conditioning, dict):
        if "guidance" in conditioning:
            try:
                guidance = float(conditioning["guidance"]) if conditioning.get("guidance") is not None else None
            except Exception:
                guidance = None
        if "conditioning" in conditioning:
            conditioning = conditioning["conditioning"]

        if "cond" in conditioning:
            cond = conditioning.get("cond")
        else:
            entry = conditioning[0] if conditioning else None
            if entry is None:
                raise RuntimeError("ASDX: positive conditioning is empty.")
            cond = entry[0]
            meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            if "guidance" in meta:
                try:
                    guidance = float(meta["guidance"])
                except Exception:
                    pass
    else:
        raise RuntimeError("ASDX: conditioning must be a list or dict.")

    cond_np = _to_numpy(cond)

    KREA2_TEXT_DIM = 2560
    KREA2_NUM_LAYERS = 12
    KREA2_FUSED_DIM = KREA2_TEXT_DIM * KREA2_NUM_LAYERS  # 30720

    if cond_np.ndim != 3:
        print(f"[ASDX] WARNING: Krea2 conditioning has unexpected ndim {cond_np.ndim}. "
              f"Expected 3.")
        cond_np = cond_np.reshape(1, cond_np.shape[0], cond_np.shape[-1]) if cond_np.ndim == 2 else cond_np

    if cond_np.shape[0] != 1:
        raise RuntimeError("ASDX: currently supports batch_size=1 only.")

    if cond_np.shape[-1] == KREA2_FUSED_DIM:
        # Already fused [B, T, 30720] — use as-is
        pass
    elif cond_np.shape[-1] == KREA2_TEXT_DIM:
        # Single layer [B, T, 2560] — repeat 12 times to fuse
        cond_np = np.tile(cond_np, (1, 1, KREA2_NUM_LAYERS))
        print(f"[ASDX] Krea2: repeating single-layer embedding to {KREA2_FUSED_DIM} dims")
    elif cond_np.shape[-1] == 4096:
        # T5 embedding — not compatible with Krea2. Print warning and truncate.
        print("[ASDX] WARNING: T5 embeddings (4096-dim) passed to Krea2 sampler. "
              "Truncating to 2560 then repeating 12x. Results will be incorrect.")
        cond_np = cond_np[:, :, :KREA2_TEXT_DIM]
        cond_np = np.tile(cond_np, (1, 1, KREA2_NUM_LAYERS))
    else:
        print(f"[ASDX] WARNING: Krea2 conditioning has unexpected dim {cond_np.shape[-1]}. "
              f"Expected {KREA2_TEXT_DIM} or {KREA2_FUSED_DIM}. Repeating to fused.")
        cond_np = np.tile(cond_np, (1, 1, KREA2_FUSED_DIM // max(cond_np.shape[-1], 1)))

    prompt_embeds = mx.array(cond_np).astype(precision)
    mx.eval(prompt_embeds)

    return prompt_embeds


def _to_numpy(value: Any) -> np.ndarray:
    """Convert any tensor-like object to a NumPy float32 array."""
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)
