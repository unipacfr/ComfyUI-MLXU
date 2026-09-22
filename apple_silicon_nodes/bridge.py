"""
PyTorch <-> MLX Bridge
======================
Conversion utilities optimized for Apple Silicon Unified Memory.

Key patterns:
  - PyTorch tensor -> MLX array:  .detach().cpu().numpy() -> mx.array()
  - MLX array -> PyTorch tensor:  np.array(mx_arr) -> torch.from_numpy()
  - Always call mx.eval() before converting MLX -> NumPy
  - Reuse NumPy buffers when possible to minimize allocations
"""

from __future__ import annotations

import gc
from typing import Any

import mlx.core as mx
import numpy as np
import torch


# ── Constants ─────────────────────────────────────────────────────────

# FLUX uses 16 latent channels; SD1.5/SDXL use 4
FLUX_LATENT_CHANNELS = 16
SDXL_LATENT_CHANNELS = 4
# Flux2/Klein: 128 latent channels, VAE downscales 16x spatially (vs FLUX's
# 8x) and patch_size=1 (no further 2x2 token packing) — see
# comfy/latent_formats.py::Flux2 (spacial_downscale_ratio=16).
FLUX2_LATENT_CHANNELS = 128
FLUX2_VAE_DOWNSCALE = 16
# Qwen Image 2.1: 64 latent channels, VAE downscales 16x spatially, DiT
# operates on the latent grid directly (no patch packing) — comfy/latent_formats.py.
QWEN_IMAGE21_LATENT_CHANNELS = 64
QWEN_IMAGE21_VAE_DOWNSCALE = 16

# ComfyUI image format: [B, H, W, C] range [0, 1] float32
# FLUX VAE input/output expects [B, C, H, W] range [-1, 1] float16


# ── PyTorch -> MLX ───────────────────────────────────────────────────

def tensor_to_mlx(tensor: Any, dtype: mx.Dtype | None = None) -> mx.array:
    """Convert a PyTorch tensor to an MLX array.

    Handles both on-CPU and on-device tensors. Calls .detach() and .cpu()
    automatically. Uses float32 intermediate to avoid precision loss during
    the numpy round-trip.
    """
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().float().numpy().astype(np.float32, copy=False)
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    out = mx.array(arr)
    if dtype is not None:
        out = out.astype(dtype)
    mx.eval(out)
    return out


def conditioning_to_mlx(
    conditioning: Any,
    precision: mx.Dtype,
) -> tuple[mx.array, mx.array, float | None]:
    """Extract T5 embeddings, pooled CLIP output, and guidance from Comfy conditioning.

    Returns (prompt_embeds, pooled_prompt_embeds, guidance).
    prompt_embeds: [B, T, 4096]  (T5 XXL)
    pooled_prompt_embeds: [B, 768]  (CLIP-L)
    """
    guidance = None

    # Handle ASDX_CLIPTextEncode's wrapper dict. It's produced for both the
    # FLUX ("type": "flux1", t5xxl filled) and single-CLIP ("type": "clip",
    # t5xxl left blank -- text is then reused for every sub-tokenizer) modes,
    # so unwrap on the "conditioning" key itself rather than gating on
    # "type" == "flux1" -- otherwise a FLUX-type dual CLIP with an empty
    # t5xxl field falls through and the wrapper dict gets indexed like a
    # list below (KeyError: 0).
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
            pooled = conditioning.get("pooled_output", conditioning.get("pooled"))
        else:
            # Standard Comfy conditioning list: [[tensor, {meta}], ...]
            entry = conditioning[0] if conditioning else None
            if entry is None:
                raise RuntimeError("ASDX: positive conditioning is empty.")
            cond = entry[0]
            meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            pooled = meta.get("pooled_output")
            if "guidance" in meta:
                try:
                    guidance = float(meta["guidance"])
                except Exception:
                    pass
    else:
        raise RuntimeError("ASDX: conditioning must be a list or dict.")

    if pooled is None:
        raise RuntimeError("ASDX: conditioning has no pooled_output.")

    # Convert to numpy then MLX
    cond_np = _to_numpy(cond)
    pooled_np = _to_numpy(pooled)

    # Validate shapes
    if cond_np.ndim != 3 or cond_np.shape[-1] != 4096:
        raise RuntimeError(
            f"ASDX: expected T5 embeddings [B,T,4096], got {cond_np.shape}"
        )
    if pooled_np.ndim != 2 or pooled_np.shape[-1] != 768:
        raise RuntimeError(
            f"ASDX: expected pooled CLIP [B,768], got {pooled_np.shape}"
        )
    if cond_np.shape[0] != 1 or pooled_np.shape[0] != 1:
        raise RuntimeError("ASDX: currently supports batch_size=1 only.")

    prompt_embeds = mx.array(cond_np).astype(precision)
    pooled_prompt_embeds = mx.array(pooled_np).astype(precision)
    mx.eval(prompt_embeds, pooled_prompt_embeds)

    return prompt_embeds, pooled_prompt_embeds, guidance


def _to_numpy(value: Any) -> np.ndarray:
    """Convert any tensor-like object to a NumPy float32 array."""
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def conditioning_sdxl_to_mlx(
    conditioning: Any,
    precision: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    """Extract cross-attn context and pooled CLIP-G from SD-style Comfy conditioning.

    `ASDX_CLIPTextEncode`'s SD/SDXL/Pony path (`conditioning.py`) produces
    `{"type": "clip", "conditioning": [[cond_tensor, {"pooled_output": ...}], ...]}`
    via ComfyUI's own dual-CLIP object (`comfy.sd.CLIP` loaded with
    `clip_type="sdxl"`) — `cond_tensor` is already the concatenated
    CLIP-L(768)+CLIP-G(1280) cross-attn context (2048-dim), matching
    `comfy/sdxl_clip.py::SDXLClipModel.encode_token_weights`.

    Returns (cond, pooled). cond: [B, T, 2048]. pooled: [B, 1280] (CLIP-G only).
    """
    if isinstance(conditioning, dict):
        conditioning = conditioning.get("conditioning", conditioning)

    entry = conditioning[0] if conditioning else None
    if entry is None:
        raise RuntimeError("ASDX: SDXL conditioning is empty.")
    cond = entry[0]
    meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
    pooled = meta.get("pooled_output")
    if pooled is None:
        raise RuntimeError("ASDX: SDXL conditioning has no pooled_output (CLIP-G pooled).")

    cond_np = _to_numpy(cond)
    pooled_np = _to_numpy(pooled)

    if cond_np.ndim != 3 or cond_np.shape[-1] != 2048:
        raise RuntimeError(f"ASDX: expected SDXL context [B,T,2048], got {cond_np.shape}")
    if pooled_np.ndim != 2 or pooled_np.shape[-1] != 1280:
        raise RuntimeError(f"ASDX: expected pooled CLIP-G [B,1280], got {pooled_np.shape}")
    if cond_np.shape[0] != 1 or pooled_np.shape[0] != 1:
        raise RuntimeError("ASDX: currently supports batch_size=1 only.")

    cond_mlx = mx.array(cond_np).astype(precision)
    pooled_mlx = mx.array(pooled_np).astype(precision)
    mx.eval(cond_mlx, pooled_mlx)
    return cond_mlx, pooled_mlx


# ── MLX -> PyTorch ───────────────────────────────────────────────────

def mlx_to_comfy_image(decoded: mx.array) -> torch.Tensor:
    """Convert MLX decoded VAE output to ComfyUI IMAGE format [B,H,W,C].

    VAE output is [B, C, H, W] in range [-1, 1]. This normalizes to [0, 1]
    and transposes to ComfyUI convention.
    """
    image = mx.clip((decoded.astype(mx.float32) / 2.0) + 0.5, 0.0, 1.0)
    image = mx.transpose(image, (0, 2, 3, 1))
    mx.eval(image)
    return torch.from_numpy(np.array(image, dtype=np.float32))


def mlx_to_comfy_latent(
    latents: mx.array,
    height: int,
    width: int,
    template: dict[str, Any],
) -> dict[str, Any]:
    """Convert MLX latent to ComfyUI LATENT dict.

    FLUX latents are packed as [B, H_lat*W_lat, 64] where 64 = 16*2*2
    (16 channels * 2*2 patch). Unpack back to [B, 16, H/8, W/8].
    """
    samples = _unpack_flux_latents(latents, height, width)
    out = dict(template)
    out["samples"] = samples
    return out


def _unpack_flux_latents(
    latents: mx.array,
    height: int,
    width: int,
) -> torch.Tensor:
    """Unpack packed FLUX latent [B, NW*NH, 64] -> [B, 16, H/8, W/8].

    The denoising loop runs entirely in the model's normalized latent space
    (matching `comfy/samplers.py:1236`'s `process_latent_out`, applied once
    at this exact sampling boundary) -- must be converted back to raw VAE
    latent space here before the caller hands this to VAE decode, or the
    decoded image comes out visibly grainy/speckled (fp16 VAE decoders are
    very sensitive to latent scale). NOT used for Z-Image, despite it
    inheriting the same `latent_formats.Flux` 16ch/0.3611/0.1159 VAE space
    -- Z-Image's own per-token patch-channel order is `[pH,pW,C]`, not
    FLUX's `[C,pH,pW]` (see `_unpack_zimage_latents`'s docstring for the
    comfy-source-verified difference); use `mlx_to_comfy_latent_zimage`.
    """
    from .native.config import process_flux_latent_out

    latent_h = height // 8
    latent_w = width // 8

    # Reshape: [B, NH, NW, 16, 2, 2] -> [B, 16, NH, 2, NW, 2] -> [B, 16, NH*2, NW*2]
    unpacked = mx.reshape(latents, (1, latent_h // 2, latent_w // 2, 16, 2, 2))
    unpacked = mx.transpose(unpacked, (0, 3, 1, 4, 2, 5))
    unpacked = mx.reshape(unpacked, (1, 16, latent_h // 2 * 2, latent_w // 2 * 2))
    unpacked = process_flux_latent_out(unpacked.astype(mx.float32))
    mx.eval(unpacked)

    return torch.from_numpy(np.array(unpacked, dtype=np.float32))


def mlx_to_comfy_latent_zimage(
    latents: mx.array,
    height: int,
    width: int,
    template: dict[str, Any],
) -> dict[str, Any]:
    """Convert MLX Z-Image latent to a ComfyUI LATENT dict.

    Z-Image latents are packed as [B, H_lat*W_lat, 64], token order
    [pH, pW, C] (see `_unpack_zimage_latents`). Unpack back to
    [B, 16, H/8, W/8].
    """
    samples = _unpack_zimage_latents(latents, height, width)
    out = dict(template)
    out["samples"] = samples
    return out


def _unpack_zimage_latents(
    latents: mx.array,
    height: int,
    width: int,
) -> torch.Tensor:
    """Unpack packed Z-Image latent [B, NH*NW, 64] -> [B, 16, H/8, W/8].

    Inverse of `prepare_noise_from_latent_zimage`'s packing, matching
    comfy/ldm/lumina/model.py::NextDiT.unpatchify exactly:
    `x[i][begin:end].view(H//pH, W//pW, pH, pW, C).permute(4,0,2,1,3)
    .flatten(3,4).flatten(1,2)` -- i.e. the packed per-token dim is
    `[pH, pW, C]` (verified against the real comfy source), NOT FLUX's
    `[C, pH, pW]` (`_unpack_flux_latents`). Also applies
    `process_flux_latent_out` (same 0.3611/0.1159 constants as FLUX,
    since Z-Image shares `latent_formats.Flux` unchanged for scale/shift
    -- only the token axis order differs, not the VAE latent space).
    """
    from .native.config import process_flux_latent_out

    latent_h = height // 8
    latent_w = width // 8

    # Reshape: [B, NH, NW, 64] -> [B, NH, NW, pH, pW, 16] -> [B, 16, NH, pH, NW, pW] -> [B, 16, NH*2, NW*2]
    unpacked = mx.reshape(latents, (1, latent_h // 2, latent_w // 2, 2, 2, 16))
    unpacked = mx.transpose(unpacked, (0, 5, 1, 3, 2, 4))
    unpacked = mx.reshape(unpacked, (1, 16, latent_h // 2 * 2, latent_w // 2 * 2))
    unpacked = process_flux_latent_out(unpacked.astype(mx.float32))
    mx.eval(unpacked)

    return torch.from_numpy(np.array(unpacked, dtype=np.float32))


def mlx_to_comfy_latent_krea2(
    latents: mx.array,
    height: int,
    width: int,
    template: dict[str, Any],
) -> dict[str, Any]:
    """Convert MLX Krea2 latent to a ComfyUI LATENT dict.

    Krea2 latents are packed as [B, H_lat*W_lat, 64], token order [C, pH, pW]
    -- confirmed identical to FLUX by reading `comfy/ldm/krea2/model.py::
    process_img` (`rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)")`,
    the exact same call FLUX's own patchify uses). See `_unpack_krea2_latents`
    for the Wan21 per-channel de-whitening applied here.
    """
    samples = _unpack_krea2_latents(latents, height, width)
    out = dict(template)
    out["samples"] = samples
    return out


def _unpack_krea2_latents(
    latents: mx.array,
    height: int,
    width: int,
) -> torch.Tensor:
    """Unpack packed Krea2 latent [B, NH*NW, 64] -> [B, 16, H/8, W/8].

    Same `[C, pH, pW]` per-token axis order as FLUX (verified above), so the
    reshape/transpose below is identical to `_unpack_flux_latents`. The
    difference is scale/shift: `comfy.supported_models.Krea2.latent_format`
    is `latent_formats.Wan21`, which DOES define a real per-channel affine
    `process_out` (`latent * latents_std + latents_mean`, 16 channel-specific
    constants) despite `scale_factor=1.0` -- confirmed by reading comfy/
    latent_formats.py::Wan21 directly, after a prior assumption that it was
    an identity no-op turned out to be wrong (that assumption produced a
    ~2x-undersized final latent relative to real ComfyUI, verified against
    a same-seed reference run). `comfy/samplers.py`'s `CFGGuider.
    inner_sample()` applies this unconditionally, once, after the whole
    Euler loop -- see `native/config.py::process_wan21_latent_out`.
    """
    from .native.config import process_wan21_latent_out

    latent_h = height // 8
    latent_w = width // 8

    unpacked = mx.reshape(latents, (1, latent_h // 2, latent_w // 2, 16, 2, 2))
    unpacked = mx.transpose(unpacked, (0, 3, 1, 4, 2, 5))
    unpacked = mx.reshape(unpacked, (1, 16, latent_h // 2 * 2, latent_w // 2 * 2))
    unpacked = unpacked.astype(mx.float32)
    unpacked = process_wan21_latent_out(unpacked)
    mx.eval(unpacked)

    return torch.from_numpy(np.array(unpacked, dtype=np.float32))


def conditioning_zimage_to_mlx(
    conditioning: Any,
    precision: mx.Dtype,
) -> mx.array:
    """Extract the Qwen3-4B text embedding from SD-style Comfy conditioning.

    Unlike Krea2 (which fuses 12 stacked Qwen3-VL layer taps into a
    30720-dim tensor via `TxtFusionTransformer`), Z-Image's `cap_embedder`
    consumes a SINGLE hidden layer directly — `comfy/text_encoders/
    z_image.py::Qwen3_4BModel` uses `layer="hidden", layer_idx=-2`
    (penultimate layer only, same convention as SDXL's CLIP-L/G) — so this
    is a plain single-tensor extraction, no fusion bridge needed. No pooled
    output is used (Z-Image's base config has no ADM-style pooled
    conditioning — `clip_text_pooled_proj` stays unset unless a variant
    declares `clip_text_dim`, which the base checkpoint on this machine
    doesn't).

    `ASDX_CLIPTextEncode`'s SD-style path (`{"type":"clip","conditioning":
    [[cond_tensor,{...}], ...]}`) already works for Z-Image: comfy's
    `load_clip` routes a detected Qwen3-4B checkpoint to
    `z_image.te()`/`ZImageTokenizer` for any `clip_type` other than
    FLUX/FLUX2 (`comfy/sd.py:1744-1750`) — no dedicated CLIPType or
    `conditioning.py` change needed, verified against the real source.

    Returns cond: [B, T, cap_feat_dim] (2560 for the real checkpoint on
    this machine).
    """
    if isinstance(conditioning, dict):
        conditioning = conditioning.get("conditioning", conditioning)

    entry = conditioning[0] if conditioning else None
    if entry is None:
        raise RuntimeError("ASDX: Z-Image conditioning is empty.")
    cond = entry[0]

    cond_np = _to_numpy(cond)
    if cond_np.ndim != 3:
        raise RuntimeError(f"ASDX: expected Z-Image context [B,T,D], got {cond_np.shape}")
    if cond_np.shape[0] != 1:
        raise RuntimeError("ASDX: currently supports batch_size=1 only.")

    cond_mlx = mx.array(cond_np).astype(precision)
    mx.eval(cond_mlx)
    return cond_mlx


def conditioning_flux2_to_mlx(
    conditioning: Any,
    precision: mx.Dtype,
) -> tuple[mx.array, float | None]:
    """Extract the tapped-and-concatenated text embedding for Flux2/Klein.

    `Flux2Tokenizer`/`KleinTokenizer` (`comfy/text_encoders/flux.py`) tap 3
    hidden layers of the text encoder (Mistral3-24B for Flux2-D, Qwen3-4B/8B
    for Klein) and concatenate them — [B,T,3*text_hidden_dim]. This is
    already what `ASDX_CLIPTextEncode`'s SD-style conditioning path hands
    over as `cond`; no fusion bridge needed (same situation as Z-Image).

    `Flux2.extra_conds` (`comfy/model_base.py:1075-1084`) left-pads the text
    sequence to a MINIMUM of 512 tokens with zeros
    (`F.pad(cross_attn, (0,0, 512-len, 0))` — padding added at the START of
    the sequence, not the end) whenever the real prompt is shorter. This is
    NOT optional/cosmetic: skipping it changes the RoPE positions assigned
    to the real tokens by `get_rope()` (whose txt_ids are simply
    `arange(txt_len)`), silently diverging from what the checkpoint saw
    during training. Longer prompts are left untouched (no truncation).

    Returns (cond, guidance). guidance is None unless the conditioning
    metadata carries one (Flux2-D's checkpoint has a guidance embedding;
    Klein's doesn't — `Flux2Transformer` silently ignores it either way if
    `guidance_in` wasn't allocated for the loaded checkpoint).
    """
    guidance = None

    if isinstance(conditioning, dict):
        conditioning = conditioning.get("conditioning", conditioning)

    entry = conditioning[0] if conditioning else None
    if entry is None:
        raise RuntimeError("ASDX: Flux2 conditioning is empty.")
    cond = entry[0]
    meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
    if "guidance" in meta:
        try:
            guidance = float(meta["guidance"])
        except Exception:
            pass

    cond_np = _to_numpy(cond)
    if cond_np.ndim != 3:
        raise RuntimeError(f"ASDX: expected Flux2 context [B,T,D], got {cond_np.shape}")
    if cond_np.shape[0] != 1:
        raise RuntimeError("ASDX: currently supports batch_size=1 only.")

    target_text_len = 512
    if cond_np.shape[1] < target_text_len:
        pad = target_text_len - cond_np.shape[1]
        cond_np = np.pad(cond_np, ((0, 0), (pad, 0), (0, 0)), mode="constant")

    cond_mlx = mx.array(cond_np).astype(precision)
    mx.eval(cond_mlx)
    return cond_mlx, guidance


def conditioning_qwen_image21_to_mlx(conditioning: Any, precision: mx.Dtype) -> mx.array:
    """Extract the Qwen3-VL-8B hidden states for Qwen Image 2.1's DiT.

    Unlike Flux2/SDXL (which route through ComfyUI's real CLIP pipeline and a
    standard conditioning list), Qwen Image 2.1's text encoder is a native MLX
    module wired through a dedicated pair of nodes
    (`ASDX_QwenImage21TextEncoderLoader`/`ASDX_QwenImage21TextEncode`, same pattern
    as MiniMax H3) -- `conditioning` here is that pair's own dict output, not a
    ComfyUI CONDITIONING list. `hidden_states` is `[S, hidden_size]` (no batch dim,
    brick 1's convention); this adds one."""
    if not isinstance(conditioning, dict) or conditioning.get("type") != "qwen_image21":
        raise RuntimeError(
            "ASDX: Qwen Image 2.1 sampler expected the output of ASDX_QwenImage21TextEncode "
            f"(dict with type='qwen_image21'), got {conditioning!r}."
        )
    hidden_states = conditioning["hidden_states"]
    cond = hidden_states[None].astype(precision)
    mx.eval(cond)
    return cond


def mlx_to_comfy_latent_flux2(
    latents: mx.array,
    height: int,
    width: int,
    template: dict[str, Any],
) -> dict[str, Any]:
    """Convert MLX Flux2 latent to a ComfyUI LATENT dict.

    Flux2 latents are flat token sequences [B, H_lat*W_lat, 128] with NO 2x2
    patch packing (`patch_size=1`, unlike FLUX.1's 64=16*2*2) — unpacking is
    just a reshape+transpose back to [B, 128, H_lat, W_lat], no interleaved
    patch un-shuffle needed.
    """
    samples = _unpack_flux2_latents(latents, height, width)
    out = dict(template)
    out["samples"] = samples
    return out


def _unpack_flux2_latents(
    latents: mx.array,
    height: int,
    width: int,
) -> torch.Tensor:
    """Unpack flat Flux2 latent [B, H_lat*W_lat, 128] -> [B, 128, H_lat, W_lat]."""
    latent_h = height // FLUX2_VAE_DOWNSCALE
    latent_w = width // FLUX2_VAE_DOWNSCALE

    unpacked = mx.reshape(latents, (1, latent_h, latent_w, FLUX2_LATENT_CHANNELS))
    unpacked = mx.transpose(unpacked, (0, 3, 1, 2)).astype(mx.float32)
    mx.eval(unpacked)

    return torch.from_numpy(np.array(unpacked, dtype=np.float32))


def mlx_to_comfy_latent_sdxl(
    latents: mx.array,
    template: dict[str, Any],
) -> dict[str, Any]:
    """Convert MLX SDXL UNet output to a ComfyUI LATENT dict.

    Unlike FLUX, SDXL's UNet operates on the latent grid directly (no 2x2
    patchify) but in MLX's channel-last layout — this only needs the NHWC
    -> NCHW transpose plus `process_latent_out` (divide by SDXL's
    `scale_factor`, applied once at the sampling boundary, matching
    `comfy/samplers.py:1236`).
    """
    samples = _unpack_sdxl_latents(latents)
    out = dict(template)
    out["samples"] = samples
    return out


def _unpack_sdxl_latents(latents: mx.array) -> torch.Tensor:
    """[B, H, W, 4] NHWC (MLX) -> [B, 4, H, W] NCHW (ComfyUI), unscaled."""
    from .native.sdxl.config import process_sdxl_latent_out

    unpacked = mx.transpose(latents, (0, 3, 1, 2)).astype(mx.float32)
    unpacked = process_sdxl_latent_out(unpacked)
    mx.eval(unpacked)
    return torch.from_numpy(np.array(unpacked, dtype=np.float32))


def mlx_to_comfy_latent_qwen_image21(latents: mx.array, template: dict[str, Any]) -> dict[str, Any]:
    """Convert MLX Qwen Image 2.1 DiT output to a ComfyUI LATENT dict.

    The DiT operates on the latent grid directly in NCHW (matching ComfyUI's own
    LATENT convention) -- no patch packing, no NHWC transpose (unlike SDXL's
    native-MLX channel-last convention). The denoising loop runs entirely in the
    model's normalized ("whitened") latent space -- must apply
    `process_qwen_image21_latent_out` (matching `comfy/samplers.py:1238`'s
    `process_latent_out`, applied once at this exact sampling boundary) before
    VAE decode, same bug class as `_unpack_krea2_latents`'s Wan21 de-whitening
    (see `native/config.py::process_wan21_latent_out`) -- missing this step
    feeds the VAE decoder out-of-distribution latent values, producing a
    systematic grid/mesh artifact from the decoder's upsampling layers."""
    from .native.config import process_qwen_image21_latent_out

    latents = process_qwen_image21_latent_out(latents.astype(mx.float32))
    mx.eval(latents)
    samples = torch.from_numpy(np.array(latents, dtype=np.float32))
    out = dict(template)
    out["samples"] = samples
    return out


# ── Noise preparation ────────────────────────────────────────────────

def prepare_noise_from_latent(
    latent: dict[str, Any],
    seed: int,
    precision: mx.Dtype,
) -> tuple[mx.array, int, int, tuple[int, int]]:
    """Prepare initial noise from a Comfy latent, converted to MLX.

    FLUX packs latents as [B, H/8, W/8, 16] -> packed to [B, NH*NW, 64].
    Returns (noise, height, width, output_shape).
    """
    if "samples" not in latent:
        raise RuntimeError("ASDX: latent must be a Comfy LATENT with 'samples'.")

    samples = latent["samples"]
    if tuple(samples.shape)[1] != FLUX_LATENT_CHANNELS:
        raise RuntimeError(
            f"ASDX: needs 16-channel FLUX latent, got {tuple(samples.shape)}"
        )

    # ComfyUI noise
    import comfy.sample
    batch_inds = latent.get("batch_index") if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(samples, int(seed), batch_inds)

    # Convert to numpy and pack
    noise_np = noise.detach().cpu().float().numpy().astype(np.float32, copy=False)

    # Pad to even dimensions if needed
    h, w = noise_np.shape[-2], noise_np.shape[-1]
    if h % 2 != 0 or w % 2 != 0:
        pad_h = h % 2
        pad_w = w % 2
        noise_np = np.pad(noise_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    height = noise_np.shape[-2] * 8
    width = noise_np.shape[-1] * 8

    # Pack: [B, C, H, W] -> [B, H/2, W/2, C*4] -> flatten spatial
    batch, channels, latent_h, latent_w = noise_np.shape
    packed = noise_np.reshape(batch, channels, latent_h // 2, 2, latent_w // 2, 2)
    packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
    packed = packed.reshape(batch, (latent_h // 2) * (latent_w // 2), channels * 4)

    noise_mlx = mx.array(packed).astype(precision)
    mx.eval(noise_mlx)

    output_shape = (int(samples.shape[-2]), int(samples.shape[-1]))
    return noise_mlx, height, width, output_shape


def prepare_noise_from_latent_zimage(
    latent: dict[str, Any],
    seed: int,
    precision: mx.Dtype,
) -> tuple[mx.array, int, int, tuple[int, int]]:
    """Prepare initial noise from a Comfy latent for Z-Image, packed MLX.

    Same 16ch/patch=2/8x VAE latent space as FLUX (`latent_formats.Flux`,
    verified against the real comfy source), but a DIFFERENT per-token
    patch-channel order: comfy/ldm/lumina/model.py's own patchify
    (`embed_all`: `.permute(0,2,4,3,5,1).flatten(3)` on
    `[B,C,H/pH,pH,W/pW,pW]`) packs each token as `[pH, pW, C]`, whereas
    FLUX's `rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)")`
    (comfy/ldm/flux/model.py:319) packs as `[C, pH, pW]` -- a different
    axis order, despite the identical channel count/scale/shift. Reusing
    `prepare_noise_from_latent` (FLUX's `[C,pH,pW]` order) here silently
    scrambled every 2x2 patch's channel-vs-position mapping, producing a
    visibly grainy/pixelated image even though the model ran NaN-free.
    """
    if "samples" not in latent:
        raise RuntimeError("ASDX: latent must be a Comfy LATENT with 'samples'.")

    samples = latent["samples"]
    if tuple(samples.shape)[1] != FLUX_LATENT_CHANNELS:
        raise RuntimeError(
            f"ASDX: needs 16-channel Z-Image latent, got {tuple(samples.shape)}"
        )

    import comfy.sample
    batch_inds = latent.get("batch_index") if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(samples, int(seed), batch_inds)

    noise_np = noise.detach().cpu().float().numpy().astype(np.float32, copy=False)

    h, w = noise_np.shape[-2], noise_np.shape[-1]
    if h % 2 != 0 or w % 2 != 0:
        pad_h = h % 2
        pad_w = w % 2
        noise_np = np.pad(noise_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    height = noise_np.shape[-2] * 8
    width = noise_np.shape[-1] * 8

    # Pack: [B, C, H, W] -> [B, H/2, W/2, pH*pW*C], token order [pH, pW, C]
    batch, channels, latent_h, latent_w = noise_np.shape
    packed = noise_np.reshape(batch, channels, latent_h // 2, 2, latent_w // 2, 2)
    packed = np.transpose(packed, (0, 2, 4, 3, 5, 1))
    packed = packed.reshape(batch, (latent_h // 2) * (latent_w // 2), channels * 4)

    noise_mlx = mx.array(packed).astype(precision)
    mx.eval(noise_mlx)

    output_shape = (int(samples.shape[-2]), int(samples.shape[-1]))
    return noise_mlx, height, width, output_shape


def prepare_noise_from_latent_sdxl(
    latent: dict[str, Any],
    seed: int,
    precision: mx.Dtype,
) -> tuple[mx.array, int, int, tuple[int, int]]:
    """Prepare unit-gaussian noise from a Comfy latent for SDXL, converted to MLX.

    Unlike FLUX (packed to [B, N, 64] tokens), SDXL's UNet consumes the
    latent grid directly — this only transposes NCHW -> NHWC (MLX's
    channel-last convention). Returns unscaled unit-gaussian noise; the
    caller (`_run_sdxl`) is responsible for scaling by the schedule's
    `sigma_max` to form the initial denoising state (this function mirrors
    `prepare_noise_from_latent`'s scope: shape/layout prep only, no
    schedule-specific scaling).

    Returns (noise, height, width, output_shape).
    """
    if "samples" not in latent:
        raise RuntimeError("ASDX: latent must be a Comfy LATENT with 'samples'.")

    samples = latent["samples"]
    if tuple(samples.shape)[1] != SDXL_LATENT_CHANNELS:
        raise RuntimeError(
            f"ASDX: needs 4-channel SDXL latent, got {tuple(samples.shape)}"
        )

    import comfy.sample
    batch_inds = latent.get("batch_index") if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(samples, int(seed), batch_inds)
    noise_np = _to_numpy(noise)

    # Pad to a multiple of 4 (SDXL's UNet has 2 downsamples -> factor 4)
    h, w = noise_np.shape[-2], noise_np.shape[-1]
    pad_h = (-h) % 4
    pad_w = (-w) % 4
    if pad_h or pad_w:
        noise_np = np.pad(noise_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    height = noise_np.shape[-2] * 8
    width = noise_np.shape[-1] * 8

    noise_nhwc = np.transpose(noise_np, (0, 2, 3, 1))
    noise_mlx = mx.array(noise_nhwc).astype(precision)
    mx.eval(noise_mlx)

    output_shape = (int(samples.shape[-2]), int(samples.shape[-1]))
    return noise_mlx, height, width, output_shape


def prepare_noise_from_latent_flux2(
    latent: dict[str, Any],
    seed: int,
    precision: mx.Dtype,
) -> tuple[mx.array, int, int, tuple[int, int]]:
    """Prepare initial noise from a Comfy latent for Flux2/Klein, converted to MLX.

    Flux2 flattens the latent grid directly to token sequence [B, H_lat*W_lat,
    128] — no 2x2 patch packing (`patch_size=1`), so this is a plain
    transpose+reshape rather than FLUX.1's interleaved patch-pack. No padding
    to even dimensions is needed either, for the same reason.

    Returns (noise, height, width, output_shape).
    """
    if "samples" not in latent:
        raise RuntimeError("ASDX: latent must be a Comfy LATENT with 'samples'.")

    samples = latent["samples"]
    if tuple(samples.shape)[1] != FLUX2_LATENT_CHANNELS:
        raise RuntimeError(
            f"ASDX: needs 128-channel Flux2 latent, got {tuple(samples.shape)}"
        )

    import comfy.sample
    batch_inds = latent.get("batch_index") if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(samples, int(seed), batch_inds)
    noise_np = _to_numpy(noise)

    batch, channels, latent_h, latent_w = noise_np.shape
    height = latent_h * FLUX2_VAE_DOWNSCALE
    width = latent_w * FLUX2_VAE_DOWNSCALE

    # [B, C, H, W] -> [B, H, W, C] -> flatten spatial -> [B, H*W, C]
    flat = np.transpose(noise_np, (0, 2, 3, 1)).reshape(batch, latent_h * latent_w, channels)

    noise_mlx = mx.array(flat).astype(precision)
    mx.eval(noise_mlx)

    output_shape = (int(samples.shape[-2]), int(samples.shape[-1]))
    return noise_mlx, height, width, output_shape


def prepare_noise_from_latent_qwen_image21(
    latent: dict[str, Any], seed: int, precision: mx.Dtype
) -> tuple[mx.array, int, int, tuple[int, int]]:
    """Prepare unit-gaussian noise from a Comfy latent for Qwen Image 2.1, in MLX.

    NCHW throughout, matching the DiT's own input convention -- no layout
    conversion needed, unlike SDXL's NCHW->NHWC bridge.

    Returns (noise, height, width, output_shape)."""
    if "samples" not in latent:
        raise RuntimeError("ASDX: latent must be a Comfy LATENT with 'samples'.")

    samples = latent["samples"]
    if tuple(samples.shape)[1] != QWEN_IMAGE21_LATENT_CHANNELS:
        raise RuntimeError(
            f"ASDX: needs {QWEN_IMAGE21_LATENT_CHANNELS}-channel Qwen Image 2.1 latent, "
            f"got {tuple(samples.shape)}"
        )

    import comfy.sample
    batch_inds = latent.get("batch_index") if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(samples, int(seed), batch_inds)
    noise_np = _to_numpy(noise)

    batch, channels, latent_h, latent_w = noise_np.shape
    height = latent_h * QWEN_IMAGE21_VAE_DOWNSCALE
    width = latent_w * QWEN_IMAGE21_VAE_DOWNSCALE

    noise_mlx = mx.array(noise_np).astype(precision)
    mx.eval(noise_mlx)
    return noise_mlx, height, width, (latent_h, latent_w)


# ── Memory management ────────────────────────────────────────────────

def collect_mlx_memory() -> dict[str, float]:
    """Return current MLX memory stats in GB."""
    active = mx.get_active_memory() / (1024 ** 3)
    cached = mx.get_cache_memory() / (1024 ** 3)
    peak = mx.get_peak_memory() / (1024 ** 3)
    return {"active_gb": round(active, 2), "cache_gb": round(cached, 2), "peak_gb": round(peak, 2)}


def _process_rss_gb() -> float | None:
    """Current process resident set size, in GB, or None if psutil is unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return None


def _mps_allocator_gb() -> tuple[float, float] | None:
    """Current/driver MPS allocator memory, in GB, or None if unavailable.

    `current_allocated_memory` is what PyTorch's MPS allocator has handed
    out; `driver_allocated_memory` is what the Metal driver has reserved
    for that allocator, which can stay high even after `empty_cache()` if
    the driver hasn't returned the underlying pages to the OS.
    """
    if not (hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory")):
        return None
    try:
        current = torch.mps.current_allocated_memory() / (1024 ** 3)
        driver = torch.mps.driver_allocated_memory() / (1024 ** 3)
        return current, driver
    except Exception:
        return None


def clear_mlx_cache() -> None:
    """Clear MLX constant cache and trigger GC. Call between major phases."""
    rss_before = _process_rss_gb()
    mps_before = _mps_allocator_gb()
    mx.clear_cache()
    gc.collect()
    # Also clear MPS cache if available
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    rss_after = _process_rss_gb()
    mps_after = _mps_allocator_gb()
    if rss_before is not None and rss_after is not None:
        print(f"[ASDX] clear_mlx_cache: process RSS {rss_before:.1f}GB -> {rss_after:.1f}GB")
    if mps_before is not None and mps_after is not None:
        print(
            f"[ASDX] clear_mlx_cache: MPS allocator current "
            f"{mps_before[0]:.1f}GB -> {mps_after[0]:.1f}GB, driver "
            f"{mps_before[1]:.1f}GB -> {mps_after[1]:.1f}GB"
        )


def set_mlx_cache_limit_gb(gb: float) -> None:
    """Set MLX constant cache limit in GB."""
    mx.set_cache_limit(int(gb * 1024 ** 3))
