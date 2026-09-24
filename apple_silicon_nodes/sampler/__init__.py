"""
MLX Native Sampler
==================
Euler sampler running entirely in MLX on Apple Silicon.

Submodules:
  - cache: TeaCacheState (output-level step skipping)
  - core: core denoising loop, sigma scheduling, acceleration helpers

This module provides the ComfyUI node interface (ASDX_MLXSampler)
that wraps the core sampling logic.
"""

from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io

from .. import metadata as metadata_util
from .. import bridge
from . import solvers
from .core import _SamplerCore
from .scheduling import SCHEDULER_NAMES


# ── Preview cache ─────────────────────────────────────────────────────

_PREVIEW_CACHE: dict[str, Any] = {}


# ── Sampler Node ──────────────────────────────────────────────────────

class ASDX_MLXSampler(io.ComfyNode):
    """MLX-native FLUX sampler with SeaCache acceleration.

    Runs the full denoising loop in MLX, only bridging to PyTorch
    for the final latent output (for ComfyUI compatibility).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MLXSampler",
            display_name="🍏 ASDX MLX Native Sampler",
            category="ASDX/Sampling",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Custom("mlx_conditioning").Input("positive"),
                io.Latent.Input("latent_image"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=100),
                io.Combo.Input("sampler_name", options=solvers.SAMPLER_NAMES, default="euler"),
                io.Combo.Input("scheduler_name", options=SCHEDULER_NAMES, default="normal"),
                io.Float.Input("guidance", default=3.5, min=0.0, max=20.0, step=0.1),
                io.Boolean.Input("teacache", default=False),
                io.Float.Input("teacache_threshold", default=0.08, min=0.01, max=1.0, step=0.01),
                io.Boolean.Input("kontext", default=False),
                io.Float.Input("kontext_reference_strength", default=1.0, min=0.0, max=2.0, step=0.01),
                io.Boolean.Input("seacache", default=False),
                io.Boolean.Input("preview", default=False),
                io.Boolean.Input("low_memory_mode", default=False),
                io.Boolean.Input("save_metadata", default=False),
                # Kontext reference — only meaningful when "kontext" above is
                # True; must be optional so the graph doesn't force a link on
                # every generation, including non-Kontext ones.
                io.Latent.Input("kontext_reference_latent", optional=True),
                # Mode routing (Phase 2)
                io.Combo.Input(
                    "mode",
                    options=["auto", "text2img", "img2img", "inpaint", "fill", "depth"],
                    default="auto", optional=True,
                ),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Float.Input("image_strength", default=0.8, min=0.0, max=1.0, step=0.01, optional=True),
                io.Int.Input("mask_blur", default=0, min=0, max=64, optional=True),
                io.Int.Input("mask_padding", default=48, min=32, max=256, optional=True),
                io.Image.Input("depth_image", optional=True),
                io.Float.Input("depth_strength", default=1.0, min=0.0, max=2.0, step=0.01, optional=True),
                io.Float.Input("noise_aug", default=0.0, min=0.0, max=1.0, step=0.01, optional=True),
                # Krea2 Identity Edit
                io.Latent.Input("source_latent", optional=True),
                # Pixel-space Identity Edit path (preferred over source_latent
                # -- see core.py::_prepare_krea2_identity_edit): only used
                # when a Krea2 Identity Edit LoRA is actually loaded, so it
                # must stay optional and not force a link on every graph.
                io.Image.Input("source_image", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Float.Input(
                    "ref_boost", default=1.0, min=0.0, max=1000.0, step=0.01, optional=True,
                    tooltip="reference-fidelity dial: multiplies target->source attention. "
                            "1.0 = off, >1 pulls harder toward the source's appearance.",
                ),
                io.Float.Input(
                    "krea2_enhancer_strength", default=1.0, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Krea2-only: community txtfusion conditioning-strength "
                            "boost (ported from ComfyUI-Krea2T-Enhancer). 0 = off "
                            "(vanilla Krea2, weaker prompt adherence and lower final "
                            "latent amplitude); 1.0 matches the reference node's "
                            "default (max). Its internal ~x75 amplification can "
                            "overflow in float16 (silently falls back to unamplified "
                            "conditioning if it does) -- use bfloat16 precision for "
                            "the full effect at strength=1.0. No effect on "
                            "non-Krea2 models.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: dict,
        positive: dict,
        latent_image: dict,
        seed: int,
        steps: int,
        sampler_name: str,
        scheduler_name: str,
        guidance: float,
        teacache: bool,
        teacache_threshold: float,
        kontext: bool,
        kontext_reference_strength: float,
        seacache: bool,
        preview: bool,
        low_memory_mode: bool,
        save_metadata: bool,
        kontext_reference_latent: dict | None = None,
        # Mode routing params (Phase 2)
        mode: str = "auto",
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        image_strength: float = 0.8,
        mask_blur: int = 0,
        mask_padding: int = 48,
        depth_image: torch.Tensor | None = None,
        depth_strength: float = 1.0,
        noise_aug: float = 0.0,
        # Krea2 Identity Edit
        source_latent: dict | None = None,
        ref_boost: float = 1.0,
        krea2_enhancer_strength: float = 1.0,
        # Krea2 Identity Edit pixel path: the source IMAGE + the VAE used to
        # encode it. When both are wired, the sampler fits the image in pixel
        # space and VAE-encodes it itself (the blur-proof path the reference
        # and the v1_2 LoRA were trained with), instead of fitting a
        # pre-encoded latent.
        source_image: torch.Tensor | None = None,
        vae: Any | None = None,
    ) -> io.NodeOutput:
        """Execute the MLX-native sampling loop via _SamplerCore."""
        transformer = model["transformer"]
        config = model["config"]
        model_type = model.get("model_type", "dev")
        capability = model.get("capability")
        controlnet = model.get("controlnet")
        memory_shape = model.get("memory_shape")
        # ASDX_LoraSchedule stores its config in the model dict.
        lora_schedule = model.get("lora_schedule")
        # ASDX_Krea2Edit (Task A1) pre-computes the Identity Edit source and
        # stores it in the model dict. Read it from there, in priority over the
        # legacy source_latent / source_image inputs below (which stay as a
        # documented fallback so saved workflows keep working).
        identity_edit = model.get("identity_edit")
        if identity_edit is not None and (source_latent is not None or source_image is not None):
            print("[ASDX] Identity Edit: both ASDX_Krea2Edit and legacy source_latent/"
                  "source_image inputs are wired -- ASDX_Krea2Edit wins.")

        # ASDX_DepthConditioning (Phase D) pre-validates the capability and
        # stores its raw ingredients in the model dict. Read it in priority
        # over the legacy depth_image/depth_strength inputs, which stay as a
        # documented fallback so saved workflows keep working.
        depth_cond = model.get("depth_cond")
        if depth_cond is not None and depth_image is not None:
            print("[ASDX] Depth: both ASDX_DepthConditioning and the legacy "
                  "depth_image input are wired -- ASDX_DepthConditioning wins.")

        # ASDX_KontextReference (Phase E) stores its reference latent + strength
        # in the model dict. Read it in priority over the sampler's own
        # kontext/kontext_reference_latent/kontext_reference_strength inputs,
        # which stay as a documented fallback so saved workflows keep working.
        kontext_cfg = model.get("kontext")
        if kontext_cfg is not None:
            if kontext_reference_latent is not None:
                print("[ASDX] Kontext: both ASDX_KontextReference and the legacy "
                      "kontext_reference_latent input are wired -- "
                      "ASDX_KontextReference wins.")
            kontext = True
            kontext_reference_latent = kontext_cfg["reference_latent"]
            kontext_reference_strength = kontext_cfg["strength"]

        # ASDX_LatentNoisePrep (Phase E) validates and mode-tags the raw
        # img2img/inpaint/fill ingredients. Read it in priority over the
        # sampler's own mode/image/mask/image_strength/mask_blur/mask_padding
        # inputs, which stay as a documented fallback so saved workflows keep
        # working. The actual noise blend still happens in _SamplerCore (it
        # needs the seeded noise tensor built below, not available here).
        latent_prep = model.get("latent_prep")
        if latent_prep is not None:
            if image is not None or mask is not None:
                print("[ASDX] Latent prep: both ASDX_LatentNoisePrep and the "
                      "sampler's own mode/image/mask inputs are wired -- "
                      "ASDX_LatentNoisePrep wins.")
            mode = latent_prep["mode"]
            image = latent_prep["image"]
            mask = latent_prep["mask"]
            image_strength = latent_prep["image_strength"]
            mask_blur = latent_prep["mask_blur"]
            mask_padding = latent_prep["mask_padding"]

        # Diagnostic: snapshot memory at the very start of every generation
        # (this node always re-executes, unlike loader/LoRA nodes which may
        # be cache-hit) -- lets us see whether the floor left over from the
        # previous generation is growing across repeated queues, or whether
        # a single generation's own peak is what's pushing past the jetsam
        # ceiling.
        _mps = bridge._mps_allocator_gb()
        _rss = bridge._process_rss_gb()
        if _mps is not None and _rss is not None:
            print(
                f"[ASDX] Sampler start: process RSS {_rss:.1f}GB, MPS allocator "
                f"current {_mps[0]:.1f}GB, driver {_mps[1]:.1f}GB"
            )

        # TeaCache/SeaCache's skip heuristic (reuse the previous step's
        # noise_pred) assumes a solver with one model call per step and no
        # cross-step state -- only true for "euler"/"ddim" among the
        # supported samplers (see sampler/solvers.py module docstring).
        if (teacache or seacache) and sampler_name not in solvers.STATELESS_SINGLE_EVAL_SAMPLERS:
            raise RuntimeError(
                f"ASDX: teacache/seacache are only supported with sampler_name "
                f"in {sorted(solvers.STATELESS_SINGLE_EVAL_SAMPLERS)!r} (got "
                f"{sampler_name!r}) -- other samplers have per-step state or "
                f"multiple model calls that the cache skip heuristic would "
                f"silently corrupt. Disable teacache/seacache or switch sampler."
            )

        # Prepare noise. SDXL's UNet consumes the latent grid directly (no
        # 2x2 patchify like FLUX/Krea2), so it needs its own noise-prep path.
        # Flux2/Klein has 128 channels / patch_size=1 / 16x VAE downscale
        # (vs FLUX's 16ch/patch=2/8x) — also its own path. Z-Image shares
        # FLUX's channel count/scale/shift but NOT its patch-token axis
        # order (comfy/ldm/lumina/model.py packs [pH,pW,C], FLUX packs
        # [C,pH,pW] — verified against the real comfy source), so it needs
        # its own noise-prep path too, despite the identical latent shape.
        if model_type == "sdxl":
            noise, height, width, output_shape = bridge.prepare_noise_from_latent_sdxl(
                latent_image, int(seed), config.mlx_dtype
            )
        elif model_type == "flux2":
            noise, height, width, output_shape = bridge.prepare_noise_from_latent_flux2(
                latent_image, int(seed), config.mlx_dtype
            )
        elif model_type in ("zimage", "zimage_turbo"):
            noise, height, width, output_shape = bridge.prepare_noise_from_latent_zimage(
                latent_image, int(seed), config.mlx_dtype
            )
        elif model_type == "qwen_image21":
            noise, height, width, output_shape = bridge.prepare_noise_from_latent_qwen_image21(
                latent_image, int(seed), config.mlx_dtype
            )
        else:
            noise, height, width, output_shape = bridge.prepare_noise_from_latent(
                latent_image, int(seed), config.mlx_dtype
            )

        # Get previewer for real-time output
        previewer, preview_device = cls._get_previewer() if preview else (None, None)

        # Create core sampler with mode routing params
        core = _SamplerCore(
            transformer=transformer,
            config=config,
            positive=positive,
            noise=noise,
            height=height,
            width=width,
            output_shape=output_shape,
            model_type=model_type,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            guidance=guidance,
            teacache=teacache,
            teacache_threshold=teacache_threshold,
            kontext=kontext,
            kontext_reference_latent=kontext_reference_latent,
            kontext_reference_strength=kontext_reference_strength,
            seacache=seacache,
            preview=preview,
            lora_schedule=lora_schedule,
            previewer=previewer,
            preview_device=preview_device,
            capability=capability,
            # Mode routing
            mode=mode,
            latent_image=latent_image,
            # Low memory mode
            low_memory_mode=low_memory_mode,
            image=image,
            image_strength=image_strength,
            mask=mask,
            mask_blur=mask_blur,
            mask_padding=mask_padding,
            depth_image=depth_image,
            depth_strength=depth_strength,
            depth_cond=depth_cond,
            noise_aug=noise_aug,
            # Krea2 Identity Edit
            identity_edit=identity_edit,
            source_latent=source_latent,
            source_image=source_image,
            vae=vae,
            ref_boost=ref_boost,
            krea2_enhancer_strength=krea2_enhancer_strength,
            controlnet=controlnet,
            memory_shape=memory_shape,
        )

        # Run sampling
        out_latent = core.run(steps, seed)

        # Attach generation metadata to the latent output
        if save_metadata:
            out_latent["asdx_metadata"] = metadata_util.build_generation_metadata(
                model_name=model.get("name", "unknown"),
                model_type=model_type,
                precision=config.dtype,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                cfg=guidance,
                mode=mode,
                sampler_name=sampler_name,
                scheduler_name=scheduler_name,
            )

        return io.NodeOutput(out_latent)

    @staticmethod
    def _get_previewer():
        """Get latent previewer from ComfyUI."""
        try:
            import comfy.model_management
            import latent_preview
            from comfy.cli_args import LatentPreviewMethod

            device = comfy.model_management.get_torch_device()
            preview_method = getattr(latent_preview, 'args', None)
            if preview_method is not None:
                preview_method = preview_method.preview_method

            cache_key = f"preview:{preview_method}:{device}"
            if cache_key in _PREVIEW_CACHE:
                return _PREVIEW_CACHE[cache_key]

            import comfy.latent_formats
            latent_format = comfy.latent_formats.Flux()
            if preview_method == LatentPreviewMethod.NoPreviews:
                _PREVIEW_CACHE[cache_key] = (None, None)
                return _PREVIEW_CACHE[cache_key]

            previewer = latent_preview.get_previewer(device, latent_format)
            _PREVIEW_CACHE[cache_key] = (previewer, device)
            return _PREVIEW_CACHE[cache_key]
        except Exception:
            return (None, None)


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_LIST = [
    ASDX_MLXSampler,
]
