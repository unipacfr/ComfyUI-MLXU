"""Core sampling logic for the MLX-native FLUX sampler.

Contains the denoising loop, sigma scheduling, and acceleration helpers.
The ComfyUI node (sampler.py) wraps this to provide the node interface.
"""

from __future__ import annotations

import enum
import math
import time
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from .. import capability as cap_module
from ..memory_calibration import record_observation
from ..native.config import FLUX_LATENT_SCALE, FLUX_LATENT_SHIFT
from . import bridge
from .cache import TeaCacheState
from . import solvers
from .scheduling import SDXLSampling, calculate_sigmas


class SamplerMode(enum.Enum):
    """Sampling mode: text2img, img2img, inpainting, fill, depth control."""

    TEXT_TO_IMAGE = "text2img"
    IMAGE_TO_IMAGE = "img2img"
    INPAINTING = "inpaint"
    FILL = "fill"
    DEPTH_CONTROL = "depth"


class _SamplerCore:
    """Core denoising loop and acceleration logic.

    This class contains the sampling logic without ComfyUI node boilerplate.
    It is instantiated by ASDX_MLXSampler for each sample call.
    """

    def __init__(
        self,
        transformer: Any,
        config: Any,
        positive: dict,
        noise: mx.array,
        height: int,
        width: int,
        output_shape: tuple[int, int],
        model_type: str,
        guidance: float,
        teacache: bool,
        teacache_threshold: float,
        kontext: bool,
        kontext_reference_latent: dict | None,
        kontext_reference_strength: float,
        seacache: bool,
        preview: bool,
        lora_schedule: dict | None,
        previewer: Any | None,
        preview_device: str | None,
        capability: Any | None = None,
        # Mode routing (Phase 2)
        mode: str = "auto",
        latent_image: dict | None = None,
        image: torch.Tensor | None = None,
        image_strength: float = 0.8,
        mask: torch.Tensor | None = None,
        mask_blur: int = 0,
        mask_padding: int = 48,
        depth_image: torch.Tensor | None = None,
        depth_strength: float = 1.0,
        noise_aug: float = 0.0,
        # Low memory mode (DiffusionKit pattern)
        low_memory_mode: bool = False,
        # Krea2 Identity Edit
        source_latent: dict | None = None,
        source_image: torch.Tensor | None = None,
        vae: Any | None = None,
        ref_boost: float = 1.0,
        krea2_enhancer_strength: float = 1.0,
        controlnet: dict | None = None,
        sampler_name: str = "euler",
        scheduler_name: str = "normal",
        memory_shape: Any | None = None,
        nag_phi: float = 4.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
        nag_sigma_start: float = 1.0,
        nag_sigma_end: float = 0.0,
    ):
        self.transformer = transformer
        self.config = config
        self.positive = positive
        self.noise = noise
        self.height = height
        self.width = width
        self.output_shape = output_shape
        self.model_type = model_type
        self.guidance = guidance
        self.teacache = teacache
        self.teacache_threshold = teacache_threshold
        self.kontext = kontext
        self.kontext_reference_latent = kontext_reference_latent
        self.kontext_reference_strength = kontext_reference_strength
        self.seacache = seacache
        self.preview = preview
        self.lora_schedule = lora_schedule
        self.previewer = previewer
        self.preview_device = preview_device
        self.capability = capability
        # Mode routing
        self.mode = mode
        self.latent_image = latent_image
        self.image = image
        self.image_strength = image_strength
        self.mask = mask
        self.mask_blur = mask_blur
        self.mask_padding = mask_padding
        self.depth_image = depth_image
        self.depth_strength = depth_strength
        self.noise_aug = noise_aug
        self.low_memory_mode = low_memory_mode
        # Krea2 Identity Edit
        self.source_latent = source_latent
        self.source_image = source_image
        self.vae = vae
        self.ref_boost = ref_boost
        self.krea2_enhancer_strength = krea2_enhancer_strength
        self.controlnet = controlnet
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.memory_shape = memory_shape
        if nag_sigma_start < nag_sigma_end:
            raise ValueError(
                f"ASDX: nag_sigma_start ({nag_sigma_start}) must be >= "
                f"nag_sigma_end ({nag_sigma_end})."
            )
        self.nag_phi = nag_phi
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.nag_sigma_start = nag_sigma_start
        self.nag_sigma_end = nag_sigma_end
        self._is_flow_matching = model_type != "sdxl"

    def run(self, steps: int, seed: int) -> dict:
        """Execute the MLX-native sampling loop and return the result latent."""
        precision = self.config.mlx_dtype
        model_type = self.model_type

        # ── Z-Image routing ──────────────────────────────────────────
        # Z-Image uses a different conditioning pipeline (single-layer
        # Qwen3-4B embeddings, no txt_in) and a 3-stage transformer
        # (context_refiner/noise_refiner/layers, encapsulated inside
        # NextDiT.__call__). Route before any FLUX-specific calls below.
        if model_type in ("zimage", "zimage_turbo"):
            return self._run_zimage(steps, seed)

        # ── Flux2/Klein routing ─────────────────────────────────────────
        # Flux2 uses its own conditioning (tapped-and-concatenated text
        # embeddings, left-padded to 512 tokens), latent packing (128ch,
        # patch_size=1, no 2x2 patchify, 16x VAE downscale instead of 8x),
        # and 4-axis RoPE with a dedicated text axis. Route before any
        # FLUX.1-specific calls below, which assume 3-axis RoPE and 64=16*2*2
        # packed tokens that don't apply here.
        if model_type == "flux2":
            return self._run_flux2(steps, seed)

        # ── SDXL routing ──────────────────────────────────────────────
        # SDXL is an EPS-prediction conv UNet on a discrete DDPM schedule,
        # not a flow-matching DiT — completely different noise shape (NHWC
        # grid, not packed tokens), conditioning (dual CLIP-L/G context +
        # ADM vector, not T5+pooled-CLIP), and denoise update (two-pass true
        # CFG + EPS x0=x-eps*sigma, not the Euler x+=pred*dt loop below).
        # Route before mode-detection/img2img prep, which assume FLUX's
        # packed-token noise shape (txt2img only for SDXL in this phase).
        if model_type == "sdxl":
            return self._run_sdxl(steps, seed)

        # Detect mode and prepare noise (Phase 2: mode routing)
        self._mode = self._detect_mode()
        if self._mode != SamplerMode.TEXT_TO_IMAGE:
            print(f"[ASDX] Mode: {self._mode.value}")
            if self._mode == SamplerMode.IMAGE_TO_IMAGE:
                self.noise = self._prepare_img2img_noise()
            elif self._mode in (SamplerMode.INPAINTING, SamplerMode.FILL):
                self.noise = self._prepare_inpainting_noise()
            elif self._mode == SamplerMode.DEPTH_CONTROL:
                self.noise = self._prepare_depth_noise()

        # ── Krea2 routing ───────────────────────────────────────────
        # Krea2 uses a different conditioning pipeline (Qwen3-VL fused embeddings,
        # txtfusion instead of txt_in) and RoPE (3-axis grid, not FLUX's flat
        # get_rope). Route BEFORE any FLUX-specific calls below, which assume
        # methods (get_rope(seq,txt), txt_in) that Krea2Transformer doesn't have.
        if model_type in ("krea2", "krea2_turbo"):
            effective_guidance = float(self.guidance) if self.guidance > 0 else 1.0
            if self.capability is not None:
                candidate_params = {
                    "guidance": self.guidance,
                    "width": self.width,
                    "height": self.height,
                    "steps": steps,
                }
                try:
                    valid_params, dropped = cap_module.filter_params_for_model(
                        self.capability, candidate_params
                    )
                    if dropped:
                        print(f"[ASDX] Dropped params for {self.capability.name}: {dropped}")
                    if "guidance" in valid_params and valid_params["guidance"] is not None:
                        effective_guidance = float(valid_params["guidance"])
                except ValueError as e:
                    print(f"[ASDX] Capability filter warning: {e}")
            return self._run_krea2(steps, seed, None, None, effective_guidance)

        # Prepare conditioning
        prompt_embeds, pooled_embeds, cond_guidance = bridge.conditioning_to_mlx(
            self.positive, precision
        )
        effective_guidance = float(self.guidance) if self.guidance > 0 else (
            cond_guidance if cond_guidance is not None else 3.5
        )

        # Capability-aware parameter filtering (mflux-AnyModel pattern)
        if self.capability is not None:
            candidate_params = {
                "guidance": self.guidance,
                "width": self.width,
                "height": self.height,
                "steps": steps,
            }
            try:
                valid_params, dropped = cap_module.filter_params_for_model(
                    self.capability, candidate_params
                )
                if dropped:
                    print(f"[ASDX] Dropped params for {self.capability.name}: {dropped}")
                # Update effective_guidance from filtered params if present
                if "guidance" in valid_params and valid_params["guidance"] is not None:
                    effective_guidance = float(valid_params["guidance"])
            except ValueError as e:
                print(f"[ASDX] Capability filter warning: {e}")

        # Precompute rope embeddings on the real 2D image token grid (FLUX packs
        # latents as 2x2 patches, so the grid is (height//8//2) x (width//8//2))
        cap_module.require_divisible_dims(self.width, self.height, 16, "flux1")
        img_h = (self.height // 8) // 2
        img_w = (self.width // 8) // 2
        if img_h * img_w != self.noise.shape[1]:
            raise RuntimeError(
                f"ASDX: FLUX token grid mismatch: {img_h}x{img_w}={img_h * img_w} "
                f"vs noise token count {self.noise.shape[1]}"
            )

        # Kontext reference latent: pack it in the same raw-patch space as the
        # main image (before img_in — FluxTransformer projects the whole
        # [img, ref] sequence together, matching comfy's `_forward`). Must run
        # before get_rope() below since the ref grid extends the RoPE table.
        kontext_ref_packed: mx.array | None = None
        kontext_ref_grid: tuple[int, int] | None = None
        if self.kontext and self.kontext_reference_latent is not None:
            self._prepare_kontext_reference(self.kontext_reference_latent, precision)
            kontext_ref_packed = getattr(self, "_kontext_ref_packed", None)
            kontext_ref_grid = getattr(self, "_kontext_ref_grid", None)

        rope = self.transformer.get_rope(
            img_h, img_w, prompt_embeds.shape[1],
            ref_grids=[kontext_ref_grid] if kontext_ref_grid is not None else None,
        )

        # ControlNet: VAE-encode + pack the control image once (the model's
        # own weights are fixed across steps; only the residuals it produces
        # depend on the current noisy latent, so they're recomputed per step)
        controlnet_model = self.controlnet.get("control_net") if self.controlnet else None
        controlnet_latent = self._prepare_controlnet_latent(precision) if controlnet_model else None
        controlnet_strength = self.controlnet.get("strength", 1.0) if self.controlnet else 1.0
        controlnet_type = self.controlnet.get("control_type") if self.controlnet else None

        # Setup sigmas for the model type (adapted from DiffusionKit)
        sigmas = calculate_sigmas(model_type, self.scheduler_name, steps, self.width, self.height)
        solver_state: dict[str, Any] = {}

        # Setup TeaCache
        teacache_state: TeaCacheState | None = None
        if self.teacache:
            teacache_state = TeaCacheState(
                threshold=self.teacache_threshold,
                warmup_steps=5,
                final_steps=3,
            )

        # SeaCache state
        seacache_enabled = self.seacache and model_type == "dev"
        previous_output: mx.array | None = None
        cache_threshold = 0.1 if seacache_enabled else 100.0
        warmup_steps = 3

        # Memory profiling
        mx.reset_peak_memory()
        step_times: list[float] = []

        t_sampling_start = time.perf_counter()

        for t in range(steps):
            step_start = time.perf_counter()

            # Current sigma
            sigma_t = sigmas[t]
            sigma_next = sigmas[t + 1] if t + 1 < len(sigmas) else 0.0

            # Update LoRA schedule
            if self.lora_schedule is not None:
                self.lora_schedule["step"] = t
                self.transformer = self._update_lora_schedule(
                    self.transformer, self.config, self.lora_schedule, t, steps
                )

            # ControlNet: recompute residuals each step (they depend on the
            # current noisy latent, unlike the frozen ControlNet weights).
            # Do NOT reuse the base model's `rope`: when control_type is set,
            # ControlNetFlux prefixes a control-mode token to txt, shifting
            # the sequence length — let it compute its own RoPE table.
            control = None
            if controlnet_model is not None and controlnet_latent is not None:
                control = controlnet_model(
                    img=self.noise,
                    control_latent=controlnet_latent,
                    txt=prompt_embeds,
                    t=mx.array([sigma_t], dtype=mx.float32),
                    img_h=img_h,
                    img_w=img_w,
                    guidance=mx.array([effective_guidance], dtype=mx.float32),
                    pooled=pooled_embeds,
                    control_type=controlnet_type,
                )
                if controlnet_strength != 1.0:
                    control = {
                        k: [r * controlnet_strength for r in v] for k, v in control.items()
                    }

            # --- Compute transformer output ---
            if teacache_state is not None:
                current_output = self.transformer.predict(
                    img=self.noise,
                    txt=prompt_embeds,
                    timestep=sigma_t,
                    guidance=effective_guidance,
                    pooled=pooled_embeds,
                    rope=rope,
                    control=control,
                    ref_img=kontext_ref_packed,
                )
                mx.eval(current_output)

                reused, reason = self._teacache_check(teacache_state, current_output, t, steps)

                if reused:
                    noise_pred = teacache_state.last_feature if teacache_state.last_feature is not None else current_output
                    skip_step = True
                else:
                    skip_step = False
                    noise_pred = current_output
            else:
                current_output = self.transformer.predict(
                    img=self.noise,
                    txt=prompt_embeds,
                    timestep=sigma_t,
                    guidance=effective_guidance,
                    pooled=pooled_embeds,
                    rope=rope,
                    control=control,
                    ref_img=kontext_ref_packed,
                )
                mx.eval(current_output)
                noise_pred = current_output
                skip_step = False

            # SeaCache: additional skip check (only if not already skipped)
            if not skip_step and seacache_enabled and t >= warmup_steps and previous_output is not None:
                diff = float(mx.mean(mx.abs(current_output - previous_output)).item())
                if diff < cache_threshold:
                    skip_step = True
                    noise_pred = previous_output
                else:
                    previous_output = current_output

            if skip_step and previous_output is not None:
                noise_pred = previous_output

            if not skip_step:
                previous_output = current_output

            # denoised = x - noise_pred*sigma (same formula for every family
            # in this project, see solvers.py module docstring); solver step
            # advances x from sigma_t to sigma_next given that estimate.
            denoised = self.noise - noise_pred * sigma_t

            def _model_call(x_at, sigma_at):
                # Second model evaluation for multi-eval solvers (e.g.
                # dpmpp_2s_ancestral). Reuses this step's ControlNet residuals
                # rather than recomputing them at the intermediate point.
                out = self.transformer.predict(
                    img=x_at, txt=prompt_embeds, timestep=sigma_at,
                    guidance=effective_guidance, pooled=pooled_embeds, rope=rope,
                    control=control, ref_img=kontext_ref_packed,
                )
                mx.eval(out)
                return x_at - out * sigma_at

            self.noise, solver_state = solvers.step(
                self.sampler_name,
                x=self.noise,
                sigma=sigma_t,
                sigma_next=sigma_next,
                denoised=denoised,
                state=solver_state,
                seed=seed,
                step_index=t,
                is_flow_matching=self._is_flow_matching,
                model_call=_model_call,
            )
            mx.eval(self.noise)

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            # Preview
            if self.preview and self.previewer is not None:
                preview_latent = self._denoise_to_latent(self.noise, self.height, self.width, self.output_shape)
                if preview_latent is not None:
                    preview_bytes = self.previewer.decode_latent_to_preview_image(
                        "JPEG", preview_latent
                    )
                    if preview_bytes:
                        import comfy.utils
                        comfy.utils.ProgressBar(steps).update_absolute(t + 1, steps, preview_bytes)

            # Progress logging
            if (t + 1) % 5 == 0 or t == 0:
                cache_tag = ""
                if skip_step:
                    cache_tag = " [cache]"
                if kontext_ref_packed is not None:
                    cache_tag += " [kontext]"
                print(f"[ASDX] Step {t + 1}/{steps} - {step_time:.3f}s{cache_tag}")

        total_time = time.perf_counter() - t_sampling_start

        # ── Low memory mode: evict the cached base model ─────────────
        # self.transformer is the SAME object as loader.py's _MODEL_CACHE
        # entry -- dropping only this local reference never freed anything
        # (and _SamplerCore is local to sample() anyway, released on return
        # regardless). Evict the cache entry itself so the memory is
        # actually returned; the next generation will reload the checkpoint
        # from disk instead of hitting the cache.
        if self.low_memory_mode:
            from ..loader import clear_model_cache
            clear_model_cache()
            self.transformer = None
            print("[ASDX] Low memory: model cache evicted, next load will read from disk")

        # Convert result to ComfyUI latent
        out_latent = bridge.mlx_to_comfy_latent(self.noise, self.height, self.width,
                                                {"samples": self.noise})
        out_latent["sdmlx_model_type"] = model_type
        out_latent["sdmlx_model_name"] = "unknown"

        # Memory stats
        mem = bridge.collect_mlx_memory()
        avg_step = sum(step_times) / len(step_times) if step_times else 0

        # Build acceleration summary
        accel_parts = []
        if teacache_state is not None:
            accel_parts.append(f"TeaCache({teacache_state.hits}h/{teacache_state.real_steps}real)")
        if kontext_ref_packed is not None:
            accel_parts.append("Kontext(ref_latents)")
        if seacache_enabled:
            accel_parts.append("SeaCache")
        accel_str = "+".join(accel_parts) if accel_parts else "standard"

        print(f"[ASDX] Sampling complete: {total_time:.1f}s total, "
              f"{avg_step:.3f}s/step, {mem['peak_gb']:.1f}GB peak, {accel_str}")

        if self.memory_shape is not None:
            record_observation(self.memory_shape, mx.get_peak_memory())

        bridge.clear_mlx_cache()

        return out_latent

    @staticmethod
    def _teacache_check(
        state: TeaCacheState,
        current_output: mx.array,
        step: int,
        total_steps: int,
    ) -> tuple[bool, str]:
        """Check if TeaCache allows skipping. Returns (skip, reason)."""
        _, reused, reason = state.try_reuse(current_output, step, total_steps)
        return reused, reason

    def _prepare_kontext_reference(
        self,
        reference_latent: dict,
        precision: mx.Dtype,
    ) -> None:
        """Pack the Kontext reference latent into raw 2x2-patch space.

        Same patchify as the main image latent (`_prepare_krea2_identity_edit`
        follows the identical pattern for Krea2), in FLUX model space
        (scale/shift applied via comfy.latent_formats.Flux().process_in,
        matching how the noise itself enters the model). Stored as raw,
        un-projected patches: FluxTransformer.__call__ concatenates it onto
        `img` BEFORE running `img_in`, matching comfy's `_forward`
        (`img = torch.cat([img, kontext], dim=1)` happens on raw patches;
        `img_in` then runs once on the whole sequence).
        """
        self._kontext_ref_packed: mx.array | None = None
        self._kontext_ref_grid: tuple[int, int] | None = None
        try:
            import comfy.latent_formats

            latent = reference_latent.get("samples", reference_latent)
            if hasattr(latent, "numpy"):
                ref_np = latent.detach().cpu().float().numpy().astype(np.float32, copy=False)
            else:
                ref_np = np.asarray(latent, dtype=np.float32)

            # Process through FLUX latent format
            model_space = comfy.latent_formats.Flux().process_in(
                torch.from_numpy(ref_np)
            )
            ref_np = model_space.numpy().astype(np.float32, copy=False)

            batch, channels, ref_h, ref_w = ref_np.shape
            if channels != bridge.FLUX_LATENT_CHANNELS:
                return

            packed = ref_np.reshape(
                batch, channels, ref_h // 2, 2, ref_w // 2, 2
            )
            packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
            packed = packed.reshape(
                batch, (ref_h // 2) * (ref_w // 2), channels * 4
            )

            ref_mlx = mx.array(packed).astype(precision)
            mx.eval(ref_mlx)

            self._kontext_ref_packed = ref_mlx
            self._kontext_ref_grid = (ref_h // 2, ref_w // 2)

            print(
                f"[ASDX] Kontext reference packed "
                f"[1, {ref_h // 2 * ref_w // 2}, {channels * 4}] grid={ref_h // 2}x{ref_w // 2}"
            )
        except Exception as e:
            print(f"[ASDX] Kontext reference prep failed: {e}")

    @staticmethod
    def _text_lora_state(transformer: Any) -> list[tuple[int, int, float]] | None:
        """Snapshot of the LoRA scales on Krea2's TEXT-side modules.

        Krea2 precomputes `context = encode_text(...)` once before the
        sampling loop (it costs ~86ms at seq=256 and grows with sequence
        length, up to the ~12s/step documented on `encode_text` at seq=4444
        with the enhancer), so a schedule targeting `txtfusion`/`txtmlp`
        would otherwise never show after step 0. Re-encoding unconditionally
        would hand that cost back on every step; comparing two snapshots
        narrows it to the steps that actually moved a text-side weight.

        Must be captured BEFORE the schedule update and compared with a
        snapshot taken after: `_rescale_attached_lora`'s fast path mutates
        scales IN PLACE on the same object, so holding a reference to the
        pre-update transformer would read the already-updated state and never
        detect a change. Returns None when the state can't be read, which the
        caller treats as "changed" -- re-encoding is the safe direction.
        """
        try:
            from ..lora import _iter_adaptable_leaves
        except ImportError:
            return None
        out: list[tuple[int, int, float]] = []
        for attr in ("txtfusion", "txtmlp"):
            root = getattr(transformer, attr, None)
            if root is None:
                continue
            for leaf in _iter_adaptable_leaves(root):
                for a, b, scale in leaf._lora_factors:
                    out.append((id(a), id(b), scale))
        return sorted(out)

    @staticmethod
    def _update_lora_schedule(
        transformer: Any, config: Any, schedule: dict, step: int, total_steps: int
    ) -> Any:
        """Recompute this step's LoRA strength from the schedule and re-apply it.

        Returns the (possibly new) transformer — callers must reassign
        `self.transformer` to the result.

        `schedule["lora"]` is the same LoRAAdapter set up by ASDX_LoraSchedule.
        Its deltas were already baked into the transformer once at
        `strength_start` when the node ran; here we UNDO that previous step's
        contribution and apply the new strength, so repeated calls don't
        compound (delta * scale_prev, then delta * scale_new, not additive).
        `transformer` here is already a private, non-cached object (the node
        graph's ASDX_LoraSchedule returns a fresh transformer, never the one
        shared via loader.py's _MODEL_CACHE — see lora.py's
        _apply_lora_to_transformer), so per-step reapplication is safe.
        """
        lora = schedule.get("lora")
        if lora is None:
            return transformer

        strength_curve = schedule.get("strength_curve", "linear")
        start = schedule.get("strength_start", 1.0)
        end = schedule.get("strength_end", 0.5)
        middle = schedule.get("strength_middle", 1.0)

        progress = step / max(total_steps, 1)

        if strength_curve == "linear":
            if progress <= 0.5:
                strength = start + (middle - start) * (progress * 2)
            else:
                strength = middle + (end - middle) * ((progress - 0.5) * 2)
        elif strength_curve == "cosine":
            import math
            mid = (start + middle) / 2
            if progress <= 0.5:
                strength = mid + (start - mid) * 0.5 * (1 - math.cos(progress * math.pi))
            else:
                end_val = (middle + end) / 2
                strength = end_val + (end - end_val) * 0.5 * (1 - math.cos((progress - 0.5) * math.pi * 2))
        elif strength_curve == "ease_in_out":
            t = 3 * progress ** 2 - 2 * progress ** 3
            if progress <= 0.5:
                strength = start + (middle - start) * t
            else:
                strength = middle + (end - middle) * t
        else:
            strength = start

        from ..lora import ASDX_LoraLoader, base_lora_scale

        new_scale = base_lora_scale(lora.alpha, lora.rank) * strength
        delta_scale = new_scale - lora.scale
        if delta_scale != 0:
            # Apply only the incremental change since the last step's scale,
            # so the cumulative effect matches `deltas * new_scale` without
            # first subtracting out the old contribution separately.
            lora.scale = delta_scale
            transformer = ASDX_LoraLoader._apply_lora_to_transformer(transformer, lora, config)
            lora.scale = new_scale

        if step % 5 == 0:
            print(f"[ASDX] LoRA schedule: step {step}/{total_steps}, strength={strength:.3f}")

        return transformer

    @staticmethod
    def _denoise_to_latent(
        noise: mx.array,
        height: int,
        width: int,
        output_shape: tuple[int, int],
        model_type: str = "flux",
    ) -> torch.Tensor | None:
        """Convert current MLX latent to a decodable PyTorch latent for preview.

        `_unpack_flux_latents`/`_unpack_zimage_latents` already apply
        `process_flux_latent_out` (the FLUX/Z-Image scale+shift) internally
        -- do not additionally call `comfy.latent_formats.Flux().process_out`
        here, or the conversion is applied twice. Krea2 uses its own
        `_unpack_krea2_latents`, which applies `process_wan21_latent_out`
        (Wan21's real per-channel de-whitening -- mean/std, not a plain
        scale_factor) internally for the same reason -- do not reapply it.
        """
        try:
            if model_type in ("zimage", "zimage_turbo"):
                samples = bridge._unpack_zimage_latents(noise, height, width)
            elif model_type in ("krea2", "krea2_turbo"):
                samples = bridge._unpack_krea2_latents(noise, height, width)
            else:
                samples = bridge._unpack_flux_latents(noise, height, width)
            if torch.backends.mps.is_available():
                return samples.to(device="mps")
            return samples
        except Exception:
            return None

    # ── Mode routing (Phase 2) ────────────────────────────────────────

    def _detect_mode(self) -> SamplerMode:
        """Detect sampling mode from connected inputs.

        Priority: inpainting (image + mask) > img2img (image only) >
        depth (depth_image) > text2img (default).
        """
        if self.mode != "auto":
            return SamplerMode(self.mode)

        has_image = self.image is not None and self.image.numel() > 0
        has_mask = self.mask is not None and self.mask.numel() > 0
        has_depth = (
            self.depth_image is not None and self.depth_image.numel() > 0
        )

        if has_depth:
            return SamplerMode.DEPTH_CONTROL
        if has_image and has_mask:
            return SamplerMode.INPAINTING
        if has_image:
            return SamplerMode.IMAGE_TO_IMAGE
        return SamplerMode.TEXT_TO_IMAGE

    def _encode_image_to_latent(self, image: torch.Tensor) -> mx.array:
        """VAE-encode an image to FLUX latent packed format.

        Returns MLX packed latent [B, NH*NW, 64].
        """
        import comfy.latent_formats

        # Transpose [B, H, W, C] -> [B, C, H, W]
        img = image.permute(0, 3, 1, 2) if image.ndim == 4 else image
        # Normalize [0, 1] -> [-1, 1] for VAE
        img = (img - 0.5) / 0.5

        # Use MLX VAE encoder
        from .mlx_vae import MLXVAE

        vae = MLXVAE()
        latent_mlx = vae.encode(img)

        # Convert to numpy and pack
        latent_np = (
            latent_mlx.detach().cpu().numpy()
            if hasattr(latent_mlx, "detach")
            else np.array(latent_mlx, dtype=np.float32)
        )

        # Pack: [B, C, H, W] -> [B, H/2, W/2, C*4] -> flatten spatial
        batch, channels, latent_h, latent_w = latent_np.shape
        packed = latent_np.reshape(
            batch, channels, latent_h // 2, 2, latent_w // 2, 2
        )
        packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
        packed = packed.reshape(
            batch, (latent_h // 2) * (latent_w // 2), channels * 4
        )

        precision = self.config.mlx_dtype
        packed_mlx = mx.array(packed).astype(precision)
        mx.eval(packed_mlx)
        return packed_mlx

    def _prepare_img2img_noise(self) -> mx.array:
        """Blend the real encoded `latent_image` with noise at the requested strength.

        Previously this re-derived latent content from raw `image` pixels via
        `_encode_image_to_latent()`, which routes through `MLXVAE()` -- an
        untrained placeholder with no real weights (same class of bug fixed in
        `vae.py`'s `ASDX_VAEEncode`) that silently produced garbage, and also
        skipped the model-space scale/shift (`process_in`) that the noise/
        Kontext-reference/ControlNet paths all apply. `latent_image` is a
        required node input and is always a real VAE-encoded latent (e.g. via
        the now-fixed `ASDX_VAEEncode`), and `_detect_mode()` only calls this
        method when img2img was actually requested (image connected or
        mode="img2img" set) -- so using it here, the same way
        `_prepare_kontext_reference` already does, is correct and free of the
        placeholder-VAE dependency. `image`/`_encode_image_to_latent` remain
        for `_prepare_inpainting_noise`, which still has the same placeholder
        issue (not fixed here -- see ROADMAP.md).

        Returns MLX packed noise for the denoising loop.
        """
        if self.latent_image is None:
            return self.noise

        import comfy.latent_formats

        samples = self.latent_image.get("samples", self.latent_image)
        latent_np = (
            samples.detach().cpu().float().numpy().astype(np.float32, copy=False)
            if hasattr(samples, "detach")
            else np.asarray(samples, dtype=np.float32)
        )

        # Model-space scale/shift, matching how the noise/Kontext-reference/
        # ControlNet paths all enter the transformer.
        model_space = comfy.latent_formats.Flux().process_in(torch.from_numpy(latent_np))
        latent_np = model_space.numpy().astype(np.float32, copy=False)

        # Pack: [B, C, H, W] -> [B, H/2, W/2, C*4] -> flatten spatial
        batch, channels, latent_h, latent_w = latent_np.shape
        packed = latent_np.reshape(
            batch, channels, latent_h // 2, 2, latent_w // 2, 2
        )
        packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
        packed = packed.reshape(
            batch, (latent_h // 2) * (latent_w // 2), channels * 4
        )

        # Use ComfyUI's noise addition with sqrt(strength) for proper blending
        noise_strength = math.sqrt(self.image_strength)
        noise_np = np.array(self.noise, dtype=np.float32)
        blended = packed * (1.0 - noise_strength) + noise_np * noise_strength

        precision = self.config.mlx_dtype
        return mx.array(blended).astype(precision)

    def _prepare_inpainting_noise(self) -> mx.array:
        """Prepare inpainting noise: encode image, apply mask, add noise.

        The masked region retains the original image (no noise), while
        the unmasked region receives noise. This allows in-painting.
        """
        if self.image is None:
            return self.noise

        # Encode input image to latent
        input_latent = self._encode_image_to_latent(self.image)

        # Prepare mask in packed latent space
        mask_latent = self._prepare_mask_latent()
        if mask_latent is None:
            return self.noise

        # Get raw noise
        noise_np = np.array(self.noise, dtype=np.float32)

        # Apply mask: masked region = original image, unmasked = noise
        # mask=1 means "keep original", mask=0 means "generate new"
        blended = input_latent * mask_latent + noise_np * (1.0 - mask_latent)

        precision = self.config.mlx_dtype
        return mx.array(blended).astype(precision)

    def _prepare_mask_latent(self) -> np.ndarray | None:
        """Convert mask tensor to packed latent-space mask.

        Returns a numpy array in packed format [B, NH*NW, 64] matching
        the noise layout, or None if mask is unavailable.
        """
        if self.mask is None:
            return None

        mask = self.mask
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = mask[0]  # Take first batch item

        h, w = mask.shape[-2:] if mask.ndim == 3 else (mask.shape[0], mask.shape[1])

        # Resize mask to latent space (H/8, W/8)
        latent_h = h // 8
        latent_w = w // 8

        # Resize using bilinear interpolation
        mask_pt = mask.unsqueeze(0).unsqueeze(0) if mask.ndim == 2 else mask
        if mask_pt.shape[-1] != latent_w or mask_pt.shape[-2] != latent_h:
            import torch.nn.functional as F
            mask_pt = F.interpolate(
                mask_pt.float(), size=(latent_h, latent_w), mode="bilinear", align_corners=False
            )
        else:
            mask_pt = mask_pt.float()

        # Blur mask if requested
        if self.mask_blur > 0:
            kernel_size = self.mask_blur if self.mask_blur % 2 == 1 else self.mask_blur + 1
            mask_pt = F.gaussian_blur(mask_pt, kernel_size, sigma=self.mask_blur / 3.0)

        # Squeeze and convert to numpy
        mask_np = mask_pt.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)

        # Expand to packed latent format: [NH, NW] -> [NH, NW, 64]
        latent_np = np.tile(
            mask_np[:, :, np.newaxis, np.newaxis],
            (1, 1, 2, 2)
        ).reshape(latent_h // 2, latent_w // 2, 16 * 4)

        # Pack to [NH/2, NW/2, 64]
        packed = latent_np.reshape(
            latent_h // 2, 2, latent_w // 2, 2, 16 * 4
        )
        packed = np.transpose(packed, (0, 2, 1, 3, 4))
        packed = packed.reshape(
            (latent_h // 2) * (latent_w // 2), 16 * 4
        )

        return packed

    def _prepare_depth_noise(self) -> mx.array:
        """Prepare depth-controlled noise.

        For depth control, the initial noise is the same as text2img,
        but depth conditioning is injected during the transformer forward
        pass (handled by the transformer's depth conditioning path).
        Returns the original noise.
        """
        # Depth conditioning will be handled in the transformer loop
        # via the depth_image passed through the model dict
        return self.noise

    def _prepare_controlnet_latent(self, precision: mx.Dtype) -> mx.array | None:
        """VAE-encode and pack the ControlNet control image into FLUX tokens.

        Matches the reference (comfy/controlnet.py's ControlNet.get_control):
        the control hint is VAE-encoded, then run through the base latent
        format's process_in (same scale/shift as the noisy latent), then
        packed into 2x2 patches exactly like the noise/img input.
        """
        if self.controlnet is None:
            return None
        try:
            vae = self.controlnet.get("vae")
            image = self.controlnet.get("image")
            if vae is None or image is None:
                print("[ASDX] ControlNet: missing vae or image, skipping")
                return None

            # ComfyUI's VAE.encode expects [B,H,W,C] pixels and returns the
            # latent tensor directly (not wrapped in a dict).
            samples = vae.encode(image)
            ctrl_np = (
                samples.detach().cpu().float().numpy().astype(np.float32, copy=False)
                if hasattr(samples, "detach")
                else np.asarray(samples, dtype=np.float32)
            )
            ctrl_np = (ctrl_np - FLUX_LATENT_SHIFT) * FLUX_LATENT_SCALE

            batch, channels, ctrl_h, ctrl_w = ctrl_np.shape
            packed = ctrl_np.reshape(batch, channels, ctrl_h // 2, 2, ctrl_w // 2, 2)
            packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
            packed = packed.reshape(batch, (ctrl_h // 2) * (ctrl_w // 2), channels * 4)

            ctrl_mlx = mx.array(packed).astype(precision)
            mx.eval(ctrl_mlx)
            return ctrl_mlx
        except Exception as e:
            print(f"[ASDX] ControlNet latent prep failed: {e}")
            return None

    @staticmethod
    def _fit_source_latent(source: np.ndarray, tgt_h: int, tgt_w: int) -> np.ndarray:
        """Center-crop a source latent to the target's aspect ratio, then resize
        it onto the target's exact latent grid.

        Ported from comfyui-krea2edit's `_fit_src`, which `krea2_edit_forward`
        applies to every source whose grid differs from the target's; SceneWorks
        fits every edit reference the same way (`fit_edit_references` ->
        `fit_rgb` "crop": scale-to-cover, then center-crop). Training put source
        and target on the SAME grid, so an unfitted source is wrong twice over:
        its RoPE coordinates run past the target's, and a source larger than the
        target dominates the sequence (an 83x125 source against a 48x84 target
        is 72% of the image tokens), pulling the result back toward the
        reference and away from the edit instruction.

        Crop before resizing: a plain resize stretches a mixed-aspect source.

        Runs before the Wan21 whitening below, matching the reference, which
        fits the raw VAE latent. The order is free either way -- whitening is a
        per-channel affine and bilinear interpolation is a weighted mean with
        unit weights, so the two commute exactly.

        Deliberately NOT on MLX, as the exception the canon record `Porting from
        a reference means converting it to Apple Silicon` requires to be stated
        where it is taken. `mlx.nn.Upsample(mode="linear")` does produce the
        exact target grid (NHWC layout, fractional scale_factor) but lands
        2.2e-05 relative away from torch's bilinear -- a real algorithmic
        difference, measured against a harness checked to be exact at scale=1
        and bit-identical on `nearest`. This path runs ONCE per generation on
        ~1.6M elements, milliseconds against minutes of sampling, so converting
        it buys nothing and costs the property that validates it: bit-exact
        equality with the reference's own `_fit_src`. Revisit only if this ever
        moves onto a per-step path.

        This is the LATENT-space fallback (the reference's "crop (legacy)"
        geometry): center-crop to the target AR, then resize onto the target's
        exact grid. It is only used when no source_image + vae are wired. The
        preferred path fits in PIXEL space (`_prepare_krea2_identity_edit_pixels`),
        which is what the v1_2 LoRA was trained with and what the reference
        prefers -- latent-space resizing softens VAE latents and is the source
        of the face artifacts this fallback is meant to avoid.
        """
        _, _, src_h, src_w = source.shape
        if (src_h, src_w) == (tgt_h, tgt_w):
            return source

        # Center-crop to the target aspect ratio, then resize onto the target's
        # exact grid. A plain resize would stretch a mixed-AR source.
        scale = max(tgt_h / src_h, tgt_w / src_w)
        crop_h = min(src_h, int(round(tgt_h / scale)))
        crop_w = min(src_w, int(round(tgt_w / scale)))
        y0, x0 = (src_h - crop_h) // 2, (src_w - crop_w) // 2
        cropped = source[:, :, y0:y0 + crop_h, x0:x0 + crop_w]
        fitted = torch.nn.functional.interpolate(
            torch.from_numpy(np.ascontiguousarray(cropped)),
            size=(tgt_h, tgt_w), mode="bilinear", align_corners=False,
        ).numpy()
        print(
            f"[ASDX] Identity Edit: source latent fitted {src_h}x{src_w} -> "
            f"{tgt_h}x{tgt_w} (center-crop to target aspect, then resize)"
        )
        return fitted

    def _prepare_krea2_identity_edit(self) -> None:
        """Prepare the Identity Edit source for Krea2.

        Two paths, matching comfyui-krea2edit:
        - Pixel path (preferred): fit the source IMAGE in pixel space, VAE-encode,
          whiten, pack. This is what the v1_2 LoRA was trained with and what the
          reference prefers ("blur-proof"); it yields a content-only ref grid
          centered with a RoPE offset. Requires source_image + vae.
        - Latent path (fallback): fit the source LATENT (center-crop + bilinear),
          whiten, pack. Yields a full-size ref grid, no offset. Used when only
          source_latent is wired.

        The frame index in RoPE distinguishes source (frame=1) from target
        (frame=0) tokens for identity preservation.
        """
        if self.source_image is not None and self.vae is not None:
            self._prepare_krea2_identity_edit_pixels()
            return

        if self.source_latent is None:
            return

        try:
            source_samples = self.source_latent.get("samples")
            if source_samples is None:
                return

            source_np = (
                source_samples.detach().cpu().float().numpy().astype(np.float32, copy=False)
                if hasattr(source_samples, "detach")
                else np.asarray(source_samples, dtype=np.float32)
            )

            source_np = self._fit_source_latent(
                source_np, self.height // 8, self.width // 8
            )

            # Whiten into the model's internal Wan21 latent space -- matches
            # comfy/model_base.py::Krea2.extra_conds's
            # `latents.append(self.process_latent_in(lat))` for reference
            # latents, and this project's own FLUX Kontext path
            # (`_prepare_kontext_reference`), which applies
            # `latent_formats.Flux().process_in(...)` for the same reason.
            from ..native.config import process_wan21_latent_in

            source_mlx = mx.array(source_np).astype(mx.float32)
            source_mlx = process_wan21_latent_in(source_mlx)
            mx.eval(source_mlx)
            source_np = np.array(source_mlx, dtype=np.float32)

            # [B, C, H, W] -> pack to [B, NH*NW, 64]
            batch, channels, src_h, src_w = source_np.shape
            packed = source_np.reshape(
                batch, channels, src_h // 2, 2, src_w // 2, 2
            )
            packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
            packed = packed.reshape(
                batch, (src_h // 2) * (src_w // 2), channels * 4
            )

            precision = self.config.mlx_dtype
            source_packed = mx.array(packed).astype(precision)
            mx.eval(source_packed)

            # The latent path fits onto the target's exact grid, so the ref grid
            # equals the target grid -- no centered offset needed.
            self._identity_edit_source = source_packed
            self._identity_edit_src_grid = (src_h // 2, src_w // 2)
            self._identity_edit_src_offset = (0, 0)

            print(
                f"[ASDX] Identity Edit: source latent packed "
                f"[1, {src_h//2*src_w//2}, {channels*4}] grid={src_h//2}x{src_w//2}"
            )
        except Exception as e:
            # Don't swallow this: a `source_latent` was explicitly wired in, so
            # silently disabling Identity Edit and continuing would run the full
            # (multi-minute) sampling pass while quietly ignoring the source
            # image -- a misleading "success" instead of a fast, clear failure.
            raise RuntimeError(f"ASDX: Identity Edit prep failed: {e}") from e

    def _prepare_krea2_identity_edit_pixels(self) -> None:
        """Pixel-space source prep (the blur-proof path), matching
        comfyui-krea2edit's `_fit_encode_image` fit mode.

        Fits the source IMAGE in pixel space (contain + /16 floor + bicubic, or a
        minimal center-crop when the AR nearly matches), VAE-encodes the fitted
        image, whitens, and packs. The ref grid is content-only (no zero padding)
        and centered in the target grid with a RoPE offset -- the geometry the
        v1_2 LoRA was trained with.
        """
        try:
            img = self.source_image  # [B, H, W, C] in [0, 1]
            if img is None:
                return
            px_h, px_w = self.height, self.width  # target pixel dims
            img4 = img.movedim(-1, 1).float()  # [B, C, H, W]
            ih, iw = img4.shape[-2:]

            sc = min(px_h / ih, px_w / iw)
            CROP_TOL = 0.08
            if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
                # Near-matched AR: fill the target grid exactly via a minimal
                # center-crop (avoids 1-2 token fit-inside margins that the
                # reference found cause edge-duplication seams).
                s = max(px_h / ih, px_w / iw)
                ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
                y0, x0 = (ih - ch) // 2, (iw - cw) // 2
                img4 = img4[..., y0:y0 + ch, x0:x0 + cw]
                nh, nw = px_h, px_w
            else:
                # Genuine AR mismatch: snap to /16 (NOT /8) to stay byte-identical
                # to the trainer's _fit_prep, capped at the target's /16 floor.
                nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
                nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))

            fitted = torch.nn.functional.interpolate(
                img4, size=(nh, nw), mode="bicubic", antialias=True
            )

            # VAE-encode the fitted image. comfy's VAE.encode takes [B, H, W, C]
            # in [0, 1]; the reference does exactly this.
            fitted_hwc = fitted.movedim(1, -1)[..., :3].clamp(0, 1)
            latent = self.vae.encode(fitted_hwc)
            if getattr(self.vae, "latent_dim", 2) == 3 and latent.dim() == 5:
                latent = latent.squeeze(2)
            latent_np = latent.detach().cpu().float().numpy().astype(np.float32, copy=False)

            # Whiten into the model's internal Wan21 latent space.
            from ..native.config import process_wan21_latent_in
            latent_mlx = mx.array(latent_np).astype(mx.float32)
            latent_mlx = process_wan21_latent_in(latent_mlx)
            mx.eval(latent_mlx)
            latent_np = np.array(latent_mlx, dtype=np.float32)

            # [B, C, H, W] -> pack to [B, NH*NW, 64]
            batch, channels, src_lat_h, src_lat_w = latent_np.shape
            packed = latent_np.reshape(
                batch, channels, src_lat_h // 2, 2, src_lat_w // 2, 2
            )
            packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
            packed = packed.reshape(
                batch, (src_lat_h // 2) * (src_lat_w // 2), channels * 4
            )

            precision = self.config.mlx_dtype
            source_packed = mx.array(packed).astype(precision)
            mx.eval(source_packed)

            src_h = src_lat_h // 2
            src_w = src_lat_w // 2
            img_h = (self.height // 8) // 2
            img_w = (self.width // 8) // 2
            self._identity_edit_source = source_packed
            self._identity_edit_src_grid = (src_h, src_w)
            # Center the content-only ref grid in the target grid (matching the
            # reference `_imgids_offset`); (0, 0) when it fills the target.
            self._identity_edit_src_offset = (
                max(0, (img_h - src_h) // 2),
                max(0, (img_w - src_w) // 2),
            )
            print(
                f"[ASDX] Identity Edit (pixel path): source {ih}x{iw} -> fitted "
                f"{nh}x{nw} px, latent {src_lat_h}x{src_lat_w}, grid {src_h}x{src_w} "
                f"centered offset {self._identity_edit_src_offset}"
            )
        except Exception as e:
            raise RuntimeError(f"ASDX: Identity Edit pixel prep failed: {e}") from e

    def _krea2_ref_attn_bias(
        self, txt_len: int, src_h: int, src_w: int, tgt_h: int, tgt_w: int, boost: float
    ) -> mx.array:
        """Additive attention-logit bias on the [text | source | target] sequence.

        Matches the ComfyUI reference `_ref_attn_bias`: only target rows get a
        log(boost) bias added on source columns (equivalent to multiplying the
        post-softmax target->source attention weight by `boost` before
        renormalization). All other entries are 0.
        """
        src_len = src_h * src_w
        tgt_len = tgt_h * tgt_w
        total = txt_len + src_len + tgt_len
        bias = mx.zeros((1, 1, total, total), dtype=mx.float32)
        log_boost = math.log(max(boost, 1e-4))
        bias[:, :, txt_len + src_len:, txt_len:txt_len + src_len] = log_boost
        return bias

    def _nag_is_active(self, sigma_t: float) -> bool:
        """Mirrors krea2-nag's _nag_is_active: NAG is a no-op if phi/alpha
        is 0, or if sigma_t falls outside [nag_sigma_end, nag_sigma_start].
        ASDX's Krea2 sigmas are already normalized (linear 1->0, see
        calculate_sigmas), unlike the reference's raw 0-1000 scale -- the
        sampler node's nag_sigma_start/nag_sigma_end defaults (1.0/0.0)
        are on THIS scale, not the reference's.
        """
        negative = self.positive.get("_negative") if isinstance(self.positive, dict) else None
        if negative is None:
            return False
        if self.nag_phi == 0.0 or self.nag_alpha == 0.0:
            return False
        return self.nag_sigma_end <= sigma_t <= self.nag_sigma_start

    def _run_krea2(
        self,
        steps: int,
        seed: int,
        prompt_embeds: mx.array | None,
        pooled_embeds: mx.array | None,
        guidance: float,
    ) -> dict:
        """Run the Krea2 (SingleStreamDiT) sampling loop.

        Krea2 differs from FLUX:
          - Uses Qwen3-VL text embeddings (12-layer fused, 30720-dim)
          - txtfusion adapter fuses layers internally
          - txtmlp projects to hidden_dim (6144)
          - 3-axis RoPE (frame, height, width) for Identity Edit, on a real 2D
            token grid (not a flat sequential index)
          - Flow matching schedule (linear 1→0)
          - DoubleSharedModulation: timestep vec → 6 params, SHARED across the
            whole sequence (text, source, target) — source/target are
            distinguished only by the RoPE frame index, matching the ComfyUI
            reference `krea2_edit_forward`
          - GQA attention with per-head QK norm + sigmoid gate
          - SwiGLU MLP
          - Same Euler update: noise += output * (sigma_next - sigma_t)
        """
        from .bridge import conditioning_krea2_to_mlx

        precision = self.config.mlx_dtype
        model_type = self.model_type

        # ── Krea2 conditioning ──────────────────────────────────────
        # Krea2 expects fused Qwen3-VL embeddings [B, T, 12*2560] = [B, T, 30720]
        # The bridge handles single-layer → fused conversion
        txt_fused = conditioning_krea2_to_mlx(self.positive, precision)
        txt_len = txt_fused.shape[1]

        # Target image token grid (in patches): latent is height//8 x width//8,
        # each Krea2 token packs a 2x2 patch, so the grid is (latent//2) x (latent//2).
        cap_module.require_divisible_dims(self.width, self.height, 16, "krea2")
        img_h = (self.height // 8) // 2
        img_w = (self.width // 8) // 2
        if img_h * img_w != self.noise.shape[1]:
            raise RuntimeError(
                f"ASDX: Krea2 token grid mismatch: {img_h}x{img_w}={img_h * img_w} "
                f"vs noise token count {self.noise.shape[1]}"
            )

        # Setup sigmas (flow matching: linear 1→0)
        sigmas = calculate_sigmas(model_type, self.scheduler_name, steps, self.width, self.height)
        solver_state: dict[str, Any] = {}

        # Setup TeaCache
        teacache_state: TeaCacheState | None = None
        if self.teacache:
            teacache_state = TeaCacheState(
                threshold=self.teacache_threshold,
                warmup_steps=5,
                final_steps=3,
            )

        # SeaCache state (not supported for Krea2)
        previous_output: mx.array | None = None

        # Memory profiling
        mx.reset_peak_memory()
        step_times: list[float] = []

        t_sampling_start = time.perf_counter()

        # Source latent prepending for Identity Edit
        has_latent = self.source_latent is not None and self.source_latent.get("samples") is not None
        has_pixels = self.source_image is not None and self.vae is not None
        if has_latent or has_pixels:
            self._prepare_krea2_identity_edit()

        src_tokens = getattr(self, "_identity_edit_source", None)
        src_grid = getattr(self, "_identity_edit_src_grid", None)
        src_offset = getattr(self, "_identity_edit_src_offset", (0, 0))
        src_h, src_w = src_grid if src_grid is not None else (None, None)

        # RoPE is the same at every step (text/source/target positions don't
        # change across the denoising loop) — precompute once. The pixel path
        # yields a content-only ref grid centered in the target (offset != 0);
        # the latent path fills the target grid (offset == 0).
        if src_tokens is not None and src_h is not None and src_w is not None:
            src_grids = [(src_h, src_w)]
            src_offsets = [src_offset]
        else:
            src_grids = None
            src_offsets = None
        rope_freqs = self.transformer.get_rope_grid(
            img_h, img_w, txt_len, src_grids, src_offsets
        )

        # ref_boost: additive attention-logit bias favoring source tokens,
        # only meaningful (and only computable) when Identity Edit is active.
        ref_boost = None
        if src_tokens is not None and self.ref_boost and self.ref_boost != 1.0:
            ref_boost = self._krea2_ref_attn_bias(
                txt_len, src_h, src_w, img_h, img_w, self.ref_boost
            )

        # Text conditioning (txtfusion + optional Krea2T enhancer) doesn't
        # depend on the noisy image latent or timestep, so it's identical on
        # every step -- precompute once instead of letting each predict()
        # call recompute it (was costing up to ~12s/step, see
        # SingleStreamDiT.encode_text's docstring).
        context = self.transformer.encode_text(txt_fused, self.krea2_enhancer_strength)
        mx.eval(context)

        # NAG: encode the negative conditioning once, same rationale as the
        # positive `context` precompute above -- it doesn't depend on the
        # noisy image latent or timestep either.
        negative = self.positive.get("_negative") if isinstance(self.positive, dict) else None
        neg_context = None
        neg_freqs = None
        if negative is not None:
            neg_txt_fused = conditioning_krea2_to_mlx(negative, precision)
            neg_context = self.transformer.encode_text(neg_txt_fused, self.krea2_enhancer_strength)
            mx.eval(neg_context)
            neg_txt_len = neg_context.shape[1]
            neg_freqs = self.transformer.get_rope_grid(
                img_h, img_w, neg_txt_len, src_grids, src_offsets
            )

        def _predict(img_at, sigma_at):
            if neg_context is not None and self._nag_is_active(float(sigma_at)):
                return self.transformer.predict_nag(
                    img=img_at, context=context, neg_context=neg_context,
                    timestep=mx.array([float(sigma_at)], dtype=mx.float32),
                    img_h=img_h, img_w=img_w,
                    freqs=rope_freqs, neg_freqs=neg_freqs, ref_boost=ref_boost,
                    src=src_tokens, src_h=src_h, src_w=src_w,
                    phi=self.nag_phi, tau=self.nag_tau, alpha=self.nag_alpha,
                )
            return self.transformer.predict(
                img=img_at, context=context, timestep=sigma_at, img_h=img_h, img_w=img_w,
                freqs=rope_freqs, ref_boost=ref_boost, src=src_tokens, src_h=src_h, src_w=src_w,
            )

        for t in range(steps):
            step_start = time.perf_counter()

            sigma_t = sigmas[t]
            sigma_next = sigmas[t + 1] if t + 1 < len(sigmas) else 0.0

            # Update LoRA schedule, re-encoding the precomputed `context`
            # only on steps that actually moved a text-side weight (see below).
            if self.lora_schedule is not None:
                self.lora_schedule["step"] = t
                text_state_before = self._text_lora_state(self.transformer)
                self.transformer = self._update_lora_schedule(
                    self.transformer, self.config, self.lora_schedule, t, steps
                )
                # `context` was precomputed once, before this loop. A schedule
                # that changes a `txtfusion`/`txtmlp` weight therefore has NO
                # effect after step 0 unless the text conditioning is re-encoded
                # -- and essentially every real Krea2 LoRA targets txtfusion
                # (35/35 in this machine's library), so this is the common case,
                # not an edge case. Re-encode only on steps that actually moved
                # a text-side weight: a step whose `delta_scale` was 0, or one
                # touching only image-side blocks, still costs nothing.
                text_state_after = self._text_lora_state(self.transformer)
                if text_state_before != text_state_after:
                    context = self.transformer.encode_text(
                        txt_fused, self.krea2_enhancer_strength
                    )
                    mx.eval(context)

            # ── Compute transformer output ──────────────────────────
            if teacache_state is not None:
                current_output = _predict(self.noise, sigma_t)
                mx.eval(current_output)

                reused, reason = self._teacache_check(
                    teacache_state, current_output, t, steps
                )

                if reused:
                    noise_pred = (
                        teacache_state.last_feature
                        if teacache_state.last_feature is not None
                        else current_output
                    )
                    skip_step = True
                else:
                    skip_step = False
                    noise_pred = current_output
            else:
                current_output = _predict(self.noise, sigma_t)
                mx.eval(current_output)
                noise_pred = current_output
                skip_step = False

            # SeaCache: not supported for Krea2 (skip)

            if skip_step and previous_output is not None:
                noise_pred = previous_output

            if not skip_step:
                previous_output = current_output

            # denoised = x - noise_pred*sigma (see solvers.py module docstring)
            denoised = self.noise - noise_pred * sigma_t

            def _model_call(x_at, sigma_at):
                out = _predict(x_at, sigma_at)
                mx.eval(out)
                return x_at - out * sigma_at

            self.noise, solver_state = solvers.step(
                self.sampler_name,
                x=self.noise,
                sigma=sigma_t,
                sigma_next=sigma_next,
                denoised=denoised,
                state=solver_state,
                seed=seed,
                step_index=t,
                is_flow_matching=self._is_flow_matching,
                model_call=_model_call,
            )
            mx.eval(self.noise)

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            # Preview
            if self.preview and self.previewer is not None:
                preview_latent = self._denoise_to_latent(
                    self.noise, self.height, self.width, self.output_shape,
                    model_type=model_type,
                )
                if preview_latent is not None:
                    preview_bytes = self.previewer.decode_latent_to_preview_image(
                        "JPEG", preview_latent
                    )
                    if preview_bytes:
                        import comfy.utils
                        comfy.utils.ProgressBar(steps).update_absolute(
                            t + 1, steps, preview_bytes
                        )

            # Progress logging
            if (t + 1) % 5 == 0 or t == 0:
                cache_tag = ""
                if skip_step:
                    cache_tag = " [cache]"
                print(f"[ASDX] Step {t + 1}/{steps} - {step_time:.3f}s{cache_tag}")

        total_time = time.perf_counter() - t_sampling_start

        # ── Low memory mode ─────────────────────────────────────────
        if self.low_memory_mode:
            # self.transformer is the SAME object as loader.py's
            # _MODEL_CACHE entry -- dropping only this local reference never
            # freed anything (and _SamplerCore is local to sample() anyway,
            # released on return regardless). Evict the cache entry itself
            # so the memory is actually returned; the next generation will
            # reload the checkpoint from disk instead of hitting the cache.
            from ..loader import clear_model_cache
            clear_model_cache()
            self.transformer = None
            print("[ASDX] Low memory: model cache evicted, next load will read from disk")

        # Convert result to ComfyUI latent. Krea2's own `latent_formats.Wan21`
        # applies a real per-channel affine de-whitening in `_unpack_krea2_
        # latents` (see that function's docstring) -- NOT FLUX's scalar
        # scale/shift, despite sharing the same [C,pH,pW] patch order.
        out_latent = bridge.mlx_to_comfy_latent_krea2(
            self.noise, self.height, self.width, {"samples": self.noise}
        )
        out_latent["sdmlx_model_type"] = model_type
        out_latent["sdmlx_model_name"] = "unknown"

        # Memory stats
        mem = bridge.collect_mlx_memory()
        avg_step = sum(step_times) / len(step_times) if step_times else 0

        print(
            f"[ASDX] Krea2 Sampling complete: {total_time:.1f}s total, "
            f"{avg_step:.3f}s/step, {mem['peak_gb']:.1f}GB peak"
        )

        if self.memory_shape is not None:
            record_observation(self.memory_shape, mx.get_peak_memory())

        bridge.clear_mlx_cache()

        return out_latent

    def _run_sdxl(self, steps: int, seed: int) -> dict:
        """Run the SDXL (conv UNet, EPS/discrete-DDPM) sampling loop.

        Fundamentally different from the flow-matching DiT loop above:
          - True classifier-free guidance: two forward passes per step
            (positive + negative conditioning), combined via
            `noise_pred = uncond + cfg_scale*(cond-uncond)` — the FLUX/Krea2
            loop only ever does a single conditional pass (guidance is baked
            into the model via an embedding, not sampled twice). This is the
            first real consumer of `_negative` (stored by
            `ASDX_ConditioningMerger` but never read before this).
          - EPS preconditioning (`SDXLSampling.calculate_input`): the UNet
            input is `x / sqrt(sigma^2+1)`, not raw `x`; its output is a
            noise (eps) prediction, denoised via `x - eps*sigma`.
          - ADM/"y" vector (pooled CLIP-G + size/crop sinusoidal embeddings)
            drives an additive embedding branch, not cross-attention.
          - `self.guidance` is reused as the CFG scale (no separate node
            input) — same pattern as Krea2 reusing it for its own guidance.

        Scope: txt2img only. img2img/inpainting/depth/ControlNet mode
        routing (`_detect_mode()` and friends) is FLUX-specific and not
        wired up for SDXL in this phase — `run()` dispatches here before
        reaching that code.
        """
        from ..native.sdxl.model import encode_adm

        precision = self.config.mlx_dtype
        sampling = SDXLSampling()

        negative = self.positive.get("_negative") if isinstance(self.positive, dict) else None
        if negative is None:
            raise RuntimeError(
                "ASDX: SDXL requires a negative prompt for true classifier-free "
                "guidance. Provide one via ASDX_ConditioningMerger (merge the "
                "positive and negative ASDX_CLIPTextEncode outputs before the sampler)."
            )

        cond_pos, pooled_pos = bridge.conditioning_sdxl_to_mlx(self.positive, precision)
        cond_neg, pooled_neg = bridge.conditioning_sdxl_to_mlx(negative, precision)

        y_pos = encode_adm(pooled_pos, height=self.height, width=self.width).astype(precision)
        y_neg = encode_adm(pooled_neg, height=self.height, width=self.width).astype(precision)

        cfg_scale = float(self.guidance) if self.guidance and self.guidance > 0 else 7.0

        sigmas = calculate_sigmas("sdxl", self.scheduler_name, steps)
        solver_state: dict[str, Any] = {}

        # Initial state: sigma_max * unit-gaussian noise (txt2img: no seed
        # latent to add, matching EPS.noise_scaling at sigma=sigma_max).
        x = self.noise * sigmas[0]

        mx.reset_peak_memory()
        step_times: list[float] = []
        t_sampling_start = time.perf_counter()

        for t in range(steps):
            step_start = time.perf_counter()
            sigma_t = sigmas[t]
            sigma_next = sigmas[t + 1] if t + 1 < len(sigmas) else 0.0

            # Update LoRA schedule
            if self.lora_schedule is not None:
                self.lora_schedule["step"] = t
                self.transformer = self._update_lora_schedule(
                    self.transformer, self.config, self.lora_schedule, t, steps
                )

            xc = sampling.calculate_input(sigma_t, x)
            timestep = mx.array([sampling.timestep(sigma_t)], dtype=mx.float32)

            eps_pos = self.transformer(xc, timestep, cond_pos, y_pos)
            eps_neg = self.transformer(xc, timestep, cond_neg, y_neg)
            mx.eval(eps_pos, eps_neg)

            eps = eps_neg + cfg_scale * (eps_pos - eps_neg)

            denoised = sampling.calculate_denoised(sigma_t, eps, x)

            def _model_call(x_at, sigma_at):
                xc_at = sampling.calculate_input(sigma_at, x_at)
                timestep_at = mx.array([sampling.timestep(sigma_at)], dtype=mx.float32)
                eps_pos_at = self.transformer(xc_at, timestep_at, cond_pos, y_pos)
                eps_neg_at = self.transformer(xc_at, timestep_at, cond_neg, y_neg)
                mx.eval(eps_pos_at, eps_neg_at)
                eps_at = eps_neg_at + cfg_scale * (eps_pos_at - eps_neg_at)
                return sampling.calculate_denoised(sigma_at, eps_at, x_at)

            x, solver_state = solvers.step(
                self.sampler_name,
                x=x,
                sigma=sigma_t,
                sigma_next=sigma_next,
                denoised=denoised,
                state=solver_state,
                seed=seed,
                step_index=t,
                is_flow_matching=self._is_flow_matching,
                model_call=_model_call,
            )
            mx.eval(x)

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            if (t + 1) % 5 == 0 or t == 0:
                print(f"[ASDX] SDXL Step {t + 1}/{steps} - {step_time:.3f}s")

        total_time = time.perf_counter() - t_sampling_start

        if self.low_memory_mode:
            # self.transformer is the SAME object as loader.py's
            # _MODEL_CACHE entry -- dropping only this local reference never
            # freed anything (and _SamplerCore is local to sample() anyway,
            # released on return regardless). Evict the cache entry itself
            # so the memory is actually returned; the next generation will
            # reload the checkpoint from disk instead of hitting the cache.
            from ..loader import clear_model_cache
            clear_model_cache()
            self.transformer = None
            print("[ASDX] Low memory: model cache evicted, next load will read from disk")

        out_latent = bridge.mlx_to_comfy_latent_sdxl(x, {"samples": x})
        out_latent["sdmlx_model_type"] = "sdxl"
        out_latent["sdmlx_model_name"] = "unknown"

        mem = bridge.collect_mlx_memory()
        avg_step = sum(step_times) / len(step_times) if step_times else 0

        print(
            f"[ASDX] SDXL Sampling complete: {total_time:.1f}s total, "
            f"{avg_step:.3f}s/step, {mem['peak_gb']:.1f}GB peak, cfg={cfg_scale:.1f}"
        )

        if self.memory_shape is not None:
            record_observation(self.memory_shape, mx.get_peak_memory())

        bridge.clear_mlx_cache()

        return out_latent

    def _run_zimage(self, steps: int, seed: int) -> dict:
        """Run the Z-Image (NextDiT) sampling loop.

        Flow-matching, same Euler update as FLUX/Krea2 (`sampler/
        scheduling.py::time_snr_shift` — a FIXED shift, not FLUX-dev's
        resolution-dependent `mu`). Z-Image inherits FLUX's 16-channel/
        patch=2 `latent_formats.Flux` VAE space (scale/shift constants,
        verified via comfy's class chain: `ZImage(Lumina2)`,
        `Lumina2.latent_format = latent_formats.Flux`) but NOT its
        per-token patch-channel axis order: comfy/ldm/lumina/model.py
        packs `[pH,pW,C]`, FLUX packs `[C,pH,pW]` (comfy/ldm/flux/
        model.py:319) — reusing FLUX's pack/unpack functions here produced
        a visibly grainy/pixelated image despite a NaN-free, otherwise
        correct forward pass. Uses `bridge.prepare_noise_from_latent_zimage`/
        `mlx_to_comfy_latent_zimage` (own `[pH,pW,C]` packing) instead.

        Single conditional forward pass per step: `NextDiT` has no built-in
        guidance-embedding mechanism (unlike FLUX-dev's `guidance_in`), and
        no CFG is wired here — if the base (non-turbo) checkpoint is found
        to need it for quality, add the same two-pass cond/uncond approach
        `_run_sdxl()` uses, driven by `positive["_negative"]`.
        """
        precision = self.config.mlx_dtype
        model_type = self.model_type

        context = bridge.conditioning_zimage_to_mlx(self.positive, precision)

        cap_module.require_divisible_dims(self.width, self.height, 16, "zimage")
        img_h = (self.height // 8) // 2
        img_w = (self.width // 8) // 2
        if img_h * img_w != self.noise.shape[1]:
            raise RuntimeError(
                f"ASDX: Z-Image token grid mismatch: {img_h}x{img_w}={img_h * img_w} "
                f"vs noise token count {self.noise.shape[1]}"
            )

        sigmas = calculate_sigmas(model_type, self.scheduler_name, steps, self.width, self.height)
        solver_state: dict[str, Any] = {}

        teacache_state: TeaCacheState | None = None
        if self.teacache:
            teacache_state = TeaCacheState(
                threshold=self.teacache_threshold, warmup_steps=5, final_steps=3,
            )

        mx.reset_peak_memory()
        step_times: list[float] = []
        t_sampling_start = time.perf_counter()

        for t in range(steps):
            step_start = time.perf_counter()
            sigma_t = sigmas[t]
            sigma_next = sigmas[t + 1] if t + 1 < len(sigmas) else 0.0

            # Update LoRA schedule
            if self.lora_schedule is not None:
                self.lora_schedule["step"] = t
                self.transformer = self._update_lora_schedule(
                    self.transformer, self.config, self.lora_schedule, t, steps
                )

            current_output = self.transformer.predict(self.noise, context, sigma_t, img_h, img_w)
            mx.eval(current_output)

            if teacache_state is not None:
                reused, reason = self._teacache_check(teacache_state, current_output, t, steps)
                if reused:
                    noise_pred = (
                        teacache_state.last_feature
                        if teacache_state.last_feature is not None
                        else current_output
                    )
                    skip_step = True
                else:
                    skip_step = False
                    noise_pred = current_output
            else:
                noise_pred = current_output
                skip_step = False

            # denoised = x - noise_pred*sigma (see solvers.py module docstring)
            denoised = self.noise - noise_pred * sigma_t

            def _model_call(x_at, sigma_at):
                out = self.transformer.predict(x_at, context, sigma_at, img_h, img_w)
                mx.eval(out)
                return x_at - out * sigma_at

            self.noise, solver_state = solvers.step(
                self.sampler_name,
                x=self.noise,
                sigma=sigma_t,
                sigma_next=sigma_next,
                denoised=denoised,
                state=solver_state,
                seed=seed,
                step_index=t,
                is_flow_matching=self._is_flow_matching,
                model_call=_model_call,
            )
            mx.eval(self.noise)

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            if self.preview and self.previewer is not None:
                preview_latent = self._denoise_to_latent(
                    self.noise, self.height, self.width, self.output_shape,
                    model_type="zimage",
                )
                if preview_latent is not None:
                    preview_bytes = self.previewer.decode_latent_to_preview_image(
                        "JPEG", preview_latent
                    )
                    if preview_bytes:
                        import comfy.utils
                        comfy.utils.ProgressBar(steps).update_absolute(t + 1, steps, preview_bytes)

            if (t + 1) % 5 == 0 or t == 0:
                cache_tag = " [cache]" if skip_step else ""
                print(f"[ASDX] Z-Image Step {t + 1}/{steps} - {step_time:.3f}s{cache_tag}")

        total_time = time.perf_counter() - t_sampling_start

        if self.low_memory_mode:
            # self.transformer is the SAME object as loader.py's
            # _MODEL_CACHE entry -- dropping only this local reference never
            # freed anything (and _SamplerCore is local to sample() anyway,
            # released on return regardless). Evict the cache entry itself
            # so the memory is actually returned; the next generation will
            # reload the checkpoint from disk instead of hitting the cache.
            from ..loader import clear_model_cache
            clear_model_cache()
            self.transformer = None
            print("[ASDX] Low memory: model cache evicted, next load will read from disk")

        out_latent = bridge.mlx_to_comfy_latent_zimage(
            self.noise, self.height, self.width, {"samples": self.noise}
        )
        out_latent["sdmlx_model_type"] = model_type
        out_latent["sdmlx_model_name"] = "unknown"

        mem = bridge.collect_mlx_memory()
        avg_step = sum(step_times) / len(step_times) if step_times else 0
        accel = (
            f"TeaCache({teacache_state.hits}h/{teacache_state.real_steps}real)"
            if teacache_state is not None else "standard"
        )

        print(
            f"[ASDX] Z-Image Sampling complete: {total_time:.1f}s total, "
            f"{avg_step:.3f}s/step, {mem['peak_gb']:.1f}GB peak, {accel}"
        )

        if self.memory_shape is not None:
            record_observation(self.memory_shape, mx.get_peak_memory())

        bridge.clear_mlx_cache()

        return out_latent

    def _run_flux2(self, steps: int, seed: int) -> dict:
        """Run the Flux2/Klein sampling loop.

        Flow-matching, same Euler update as FLUX.1/Krea2/Z-Image, but with a
        FIXED shift of 2.02 (`sampler/scheduling.py::generate_sigmas`'s
        `model_type == "flux2"` branch) — not FLUX-dev's resolution-dependent
        mu. Latent packing is Flux2-specific (128ch, patch_size=1, 16x VAE
        downscale — `bridge.prepare_noise_from_latent_flux2`/
        `mlx_to_comfy_latent_flux2`), NOT reused from FLUX.1 (unlike
        Z-Image, which shares FLUX.1's exact 16ch/patch=2/8x-downscale
        latent space — Flux2's VAE is architecturally different, confirmed
        via `comfy/latent_formats.py::Flux2.spacial_downscale_ratio=16`).

        Single conditional forward pass per step (no two-pass CFG) — same
        as Z-Image. `guidance` is threaded through regardless of whether the
        loaded checkpoint has a `guidance_in` (Klein doesn't; the larger
        Flux2-D does): `Flux2Transformer.time_embed` silently ignores it
        when `guidance_in` wasn't allocated for this checkpoint, so passing
        a value is always safe.
        """
        precision = self.config.mlx_dtype
        model_type = self.model_type

        context, cond_guidance = bridge.conditioning_flux2_to_mlx(self.positive, precision)
        txt_len = context.shape[1]

        effective_guidance = float(self.guidance) if self.guidance and self.guidance > 0 else (
            cond_guidance if cond_guidance is not None else 3.5
        )

        if self.capability is not None:
            candidate_params = {
                "guidance": self.guidance,
                "width": self.width,
                "height": self.height,
                "steps": steps,
            }
            try:
                valid_params, dropped = cap_module.filter_params_for_model(
                    self.capability, candidate_params
                )
                if dropped:
                    print(f"[ASDX] Dropped params for {self.capability.name}: {dropped}")
                if "guidance" in valid_params and valid_params["guidance"] is not None:
                    effective_guidance = float(valid_params["guidance"])
            except ValueError as e:
                print(f"[ASDX] Capability filter warning: {e}")

        # Flux2's VAE downscales 16x spatially with patch_size=1 (no further
        # 2x2 token packing) — the image token grid IS the latent grid,
        # unlike FLUX.1/Z-Image's extra //2 for their 2x2 patchify.
        cap_module.require_divisible_dims(
            self.width, self.height, bridge.FLUX2_VAE_DOWNSCALE, "flux2"
        )
        img_h = self.height // bridge.FLUX2_VAE_DOWNSCALE
        img_w = self.width // bridge.FLUX2_VAE_DOWNSCALE
        if img_h * img_w != self.noise.shape[1]:
            raise RuntimeError(
                f"ASDX: Flux2 token grid mismatch: {img_h}x{img_w}={img_h * img_w} "
                f"vs noise token count {self.noise.shape[1]}"
            )

        rope = self.transformer.get_rope(img_h, img_w, txt_len)

        sigmas = calculate_sigmas(model_type, self.scheduler_name, steps, self.width, self.height)
        solver_state: dict[str, Any] = {}

        teacache_state: TeaCacheState | None = None
        if self.teacache:
            teacache_state = TeaCacheState(
                threshold=self.teacache_threshold, warmup_steps=5, final_steps=3,
            )

        mx.reset_peak_memory()
        step_times: list[float] = []
        t_sampling_start = time.perf_counter()

        for t in range(steps):
            step_start = time.perf_counter()
            sigma_t = sigmas[t]
            sigma_next = sigmas[t + 1] if t + 1 < len(sigmas) else 0.0

            # Update LoRA schedule
            if self.lora_schedule is not None:
                self.lora_schedule["step"] = t
                self.transformer = self._update_lora_schedule(
                    self.transformer, self.config, self.lora_schedule, t, steps
                )

            current_output = self.transformer.predict(
                img=self.noise,
                txt=context,
                timestep=sigma_t,
                guidance=effective_guidance,
                rope=rope,
            )
            mx.eval(current_output)

            if teacache_state is not None:
                reused, reason = self._teacache_check(teacache_state, current_output, t, steps)
                if reused:
                    noise_pred = (
                        teacache_state.last_feature
                        if teacache_state.last_feature is not None
                        else current_output
                    )
                    skip_step = True
                else:
                    skip_step = False
                    noise_pred = current_output
            else:
                noise_pred = current_output
                skip_step = False

            # denoised = x - noise_pred*sigma (see solvers.py module docstring)
            denoised = self.noise - noise_pred * sigma_t

            def _model_call(x_at, sigma_at):
                out = self.transformer.predict(
                    img=x_at, txt=context, timestep=sigma_at, guidance=effective_guidance, rope=rope,
                )
                mx.eval(out)
                return x_at - out * sigma_at

            self.noise, solver_state = solvers.step(
                self.sampler_name,
                x=self.noise,
                sigma=sigma_t,
                sigma_next=sigma_next,
                denoised=denoised,
                state=solver_state,
                seed=seed,
                step_index=t,
                is_flow_matching=self._is_flow_matching,
                model_call=_model_call,
            )
            mx.eval(self.noise)

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            if (t + 1) % 5 == 0 or t == 0:
                cache_tag = " [cache]" if skip_step else ""
                print(f"[ASDX] Flux2 Step {t + 1}/{steps} - {step_time:.3f}s{cache_tag}")

        total_time = time.perf_counter() - t_sampling_start

        if self.low_memory_mode:
            # self.transformer is the SAME object as loader.py's
            # _MODEL_CACHE entry -- dropping only this local reference never
            # freed anything (and _SamplerCore is local to sample() anyway,
            # released on return regardless). Evict the cache entry itself
            # so the memory is actually returned; the next generation will
            # reload the checkpoint from disk instead of hitting the cache.
            from ..loader import clear_model_cache
            clear_model_cache()
            self.transformer = None
            print("[ASDX] Low memory: model cache evicted, next load will read from disk")

        out_latent = bridge.mlx_to_comfy_latent_flux2(
            self.noise, self.height, self.width, {"samples": self.noise}
        )
        out_latent["sdmlx_model_type"] = model_type
        out_latent["sdmlx_model_name"] = "unknown"

        mem = bridge.collect_mlx_memory()
        avg_step = sum(step_times) / len(step_times) if step_times else 0
        accel = (
            f"TeaCache({teacache_state.hits}h/{teacache_state.real_steps}real)"
            if teacache_state is not None else "standard"
        )

        print(
            f"[ASDX] Flux2 Sampling complete: {total_time:.1f}s total, "
            f"{avg_step:.3f}s/step, {mem['peak_gb']:.1f}GB peak, {accel}, "
            f"guidance={effective_guidance:.1f}"
        )

        if self.memory_shape is not None:
            record_observation(self.memory_shape, mx.get_peak_memory())

        bridge.clear_mlx_cache()

        return out_latent
