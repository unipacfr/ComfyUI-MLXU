"""ASDX_Krea2Edit — Krea2 Identity Edit as a focused, composable node.

Extracts the Identity Edit feature out of ``ASDX_MLXSampler`` (a god-node)
into its own upstream node that:

1. Loads and applies the Krea2 Identity Edit LoRA, reusing ``ASDX_LoraLoader``'s
   helpers (never duplicating them).
2. Fits the source IMAGE in pixel space — contain + /16 floor + bicubic, or a
   minimal center-crop when the AR nearly matches — then VAE-encodes, whitens,
   and packs it. This is the geometry the v1_2 LoRA was trained with and what
   ``comfyui-krea2edit``'s ``fit`` mode does; see canon "Krea2 Identity Edit
   source is fitted in pixel space, never in latent space".
3. Writes the prepared source into the model dict as ``identity_edit``, which
   ``ASDX_MLXSampler`` reads in priority over its legacy ``source_latent`` /
   ``source_image`` inputs — the same model-dict pattern as ``lora_schedule``.

The sampler's legacy inputs stay in place as a documented fallback, so every
saved workflow keeps working unchanged while new workflows get this small,
focused node.

The pixel fit runs in torch (bicubic) on purpose: it sits in the same torch
segment as the VAE encode (a canon-recognized PyTorch-MPS exception), so it is
not a needless host round-trip, and it stays byte-identical to the reference's
``F.interpolate(..., mode="bicubic")`` and to the sampler's own verified pixel
path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from comfy_api.latest import io

from .lora import ASDX_LoraLoader, _check_lora_compatibility, base_lora_scale
from .vae import ASDX_VAEEncode
from .native.config import process_wan21_latent_in

# Identity Edit LoRA files are recognized by NAME — the only available
# discriminator (the spec verified the key structure does not discriminate, and
# most identity-edit files carry no metadata). Case-insensitive; reused for both
# the ``lora_name`` sorting and the double-LoRA guard.
_IDENTITY_EDIT_RE = re.compile(r"identity[_-]?edit", re.IGNORECASE)


class ASDX_Krea2Edit(io.ComfyNode):
    """Prepare a Krea2 Identity Edit source and apply its LoRA.

    Emits the model dict with an ``identity_edit`` entry the sampler consumes
    in priority over its legacy ``source_latent`` / ``source_image`` inputs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_Krea2Edit",
            display_name="🍏 ASDX Krea2 Identity Edit",
            category="ASDX/Conditioning",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Image.Input("image"),
                io.Vae.Input("vae"),
                io.Combo.Input("lora_name", options=cls._get_loras()),
                io.Float.Input(
                    "lora_strength", default=1.0, min=-10.0, max=10.0, step=0.01
                ),
                io.Float.Input(
                    "ref_boost",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    tooltip=(
                        "Attention bias pulling the target toward the source. "
                        "1.0 = off (the ASDX default everywhere). The reference "
                        "workflow ships 4.0 for much stronger face/body likeness "
                        "and more reliable edits — raise it to opt in."
                    ),
                ),
                io.Latent.Input("target_latent", optional=True),
            ],
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
            ],
        )

    @staticmethod
    def _get_loras() -> list[str]:
        """All LoRA files, with identity-edit files (name match) sorted to the
        top — most recent first, so the newest Identity Edit LoRA is the default
        and a future update is picked up automatically. Non-identity-edit files
        (e.g. ``krea2_style_reference``, which rides the same source-token
        mechanism) stay listed and usable, never masked."""
        names = ASDX_LoraLoader._get_loras()
        identity: list[tuple[float, str]] = []
        others: list[str] = []
        for name in names:
            if _IDENTITY_EDIT_RE.search(name):
                mtime = _lora_mtime(name)
                identity.append((mtime, name))
            else:
                others.append(name)
        # Most recent first (mtime desc), name as a stable tiebreak.
        identity.sort(key=lambda t: (-t[0], t[1]))
        return [name for _, name in identity] + others

    @classmethod
    def execute(
        cls,
        model: dict,
        image: torch.Tensor,
        vae: Any,
        lora_name: str,
        lora_strength: float,
        ref_boost: float,
        target_latent: dict | None = None,
    ) -> io.NodeOutput:
        transformer = model["transformer"]

        # 1. Refuse a non-Krea2 model with an explicit message (never a silent
        #    no-op) — same principle as _prepare_krea2_identity_edit's raise.
        family = model["capability"].family
        if family != "krea2":
            raise RuntimeError(
                f"ASDX_Krea2Edit: requires a Krea2 model, got family "
                f"'{family}'. Identity Edit is Krea2-only."
            )

        # 2. Load + apply the LoRA (reusing ASDX_LoraLoader's helpers).
        if lora_name:
            # Double-LoRA guard: refuse to stack a second copy of an Identity
            # Edit LoRA that is already attached (e.g. by a separate
            # ASDX_LoraLoader still in the graph). Two copies of the SAME
            # adapter silently double its effect; two DIFFERENT adapters
            # (identity-edit + style-reference) are a legitimate stack and are
            # allowed through.
            if _IDENTITY_EDIT_RE.search(lora_name):
                applied = getattr(transformer, "_applied_lora_names", [])
                existing = [n for n in applied if _IDENTITY_EDIT_RE.search(n)]
                if existing:
                    raise RuntimeError(
                        "ASDX_Krea2Edit: the transformer already has an Identity "
                        f"Edit LoRA attached ({', '.join(existing)}). This node "
                        f"would apply '{lora_name}' — a second copy of the same "
                        "adapter. Remove the separate ASDX_LoraLoader that "
                        "applies it, or select a different LoRA here."
                    )
            lora_path = ASDX_LoraLoader._resolve_lora_path(lora_name)
            _check_lora_compatibility(lora_path, model)
            lora = ASDX_LoraLoader._load_lora_file(lora_path)
            lora.scale = base_lora_scale(lora.alpha, lora.rank) * lora_strength
            transformer = ASDX_LoraLoader._apply_lora_to_transformer(
                transformer, lora, model["config"]
            )
            # Drop the raw factor arrays now that they are attached, so the
            # (potentially large) A/B buffers are reclaimable — mirrors
            # ASDX_LoraLoader.execute's own memory hygiene.
            lora.deltas = {}
            lora.factors = {}

        # 3. Prepare the source (pixel fit + VAE encode + whiten + pack).
        identity_edit = cls._prepare_source(
            image, vae, target_latent, ref_boost, model["config"]
        )

        # 4. Shallow copy — never mutate the input dict (it may be the cached
        #    model shared across executions), exactly like ASDX_LoraLoader.
        new_model = {**model, "transformer": transformer, "identity_edit": identity_edit}
        return io.NodeOutput(new_model)

    @classmethod
    def _prepare_source(
        cls,
        image: torch.Tensor,
        vae: Any,
        target_latent: dict | None,
        ref_boost: float,
        config: Any,
    ) -> dict:
        """Fit the source image in pixel space, VAE-encode, whiten, and pack it.

        Returns the ``identity_edit`` dict the sampler consumes. Two shapes:

        - ``target_latent`` wired → the source is fitted to the target grid in
          pixel space (canon geometry), encoded, whitened, and PACKED.
          ``fitted=True``; ``source_latent`` is a ``[B, N, 64]`` token tensor.
        - ``target_latent`` absent → the node cannot know the target grid, so it
          VAE-encodes the source as-is (no fit) and leaves the fit to the
          sampler's ``_fit_source_latent`` safety net. ``fitted=False``;
          ``source_latent`` is a raw ``[B, C, H, W]`` latent (unwhitened,
          unpacked) for the sampler's existing latent path to finish.
        """
        img = image  # [B, H, W, C] in [0, 1]
        img4 = img.movedim(-1, 1).float()  # [B, C, H, W]
        ih, iw = img4.shape[-2:]

        if target_latent is not None:
            samples = target_latent.get("samples")
            if samples is None:
                raise RuntimeError(
                    "ASDX_Krea2Edit: target_latent has no 'samples'."
                )
            px_h, px_w = samples.shape[-2] * 8, samples.shape[-1] * 8
            return cls._fit_encode_pack(
                img4, ih, iw, px_h, px_w, vae, ref_boost, config, fitted=True
            )

        # No target grid: encode the source at its own size, leave the fit to
        # the sampler's latent-space safety net.
        fitted_hwc = img4.movedim(1, -1)[..., :3].clamp(0, 1)
        latent = ASDX_VAEEncode._fallback_encode(fitted_hwc, vae)
        latent_np = latent.detach().cpu().float().numpy().astype(np.float32, copy=False)
        latent_mlx = mx.array(latent_np).astype(config.mlx_dtype)
        mx.eval(latent_mlx)
        _, _, src_lat_h, src_lat_w = latent_np.shape
        print(
            f"[ASDX] Krea2Edit: no target_latent wired — source encoded at its "
            f"own size ({ih}x{iw} px); the sampler will fit it to the target grid."
        )
        return {
            "source_latent": latent_mlx,
            "source_grid": (src_lat_h // 2, src_lat_w // 2),
            "src_offset": (0, 0),
            "ref_boost": ref_boost,
            "fitted": False,
        }

    @staticmethod
    def _fit_encode_pack(
        img4: torch.Tensor,
        ih: int,
        iw: int,
        px_h: int,
        px_w: int,
        vae: Any,
        ref_boost: float,
        config: Any,
        fitted: bool,
    ) -> dict:
        """Pixel-space fit (canon geometry) + VAE encode + whiten + pack.

        An exact port of ``_SamplerCore._prepare_krea2_identity_edit_pixels``:
        contain + /16 floor + bicubic (or a minimal center-crop when the AR
        nearly matches), then VAE-encode, whiten into the model's internal
        Wan21 latent space, and pack to ``[B, N, 64]``.
        """
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
            # `px_h`/`px_w` are only guaranteed divisible by 8 (the target
            # latent's own grid, e.g. from a Resolution node's
            # divisible_by=8) -- not by 16. The 2x2 patchify pack below needs
            # an EVEN latent grid, i.e. a /16 pixel grid; snap down like the
            # AR-mismatch branch does, or an odd target latent dimension
            # (e.g. width 153) crashes the reshape below. `src_offset`
            # (computed further down) already centers a smaller-than-target
            # grid, so this is safe.
            nh, nw = max(16, px_h // 16 * 16), max(16, px_w // 16 * 16)
        else:
            # Genuine AR mismatch: snap to /16 (NOT /8) to stay byte-identical
            # to the trainer's _fit_prep, capped at the target's /16 floor.
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))

        # NB: named ``resized`` (not ``fitted``) so it does not shadow the
        # ``fitted`` boolean param that the return dict reports.
        resized = torch.nn.functional.interpolate(
            img4, size=(nh, nw), mode="bicubic", antialias=True
        )
        fitted_hwc = resized.movedim(1, -1)[..., :3].clamp(0, 1)

        # VAE-encode the fitted image (real ComfyUI VAE, never MLXVAE — canon).
        # _fallback_encode handles the 5D Wan21 return and the tiled MPS retry.
        latent = ASDX_VAEEncode._fallback_encode(fitted_hwc, vae)
        latent_np = latent.detach().cpu().float().numpy().astype(np.float32, copy=False)

        # Whiten into the model's internal Wan21 latent space.
        latent_mlx = mx.array(latent_np).astype(mx.float32)
        latent_mlx = process_wan21_latent_in(latent_mlx)
        mx.eval(latent_mlx)
        latent_np = np.array(latent_mlx, dtype=np.float32)

        # [B, C, H, W] -> pack to [B, NH*NW, 64]
        batch, channels, src_lat_h, src_lat_w = latent_np.shape
        packed = latent_np.reshape(batch, channels, src_lat_h // 2, 2, src_lat_w // 2, 2)
        packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
        packed = packed.reshape(batch, (src_lat_h // 2) * (src_lat_w // 2), channels * 4)

        source_packed = mx.array(packed).astype(config.mlx_dtype)
        mx.eval(source_packed)

        src_h = src_lat_h // 2
        src_w = src_lat_w // 2
        img_h = (px_h // 8) // 2
        img_w = (px_w // 8) // 2
        src_offset = (
            max(0, (img_h - src_h) // 2),
            max(0, (img_w - src_w) // 2),
        )
        print(
            f"[ASDX] Krea2Edit: source {ih}x{iw} -> fitted {nh}x{nw} px, latent "
            f"{src_lat_h}x{src_lat_w}, grid {src_h}x{src_w} centered offset "
            f"{src_offset}"
        )
        return {
            "source_latent": source_packed,
            "source_grid": (src_h, src_w),
            "src_offset": src_offset,
            "ref_boost": ref_boost,
            "fitted": fitted,
        }


def _lora_mtime(name: str) -> float:
    """Best-effort mtime of a LoRA file, for 'most recent first' sorting.

    Returns 0.0 when the file cannot be resolved (e.g. in a test without a real
    loras folder) — such files sort last among the identity-edit group.
    """
    try:
        path = ASDX_LoraLoader._resolve_lora_path(name)
        if path.exists():
            return path.stat().st_mtime
    except Exception:
        pass
    return 0.0


NODE_LIST = [ASDX_Krea2Edit]
