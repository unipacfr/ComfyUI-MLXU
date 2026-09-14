"""Capability profiles for mflux-AnyModel-inspired model dispatch.

Each diffusion model family has a CapabilityProfile that declares which
parameters its generate_image() / predict() method accepts. The sampler
reads this profile to filter parameters, warn about dropped ones, and
block invalid ones — mirroring the CapabilityProfile pattern from
ComfyUI-mflux-AnyModel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── CapabilityProfile ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CapabilityProfile:
    """Describes what parameters a model family supports.

    Mirrors the CapabilityProfile pattern from ComfyUI-mflux-AnyModel:
    each model family declares which parameters its inference method
    accepts, which should be silently dropped, and which must be blocked.
    """

    family: str  # e.g. "flux1", "flux2", "flux1_fill"
    name: str  # Human-readable name
    generate_params: dict[str, str] = field(default_factory=dict)
    """param_name -> type_hint ("float", "int", "image", "mask", "latent", "string")."""

    requires: frozenset[str] = field(default_factory=frozenset)
    """Parameters that must be provided (and are not None)."""

    drop_with_warning: frozenset[str] = frozenset()
    """Params silently dropped with a log warning."""

    hard_block: frozenset[str] = frozenset()
    """Params that cause an error if forwarded."""

    latent_channels: int = 16
    supports_img2img: bool = False
    supports_inpainting: bool = False
    supports_depth: bool = False
    supports_controlnet: bool = True
    supports_mask_preserve: bool = False


# ── Predefined profiles ───────────────────────────────────────────────

CAPABILITY_PROFILES: dict[str, CapabilityProfile] = {
    "flux1_dev": CapabilityProfile(
        family="flux1",
        name="FLUX.1-dev",
        generate_params={
            "guidance": "float",
            "width": "int",
            "height": "int",
            "steps": "int",
        },
        requires=frozenset(),
        latent_channels=16,
        supports_controlnet=True,
    ),
    "flux1_schnell": CapabilityProfile(
        family="flux1",
        name="FLUX.1-schnell",
        generate_params={
            "width": "int",
            "height": "int",
            "steps": "int",
        },
        requires=frozenset(),
        # schnell doesn't use guidance
        hard_block=frozenset({"guidance"}),
        latent_channels=16,
        supports_controlnet=False,
    ),
    "flux1_fill": CapabilityProfile(
        family="flux1_fill",
        name="FLUX.1-fill",
        generate_params={
            "guidance": "float",
            "width": "int",
            "height": "int",
            "steps": "int",
            "image": "image",
            "mask": "mask",
            "mask_blur": "int",
            "mask_padding": "int",
            "strength": "float",
            "noise_aug": "float",
        },
        requires=frozenset(),
        supports_img2img=True,
        supports_inpainting=True,
        supports_mask_preserve=True,
        latent_channels=16,
        supports_controlnet=False,
    ),
    "flux1_depth": CapabilityProfile(
        family="flux1_depth",
        name="FLUX.1-depth",
        generate_params={
            "guidance": "float",
            "width": "int",
            "height": "int",
            "steps": "int",
            "depth_image": "image",
            "depth_strength": "float",
        },
        requires=frozenset(),
        supports_depth=True,
        supports_controlnet=False,
        # latent_channels=16 is the OUTPUT latent (what ASDX_VAEDecode expects
        # back), not the transformer's img_in input width. The transformer's
        # in_channels is 32 unpacked / 128 packed (16 noise + 16 depth-latent
        # channels), detected dynamically from the checkpoint's own
        # img_in.weight shape by native/weight_map.py::detect_flux_in_channels
        # -- these are two different numbers; do not "fix" one to match the
        # other.
        latent_channels=16,
    ),
    # Covers both Klein (Qwen3 text encoder, no guidance embed on this
    # machine's checkpoint) and Flux2-D (Mistral3, HAS a guidance embed) —
    # `guidance` is left as an optional (not required, not hard-blocked)
    # param because whether it's actually used is a per-checkpoint runtime
    # fact (`Flux2Config.guidance_embed`, detected from the checkpoint by
    # `detect_flux2_config`), not something the capability layer can know
    # ahead of load time; Flux2Transformer silently ignores `guidance` when
    # its checkpoint has no `guidance_in`.
    "flux2_klein": CapabilityProfile(
        family="flux2",
        name="Flux2/Klein",
        generate_params={
            "guidance": "float",
            "width": "int",
            "height": "int",
            "steps": "int",
        },
        requires=frozenset(),
        latent_channels=128,
        # No ControlNet-Flux2 architecture ported yet (Phase E, optional).
        supports_controlnet=False,
    ),
    # ── Krea2 ──────────────────────────────────────────────────────
    "krea2_base": CapabilityProfile(
        family="krea2",
        name="Krea2-Base",
        generate_params={
            "guidance": "float",
            "width": "int",
            "height": "int",
            "steps": "int",
            "source_latent": "latent",
            "ref_boost": "float",
        },
        requires=frozenset(),
        latent_channels=16,
        supports_controlnet=False,
        supports_inpainting=True,
        supports_depth=True,
    ),
    "krea2_turbo": CapabilityProfile(
        family="krea2",
        name="Krea2-Turbo",
        generate_params={
            "width": "int",
            "height": "int",
            "steps": "int",
            "source_latent": "latent",
            "ref_boost": "float",
        },
        requires=frozenset(),
        # Turbo uses CFG=1, guidance is blocked
        hard_block=frozenset({"guidance"}),
        latent_channels=16,
        supports_controlnet=False,
        supports_inpainting=True,
        supports_depth=True,
    ),
    # ── SDXL (also covers Illustrious/Pony/NoobAI — same UNet, different weights) ──
    "sdxl_base": CapabilityProfile(
        family="sdxl",
        name="SDXL",
        generate_params={
            "cfg_scale": "float",
            "negative": "string",
            "width": "int",
            "height": "int",
            "steps": "int",
        },
        # SDXL needs true classifier-free guidance (two-pass cond/uncond),
        # unlike FLUX/Krea2's single-pass guidance-embedding — a negative
        # prompt is not optional here.
        requires=frozenset({"negative"}),
        # SDXL has no guidance-embedding mechanism (unlike FLUX-dev/Krea2-raw)
        hard_block=frozenset({"guidance"}),
        latent_channels=4,
        supports_controlnet=False,
    ),
    # ── Z-Image (NextDiT/Lumina2 family) ──────────────────────────────
    "zimage_base": CapabilityProfile(
        family="zimage",
        name="Z-Image",
        generate_params={
            "width": "int",
            "height": "int",
            "steps": "int",
        },
        requires=frozenset(),
        # NextDiT has no guidance-embedding mechanism and no CFG is wired
        # in _run_zimage() yet — see sampler/core.py::_run_zimage docstring.
        hard_block=frozenset({"guidance"}),
        latent_channels=16,
        supports_controlnet=False,
    ),
    "zimage_turbo": CapabilityProfile(
        family="zimage",
        name="Z-Image-Turbo",
        generate_params={
            "width": "int",
            "height": "int",
            "steps": "int",
        },
        requires=frozenset(),
        hard_block=frozenset({"guidance"}),
        latent_channels=16,
        supports_controlnet=False,
    ),
}


# ── Alias dispatch ─────────────────────────────────────────────────────

# Maps filename patterns to capability profile keys.
# Order matters: more specific patterns first.
_CAPABILITY_DISPATCH: list[tuple[str, str]] = [
    # SDXL (and same-architecture finetunes: Illustrious, Pony, NoobAI)
    ("sdxl", "sdxl_base"),
    ("illustrious", "sdxl_base"),
    ("pony", "sdxl_base"),
    ("noobai", "sdxl_base"),
    # Z-Image
    ("zimage_turbo", "zimage_turbo"),
    ("z_image_turbo", "zimage_turbo"),
    ("zimage", "zimage_base"),
    ("z_image", "zimage_base"),
    ("z-image", "zimage_base"),
    # FLUX.1 Fill
    ("fill", "flux1_fill"),
    # FLUX.1 Depth
    ("depth", "flux1_depth"),
    # FLUX.2
    ("klein", "flux2_klein"),
    # FLUX.1 Schnell
    ("schnell", "flux1_schnell"),
    # FLUX.1 Dev (default)
    ("dev", "flux1_dev"),
    ("kontext", "flux1_dev"),
]


def _resolve_capability(model_name: str) -> CapabilityProfile:
    """Resolve a model name to its CapabilityProfile.

    Uses the alias dispatch table: checks filename patterns in priority
    order, falling back to flux1_dev.
    """
    name_lower = model_name.lower()

    for pattern, profile_key in _CAPABILITY_DISPATCH:
        if pattern in name_lower:
            profile = CAPABILITY_PROFILES.get(profile_key)
            if profile is not None:
                return profile

    # Default fallback
    return CAPABILITY_PROFILES["flux1_dev"]


def _resolve_capability_from_path(path: str | Path) -> CapabilityProfile:
    """Resolve capability from a file path (extracts basename)."""
    return _resolve_capability(Path(path).stem)


# ── Parameter filtering ────────────────────────────────────────────────


def filter_params_for_model(
    profile: CapabilityProfile,
    candidate_params: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Filter candidate params according to a CapabilityProfile.

    Returns (valid_params, dropped_warnings).

    - Params in profile.generate_params are forwarded as-is (if not None).
    - Params in hard_block raise ValueError.
    - Params in drop_with_warning are dropped with a log message.
    - Unknown params are silently ignored.

    Raises ValueError if any required param is missing (None or absent).
    """
    valid: dict[str, Any] = {}
    dropped: list[str] = []

    for name, value in candidate_params.items():
        if value is None:
            continue

        if name in profile.generate_params:
            valid[name] = value
        elif name in profile.hard_block:
            raise ValueError(
                f"Parameter '{name}' is not supported by {profile.name}. "
                f"Remove it from the workflow."
            )
        elif name in profile.drop_with_warning:
            dropped.append(name)
            logger.warning(
                "ASDX: Dropped unsupported param '%s' for %s", name, profile.name
            )
        # else: silently ignore unknown params

    # Check required
    missing = profile.requires - set(valid.keys())
    if missing:
        raise ValueError(
            f"Missing required params for {profile.name}: {sorted(missing)}"
        )

    return valid, dropped


def require_divisible_dims(width: int, height: int, divisor: int, family: str) -> None:
    """Raise if width/height is not an exact multiple of `divisor`, instead
    of letting a later `//` floor-division silently truncate to a smaller
    grid than the caller asked for. `sampler/core.py` derives its patch grid
    with `(self.height // 8) // 2` (FLUX.1/Krea2/SDXL) or `self.height //
    bridge.FLUX2_VAE_DOWNSCALE` (Flux2) -- a non-multiple height/width would
    otherwise generate at a silently different resolution than requested,
    the same class of "silent substitution" bug this project's checkpoint-
    loading gates (loader.py/native/__init__.py) exist to prevent, just for
    dimensions instead of weights.
    """
    if width % divisor != 0 or height % divisor != 0:
        raise ValueError(
            f"ASDX: {family} requires width/height divisible by {divisor} "
            f"(got {width}x{height}). Silent floor-division truncation would "
            f"produce a different resolution than requested -- adjust the "
            f"dimensions instead."
        )
