"""Metadata-collector extractors for ASDX nodes.

The ``Save Image (LoraManager)`` node does not read metadata from the image;
it calls a runtime collector (``comfyui-lora-manager``) that records each node's
inputs/outputs as the prompt executes. That collector recognises nodes by their
Python class name in a registry (``NODE_EXTRACTORS``). ASDX's nodes are not in
that registry, and the collector's generic fallback misses them too (it only
handles ``MODEL``/``CONDITIONING`` return types in uppercase, while ASDX uses
lowercase custom types like ``asdx_model``/``mlx_conditioning``, and it does not
handle ``LATENT`` at all). Result: model name, seed, size and prompt are never
captured for an all-ASDX workflow.

This module closes that gap from the ASDX side (the right dependency direction:
ASDX adapts to the collector, not vice-versa). It defines pure extractor classes
that write into the collector's metadata dict using the same category layout, and
a lazy ``register()`` that inserts them into the *live* registry at execution
time.

Why lazy and why ``sys.modules``:
  * ComfyUI loads custom nodes in filesystem order with no "all loaded" hook, and
    in this environment ASDX loads *before* ``comfyui-lora-manager``. Registering
    at import time would find no registry yet and silently no-op, so registration
    is deferred to first node execution (by then every custom node is imported).
  * The registry must be the exact dict object the collector consults at runtime.
    Importing ``py.metadata_collector...`` fresh could create a second, separate
    module copy (the classic double-import), so we locate the already-loaded
    module in ``sys.modules`` instead of importing it.

The extractors themselves have no dependency on lora-manager, so they are unit
testable in isolation. Only ``register()`` touches the collector's internals, and
it is fully guarded: if lora-manager is absent or its layout changed, ASDX simply
runs without enriched metadata (no crash).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Collector category keys ───────────────────────────────────────────
# Mirrors comfyui-lora-manager's metadata_collector.constants. Kept local (not
# imported) so the extractors stay decoupled from the collector's import path and
# testable without it installed. These string values are stable contract points.
MODELS = "models"
PROMPTS = "prompts"
SAMPLING = "sampling"
SIZE = "size"
IS_SAMPLER = "is_sampler"

# Filenames that look like real model checkpoints (avoids capturing unrelated
# string fields), matching the collector's own convention.
_MODEL_EXTENSIONS = (
    ".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft", ".gguf",
)


def _looks_like_model_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    name = value.strip()
    return bool(name) and name.lower().endswith(_MODEL_EXTENSIONS)


def _store_model(metadata: dict, node_id: str, model_name: str) -> None:
    metadata.setdefault(MODELS, {})
    metadata[MODELS][node_id] = {
        "name": model_name,
        "type": "checkpoint",
        "node_id": node_id,
    }


# ── Extractors ────────────────────────────────────────────────────────
# Each matches the collector's non-generic dispatch signature:
#   extract(node_id, inputs, outputs, metadata)
# ``inputs`` is the node's processed input dict (wired values + widgets).


class ASDXSamplerExtractor:
    """ASDX_MLXSampler -> sampling params (seed/steps/sampler/scheduler/cfg) + size.

    FLUX-style ``guidance`` is the CFG-equivalent, so it is mapped onto the
    canonical ``cfg`` key the processor reads for ``cfg_scale`` (and kept as
    ``guidance`` too, which the processor also scans for).

    Size is read from the wired ``latent_image`` and stored under the *sampler's*
    node_id -- that is where the processor looks (``metadata[SIZE][sampler_id]``),
    so a size recorded under an EmptyLatent node would be ignored. The latent dict
    carries ``downscale_ratio_spacial`` (set by ASDX_EmptyLatent), which resolves
    the 8x-vs-16x VAE ambiguity exactly; we fall back to 8 when it is absent.
    """

    @staticmethod
    def extract(node_id: str, inputs: dict, outputs: Any, metadata: dict) -> None:
        if not inputs:
            return

        params: dict[str, Any] = {}
        for key in ("seed", "steps", "sampler_name"):
            if inputs.get(key) is not None:
                params[key] = inputs[key]

        guidance = inputs.get("guidance")
        if guidance is not None:
            params["cfg"] = guidance
            params["guidance"] = guidance

        scheduler = inputs.get("scheduler_name")
        if scheduler is not None:
            params["scheduler"] = scheduler

        metadata.setdefault(SAMPLING, {})
        metadata[SAMPLING][node_id] = {
            "parameters": params,
            "node_id": node_id,
            IS_SAMPLER: True,
        }

        ASDXSamplerExtractor._extract_size(node_id, inputs.get("latent_image"), metadata)

    @staticmethod
    def _extract_size(node_id: str, latent: Any, metadata: dict) -> None:
        if not isinstance(latent, dict):
            return
        samples = latent.get("samples")
        shape = getattr(samples, "shape", None)
        if not shape or len(shape) < 4:
            return
        downscale = latent.get("downscale_ratio_spacial", 8) or 8
        height = int(shape[2]) * int(downscale)
        width = int(shape[3]) * int(downscale)
        metadata.setdefault(SIZE, {})
        metadata[SIZE][node_id] = {
            "width": width,
            "height": height,
            "node_id": node_id,
        }


class ASDXModelLoaderExtractor:
    """ASDX_DiffusionLoader -> checkpoint name from ``model_name``."""

    @staticmethod
    def extract(node_id: str, inputs: dict, outputs: Any, metadata: dict) -> None:
        if not inputs:
            return
        name = inputs.get("model_name")
        if _looks_like_model_name(name):
            _store_model(metadata, node_id, name.strip())


class ASDXCheckpointLoaderExtractor:
    """ASDX_CheckpointLoader -> checkpoint name from ``ckpt_name``."""

    @staticmethod
    def extract(node_id: str, inputs: dict, outputs: Any, metadata: dict) -> None:
        if not inputs:
            return
        name = inputs.get("ckpt_name")
        if _looks_like_model_name(name):
            _store_model(metadata, node_id, name.strip())


class ASDXClipTextEncodeExtractor:
    """ASDX_CLIPTextEncode -> prompt text (+ guidance into sampling params).

    Uses ``text`` when present, otherwise the FLUX-mode ``t5xxl`` field.
    """

    @staticmethod
    def extract(node_id: str, inputs: dict, outputs: Any, metadata: dict) -> None:
        if not inputs:
            return

        text = (inputs.get("text") or "").strip()
        if not text:
            text = (inputs.get("t5xxl") or "").strip()
        if text:
            metadata.setdefault(PROMPTS, {})
            metadata[PROMPTS][node_id] = {"text": text, "node_id": node_id}

        guidance = inputs.get("guidance")
        if guidance is not None:
            metadata.setdefault(SAMPLING, {})
            entry = metadata[SAMPLING].setdefault(
                node_id, {"parameters": {}, "node_id": node_id}
            )
            entry.setdefault("parameters", {})["guidance"] = guidance


# Class name (as the collector sees it via obj.__class__.__name__) -> extractor.
ASDX_EXTRACTORS = {
    "ASDX_MLXSampler": ASDXSamplerExtractor,
    "ASDX_DiffusionLoader": ASDXModelLoaderExtractor,
    "ASDX_CheckpointLoader": ASDXCheckpointLoaderExtractor,
    "ASDX_CLIPTextEncode": ASDXClipTextEncodeExtractor,
}


# ── Registration into the live collector registry ─────────────────────

_registered = False
_log_done = False


def _find_live_node_extractors_module() -> Optional[Any]:
    """Return the already-loaded lora-manager node_extractors module, or None.

    Matches both import styles the collector uses (package-relative
    ``<pkg>.py.metadata_collector.node_extractors`` and the bare fallback
    ``py.metadata_collector.node_extractors``). When both are present we prefer
    the package-qualified copy, since that is the one the collector's hook and
    registry actually use at runtime.
    """
    candidates = []
    for name, mod in list(sys.modules.items()):
        if not (
            name == "py.metadata_collector.node_extractors"
            or name.endswith(".py.metadata_collector.node_extractors")
        ):
            continue
        registry = getattr(mod, "NODE_EXTRACTORS", None)
        if isinstance(registry, dict):
            candidates.append((name, mod))

    if not candidates:
        return None

    def _rank(item):
        # 0 = package-qualified (top-level != "py"), 1 = bare "py.*" copy.
        return 0 if item[0].split(".")[0] != "py" else 1

    candidates.sort(key=_rank)
    return candidates[0][1]


def register() -> bool:
    """Insert ASDX extractors into the live collector registry.

    Idempotent and safe to call repeatedly (cheap flag check after success).
    Returns True once registration has taken effect, False if the collector is
    not (yet) available. Never raises: a missing or changed collector must not
    break ASDX node execution.
    """
    global _registered, _log_done

    if _registered:
        return True

    try:
        mod = _find_live_node_extractors_module()
        if mod is None:
            return False

        registry = mod.NODE_EXTRACTORS

        # setdefault so a user's manual override of an ASDX entry is preserved.
        for class_name, extractor in ASDX_EXTRACTORS.items():
            registry.setdefault(class_name, extractor)

        # ASDX_LoraLoader uses the same field names as ComfyUI's LoraLoader
        # (lora_name / strength_model), so reuse the collector's own extractor.
        lora_extractor = getattr(mod, "LoraLoaderExtractor", None)
        if lora_extractor is not None:
            registry.setdefault("ASDX_LoraLoader", lora_extractor)

        _registered = True
        if not _log_done:
            logger.info(
                "[ASDX] Registered %d metadata extractors with "
                "comfyui-lora-manager", len(ASDX_EXTRACTORS) + 1
            )
            _log_done = True
        return True

    except Exception as e:  # pragma: no cover - defensive; must never break exec
        if not _log_done:
            logger.warning("[ASDX] Metadata extractor registration skipped: %s", e)
            _log_done = True
        return False


def ensure_registered() -> None:
    """Best-effort registration hook for node execute() entry points.

    Called at the start of each metadata-relevant ASDX node's execute so that,
    regardless of custom-node load order, the registry is populated before any
    later node in the prompt (and every subsequent prompt) is recorded.
    """
    register()
