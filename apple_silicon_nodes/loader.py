"""
Diffusion Model Loader
======================
Loads FLUX.1 checkpoints into MLX-native transformers.

Features:
  - Automatic checkpoint type detection (dev vs schnell)
  - Quantization support: dense, FP8, GGUF (via sdmlx native)
  - Model weight caching to avoid reload
  - Memory profiling on load
"""

from __future__ import annotations

import os
import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import torch

from comfy_api.latest import io

from . import bridge
from . import metadata_extractors
from .capability import CAPABILITY_PROFILES, CapabilityProfile, _resolve_capability_from_path
from .memory_calibration import LoadShape, check_fits_or_warn
from .native import FluxConfig, FluxTransformer, load_transformer
from .native.safetensors_header import read_safetensors_header
from .native.weight_format import Unrecognized, classify_quant_format


# ── Globals ───────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    value: dict[str, Any]
    last_used: float  # time.monotonic()


class _InactivityCache(MutableMapping):
    """Dict-like cache that evicts entries idle for longer than
    `idle_timeout_s`, checked at access time (no background thread --
    ComfyUI gives nodes no natural hook to run one, and this project has no
    other async infra to hang it off). Implements the standard mapping
    protocol so every existing `_MODEL_CACHE[...]`/`in`/`len`/`clear()`
    call site keeps working unchanged; only the class definition and
    construction below change.
    """

    def __init__(self, idle_timeout_s: float):
        self._store: dict[str, _CacheEntry] = {}
        self.idle_timeout_s = idle_timeout_s

    def _evict_idle(self) -> None:
        now = time.monotonic()
        stale = [k for k, e in self._store.items() if now - e.last_used > self.idle_timeout_s]
        for k in stale:
            idle_for = now - self._store[k].last_used
            print(f"[ASDX] Evicting idle cache entry '{k}' (idle {idle_for:.0f}s)")
            del self._store[k]
        if stale:
            bridge.clear_mlx_cache()

    def __contains__(self, key: object) -> bool:
        self._evict_idle()
        return key in self._store

    def __getitem__(self, key: str) -> dict[str, Any]:
        self._evict_idle()
        entry = self._store[key]
        entry.last_used = time.monotonic()
        return entry.value

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        self._store[key] = _CacheEntry(value=value, last_used=time.monotonic())

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __bool__(self) -> bool:
        self._evict_idle()
        return bool(self._store)

    def clear(self) -> None:
        self._store.clear()


_MODEL_CACHE = _InactivityCache(
    idle_timeout_s=float(os.environ.get("ASDX_MODEL_CACHE_IDLE_TIMEOUT_S", "900"))
)


def clear_model_cache() -> None:
    """Evict every cached diffusion model, freeing its memory.

    `_MODEL_CACHE` holds at most one entry at a time by design (`load()`
    below clears it before loading a different checkpoint/precision), so
    there's nothing to key this by — callers that want the currently
    resident model's memory back (e.g. the sampler's `low_memory_mode`)
    just call this once the sampling run that needed it is done. The next
    `ASDX_DiffusionLoader.load()` call will reload from disk instead of
    hitting the cache.
    """
    if _MODEL_CACHE:
        _MODEL_CACHE.clear()
        bridge.clear_mlx_cache()


_TYPE_HINTS = {
    "schnell": "schnell",
    "dev": "dev",
    "kontext": "dev",
}

_KREA2_HINTS = {
    "krea2": "krea2",
    "krea": "krea2",
}

_SDXL_HINTS = {
    "sdxl": "sdxl",
    "illustrious": "sdxl",
    "pony": "sdxl",
    "noobai": "sdxl",
}

_ZIMAGE_HINTS = {
    "zimage": "zimage",
    "z_image": "zimage",
    "z-image": "zimage",
}

_FLUX2_HINTS = {
    "klein": "flux2",
    "flux.2": "flux2",
    "flux2": "flux2",
    "flux_2": "flux2",
}

_QWEN_IMAGE21_HINTS = {
    "qwen_image_2.1": "qwen_image21",
    "qwen-image-2.1": "qwen_image21",
    "qwen_image21": "qwen_image21",
}

# Distinctive tensor key for the DiT (verified against the real checkpoint header in
# brick 2: `transformer_blocks.0.img_mlp.gate_up.weight`, unique to this family's fused
# SwiGLU MLP naming -- SDXL/Z-Image/Flux2/Krea2 use their own distinct markers, none of
# which contain or are contained in this string).
_QWEN_IMAGE21_STRUCTURAL_KEY = "transformer_blocks.0.img_mlp.gate_up."


def _detect_model_type(path: Path) -> str:
    """Detect model type from filename, falling back to checkpoint key
    inspection when the filename gives no hint.

    Filename hints are checked first (cheap, matches established
    convention). `_FLUX2_HINTS` is checked BEFORE the generic
    `_TYPE_HINTS` (schnell/dev/kontext): Flux2-D's real filename on this
    machine (`flux2_dev_fp8mixed.safetensors`) contains "dev" too, which
    would otherwise misroute it to the FLUX.1-dev architecture (the exact
    class of bug `_TYPE_HINTS["klein"]="schnell"` used to be, for Klein).
    Many SDXL finetunes (Illustrious/Pony/NoobAI merges in particular)
    don't include an obvious marker in their filename, so if nothing
    matches, peek at the checkpoint's own tensor keys (header-only read via
    safetensors, no weight data loaded) — SDXL's conv UNet has a
    structurally distinctive `input_blocks.` key that FLUX/Krea2 never have,
    Z-Image has an equally distinctive `noise_refiner.` key, and Flux2 has
    `double_stream_modulation_img.` (comfy's own detection marker for this
    exact branch — see `comfy/model_detection.py:237`).

    A filename hint that resolves to one of these structurally distinctive
    types (sdxl/zimage/flux2/krea2) is still verified against the same
    key-based check before being trusted: a checkpoint can be named after
    its training data/style rather than its base architecture (a
    "Pony"-tagged FLUX finetune matched `_SDXL_HINTS["pony"]` despite having
    FLUX `double_blocks.` keys, not SDXL's `input_blocks.`), which used to
    silently route it into the wrong native loader (0 matched params on
    load, garbage output with no error).
    """
    name = path.name.lower()

    if path.suffix.lower() == ".gguf":
        # `_detect_model_type_from_keys` opens the file via `safe_open(path,
        # framework="pt")`, which only understands safetensors -- it can't
        # read a GGUF file's keys at all, so the structural fallback below is
        # unreachable for this branch. No other family in this generic
        # dispatch has a GGUF variant yet, so an unrecognized `.gguf` name
        # must raise rather than silently default to "dev".
        for hint in _QWEN_IMAGE21_HINTS:
            if hint in name:
                return "qwen_image21"
        raise RuntimeError(
            f"ASDX: {path.name} is a .gguf file but its name doesn't match any known "
            "GGUF-supporting family (qwen_image21) -- no other family in this dispatch "
            "supports GGUF yet."
        )

    hint_type: str | None = None
    for hint in _KREA2_HINTS:
        if hint in name:
            hint_type = "krea2"
            break
    if hint_type is None:
        for hint in _SDXL_HINTS:
            if hint in name:
                hint_type = "sdxl"
                break
    if hint_type is None:
        for hint in _ZIMAGE_HINTS:
            if hint in name:
                hint_type = "zimage_turbo" if "turbo" in name else "zimage"
                break
    if hint_type is None:
        for hint in _QWEN_IMAGE21_HINTS:
            if hint in name:
                hint_type = "qwen_image21"
                break
    if hint_type is None:
        for hint in _FLUX2_HINTS:
            if hint in name:
                hint_type = "flux2"
                break

    if hint_type is not None:
        # Unlike the generic dev/schnell/kontext hints below, these claim a
        # structurally distinctive architecture -- verify against the
        # checkpoint's own keys before trusting it. Some finetunes are named
        # after their training data/style rather than their base architecture
        # (e.g. a "Pony"-tagged FLUX checkpoint matching `_SDXL_HINTS["pony"]`
        # despite having FLUX `double_blocks.`/`img_attn.` keys, not SDXL's
        # `input_blocks.`), which used to silently route such a checkpoint
        # into the wrong native loader (0 matched params, garbage output).
        base_hint_type = "zimage" if hint_type == "zimage_turbo" else hint_type
        detected = _detect_model_type_from_keys(path)
        if detected == base_hint_type:
            return hint_type
        print(f"[ASDX] Filename suggests '{hint_type}' but checkpoint keys indicate "
              f"'{detected}' -- using key-based detection for {path.name}")
        return detected

    for hint in _TYPE_HINTS:
        if hint in name:
            return _TYPE_HINTS[hint]
    return _detect_model_type_from_keys(path)


def _detect_model_type_from_keys(path: Path) -> str:
    """Fallback: distinguish SDXL/Z-Image/Flux2 from FLUX.1/Krea2 by checkpoint tensor keys."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
    except Exception as e:
        print(f"[ASDX] Model type key-detection failed ({e}), defaulting to 'dev'")
        return "dev"

    if any("diffusion_model.input_blocks." in k or k.startswith("input_blocks.") for k in keys):
        return "sdxl"
    if any("noise_refiner." in k for k in keys):
        return "zimage"
    if any("double_stream_modulation_img." in k for k in keys):
        return "flux2"
    if any("txtfusion." in k for k in keys):
        return "krea2"
    if any(_QWEN_IMAGE21_STRUCTURAL_KEY in k for k in keys):
        return "qwen_image21"
    return "dev"


def _contains_asdx_model(value: Any) -> bool:
    """Recurse into ComfyUI's nested output structure -- `execution.py`'s
    `merge_result_data` wraps each output socket's value in its own list
    (`output.append([o[i] for o in results])`), so a cached entry's
    `.outputs` is `[[model_desc]]`, not `[model_desc]`. Same recursion shape
    as `RAMPressureCache.ram_release`'s own `scan_list_for_ram_usage`.
    """
    if isinstance(value, dict):
        return value.get("type") == "asdx_model"
    if isinstance(value, (list, tuple)):
        return any(_contains_asdx_model(v) for v in value)
    return False


def _cache_entry_holds_asdx_model(entry: Any) -> bool:
    outputs = getattr(entry, "outputs", None)
    if not outputs:
        return False
    return _contains_asdx_model(outputs)


def _purge_stale_asdx_cache_entries() -> int:
    """Drop ComfyUI's own execution-level cache entries that still hold a
    previous "asdx_model" payload, so its MLX arrays actually become
    unreachable and `bridge.clear_mlx_cache()` can free them.

    ComfyUI's node-output cache (`comfy_execution/caching.py`, RAMPressureCache
    by default) keys entries by (node_id, input signature) -- switching
    `model_name`/`ckpt_name` produces a *new* key, so the old entry (with the
    old MLX transformer inside) is never overwritten, only left alongside the
    new one. Its own RAM-pressure eviction (`RAMPressureCache.ram_release`)
    only scores `torch.Tensor` (cpu) and `ModelPatcher` outputs; our plain
    "asdx_model" dict falls through both checks and gets the ~0-byte default
    weight, so it's effectively never chosen for eviction -- its MLX arrays
    stay reachable from that cache regardless of what `_MODEL_CACHE.clear()`
    does on our side, and `mx.clear_cache()`/`gc.collect()` can't release
    memory still referenced from outside our own module.

    Technique ported from ComfyUI-DistorchMemoryManager's
    `_purge_detailer_segs_and_executor_cache()`: walk `gc.get_objects()` for
    the live `PromptExecutor` and drop matching entries from its caches in
    place. Never call `PromptExecutor.reset()` to do this instead -- that
    swaps in a fresh cache with no `cache_key_set` initialized yet, and the
    next `caches.outputs.get()` raises `AttributeError`. Scoped to entries
    that actually hold an "asdx_model" payload (unlike the Distorch original,
    which clears every cache entry) so unrelated node outputs elsewhere in
    the graph aren't forced to recompute.
    """
    import gc

    purged = 0
    for obj in gc.get_objects():
        if type(obj).__name__ != "PromptExecutor":
            continue
        caches = getattr(obj, "caches", None)
        if caches is None:
            continue
        for cache in getattr(caches, "all", None) or []:
            cache_dict = getattr(cache, "cache", None)
            if not isinstance(cache_dict, dict):
                continue
            stale_keys = [k for k, e in cache_dict.items() if _cache_entry_holds_asdx_model(e)]
            for key in stale_keys:
                del cache_dict[key]
                purged += 1
                for bag_name in ("timestamps", "used_generation", "children", "subcaches"):
                    bag = getattr(cache, bag_name, None)
                    if isinstance(bag, dict):
                        bag.pop(key, None)
    return purged


def _gate_memory_before_load(
    path: Path, model_type: str, precision: str, low_memory_mode: bool
) -> LoadShape | None:
    """Predict the checkpoint's peak memory footprint and refuse to load if it
    clearly cannot fit -- see `memory_calibration.py` for the two-tier
    (measured vs. heuristic) prediction and the two-level refuse/warn
    threshold. Header-only read (no tensor data), independent of the
    `_load_safetensors` gate that runs later in `_load_transformer_for_type`.

    Returns the `LoadShape` used for the gate (or None if the header couldn't
    be read) so the caller can stash it on the model descriptor -- the
    sampler needs the exact same shape later to record a real observed peak
    against this same key, see `record_observation`.
    """
    try:
        header = read_safetensors_header(path)
        quant_format = classify_quant_format(header)
        quant_format_str = "unknown" if isinstance(quant_format, Unrecognized) else quant_format.value
    except Exception as e:
        print(f"[ASDX] memory_calibration: header read failed ({e}), skipping memory gate")
        return None

    shape = LoadShape(
        family=model_type,
        quant_format=quant_format_str,
        precision=precision,
        low_memory_mode=low_memory_mode,
        file_size_bytes=path.stat().st_size,
    )
    check_fits_or_warn(shape)
    return shape


def _load_transformer_for_type(
    path: Path, model_type: str, dtype: str
):
    """Load transformer weights based on detected model type.

    Returns (transformer, config) tuple.
    """
    if model_type == "krea2":
        from .native.krea2 import (
            Krea2Config,
            SingleStreamDiT,
            load_krea2_transformer,
        )
        config = Krea2Config(dtype=dtype)
        transformer = load_krea2_transformer(path, dtype=dtype)
        return transformer, config
    elif model_type == "sdxl":
        from .native.sdxl import SDXLConfig, load_sdxl_unet
        config = SDXLConfig(dtype=dtype)
        transformer = load_sdxl_unet(path, dtype=dtype)
        return transformer, config
    elif model_type in ("zimage", "zimage_turbo"):
        from .native.zimage import ZImageConfig, load_zimage_transformer
        config = ZImageConfig(dtype=dtype)
        transformer = load_zimage_transformer(path, dtype=dtype)
        return transformer, config
    elif model_type == "flux2":
        from .native.flux2 import load_flux2_transformer
        transformer = load_flux2_transformer(path, dtype=dtype)
        # Unlike the other families, Flux2's config is DETECTED from the
        # checkpoint (hidden_size/depth/guidance_embed differ between Klein
        # and Flux2-D) — reuse the config load_flux2_transformer already
        # derived, don't construct a fresh default one.
        return transformer, transformer.config
    elif model_type == "qwen_image21":
        from .native.qwen_image21 import (
            load_qwen_image21_dit_checkpoint,
            load_qwen_image21_dit_from_gguf,
        )
        if path.suffix.lower() == ".gguf":
            transformer = load_qwen_image21_dit_from_gguf(path, dtype=dtype)
        else:
            transformer = load_qwen_image21_dit_checkpoint(path, dtype=dtype)
        return transformer, transformer.config
    else:
        # FLUX.1 path
        guidance_embed = model_type == "dev"
        config = FluxConfig(dtype=dtype, guidance_embed=guidance_embed)
        transformer = load_transformer(path, dtype=dtype)
        return transformer, config


_MODEL_TYPE_CAPABILITY = {
    "sdxl": "sdxl_base",
    "zimage": "zimage_base",
    "zimage_turbo": "zimage_turbo",
    "flux2": "flux2_klein",
    # Pre-existing gap (not new this session): "krea2" had no entry here, so
    # every Krea2 load fell through to `_resolve_capability_from_path`, whose
    # `_CAPABILITY_DISPATCH` (capability.py) also has no "krea"/"krea2"
    # pattern -- every Krea2 checkpoint silently resolved to flux1_dev's
    # profile. `_detect_model_type` never distinguishes krea2_turbo from
    # krea2 today (unlike zimage_turbo), so krea2_base is the only reachable
    # Krea2 profile; krea2_turbo (capability.py) stays unreachable until that
    # detection is added -- noted, not fixed here, to keep this a minimal
    # unblock rather than a redesign of Krea2 turbo/raw detection.
    "krea2": "krea2_base",
    "qwen_image21": "qwen_image21_base",
}


def _capability_for_model_type(model_type: str, path: Path) -> CapabilityProfile:
    """Resolve a capability profile, preferring the already-known `model_type`
    (from `_detect_model_type`, which includes a content-based fallback for
    ambiguous filenames) over re-guessing from the filename alone — avoids
    the two detection systems disagreeing for checkpoints whose name gave
    no hint (e.g. Illustrious/Pony/NoobAI SDXL finetunes)."""
    profile_key = _MODEL_TYPE_CAPABILITY.get(model_type)
    if profile_key is not None:
        return CAPABILITY_PROFILES[profile_key]
    return _resolve_capability_from_path(path)



# ── Node ──────────────────────────────────────────────────────────────

class ASDX_DiffusionLoader(io.ComfyNode):
    """Load a FLUX.1 diffusion model checkpoint into MLX.

    Reads the checkpoint, creates a FluxTransformer, and caches it.
    The returned model object is passed to the sampler node.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_DiffusionLoader",
            display_name="🍏 ASDX Diffusion Loader",
            category="ASDX/Loaders",
            inputs=[
                io.Combo.Input("model_name", options=cls._get_models()),
                io.Combo.Input("precision", options=["float16", "bfloat16"], default="float16"),
                io.Boolean.Input("low_memory_mode", default=False, optional=True),
            ],
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @staticmethod
    def _get_models() -> list[str]:
        """Get list of available diffusion models."""
        try:
            import folder_paths
            models: dict[str, None] = {}
            for folder in ("diffusion_models", "unet"):
                try:
                    for name in folder_paths.get_filename_list(folder):
                        models[name] = None
                except Exception:
                    pass
            if models:
                return list(models)
        except Exception:
            pass
        return ["flux1-dev-fp16.safetensors"]

    @classmethod
    def execute(
        cls,
        model_name: str,
        precision: str,
        low_memory_mode: bool = False,
    ) -> io.NodeOutput:
        metadata_extractors.ensure_registered()
        t0 = time.perf_counter()

        cache_key = f"{model_name}:{precision}"

        if cache_key in _MODEL_CACHE:
            cached = _MODEL_CACHE[cache_key]
            print(f"[ASDX] Model cache hit: {model_name} ({precision})")
            return io.NodeOutput(cached)

        # _MODEL_CACHE never evicted past entries -- switching the checkpoint
        # or precision across separate prompt runs in the same ComfyUI session
        # kept every previously loaded transformer resident (each one is
        # several GB to tens of GB), silently accumulating until unified
        # memory was exhausted. Only one model is meaningfully "current" for
        # this node at a time, so drop everything else before loading the
        # new one -- matches how switching a checkpoint dropdown is expected
        # to free the old model.
        if _MODEL_CACHE:
            print(f"[ASDX] Evicting {len(_MODEL_CACHE)} cached model(s) before loading {model_name}")
            _MODEL_CACHE.clear()

        purged = _purge_stale_asdx_cache_entries()
        if purged:
            print(f"[ASDX] Purged {purged} stale ComfyUI executor cache entr"
                  f"{'y' if purged == 1 else 'ies'} holding old model(s)")
        bridge.clear_mlx_cache()

        # Find model file
        path = cls._resolve_model_path(model_name)
        model_type = _detect_model_type(path)

        # Resolve capability profile (see _capability_for_model_type).
        capability = _capability_for_model_type(model_type, path)

        memory_shape = _gate_memory_before_load(path, model_type, precision, low_memory_mode)

        # Load transformer based on model type (FLUX.1, Krea2, SDXL, or Z-Image)
        transformer, config = _load_transformer_for_type(
            path, model_type, precision
        )

        # _load_safetensors() upcasts BF16 checkpoint tensors to float32 before
        # the loader casts them down to the requested precision; the discarded
        # float32 buffers land in MLX's cache (freed but not returned to the
        # OS) rather than active memory. Release them now instead of letting
        # them sit alongside the real active weights for the rest of the run.
        bridge.clear_mlx_cache()

        # Create model descriptor with capability profile
        model_desc = {
            "type": "asdx_model",
            "name": model_name,
            "path": str(path),
            "transformer": transformer,
            "config": config,
            "model_type": model_type,
            "precision": precision,
            "capability": capability,
            "load_time": 0.0,
            "low_memory_mode": low_memory_mode,
            "memory_shape": memory_shape,
        }

        load_time = time.perf_counter() - t0
        model_desc["load_time"] = load_time

        mem = bridge.collect_mlx_memory()
        print(f"[ASDX] Loaded {model_name} in {load_time:.1f}s "
              f"(type={model_type}, precision={precision}, "
              f"mem={mem['active_gb']:.1f}GB active, {mem['cache_gb']:.1f}GB cache)")

        _MODEL_CACHE[cache_key] = model_desc
        return io.NodeOutput(model_desc)

    @staticmethod
    def _resolve_model_path(name: str) -> Path:
        """Resolve model name to a file path."""
        try:
            import folder_paths
            for folder in ("diffusion_models", "unet"):
                try:
                    full = folder_paths.get_full_path(folder, name)
                    if full:
                        return Path(full)
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback: check common locations
        for candidate in (
            Path.home() / "models" / "diffusion_models" / name,
            Path.home() / "ComfyUI" / "models" / "diffusion_models" / name,
        ):
            if candidate.exists():
                return candidate
        return Path(name)


def _clip_model_options(type: str) -> dict[str, Any]:
    """Return model_options dict for CLIP loading, to set dtype when needed.

    Used by ASDX_CLIPLoader and ASDX_DualCLIPLoader to select the right dtype
    for Krea2 (where float16 causes overflow in vision transformer attention).
    """
    # Default model_options (empty, let comfy choose)
    if type != "krea2":
        return {}

    # Krea2 uses Qwen3-VL text encoder which can overflow float16
    # in attention layers. Use bfloat16 instead to prevent NaNs.
    # This matches the fix described in the roadmap.
    return {"dtype": torch.bfloat16}


# ── Checkpoint Loader ────────────────────────────────────────────────────

class ASDX_CheckpointLoader(io.ComfyNode):
    """Load a full checkpoint (VAE + CLIP + Diffusion) into MLX.

    Reads the checkpoint, creates MLX model handles for the diffusion
    transformer, text encoders, and VAE. Returns handles that can be
    passed to the sampler and conditioning nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_CheckpointLoader",
            display_name="🍏 ASDX Checkpoint Loader",
            category="ASDX/Loaders",
            inputs=[
                io.Combo.Input("ckpt_name", options=cls._get_checkpoints()),
                io.Combo.Input("precision", options=["float16", "bfloat16"], default="float16"),
            ],
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
                io.Custom("mlx_clip").Output(display_name="clip"),
                io.Vae.Output(display_name="vae"),
            ],
        )

    @staticmethod
    def _get_checkpoints() -> list[str]:
        """Get list of available checkpoint files."""
        try:
            import folder_paths
            checkpoints = folder_paths.get_filename_list("checkpoints")
            if checkpoints:
                return checkpoints
        except Exception:
            pass
        return ["flux1-dev-fp16.safetensors"]

    @classmethod
    def execute(cls, ckpt_name: str, precision: str) -> io.NodeOutput:
        metadata_extractors.ensure_registered()
        t0 = time.perf_counter()

        # Resolve checkpoint path
        path = cls._resolve_checkpoint_path(ckpt_name)
        model_type = _detect_model_type(path)

        # This loader has no cache of its own (reloads every execute()), so
        # the only place a previous asdx_model payload can linger is
        # ComfyUI's own executor cache -- see _purge_stale_asdx_cache_entries.
        purged = _purge_stale_asdx_cache_entries()
        if purged:
            print(f"[ASDX] Purged {purged} stale ComfyUI executor cache entr"
                  f"{'y' if purged == 1 else 'ies'} holding old model(s)")
        bridge.clear_mlx_cache()

        # ASDX_CheckpointLoader has no low_memory_mode input, so gate with the
        # strict default (False) -- see ASDX_DiffusionLoader.load() for the
        # variant that honors a user-supplied low_memory_mode.
        memory_shape = _gate_memory_before_load(path, model_type, precision, low_memory_mode=False)

        # Load diffusion model based on type
        transformer, config = _load_transformer_for_type(
            path, model_type, precision
        )

        # See ASDX_DiffusionLoader.load() — release the float32 buffers
        # _load_safetensors()/the dtype cast leave behind in MLX's cache.
        bridge.clear_mlx_cache()

        # Resolve capability profile (see _capability_for_model_type — this
        # loader is the one most likely to see merged SDXL/Illustrious/Pony
        # checkpoints, whose filenames are often ambiguous).
        capability = _capability_for_model_type(model_type, path)

        # Create model descriptor
        model_desc = {
            "type": "asdx_model",
            "name": ckpt_name,
            "path": str(path),
            "transformer": transformer,
            "config": config,
            "model_type": model_type,
            "precision": precision,
            "capability": capability,
            "memory_shape": memory_shape,
        }

        # Real comfy.sd.CLIP + comfy.sd.VAE, extracted from the same checkpoint
        # file in one pass — NOT placeholders. The diffusion transformer is
        # loaded separately above via our own MLX-native reader, so
        # `output_model=False` skips the (expensive, redundant) PyTorch UNet
        # build; only the CLIP/VAE-prefixed tensors are used. `clip` must be a
        # real `comfy.sd.CLIP` (ASDX_CLIPTextEncode does `isinstance(mlx_clip,
        # comfy.sd.CLIP)`), and `vae` must be a real "VAE"-typed comfy object
        # (ASDX_VAEDecode's `vae` input socket only accepts the standard
        # ComfyUI "VAE" type, and its decode path calls `vae.decode()`).
        # Bonus over the standalone ASDX_DualCLIPLoader path: the text-encoder
        # architecture (e.g. Klein's Qwen3-4B vs Qwen3-8B vs Flux2-D's
        # Mistral3-24B) is detected from the checkpoint's own embedded CLIP
        # weights (`model_config.clip_target(state_dict)`), not from a
        # user-selected `clip_type` dropdown — sidesteps the Klein-4B
        # misrouting footgun that dropdown has when clip_type is left at its
        # default (see Phase F notes in the multi-model plan).
        import comfy.sd
        _, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            str(path), output_model=False, output_clip=True,
            output_vae=True, output_clipvision=False,
        )
        clip_desc = clip

        load_time = time.perf_counter() - t0
        mem = bridge.collect_mlx_memory()
        print(f"[ASDX] Checkpoint loaded: {ckpt_name} in {load_time:.1f}s "
              f"(type={model_type}, precision={precision}, "
              f"mem={mem['active_gb']:.1f}GB active, {mem['cache_gb']:.1f}GB cache)")

        return io.NodeOutput(model_desc, clip_desc, vae)

    @staticmethod
    def _resolve_checkpoint_path(name: str) -> Path:
        """Resolve checkpoint name to a file path."""
        try:
            import folder_paths
            full = folder_paths.get_full_path("checkpoints", name)
            if full:
                return Path(full)
        except Exception:
            pass
        # Fallback: check common locations
        for candidate in (
            Path.home() / "ComfyUI" / "models" / "checkpoints" / name,
            Path(name),
        ):
            if candidate.exists():
                return candidate
        return Path(name)


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_LIST = [
    ASDX_DiffusionLoader,
    ASDX_CheckpointLoader,
]
