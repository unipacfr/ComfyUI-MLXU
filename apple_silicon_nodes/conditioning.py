"""
Conditioning nodes
==================
CLIP text encoding and conditioning manipulation for FLUX and SD-style models.

Nodes:
  - ASDX_DualCLIPLoader: Load two CLIP text encoders (SDXL, FLUX, SD3...)
  - ASDX_CLIPLoader: Load a single CLIP model (SD1.5, Pony, Illustrious...)
  - ASDX_CLIPTextEncode: Encode text to conditioning (auto-detect FLUX/SD mode)
  - ASDX_ConditioningMerger: Merge two conditioning inputs
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch

import comfy.sd
import comfy.utils
from comfy_api.latest import io

from . import bridge
from . import metadata_extractors


# ── Globals ───────────────────────────────────────────────────────────

_CLIP_CACHE: dict[str, Any] = {}


# llama_detect (comfy/text_encoders/hunyuan_video.py) reads dtype_llama off
# these two probe keys ALONE, then flux2_te hard-overrides the caller's dtype
# with it -- there is no model_options path to it.
_LLAMA_DTYPE_PROBE_KEYS = (
    "model.norm.weight",
    "model.layers.0.input_layernorm.weight",
)


def _load_clip_fp8_aware(ckpt_paths: list[str], clip_type: Any, model_options: dict) -> Any:
    """`comfy.sd.load_clip`, but keeping an FP8 text encoder's weights in FP8.

    Flux.2/Klein's FP8 Qwen3-8B encoder is 7.63GB on disk yet costs 15.35GB
    live, because comfy upcasts every one of its 254 F8_E4M3 tensors to
    bfloat16 at load. The cause is narrow and confirmed by reading the file's
    own header: `llama_detect` takes `dtype_llama` from the dtype of two
    LayerNorm probe keys, and in this file the 145 norms are BF16 while the
    252 linears + 2 embeddings are F8_E4M3. comfy sees BF16 and upcasts the
    lot. `flux2_te` then applies that dtype unconditionally
    (`if dtype_llama is not None: dtype = dtype_llama`), so neither
    `model_options["dtype"]` nor `--fp8_e4m3fn-text-enc` can reach it --
    both were measured to have zero effect here.

    Fix, in two halves, because forcing FP8 alone is NOT correct: cast only
    the two probe tensors so `llama_detect` reports FP8, then restore all 145
    norms to their true BF16 values on the built module. The restore matters
    -- FP8 norms alone (0.3M params) shifted every embedding by 6.4% rel. L2
    (cosine 0.9996, against 0.998 between two *different* prompts, i.e. the
    error was a meaningful fraction of real semantic distance). With the
    norms put back, output is bit-identical to the bfloat16 path (cosine
    1.000000, max abs diff 0.0) at 7.72GB live instead of 15.35GB.

    Only files that are genuinely FP8-with-BF16-norms take this path; anything
    else loads exactly as before.
    """
    embedding_directory = (
        comfy.utils.get_t2ia_paths() if hasattr(comfy.utils, "get_t2ia_paths") else []
    )
    sd_list = [comfy.utils.load_torch_file(p) for p in ckpt_paths]

    targets = [
        sd for sd in sd_list
        if any(k in sd and sd[k].dtype == torch.bfloat16 for k in _LLAMA_DTYPE_PROBE_KEYS)
        and any(v.dtype == torch.float8_e4m3fn for v in sd.values())
    ]
    if not targets:
        del sd_list
        return comfy.sd.load_clip(
            ckpt_paths=ckpt_paths, embedding_directory=embedding_directory,
            clip_type=clip_type, model_options=model_options,
        )

    saved_norms: dict[str, torch.Tensor] = {}
    for sd in targets:
        for k, v in sd.items():
            if v.dtype == torch.bfloat16:
                saved_norms[k] = v.clone()
        for k in _LLAMA_DTYPE_PROBE_KEYS:
            if k in sd:
                sd[k] = sd[k].to(torch.float8_e4m3fn)

    clip = comfy.sd.load_text_encoder_state_dicts(
        state_dicts=sd_list, embedding_directory=embedding_directory,
        clip_type=clip_type, model_options=model_options,
    )

    modules = dict(clip.cond_stage_model.named_modules())
    restored = 0
    for file_key, value in saved_norms.items():
        for prefix in ("qwen3_8b.transformer.", ""):
            module_name, _, attr = f"{prefix}{file_key}".rpartition(".")
            module = modules.get(module_name)
            if module is None:
                continue
            current = getattr(module, attr, None)
            if current is not None and current.shape == value.shape:
                setattr(module, attr, torch.nn.Parameter(
                    value.to(current.device), requires_grad=False))
                restored += 1
                break

    if restored != len(saved_norms):
        # Never ship silently-degraded norms: that is the 6.4%-drift case.
        raise RuntimeError(
            f"ASDX FP8 text encoder: restored only {restored}/{len(saved_norms)} "
            "BF16 norms after forcing FP8 -- refusing to run with degraded "
            "LayerNorm weights. Report this with the checkpoint name."
        )
    print(f"[ASDX] FP8 text encoder kept in FP8 ({restored} norms restored to bf16)")
    return clip


def _clip_model_options(type_str: str) -> dict:
    """comfy's default text-encoder dtype is float16 (`model_management.
    text_encoder_dtype`), even on CPU. That's fine for text-only encoding, but
    Krea2's Qwen3-VL vision tower only actually runs when grounded (image
    input on ASDX_CLIPTextEncode), and float16 vision-transformer attention
    overflows far more easily than bf16 (same CLAUDE.md caveat as MPS
    LayerNorm/VAE) -- observed producing NaN embeddings that decode to a black
    image. bf16 is the same 2 bytes/param as fp16, so this costs nothing.
    """
    if type_str == "krea2":
        return {"dtype": torch.bfloat16}
    return {}


# ── Dual CLIP Loader ─────────────────────────────────────────────────

class ASDX_DualCLIPLoader(io.ComfyNode):
    """Load two CLIP text encoders for dual-CLIP architectures.

    Supports SDXL, SD3, FLUX, Hunyuan, HiDream, Kandinsky, LTXV, Newbie, ACE.
    Returns an mlx_clip handle that can be used by the text encoder
    and sampler nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_DualCLIPLoader",
            display_name="🍏 ASDX Dual CLIP Loader",
            category="ASDX/Loaders",
            inputs=[
                io.Combo.Input("clip_name1", options=cls._get_clip_names()),
                io.Combo.Input("clip_name2", options=cls._get_clip_names()),
                io.Combo.Input("type", options=_DUAL_CLIP_TYPES, default="flux"),
            ],
            outputs=[
                io.Custom("mlx_clip").Output(display_name="mlx_clip"),
            ],
        )

    @staticmethod
    def _get_clip_names() -> list[str]:
        try:
            import folder_paths
            return folder_paths.get_filename_list("text_encoders")
        except Exception:
            return ["clip_l.safetensors"]

    @staticmethod
    def _get_t5_names() -> list[str]:
        try:
            import folder_paths
            return folder_paths.get_filename_list("text_encoders")
        except Exception:
            return ["t5xxl.safetensors"]

    @classmethod
    def execute(cls, clip_name1: str, clip_name2: str, type: str) -> io.NodeOutput:
        cache_key = f"{clip_name1}:{clip_name2}:{type}"

        if cache_key not in _CLIP_CACHE:
            # Only one CLIP pair is meaningfully "current" at a time -- evict
            # prior entries before loading a new one instead of accumulating
            # every distinct clip_name/type combo used in the session (same
            # fix already applied to loader.py's _MODEL_CACHE).
            if _CLIP_CACHE:
                _CLIP_CACHE.clear()
                bridge.clear_mlx_cache()

            # Load the CLIP
            clip_path1 = cls._find_file("text_encoders", clip_name1)
            clip_path2 = cls._find_file("text_encoders", clip_name2)
            clip_type_enum = _clip_type_from_string(type)

            clip = _load_clip_fp8_aware(
                ckpt_paths=[clip_path1, clip_path2],
                clip_type=clip_type_enum,
                model_options=_clip_model_options(type),
            )

            _CLIP_CACHE[cache_key] = clip
            print(f"[ASDX] Dual CLIP loaded: {clip_name1} + {clip_name2} (type={type})")

        return io.NodeOutput(_CLIP_CACHE[cache_key])

    @staticmethod
    def _find_file(folder: str, name: str) -> str:
        try:
            import folder_paths
            return folder_paths.get_full_path(folder, name) or name
        except Exception:
            return name


# ── CLIP Text Encode ─────────────────────────────────────────────────

class ASDX_CLIPTextEncode(io.ComfyNode):
    """Encode text prompts to conditioning for any model type.

    Auto-detects FLUX vs SD-style encoding:
    - If t5xxl is provided → FLUX mode (separate clip_l + t5xxl + guidance)
    - Otherwise → SD/SDXL/Pony mode (single CLIP encode)
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_CLIPTextEncode",
            display_name="🍏 ASDX CLIP Text Encode",
            category="ASDX/Conditioning",
            inputs=[
                io.Custom("mlx_clip").Input("mlx_clip"),
                io.String.Input("text", multiline=True, default=""),
                io.String.Input("t5xxl", multiline=True, default="", optional=True),
                io.Float.Input("guidance", default=3.5, min=0.0, max=100.0, step=0.1, optional=True),
            ],
            outputs=[
                io.Custom("mlx_conditioning").Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mlx_clip: Any,
        text: str,
        t5xxl: str = "",
        guidance: float = 3.5,
    ) -> io.NodeOutput:
        metadata_extractors.ensure_registered()
        if not isinstance(mlx_clip, comfy.sd.CLIP):
            raise RuntimeError("ASDX: mlx_clip must be a Comfy CLIP object.")

        if t5xxl:
            # FLUX mode: separate clip_l + t5xxl inputs. mlx_clip.tokenize(text)
            # already returns the full {"l": [...], "t5xxl": [...]} dict (a
            # dual-tokenizer CLIP tokenizes through every sub-tokenizer at
            # once) -- only the "t5xxl" entry needs overriding with its own
            # text, matching comfy's own CLIPTextEncodeFlux.execute(). Wrapping
            # both full dicts again as {"l": tokens_l, "t5xxl": tokens_t5}
            # (the previous code here) nests them one level too deep, so
            # encode_token_weights() ends up iterating dict keys ("l",
            # "t5xxl") as if they were (token, weight) pairs.
            tokens = mlx_clip.tokenize(text)
            tokens["t5xxl"] = mlx_clip.tokenize(t5xxl)["t5xxl"]
            conditioning = mlx_clip.encode_from_tokens_scheduled(
                tokens,
                add_dict={"guidance": float(guidance)},
            )
            result = {
                "type": "flux1",
                "conditioning": conditioning,
                "text": text,
                "t5xxl": t5xxl,
                "guidance": float(guidance),
            }
            print(f"[ASDX] Text encoded (FLUX): text={len(text)} chars, "
                  f"t5xxl={len(t5xxl)} chars, guidance={guidance:.1f}")
        else:
            # SD / SDXL / Pony mode: single CLIP encode
            tokens = mlx_clip.tokenize(text)
            conditioning = mlx_clip.encode_from_tokens_scheduled(tokens)
            # Fail fast on a corrupt (NaN/Inf) embedding rather than letting it
            # silently ride through ~7min of diffusion sampling and VAE decode
            # to surface only as a black output image (comfy's own PIL cast then
            # warns "invalid value encountered in cast" and clips to garbage).
            for cond, _ in conditioning:
                if not torch.isfinite(cond).all():
                    raise RuntimeError(
                        "ASDX: CLIP Text Encode produced a non-finite (NaN/Inf) "
                        "embedding -- aborting before the expensive sampling pass."
                    )
            result = {
                "type": "clip",
                "conditioning": conditioning,
                "text": text,
            }
            print(f"[ASDX] Text encoded (SD-style): {len(text)} chars, type=clip")

        return io.NodeOutput(result)


# ── CLIP Types ────────────────────────────────────────────────────────
# Complete mapping of ComfyUI's CLIPType enum to human-readable strings.
# Single types (CLIPLoader node): all 33 CLIPType values + "mage" alias
# Dual types (DualCLIPLoader node): 9 types for two-CLIP architectures

_SINGLE_CLIP_TYPES: list[str] = [
    "stable_diffusion",   # SD1.5 — clip-l
    "stable_cascade",     # Stable Cascade — clip-g
    "sd3",                # SD3 — clip-g + clip-l + t5
    "stable_audio",       # Stable Audio — t5 base
    "hunyuan_dit",        # Hunyuan DiT
    "flux",               # FLUX — clip-l + t5
    "mochi",              # Mochi — t5 xxl
    "ltxv",               # LTX-Video
    "hunyuan_video",      # Hunyuan Video
    "pixart",             # PixArt — gemma 2 2B
    "cosmos",             # Cosmos — old t5 xxl
    "lumina2",            # Lumina2 — gemma 2 2B
    "wan",                # Wan — umt5 xxl
    "hidream",            # HiDream — t5 + llama
    "chroma",             # Chroma
    "ace",                # ACE
    "omnigen2",           # OmniGen2 — qwen vl 2.5 3B
    "qwen_image",         # Qwen Image
    "hunyuan_image",      # Hunyuan Image — qwen2.5vl + byt5
    "hunyuan_video_15",   # Hunyuan Video 1.5
    "ovis",               # OVIS
    "kandinsky5",         # Kandinsky 5
    "kandinsky5_image",   # Kandinsky 5 Image
    "newbie",             # Newbie — gemma-3-4b-it + jina
    "flux2",              # FLUX 2
    "longcat_image",      # LongCat Image
    "cogvideox",          # CogVideoX — t5 xxl (226-token)
    "lens",               # Lens — gpt-oss-20b
    "pixeldit",           # PixelDIT — gemma 2 2B elm
    "ideogram4",          # Ideogram 4
    "boogu",              # Boogu
    "krea2",              # Krea2
    "joyimage",           # JoyImage — qwen3-vl 8B
    "mage",               # Mage (alias → stable_diffusion)
]

_DUAL_CLIP_TYPES: list[str] = [
    "sdxl",               # clip-l + clip-g
    "sd3",                # clip-l + clip-g / clip-l + t5 / clip-g + t5
    "flux",               # clip-l + t5
    "hunyuan_video",      # hunyuan video dual
    "hidream",            # t5 + llama
    "hunyuan_image",      # qwen2.5vl + byt5
    "hunyuan_video_15",   # hunyuan video 1.5 dual
    "kandinsky5",         # kandinsky 5 dual
    "kandinsky5_image",   # kandinsky 5 image dual
    "ltxv",               # ltxv dual
    "newbie",             # gemma-3-4b-it + jina
    "ace",                # ace dual
]


def _clip_type_from_string(s: str) -> Any:
    """Convert a string name to the corresponding CLIPType enum value.

    Matches ComfyUI's CLIPLoader.load_clip() behavior: getattr(CLIPType, name.upper()).
    Falls back to STABLE_DIFFUSION for unknown types.
    """
    # Map special aliases
    alias_map = {
        "mage": "STABLE_DIFFUSION",
        "sd1": "SD15",
        "sd2": "SD21",
        "flux_hybrid": "HYBRID",
        "pony": "PONY",
    }
    key = alias_map.get(s, s.upper())
    return getattr(comfy.sd.CLIPType, key, comfy.sd.CLIPType.STABLE_DIFFUSION)


class ASDX_CLIPLoader(io.ComfyNode):
    """Load a single CLIP text encoder model.

    Supports all ComfyUI CLIP types: SD1.5, SDXL, Pony, SD3, FLUX, FLUX2,
    Hunyuan, Mochi, Wan, PixArt, Kandinsky, Krea2, JoyImage, and more.
    Returns an mlx_clip handle that can be connected to ASDX_CLIPTextEncode.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_CLIPLoader",
            display_name="🍏 ASDX CLIP Loader",
            category="ASDX/Loaders",
            inputs=[
                io.Combo.Input("clip_name", options=cls._get_clip_names()),
                io.Combo.Input("type", options=_SINGLE_CLIP_TYPES, default="stable_diffusion"),
            ],
            outputs=[
                io.Custom("mlx_clip").Output(display_name="mlx_clip"),
            ],
        )

    @staticmethod
    def _get_clip_names() -> list[str]:
        try:
            import folder_paths
            return folder_paths.get_filename_list("text_encoders")
        except Exception:
            return ["clip_l.safetensors"]

    @classmethod
    def execute(cls, clip_name: str, type: str) -> io.NodeOutput:
        cache_key = f"{clip_name}:{type}"

        if cache_key not in _CLIP_CACHE:
            # See ASDX_DualCLIPLoader.execute for why prior entries are
            # evicted here (shared _CLIP_CACHE, same one-active-entry policy
            # as loader.py's _MODEL_CACHE).
            if _CLIP_CACHE:
                _CLIP_CACHE.clear()
                bridge.clear_mlx_cache()

            clip_path = cls._find_file("text_encoders", clip_name)
            clip_type_enum = _clip_type_from_string(type)

            clip = _load_clip_fp8_aware(
                ckpt_paths=[clip_path],
                clip_type=clip_type_enum,
                model_options=_clip_model_options(type),
            )

            _CLIP_CACHE[cache_key] = clip
            print(f"[ASDX] CLIP loaded: {clip_name} (type={type})")

        return io.NodeOutput(_CLIP_CACHE[cache_key])

    @staticmethod
    def _find_file(folder: str, name: str) -> str:
        try:
            import folder_paths
            return folder_paths.get_full_path(folder, name) or name
        except Exception:
            return name


# ── Conditioning Merger ──────────────────────────────────────────────

class ASDX_ConditioningMerger(io.ComfyNode):
    """Merge two conditioning inputs into one.

    Useful for combining positive and negative conditioning, or for
    chaining multiple text encoders.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_ConditioningMerger",
            display_name="🍏 ASDX Conditioning Merger",
            category="ASDX/Conditioning",
            inputs=[
                io.MultiType.Input(
                    "positive", types=[io.Conditioning, io.Custom("mlx_conditioning")],
                ),
                io.MultiType.Input(
                    "negative", types=[io.Conditioning, io.Custom("mlx_conditioning")],
                ),
            ],
            outputs=[
                io.Custom("mlx_conditioning").Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(cls, positive: Any, negative: Any) -> io.NodeOutput:
        """Merge conditioning - for FLUX, negative is typically ignored but accepted for compatibility."""
        # FLUX doesn't use negative conditioning in the traditional sense
        # Store both for compatibility but sampler will use positive
        result = dict(positive) if isinstance(positive, dict) else {
            "type": "flux1",
            "conditioning": positive,
        }
        result["_negative"] = negative
        return io.NodeOutput(result)


NODE_LIST = [
    ASDX_DualCLIPLoader,
    ASDX_CLIPLoader,
    ASDX_CLIPTextEncode,
    ASDX_ConditioningMerger,
]
