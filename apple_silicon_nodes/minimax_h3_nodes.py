"""MiniMax H3 nodes -- t2va only (see `native/minimax_h3/model.py`'s module
docstring for the scope restriction this mirrors exactly: no reference
conditioning, no keyframes/first_frame/last_frame, no VSA gate_compress, no
PDD head bank).

Ported from `comfy_extras/nodes_minimax_h3.py`'s `EmptyMiniMaxH3LatentAV` and
`MiniMaxH3SigmaShift`. `MiniMaxH3ImageToVideo` (the reference's third
in-scope node per this project's own workflow-derived node survey, see
`docs/plan-multi-modeles-apple-silicon.md` §5 Phase 6) is NOT ported as-is --
it accepts `first_frame`/`last_frame` keyframe conditioning that
`native/minimax_h3/layout.py::PackedLayout` does not implement, so exposing
those inputs would silently do nothing. Its t2va-only equivalent is
`ASDX_MiniMaxH3TextEncode` below.

Text tokenization: MiniMax H3's presentation is explicitly NOT chat-templated
for plain t2va (`comfy/text_encoders/minimax.py`'s module docstring: "raw
prompt/label text, no special tokens"), so `comfy.text_encoders.minimax.
MiniMaxH3Tokenizer` is a plain, weight-free BPE tokenizer -- instantiating it
directly (no `comfy.sd.CLIP`/`mlx_clip` needed) avoids loading the real
encoder's dense weights a second time just to reach its tokenizer, which
would defeat `text_encoder_weight_map.py`'s quantized-loading entirely.
Verified directly: `MiniMaxH3Tokenizer().tokenize_with_weights("a cat...")`
returns plain `(token_id, 1.0)` pairs with no vision/special tokens for a
text-only prompt.

Whether to keep this project's own MLX-native `Qwen3TextEncoder` for
conditioning, versus routing through ComfyUI's real `comfy.sd.CLIP` path
(`.encode_from_tokens_scheduled`, the same pattern `krea2_grounded_encode.py`
uses), was an open question this session: `.claude/canon.md`'s "Porting
CLIP/T5/Qwen text encoders to MLX has weak memory ROI, except FP8 sources"
record says PyTorch-CPU text-encoder RAM is already near on-disk size on
Apple Silicon, so a native MLX port usually isn't worth it. Decided to keep
the native encoder anyway for MiniMax H3 (full MLX pipeline, no PyTorch CLIP
at inference) -- see the commit introducing `ASDX_MiniMaxH3TextEncode` for
the reasoning; the canon record's general guidance still applies to other
families.

Two separate LATENT outputs (video, audio) rather than ComfyUI's own packed
`NestedTensor` pair: `ASDX_MiniMaxH3Sampler`'s own
`native/minimax_h3/sampling.py` consumes plain tensors directly, and two
standard `{"samples": tensor}` dicts already compose with the existing
`ASDX_VAEDecode`/`ASDX_VAEDecodeAudio` nodes with no new plumbing.

`ASDX_MiniMaxH3Sampler` is deliberately its own node, not a `model_type`
branch of the shared `ASDX_MLXSampler`/`_SamplerCore` (see
`native/minimax_h3/sampling.py`'s module docstring for why) -- it converts
its LATENT inputs' shapes to real Gaussian noise itself (matching stock
ComfyUI's own split: `EmptyLatentImage`-style nodes return zeros, a
`RandomNoise`/`KSampler` seed is what actually generates the noise that
gets denoised) rather than accepting the empty-latent's zero tensor as a
starting point, which would never move under the flow ODE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

import comfy.model_management
from comfy_api.latest import io

_DIT_CACHE: dict[str, Any] = {}
_TEXT_ENCODER_CACHE: dict[str, Any] = {}

# Ported from comfy_extras/nodes_minimax_h3.py -- frame-count/latent-shape
# math for the video/audio VAEs' downscale ratios (16x spatial, 4x temporal
# video; 40 latent-frames/sec audio).
FPS = 24
AUDIO_LATENT_FPS = 40
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32


def _align_frame_count(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def _video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _temporal_shape(length: int) -> tuple[int, int, int]:
    """`length` (frames at 24fps, as the user specifies it) -> `(frame_count,
    video_latent_t, audio_latent_t)`, snapped to the model's 17k+5 frame grid."""
    frame_count = _align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, _video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def _get_device() -> torch.device:
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def _register_gguf_extension(folder_key: str) -> None:
    """ComfyUI's `folder_paths.supported_pt_extensions` -- what the
    `"diffusion_models"`/`"text_encoders"` folder keys use -- does NOT
    include `.gguf` (confirmed against the real `folder_paths.py`:
    `{'.ckpt', '.pt', '.pt2', '.bin', '.pth', '.safetensors', '.pkl', '.sft'}`).
    `folder_paths.get_filename_list(folder_key)` therefore never returns a
    `.gguf` file for ANY folder by default -- this has nothing to do with
    our own filtering, the files are invisible before we ever see the list.

    The reference `calcuis/gguf` node pack works around this by registering
    brand-new folder keys (`'model_gguf'`/`'clip_gguf'`) with `{'.gguf'}` as
    their only extension, rather than extending `'diffusion_models'`/
    `'text_encoders'` -- so even with that node installed, OUR loaders
    (which query the standard keys, matching every other ASDX loader's
    convention) would still see an empty list. Extending the standard
    keys' own extension set here makes our loaders work whether or not
    `calcuis/gguf` is installed."""
    try:
        import folder_paths
    except ImportError:
        return
    paths, extensions = folder_paths.folder_names_and_paths.get(folder_key, ([], set()))
    if ".gguf" not in extensions:
        folder_paths.folder_names_and_paths[folder_key] = (paths, set(extensions) | {".gguf"})


for _folder_key in ("diffusion_models", "text_encoders"):
    _register_gguf_extension(_folder_key)


class ASDX_MiniMaxH3EmptyLatentAV(io.ComfyNode):
    """Create empty video + audio latents for MiniMax H3 t2va.

    `length` is frame count at 24fps, snapped up to the model's 17k+5 grid
    (matching `EmptyMiniMaxH3LatentAV`'s own tooltip: 124 = ~5s; the model's
    trained range is roughly 124-362 frames, longer is untested)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MiniMaxH3EmptyLatentAV",
            display_name="🍏 ASDX MiniMax H3 Empty Latent (AV)",
            category="ASDX/Latent",
            inputs=[
                io.Int.Input("width", default=1344, min=32, max=4096, step=32),
                io.Int.Input("height", default=768, min=32, max=4096, step=32),
                io.Int.Input(
                    "length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24fps, snapped up to the model's "
                            "17k+5 frame grid (124 = ~5s).",
                ),
                io.Int.Input("batch_size", default=1, min=1, max=1, optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, width: int, height: int, length: int, batch_size: int = 1) -> io.NodeOutput:
        if batch_size != 1:
            raise RuntimeError("ASDX MiniMax H3: batch_size must be 1 -- matches native/minimax_h3/model.py's precondition.")

        frame_count, latent_t, audio_t = _temporal_shape(length)
        device = _get_device()
        dtype = comfy.model_management.intermediate_dtype()

        video_latent = torch.zeros(
            [1, VIDEO_LATENT_CHANNELS, latent_t, height // 16, width // 16],
            device=device, dtype=dtype,
        )
        audio_latent = torch.zeros(
            [1, AUDIO_LATENT_CHANNELS, 2, audio_t],
            device=device, dtype=dtype,
        )

        print(
            f"[ASDX] MiniMax H3 Empty Latent (AV): {width}x{height}, {frame_count} frames "
            f"({length} requested), video_latent={list(video_latent.shape)}, "
            f"audio_latent={list(audio_latent.shape)}"
        )
        return io.NodeOutput({"samples": video_latent}, {"samples": audio_latent})


class ASDX_MiniMaxH3SigmaShift(io.ComfyNode):
    """Set the video/audio flow-matching sigma shifts.

    Ported from `MiniMaxH3SigmaShift`, adapted to this project's own MODEL
    representation: the reference patches ComfyUI's `ModelPatcher`/
    `model_sampling`/`transformer_options` machinery (`m.add_object_patch`,
    a `ModelSamplingAV` subclass) -- this project's MiniMax H3 model dict has
    none of that, so this node just overrides the two shift values that
    `native/minimax_h3/model.py::MiniMaxH3Model.__call__`'s `sigma_v`/
    `time_shift_sigma` computation reads from `config` (defaults 12.0/3.0,
    matching `comfy/supported_models.py::MiniMaxH3.sampling_settings`).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MiniMaxH3SigmaShift",
            display_name="🍏 ASDX MiniMax H3 Sigma Shift",
            category="ASDX/Loaders",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[io.Custom("asdx_model").Output()],
        )

    @classmethod
    def execute(cls, model: dict, shift_video: float, shift_audio: float) -> io.NodeOutput:
        if "transformer" not in model:
            raise RuntimeError("ASDX MiniMax H3 Sigma Shift: expected an ASDX MODEL dict.")

        config = model.get("config")
        if config is None or not hasattr(config, "sigma_shift_video"):
            raise RuntimeError(
                "ASDX MiniMax H3 Sigma Shift: model has no MiniMaxH3Config -- "
                "not a MiniMax H3 model."
            )

        new_config = _replace_sigma_shifts(config, shift_video, shift_audio)
        new_model = dict(model)
        new_model["config"] = new_config
        print(f"[ASDX] MiniMax H3 Sigma Shift: video={shift_video}, audio={shift_audio}")
        return io.NodeOutput(new_model)


def _replace_sigma_shifts(config: Any, shift_video: float, shift_audio: float) -> Any:
    """Copy `config` with the two shift fields overridden. Uses
    `dataclasses.replace` for a real (frozen) `MiniMaxH3Config`, falling back
    to `copy.copy` + attribute assignment for anything else (e.g. a test
    double) -- `MiniMaxH3Config` is the only real caller, but this avoids an
    import-order-sensitive `isinstance` check against it."""
    import copy
    import dataclasses

    if dataclasses.is_dataclass(config):
        return dataclasses.replace(config, sigma_shift_video=shift_video, sigma_shift_audio=shift_audio)
    new_config = copy.copy(config)
    new_config.sigma_shift_video = shift_video
    new_config.sigma_shift_audio = shift_audio
    return new_config


class ASDX_MiniMaxH3ModelLoader(io.ComfyNode):
    """Load a MiniMax H3 DiT GGUF checkpoint (quantized, see
    `native/minimax_h3/weight_map.py`'s module docstring for why dense
    loading would not fit in 64GB). GGUF only -- safetensors (ComfyUI
    INT8-tensorwise) loading is not implemented (see `weight_map.py`)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MiniMaxH3ModelLoader",
            display_name="🍏 ASDX MiniMax H3 Model Loader (GGUF)",
            category="ASDX/Loaders",
            inputs=[
                io.Combo.Input("model_name", options=cls._get_models()),
                io.Combo.Input("precision", options=["float16", "bfloat16", "float32"], default="float16"),
            ],
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
            ],
        )

    @staticmethod
    def _get_models() -> list[str]:
        try:
            import folder_paths
            # "diffusion_models" already scans both models/diffusion_models/
            # and models/unet/ (see folder_paths.py) -- no separate "unet" key
            # exists in stock ComfyUI, so there is nothing to loop over here.
            return [n for n in folder_paths.get_filename_list("diffusion_models") if n.lower().endswith(".gguf")]
        except Exception:
            return []

    @classmethod
    def execute(cls, model_name: str, precision: str = "float16") -> io.NodeOutput:
        import folder_paths

        found = folder_paths.get_full_path("diffusion_models", model_name)
        if not found:
            raise RuntimeError(f"ASDX MiniMax H3 Model Loader: could not find '{model_name}'.")
        path = Path(found)

        cache_key = f"{path}:{precision}"
        if cache_key in _DIT_CACHE:
            print(f"[ASDX] MiniMax H3 model cache hit: {model_name}")
            return io.NodeOutput(_DIT_CACHE[cache_key])

        from .native.minimax_h3.weight_map import load_minimax_h3_from_gguf

        _DIT_CACHE.clear()  # one resident MiniMax H3 DiT at a time -- see ASDX_DiffusionLoader's own eviction note
        model = load_minimax_h3_from_gguf(path, dtype=precision)
        model_desc = {
            "type": "asdx_model",
            "family": "minimax_h3",
            "name": model_name,
            "path": str(path),
            "transformer": model,
            "config": model.config,
            "precision": precision,
        }
        _DIT_CACHE[cache_key] = model_desc
        return io.NodeOutput(model_desc)


class ASDX_MiniMaxH3TextEncoderLoader(io.ComfyNode):
    """Load MiniMax H3's Qwen3-VL-32B text encoder GGUF checkpoint
    (quantized, see `native/minimax_h3/text_encoder_weight_map.py`)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MiniMaxH3TextEncoderLoader",
            display_name="🍏 ASDX MiniMax H3 Text Encoder Loader (GGUF)",
            category="ASDX/Loaders",
            inputs=[
                io.Combo.Input("encoder_name", options=cls._get_encoders()),
                io.Combo.Input("precision", options=["float16", "bfloat16", "float32"], default="float16"),
            ],
            outputs=[
                io.Custom("asdx_minimax_h3_text_encoder").Output(display_name="text_encoder"),
            ],
        )

    @staticmethod
    def _get_encoders() -> list[str]:
        try:
            import folder_paths
            # "text_encoders" already scans both models/text_encoders/ and
            # models/clip/ (see folder_paths.py) -- no separate "clip" key
            # exists in stock ComfyUI, so there is nothing to loop over here.
            return [n for n in folder_paths.get_filename_list("text_encoders") if n.lower().endswith(".gguf")]
        except Exception:
            return []

    @classmethod
    def execute(cls, encoder_name: str, precision: str = "float16") -> io.NodeOutput:
        import folder_paths

        found = folder_paths.get_full_path("text_encoders", encoder_name)
        if not found:
            raise RuntimeError(f"ASDX MiniMax H3 Text Encoder Loader: could not find '{encoder_name}'.")
        path = Path(found)

        cache_key = f"{path}:{precision}"
        if cache_key in _TEXT_ENCODER_CACHE:
            print(f"[ASDX] MiniMax H3 text encoder cache hit: {encoder_name}")
            return io.NodeOutput(_TEXT_ENCODER_CACHE[cache_key])

        from .native.minimax_h3.text_encoder_weight_map import load_qwen3_text_encoder_from_gguf

        _TEXT_ENCODER_CACHE.clear()
        encoder = load_qwen3_text_encoder_from_gguf(path, dtype=precision)
        result = {
            "type": "asdx_minimax_h3_text_encoder",
            "name": encoder_name,
            "path": str(path),
            "encoder": encoder,
            "precision": precision,
        }
        _TEXT_ENCODER_CACHE[cache_key] = result
        return io.NodeOutput(result)


class ASDX_MiniMaxH3TextEncode(io.ComfyNode):
    """Encode a prompt for MiniMax H3 t2va conditioning.

    Tokenizes via `comfy.text_encoders.minimax.MiniMaxH3Tokenizer` directly
    (weight-free BPE, no `comfy.sd.CLIP` needed -- see module docstring) and
    runs the token ids through the native MLX `Qwen3TextEncoder` loaded by
    `ASDX_MiniMaxH3TextEncoderLoader`. The result is the raw hidden state
    after layer 50 -- what `native/minimax_h3/model.py::MiniMaxH3Model`
    consumes as `context` directly, no further processing needed (no final
    norm/lm_head on this truncated checkpoint, matching the reference).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MiniMaxH3TextEncode",
            display_name="🍏 ASDX MiniMax H3 Text Encode",
            category="ASDX/Conditioning",
            inputs=[
                io.Custom("asdx_minimax_h3_text_encoder").Input("text_encoder"),
                io.String.Input("prompt", multiline=True, default=""),
            ],
            outputs=[
                io.Custom("asdx_minimax_h3_conditioning").Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder: dict, prompt: str) -> io.NodeOutput:
        if not isinstance(text_encoder, dict) or text_encoder.get("type") != "asdx_minimax_h3_text_encoder":
            raise RuntimeError("ASDX MiniMax H3 Text Encode: expected the output of ASDX_MiniMaxH3TextEncoderLoader.")

        import comfy.text_encoders.minimax

        tokenizer = comfy.text_encoders.minimax.MiniMaxH3Tokenizer()
        tokens = tokenizer.tokenize_with_weights(prompt)["qwen3vl_32b"][0]
        input_ids = mx.array([token_id for token_id, _weight in tokens], dtype=mx.int32)

        encoder = text_encoder["encoder"]
        hidden_states = encoder(input_ids)
        mx.eval(hidden_states)
        if not bool(mx.all(mx.isfinite(hidden_states)).item()):
            raise RuntimeError(
                "ASDX MiniMax H3 Text Encode: produced a non-finite (NaN/Inf) "
                "embedding -- aborting before the expensive sampling pass."
            )

        print(f"[ASDX] MiniMax H3 Text Encode: {len(prompt)} chars, {len(tokens)} tokens")
        return io.NodeOutput({
            "type": "minimax_h3",
            "hidden_states": hidden_states,
            "text": prompt,
        })


class ASDX_MiniMaxH3Sampler(io.ComfyNode):
    """Run MiniMax H3's flow-matching Euler sampling loop
    (`native/minimax_h3/sampling.py::run_minimax_h3_sampling`).

    Generates the starting Gaussian noise itself from `video_latent`/
    `audio_latent`'s shapes (matching stock ComfyUI's own
    empty-latent-is-zero / sampler-generates-noise split -- see module
    docstring) rather than denoising the all-zero tensors those nodes
    actually carry, which would never move under the flow ODE.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MiniMaxH3Sampler",
            display_name="🍏 ASDX MiniMax H3 Sampler",
            category="ASDX/Sampling",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Custom("asdx_minimax_h3_conditioning").Input("conditioning"),
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
                io.Int.Input("steps", default=4, min=1, max=200),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
            ],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: dict,
        conditioning: dict,
        video_latent: dict,
        audio_latent: dict,
        steps: int,
        seed: int,
    ) -> io.NodeOutput:
        if not isinstance(model, dict) or model.get("family") != "minimax_h3":
            raise RuntimeError("ASDX MiniMax H3 Sampler: expected the output of ASDX_MiniMaxH3ModelLoader.")
        if not isinstance(conditioning, dict) or conditioning.get("type") != "minimax_h3":
            raise RuntimeError("ASDX MiniMax H3 Sampler: expected the output of ASDX_MiniMaxH3TextEncode.")
        for name, latent in (("video_latent", video_latent), ("audio_latent", audio_latent)):
            if not isinstance(latent, dict) or "samples" not in latent:
                raise RuntimeError(f"ASDX MiniMax H3 Sampler: expected LATENT input for '{name}'.")

        from .native.minimax_h3.sampling import run_minimax_h3_sampling

        video_shape = tuple(video_latent["samples"].shape)
        audio_shape = tuple(audio_latent["samples"].shape)

        mx.random.seed(seed)
        video_noise = mx.random.normal(video_shape)
        audio_noise = mx.random.normal(audio_shape)

        video_out, audio_out = run_minimax_h3_sampling(
            model["transformer"], video_noise, audio_noise, conditioning["hidden_states"], steps,
        )
        mx.eval(video_out, audio_out)

        device = video_latent["samples"].device
        video_torch = torch.from_numpy(np.array(video_out)).to(device)
        audio_torch = torch.from_numpy(np.array(audio_out)).to(device)

        print(f"[ASDX] MiniMax H3 Sampler: {steps} steps, video={video_shape}, audio={audio_shape}")
        return io.NodeOutput({"samples": video_torch}, {"samples": audio_torch})


NODE_LIST = [
    ASDX_MiniMaxH3EmptyLatentAV,
    ASDX_MiniMaxH3SigmaShift,
    ASDX_MiniMaxH3ModelLoader,
    ASDX_MiniMaxH3TextEncoderLoader,
    ASDX_MiniMaxH3TextEncode,
    ASDX_MiniMaxH3Sampler,
]
