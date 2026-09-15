"""MiniMax H3 nodes -- t2va only (see `native/minimax_h3/model.py`'s module
docstring for the scope restriction this mirrors exactly: no reference
conditioning, no keyframes/first_frame/last_frame, no VSA gate_compress, no
PDD head bank).

Ported from `comfy_extras/nodes_minimax_h3.py`'s `EmptyMiniMaxH3LatentAV` and
`MiniMaxH3SigmaShift`. `MiniMaxH3ImageToVideo` (the reference's third
in-scope node per this project's own workflow-derived node survey, see
`docs/plan-multi-modeles-apple-silicon.md` §5 Phase 6) is NOT ported here --
it accepts `first_frame`/`last_frame` keyframe conditioning that
`native/minimax_h3/layout.py::PackedLayout` does not implement, so exposing
those inputs would silently do nothing. A text-only prompt-conditioning node
needs the real Qwen3-VL tokenizer wired through a `comfy.sd.CLIP` object
(same pattern `krea2_grounded_encode.py` uses: `mlx_clip.tokenize(prompt)`
for real BPE/chat-template handling, native MLX encoder for the forward
pass) -- left as a follow-up rather than guessed at.

Two separate LATENT outputs (video, audio) rather than ComfyUI's own packed
`NestedTensor` pair: this project's own (not-yet-built) MiniMax H3 sampler
consumes plain tensors directly, and two standard `{"samples": tensor}`
dicts already compose with the existing `ASDX_VAEDecode`/
`ASDX_VAEDecodeAudio` nodes with no new plumbing.
"""

from __future__ import annotations

from typing import Any

import torch

import comfy.model_management
from comfy_api.latest import io

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


NODE_LIST = [
    ASDX_MiniMaxH3EmptyLatentAV,
    ASDX_MiniMaxH3SigmaShift,
]
