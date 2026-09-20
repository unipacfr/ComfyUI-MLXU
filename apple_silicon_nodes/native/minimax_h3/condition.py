"""Condition inputs for MiniMax H3's packed DiT (fl2va keyframes, ref2va
references): the payload types, and the once-per-run preparation of the
condition rows (patchify + noise augmentation).

Ported from `comfy/ldm/minimax/model.py::MiniMaxH3Model._cond_video_rows` /
`_cond_audio_rows`. The augmented anchor is `aug * z + (1 - aug) * noise`
(aug 0.999 for video, 1.0 = clean for audio), and every condition restarts
the SAME noise stream (audio at seed + 1). The noise is identical at every
sampling step, so `prepare_condition` draws it once per run. The stream
cannot match torch's; `noise_fn` lets tests inject the reference's draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import numpy as np

from .patchify import pack_audio, patchify_video

VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


@dataclass(frozen=True)
class KeyframeCond:
    """An fl2va anchor: `latent` `[1, C, T, h, w]` and/or `audio_latent`
    `[1, A, 2, T]` (normalized), pinned at pixel frame `resolved_frame_index`: an index on
    the model's 17k+5 frame grid at 24 fps, converted to the DiT time axis by
    `FRAME_RESCALE = 5/3` (layout.py). A keyframe `latent` must share the target's spatial
    latent grid (`h`, `w`); `PackedLayout` enforces it."""

    resolved_frame_index: int
    latent: mx.array | None = None
    audio_latent: mx.array | None = None


@dataclass(frozen=True)
class RefBlock:
    """A ref2va reference. `kind`: "image" | "video" | "video_audio" | "audio".
    `latent_t/h/w` are the video latent dims (h, w before the 2x2 patching);
    `ref_audio_t` the reference audio length in latent frames."""

    kind: str
    latent: mx.array | None = None
    audio_latent: mx.array | None = None
    latent_t: int = 0
    latent_h: int = 0
    latent_w: int = 0
    ref_audio_t: int = 0


@dataclass(frozen=True)
class ConditionPayload:
    text_token_tags: np.ndarray | None = None  # [text_len], 0 = vision row (video modality), 1 = text
    keyframes: tuple[KeyframeCond, ...] = ()
    refs: tuple[RefBlock, ...] = ()
    seed: int = 0
    visual_cond_noise_aug: float = VISUAL_COND_TIMESTEP
    audio_cond_noise_aug: float = AUDIO_COND_TIMESTEP


@dataclass(frozen=True)
class PreparedCondition:
    payload: ConditionPayload
    video_rows: mx.array | None  # [Nc, patch_dim], keyframes' rows then refs' rows
    audio_rows: mx.array | None  # [Na, audio_dim]


def default_noise(shape: tuple[int, ...], seed: int) -> mx.array:
    return mx.random.normal(tuple(shape), key=mx.random.key(int(seed)))


def _augment(rows: mx.array, aug: float, seed: int, noise_fn: Callable) -> mx.array:
    if aug >= 1.0:
        return rows
    noise = noise_fn(tuple(rows.shape), seed)
    return aug * rows + (1.0 - aug) * noise


def prepare_condition(
    payload: ConditionPayload,
    patch_size: tuple[int, int, int],
    noise_fn: Callable[[tuple[int, ...], int], mx.array] = default_noise,
) -> PreparedCondition:
    video = [k.latent for k in payload.keyframes if k.latent is not None]
    video += [r.latent for r in payload.refs if r.latent is not None]
    audio = [k.audio_latent for k in payload.keyframes if k.audio_latent is not None]
    audio += [r.audio_latent for r in payload.refs if r.audio_latent is not None]

    video_rows = [
        _augment(patchify_video(z.astype(mx.float32), patch_size), payload.visual_cond_noise_aug, payload.seed, noise_fn)
        for z in video
    ]
    audio_rows = [
        _augment(pack_audio(z.astype(mx.float32)), payload.audio_cond_noise_aug, payload.seed + 1, noise_fn)
        for z in audio
    ]
    return PreparedCondition(
        payload=payload,
        video_rows=mx.concatenate(video_rows, axis=0) if video_rows else None,
        audio_rows=mx.concatenate(audio_rows, axis=0) if audio_rows else None,
    )
