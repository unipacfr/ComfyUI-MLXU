"""Packed-sequence layout: position ids (for RoPE) and segment spans (for
adaLN modulation row assignment) for MiniMax H3's `[text | audio | video]`
packed sequence.

Ported from `comfy/ldm/minimax/model.py::PackedLayout` and its helpers
(`_frame_grid`, `_video_t_grid`, `_video_grid`, `_audio_grid`,
`_axis_from_sqrt_area`), restricted to the minimal t2va path this project
targets first: no reference blocks, no extra keyframes/guide frames, no
inpainting denoise masks. Those add more segments (`cond`, `ref_img`,
`ref_audio`) before the fixed trailing `(audio, video)` pair this module
always produces -- deliberately not accepted as parameters here (rather than
accepted and silently ignored) so the API is honest about what it covers;
see `MiniMaxH3ReferenceToVideo`/`MiniMaxH3AddGuide` in
`comfy_extras/nodes_minimax_h3.py` for the nodes that would need it, both
out of scope per `docs/plan-multi-modeles-apple-silicon.md` §5 Phase 6.

Position coordinates are computed in float64 (matching the reference)
since they encode fine-grained sub-pixel/sub-frame offsets that RoPE's
`sin`/`cos` are sensitive to -- `rope_freqs` casts down to float32 only at
the last step, mirroring `MiniMaxH3Model.rope_freqs`.
"""

from __future__ import annotations

import math

import mlx.core as mx

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> mx.array:
    ratio = dim / sqrt_area
    n = dim // patch
    idx = mx.arange(n, dtype=mx.float64)
    return (idx * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def frame_grid(h: int, w: int) -> tuple[mx.array, mx.array]:
    """`[H, W]` 2x2-patch grid -> `([H//2 * W//2, 2], [W//2])` (h, w)
    area-normalized coordinates and the w-axis alone (needed separately for
    the target audio stream's stereo-channel w extremes).

    Runs on the CPU stream: float64 (needed for sub-pixel position
    precision, matching the reference) is not supported on MLX's GPU/Metal
    stream at all -- not even elementwise ops on an already-created float64
    array, confirmed by testing directly on this machine. Every function
    below that touches float64 does the same."""
    with mx.stream(mx.cpu):
        area = math.sqrt(h * w)
        h_axis = _axis_from_sqrt_area(h, 2, area)
        w_axis = _axis_from_sqrt_area(w, 2, area)
        hh = mx.repeat(h_axis[:, None], w_axis.shape[0], axis=1)  # [H//2, W//2]
        ww = mx.repeat(w_axis[None, :], h_axis.shape[0], axis=0)  # [H//2, W//2]
        frame = mx.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1)  # [H//2*W//2, 2]
        return frame, w_axis


def video_t_spans(n: int) -> list[float]:
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def video_t_grid(n: int, origin: float) -> mx.array:
    with mx.stream(mx.cpu):
        spans = mx.array(video_t_spans(n), dtype=mx.float64)
        cumsum = mx.concatenate([mx.zeros((1,), dtype=mx.float64), mx.cumsum(spans[:-1])])
        return origin + cumsum


def video_grid(vt: int, frame: mx.array, cursor: float) -> mx.array:
    """`[vt * frame_rows, 3]` (t, h, w) positions for `vt` video-latent
    frames sharing spatial grid `frame`, starting at `cursor` on the time axis."""
    with mx.stream(mx.cpu):
        frame_rows = frame.shape[0]
        t_col = mx.broadcast_to(video_t_grid(vt, cursor)[:, None, None], (vt, frame_rows, 1))
        hw = mx.broadcast_to(frame[None], (vt, frame_rows, 2))
        return mx.concatenate([t_col, hw], axis=-1).reshape(-1, 3)


def audio_grid(cursor: float, t: int, w_low: float, w_high: float) -> mx.array:
    """`[t*2, 3]` positions for `t` stereo audio-latent frames: both channels
    share the same time axis (tiled, not interleaved -- channel-major, matching
    `pack_audio`'s row order), distinguished only by their w coordinate."""
    with mx.stream(mx.cpu):
        t_vals = cursor + mx.arange(t, dtype=mx.float64)
        t_col = mx.concatenate([t_vals, t_vals])[:, None]
        h_col = mx.zeros((2 * t, 1), dtype=mx.float64)
        w_col = mx.concatenate(
            [mx.full((t, 1), w_low, dtype=mx.float64), mx.full((t, 1), w_high, dtype=mx.float64)], axis=0
        )
        return mx.concatenate([t_col, h_col, w_col], axis=-1)


class PackedLayout:
    """Static packed-sequence structure for one `(text_len, latent_t,
    latent_h, latent_w, audio_t)` shape signature: minimal t2va path only
    (`[text | audio | video]`, see module docstring)."""

    def __init__(self, text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int):
        # float64 (needed for sub-pixel/sub-frame position precision, matching
        # the reference) is CPU-only in MLX -- these are tiny, once-per-forward
        # arrays, so the whole computation runs on the CPU stream.
        with mx.stream(mx.cpu):
            frame, w_axis = frame_grid(latent_h, latent_w)
            frame_rows = frame.shape[0]
            target_audio_w = (float(w_axis[0].item()), float(w_axis[-1].item()))
            cursor = float(text_len)

            text_pos = mx.zeros((text_len, 3), dtype=mx.float64)
            text_pos[:, 0] = mx.arange(text_len, dtype=mx.float64)

            audio_pos = audio_grid(cursor, audio_t, *target_audio_w)
            n_video = latent_t * frame_rows
            video_pos = video_grid(latent_t, frame, cursor)

            self.position_ids = mx.concatenate([text_pos, audio_pos, video_pos], axis=0)

        row = 0
        segments: list[tuple[int, int, str]] = []
        for kind, n in (("text", text_len), ("audio", audio_t * 2), ("video", n_video)):
            segments.append((row, row + n, kind))
            row += n
        self.segments = segments
        self.seq_len = row
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
