"""Packed-sequence layout: position ids (for RoPE) and segment spans (for
adaLN modulation row assignment) for MiniMax H3's `[text | audio | video]`
packed sequence.

Ported from `comfy/ldm/minimax/model.py::PackedLayout` and its helpers
(`_frame_grid`, `_video_t_grid`, `_video_grid`, `_audio_grid`,
`_axis_from_sqrt_area`). Supports keyframe (fl2va) and reference (ref2va)
condition segments; see PackedLayout. No denoise masks.

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
_REF_KINDS = ("image", "audio", "video", "video_audio")


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


def ref_t_span(blk) -> float:
    """Time-axis span a reference block occupies ahead of the target streams
    (`comfy/ldm/minimax/model.py::_ref_t_span`)."""
    if blk.kind == "image":
        return 1.0
    if blk.kind == "audio":
        return float(blk.ref_audio_t)
    if blk.kind in ("video", "video_audio"):
        return max(float(blk.ref_audio_t), sum(video_t_spans(blk.latent_t)))
    raise ValueError(f"ASDX: unknown reference kind '{blk.kind}' (expected one of {_REF_KINDS})")


def _validate_conditions(keyframes, refs, latent_h: int, latent_w: int) -> None:
    """Fail closed when a block's declared dims disagree with its latents: the layout
    sizes segments from the declared dims while `prepare_condition` builds rows from the
    latents, so a mismatch would silently shift every later block's rows. Blocks without
    latents (text-only references) are legal."""
    for i, kf in enumerate(keyframes):
        if kf.latent is not None and tuple(kf.latent.shape[3:]) != (latent_h, latent_w):
            raise ValueError(
                f"ASDX: keyframe {i} latent grid {tuple(kf.latent.shape[3:])} must equal the target "
                f"latent grid {(latent_h, latent_w)}"
            )
    for i, blk in enumerate(refs):
        if blk.kind not in _REF_KINDS:
            raise ValueError(f"ASDX: reference block {i} has unknown kind '{blk.kind}' (expected one of {_REF_KINDS})")
        if blk.latent is not None:
            declared = (blk.latent_t, blk.latent_h, blk.latent_w)
            if tuple(blk.latent.shape[2:]) != declared:
                raise ValueError(
                    f"ASDX: reference block {i} latent dims {tuple(blk.latent.shape[2:])} "
                    f"differ from declared (latent_t, latent_h, latent_w) {declared}"
                )
        if blk.audio_latent is not None:
            actual = blk.audio_latent.shape[-1]
            if actual <= 0 or actual != blk.ref_audio_t:
                raise ValueError(
                    f"ASDX: reference block {i} audio latent length {actual} differs from "
                    f"declared ref_audio_t {blk.ref_audio_t} (must be > 0)"
                )


class PackedLayout:
    """Static packed-sequence structure:
    `[text | (keyframe cond rows) | (reference rows) | audio | video]`.
    Depends on the target dims and on the keyframes/refs, so a layout built without
    them must never be reused for a conditioned run (there is no cache key here).

    Ported from `comfy/ldm/minimax/model.py::PackedLayout`. `keyframes` and
    `refs` are the objects of `condition.py` (attribute access). Segment
    kinds: text, cond, cond_audio, ref_img, ref_audio, audio, video. The
    target audio then video are always the last two segments. Positions are
    float64 and built on the CPU stream (float64 is CPU-only in MLX)."""

    def __init__(
        self,
        text_len: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        keyframes=(),
        refs=(),
    ):
        _validate_conditions(keyframes, refs, latent_h, latent_w)
        with mx.stream(mx.cpu):
            frame, w_axis = frame_grid(latent_h, latent_w)
            frame_rows = frame.shape[0]
            target_audio_w = (float(w_axis[0].item()), float(w_axis[-1].item()))

            text_pos = mx.zeros((text_len, 3), dtype=mx.float64)
            text_pos[:, 0] = mx.arange(text_len, dtype=mx.float64)
            pos = [text_pos]
            sizes: list[tuple[str, int]] = [("text", text_len)]

            # references pack between text and the targets: the target timeline starts after their spans
            cursor = float(text_len) + sum(ref_t_span(b) for b in refs)

            for kf in keyframes:
                cond_t = cursor + FRAME_RESCALE * kf.resolved_frame_index
                if kf.latent is not None:
                    vt = kf.latent.shape[2]
                    sizes.append(("cond", vt * frame_rows))
                    pos.append(video_grid(vt, frame, cond_t))
                if kf.audio_latent is not None:
                    rt = kf.audio_latent.shape[-1]
                    sizes.append(("cond_audio", rt * 2))
                    pos.append(audio_grid(cond_t, rt, *target_audio_w))

            ref_cursor = float(text_len)
            for blk in refs:
                if blk.kind == "image":
                    r_frame, _ = frame_grid(blk.latent_h, blk.latent_w)
                    n = r_frame.shape[0]
                    g = mx.concatenate([mx.full((n, 1), ref_cursor, dtype=mx.float64), r_frame], axis=-1)
                    sizes.append(("ref_img", n))
                    pos.append(g)
                    ref_cursor += 1.0
                elif blk.kind == "audio":
                    rt = blk.ref_audio_t
                    if rt > 0:
                        sizes.append(("ref_audio", rt * 2))
                        pos.append(audio_grid(ref_cursor, rt, *target_audio_w))
                    ref_cursor += float(rt)
                elif blk.kind in ("video", "video_audio"):
                    # the block's audio rows pack immediately before its video rows, sharing the cursor origin
                    rt, vt = blk.ref_audio_t, blk.latent_t
                    r_frame, r_w_axis = frame_grid(blk.latent_h, blk.latent_w)
                    if rt > 0:
                        sizes.append(("ref_audio", rt * 2))
                        pos.append(audio_grid(ref_cursor, rt, float(r_w_axis[0].item()), float(r_w_axis[-1].item())))
                    sizes.append(("ref_img", vt * r_frame.shape[0]))
                    pos.append(video_grid(vt, r_frame, ref_cursor))
                    ref_cursor += max(float(rt), sum(video_t_spans(vt)))

            sizes.append(("audio", audio_t * 2))
            pos.append(audio_grid(cursor, audio_t, *target_audio_w))
            sizes.append(("video", latent_t * frame_rows))
            pos.append(video_grid(latent_t, frame, cursor))

            self.position_ids = mx.concatenate(pos, axis=0)

        row = 0
        segments: list[tuple[int, int, str]] = []
        for kind, n in sizes:
            segments.append((row, row + n, kind))
            row += n
        self.segments = segments
        self.seq_len = row
