"""M-RoPE for the vision-grounded Qwen3-VL text encoder.

`mrope_position_ids` is a literal port of
`comfy/text_encoders/qwen_vl.py::qwen2vl_mrope_position_ids` (text runs are
sequential, each vision span gets its own T/H/W grid positions, and every
later token shifts by the span's compression). `interleaved_mrope_cos_sin`
ports the `interleaved_mrope=True` branch of
`comfy/text_encoders/llama.py::precompute_freqs_cis`: T-frequencies by
default, with the H and W frequencies replacing every 3rd dimension.
Text-only prompts never reach here (their positions stay `[1, S]` and the
plain single-axis RoPE in `text_encoder_rope.py` applies).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def mrope_position_ids(infos: list[dict], seq_len: int) -> np.ndarray | None:
    """`infos`: one `{"index", "size", "grid": (t, h, w)}` per vision span, in
    sequence order (`index` = first row of the span, `size` = its row count).
    Returns float32 `[3, seq_len]`, or `None` when there is no vision span."""
    if not infos:
        return None
    position_ids = np.zeros((3, seq_len), dtype=np.float32)
    offset = 0
    for n, info in enumerate(infos):
        t, h, w = info["grid"]
        start, size = info["index"], info["size"]
        if n == 0:
            position_ids[:, :start] = np.arange(start, dtype=np.float32)
        end = start + size
        len_max = max(t, h, w) // 2
        start_next = len_max + start
        position_ids[:, end:] = np.arange(
            start_next + offset, start_next + (seq_len - end) + offset, dtype=np.float32
        )
        position_ids[0, start:end] = start + offset
        max_h = h // 2
        position_ids[1, start:end] = np.repeat(
            np.arange(start + offset, start + max_h + offset), math.ceil(size / max_h)
        )[:size]
        max_w = w // 2
        position_ids[2, start:end] = np.tile(
            np.arange(start + offset, start + max_w + offset), math.ceil(size / max_w)
        )[:size]
        offset += len_max - size
    return position_ids


def interleaved_mrope_cos_sin(
    position_ids: np.ndarray,
    head_dim: int,
    theta: float,
    rope_dims: tuple[int, int, int] = (24, 20, 20),
) -> tuple[mx.array, mx.array]:
    """`position_ids`: `[3, S]`. Returns `(cos, sin)`, float32 `[S, head_dim//2]`
    (angles are not duplicated, same convention as `qwen3_rope_cos_sin`)."""
    half = head_dim // 2
    if sum(rope_dims) != half:
        raise ValueError(f"ASDX: rope_dims {rope_dims} must sum to head_dim // 2 = {half}")
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = position_ids[:, :, None].astype(np.float32) * inv_freq[None, None, :]  # [3, S, half]
    inter = freqs[0].copy()
    for axis, offset in ((1, 1), (2, 2)):
        idx = slice(offset, rope_dims[axis] * 3, 3)
        inter[..., idx] = freqs[axis][..., idx]
    return mx.array(np.cos(inter)), mx.array(np.sin(inter))
