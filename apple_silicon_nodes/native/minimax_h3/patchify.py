"""Video/audio latent <-> packed-row conversions for MiniMax H3.

Ported literally from `comfy/ldm/minimax/model.py::patchify_video`/
`unpatchify_video`/`pack_audio`/`unpack_audio` -- these are pure
reshape/permute operations (MLX's `mx.einsum` supports the same
letter-subscript notation the reference uses, so the port is 1:1, not
reinterpreted through `.transpose()`), verified against the real reference
in `tests/native/minimax_h3/test_patchify.py`.
"""

from __future__ import annotations

import mlx.core as mx


def patchify_video(latent: mx.array, patch_size: tuple[int, int, int] = (1, 2, 2)) -> mx.array:
    """`[1, C, T, H, W]` -> `[T//pt * H//ph * W//pw, C*pt*ph*pw]`."""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = mx.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(
    rows: mx.array, t: int, h: int, w: int, c: int = 24, patch_size: tuple[int, int, int] = (1, 2, 2)
) -> mx.array:
    """Inverse of `patchify_video`: `[T//pt * H//ph * W//pw, C*pt*ph*pw]` ->
    `[1, C, T, H, W]`."""
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = mx.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent: mx.array) -> mx.array:
    """`[1, C=32, channels=2, T]` -> `[channels*T, 32]`, channel-major
    (channel 0's `T` rows first, then channel 1's)."""
    b, c, ch, t = latent.shape
    return latent[0].transpose(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows: mx.array, ch: int = 2) -> mx.array:
    """Inverse of `pack_audio`: `[channels*T, C]` -> `[1, C, channels, T]`."""
    t = rows.shape[0] // ch
    c = rows.shape[-1]
    return rows.reshape(ch, t, c).transpose(2, 0, 1)[None]
