"""Image / video block -> Qwen3-VL vision patches, ported 1:1 from
`comfy/text_encoders/qwen_vl.py::process_qwen2vl_images` (image path: one
frame repeated `temporal_patch_size` times) and
`comfy/text_encoders/minimax.py::process_video_block` (video path: two
DISTINCT frames fill the temporal patch).

Runs on the host with torch: the bilinear `F.interpolate(align_corners=False)`
is what the reference (and the released model) uses, and a handful of small
images per call makes a device round trip irrelevant next to matching its
numerics. This is the only torch use in the vision path.
"""

from __future__ import annotations

import math

import numpy as np


def _target_size(
    height: int, width: int, *, patch_size: int, merge_size: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Smart-resize target `(h_bar, w_bar)`, identical in both reference paths."""
    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _as_hwc(image: np.ndarray) -> np.ndarray:
    """Accept `[H, W, 3]` or a single ComfyUI IMAGE `[1, H, W, 3]`."""
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"ASDX: vision preprocessing needs [H, W, 3] or [1, H, W, 3] (3 channels), got shape {image.shape}"
        )
    return image


def _patchify(
    frames_chw: "torch.Tensor",  # [T, 3, h_bar, w_bar] normalized
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    h_bar, w_bar = frames_chw.shape[-2:]
    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    patches = frames_chw.reshape(
        1, temporal_patch_size, 3,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    return flat.numpy(), (1, grid_h, grid_w)


def _resize_normalize(
    frames: np.ndarray, h_bar: int, w_bar: int, image_mean, image_std
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    chw = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32)).permute(0, 3, 1, 2)
    resized = F.interpolate(chw, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(image_mean, dtype=resized.dtype).view(1, 3, 1, 1)
    std = torch.tensor(image_std, dtype=resized.dtype).view(1, 3, 1, 1)
    return (resized - mean) / std


def preprocess_image(
    image: np.ndarray,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """`image`: float32 `[H, W, 3]` (or `[1, H, W, 3]`) in `[0, 1]`. Returns
    `(patches, grid)`: `patches` float32 `[gh*gw, 3*temporal_patch_size*
    patch_size**2]` in merge-block order, `grid = (1, gh, gw)`."""
    image = _as_hwc(image)
    h_bar, w_bar = _target_size(
        image.shape[0], image.shape[1],
        patch_size=patch_size, merge_size=merge_size, min_pixels=min_pixels, max_pixels=max_pixels,
    )
    normalized = _resize_normalize(image[None], h_bar, w_bar, image_mean, image_std)  # [1, 3, h, w]
    frames = normalized.repeat(temporal_patch_size, 1, 1, 1)
    return _patchify(frames, patch_size=patch_size, temporal_patch_size=temporal_patch_size, merge_size=merge_size)


def preprocess_video_block(
    frames: np.ndarray,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """`frames`: float32 `[temporal_patch_size, H, W, 3]` in `[0, 1]` (a 2-frame
    video block, sampled at 2 fps). Same resize/normalize policy as
    `preprocess_image`, but the frames fill the temporal patch (`grid_t = 1`)."""
    if frames.ndim != 4 or frames.shape[0] != temporal_patch_size or frames.shape[-1] != 3:
        raise ValueError(
            f"ASDX: a video block needs exactly {temporal_patch_size} frames of [H, W, 3] (3 channels), "
            f"got shape {frames.shape}"
        )
    h_bar, w_bar = _target_size(
        frames.shape[1], frames.shape[2],
        patch_size=patch_size, merge_size=merge_size, min_pixels=min_pixels, max_pixels=max_pixels,
    )
    normalized = _resize_normalize(frames, h_bar, w_bar, image_mean, image_std)
    return _patchify(normalized, patch_size=patch_size, temporal_patch_size=temporal_patch_size, merge_size=merge_size)
