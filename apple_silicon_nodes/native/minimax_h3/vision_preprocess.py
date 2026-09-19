"""Image -> Qwen3-VL vision patches, ported 1:1 from
`comfy/text_encoders/qwen_vl.py::process_qwen2vl_images` (image path,
temporal_patch_size frames = the same frame repeated).

Runs on the host with torch: the bilinear `F.interpolate(align_corners=False)`
is what the reference (and the released model) uses, and a single small image
per call makes a device round trip irrelevant next to matching its numerics.
This is the only torch use in the vision path.
"""

from __future__ import annotations

import math

import numpy as np


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
    """`image`: float32 `[H, W, 3]` in `[0, 1]`. Returns `(patches, grid)`:
    `patches` float32 `[gh*gw, 3*temporal_patch_size*patch_size**2]` in
    merge-block order, `grid = (1, gh, gw)`."""
    import torch
    import torch.nn.functional as F

    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"ASDX: vision preprocessing needs 3 channels, got {channels}")

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

    img = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1)
    resized = F.interpolate(img.unsqueeze(0), size=(h_bar, w_bar), mode="bilinear", align_corners=False).squeeze(0)
    mean = torch.tensor(image_mean, dtype=resized.dtype).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=resized.dtype).view(3, 1, 1)
    normalized = (resized - mean) / std

    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    frames = normalized.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
    patches = frames.reshape(
        1, temporal_patch_size, 3,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    return flat.numpy(), (1, grid_h, grid_w)
