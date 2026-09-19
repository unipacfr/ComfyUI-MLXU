"""Parity of vision_preprocess.preprocess_image against the real
comfy.text_encoders.qwen_vl.process_qwen2vl_images."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_qwen3vl
from support.minimax_h3_module_loader import load_native_module

pre = load_native_module("minimax_h3.vision_preprocess")


@pytest.mark.parametrize("shape", [(224, 336), (250, 333), (48, 64), (1100, 900)])
def test_matches_comfyui(shape):
    _, qwen_vl, _ = load_real_comfy_qwen3vl()
    import torch

    rng = np.random.default_rng(0)
    image = rng.random((shape[0], shape[1], 3), dtype=np.float32)
    ref_patches, ref_grid = qwen_vl.process_qwen2vl_images(
        torch.from_numpy(image)[None], patch_size=16, image_mean=[0.5] * 3, image_std=[0.5] * 3
    )
    patches, grid = pre.preprocess_image(image)
    assert grid == tuple(int(v) for v in ref_grid[0].tolist())
    assert patches.shape == tuple(ref_patches.shape)
    assert np.abs(patches - ref_patches.numpy()).max() < 1e-6


def test_min_pixels_upscales_tiny_images():
    patches, grid = pre.preprocess_image(np.zeros((10, 10, 3), dtype=np.float32))
    assert grid[1] * grid[2] * 16 * 16 >= 3136


def test_rejects_non_rgb():
    with pytest.raises(ValueError, match="3 channels"):
        pre.preprocess_image(np.zeros((64, 64, 4), dtype=np.float32))
