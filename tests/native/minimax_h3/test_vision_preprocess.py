"""Parity of vision_preprocess.preprocess_image against the real
comfy.text_encoders.qwen_vl.process_qwen2vl_images."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_qwen3vl, load_real_comfy_text_encoders
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


@pytest.mark.parametrize("shape", [(224, 336), (250, 333), (48, 64)])
def test_video_block_matches_comfyui(shape):
    minimax, _, _, _ = load_real_comfy_text_encoders()
    import torch

    rng = np.random.default_rng(1)
    frames = rng.random((2, shape[0], shape[1], 3), dtype=np.float32)
    ref_patches, ref_grid = minimax.process_video_block(torch.from_numpy(frames))
    patches, grid = pre.preprocess_video_block(frames)
    assert grid == tuple(int(v) for v in ref_grid[0].tolist())
    assert patches.shape == tuple(ref_patches.shape)
    assert np.abs(patches - ref_patches.numpy()).max() < 1e-6


def test_video_block_uses_two_distinct_frames():
    rng = np.random.default_rng(2)
    a = rng.random((1, 64, 64, 3), dtype=np.float32)
    b = rng.random((1, 64, 64, 3), dtype=np.float32)
    distinct, _ = pre.preprocess_video_block(np.concatenate([a, b]))
    repeated, _ = pre.preprocess_video_block(np.concatenate([a, a]))
    assert np.abs(distinct - repeated).max() > 0.1  # null case: the frames must matter


def test_video_block_rejects_wrong_frame_count():
    with pytest.raises(ValueError, match="2 frames"):
        pre.preprocess_video_block(np.zeros((3, 64, 64, 3), dtype=np.float32))


def test_image_accepts_4d_comfy_layout():
    rng = np.random.default_rng(3)
    image = rng.random((96, 128, 3), dtype=np.float32)
    a, ga = pre.preprocess_image(image)
    b, gb = pre.preprocess_image(image[None])
    assert ga == gb and np.array_equal(a, b)


def test_image_rejects_batches_and_other_ranks():
    with pytest.raises(ValueError, match="3 channels"):
        pre.preprocess_image(np.zeros((2, 64, 64, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="3 channels"):
        pre.preprocess_image(np.zeros((64, 64), dtype=np.float32))
