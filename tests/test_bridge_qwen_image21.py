"""Bridge functions for Qwen Image 2.1: conditioning dict -> MLX, MLX latent <-> Comfy LATENT."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
bridge_mod = load_node_module("bridge")

# comfy_stub's comfy.sample submodule is empty by convention (see its
# docstring); prepare_noise_from_latent_qwen_image21 calls the real
# comfy.sample.prepare_noise at runtime, so the tests exercising it supply a
# minimal stand-in.
import comfy.sample as _comfy_sample_stub


def _stub_prepare_noise(latent_image, seed, noise_inds=None):
    return torch.randn(latent_image.size(), generator=torch.manual_seed(seed))


_comfy_sample_stub.prepare_noise = _stub_prepare_noise


def test_conditioning_qwen_image21_to_mlx_adds_batch_dim():
    hidden_states = mx.random.normal((7, 4096))
    conditioning = {"type": "qwen_image21", "hidden_states": hidden_states, "text": "a cat"}
    cond = bridge_mod.conditioning_qwen_image21_to_mlx(conditioning, mx.float16)
    assert cond.shape == (1, 7, 4096)
    assert cond.dtype == mx.float16


def test_conditioning_qwen_image21_to_mlx_rejects_wrong_type():
    with pytest.raises(RuntimeError, match="qwen_image21"):
        bridge_mod.conditioning_qwen_image21_to_mlx({"type": "flux2"}, mx.float16)


def test_mlx_to_comfy_latent_qwen_image21_shape():
    latents = mx.random.normal((1, 64, 8, 8))
    template = {"samples": None}
    out = bridge_mod.mlx_to_comfy_latent_qwen_image21(latents, template)
    assert isinstance(out["samples"], torch.Tensor)
    assert tuple(out["samples"].shape) == (1, 64, 8, 8)


def test_mlx_to_comfy_latent_qwen_image21_applies_per_channel_denormalization():
    """The DiT samples in QwenImage21's normalized ("whitened") latent space --
    `mlx_to_comfy_latent_qwen_image21` must apply the per-channel `* std + mean`
    de-whitening (matching `comfy/samplers.py`'s `process_latent_out`) before
    VAE decode, same bug class as Wan21/Krea2 (see `native/config.py::
    process_wan21_latent_out`). A zero-valued DiT output must come out as
    exactly `latents_mean` per channel, not zero."""
    from apple_silicon_nodes.native.config import QWEN_IMAGE21_LATENTS_MEAN

    latents = mx.zeros((1, 64, 4, 4))
    out = bridge_mod.mlx_to_comfy_latent_qwen_image21(latents, {"samples": None})
    samples = out["samples"]
    expected = torch.tensor(QWEN_IMAGE21_LATENTS_MEAN, dtype=torch.float32).view(1, 64, 1, 1)
    assert torch.allclose(samples, expected.expand_as(samples), atol=1e-4)


def test_prepare_noise_from_latent_qwen_image21_shape_and_dims():
    samples = torch.zeros((1, 64, 8, 8))
    latent = {"samples": samples}
    noise, height, width, out_shape = bridge_mod.prepare_noise_from_latent_qwen_image21(
        latent, seed=0, precision=mx.float32
    )
    assert noise.shape == (1, 64, 8, 8)
    assert height == 8 * 16 and width == 8 * 16
    assert out_shape == (8, 8)


def test_prepare_noise_from_latent_qwen_image21_rejects_wrong_channels():
    latent = {"samples": torch.zeros((1, 4, 8, 8))}
    with pytest.raises(RuntimeError, match="64-channel"):
        bridge_mod.prepare_noise_from_latent_qwen_image21(latent, seed=0, precision=mx.float32)
