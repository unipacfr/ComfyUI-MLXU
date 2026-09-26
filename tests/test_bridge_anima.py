"""Bridge functions for Anima: conditioning dict -> MLX, noise prep, MLX latent -> Comfy LATENT."""

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
bridge = load_node_module("bridge")

# comfy_stub's comfy.sample submodule is empty by convention (see its
# docstring); prepare_noise_from_latent_anima calls the real
# comfy.sample.prepare_noise at runtime, so the test exercising it supplies a
# minimal stand-in.
import comfy.sample as _comfy_sample_stub


def _stub_prepare_noise(latent_image, seed, noise_inds=None):
    return torch.randn(latent_image.size(), generator=torch.manual_seed(seed))


_comfy_sample_stub.prepare_noise = _stub_prepare_noise


def _cond(hidden=1024, with_ids=True):
    extra = {"pooled_output": None}
    if with_ids:
        extra["t5xxl_ids"] = torch.tensor([10, 20, 1], dtype=torch.int)
        extra["t5xxl_weights"] = torch.tensor([1.0, 1.2, 1.0])
    return [[torch.randn(1, 7, hidden), extra]]


def test_conditioning_extracts_all_three():
    h, ids, w = bridge.conditioning_anima_to_mlx(_cond(), mx.bfloat16)
    assert h.shape == (1, 7, 1024) and h.dtype == mx.bfloat16
    assert ids.shape == (1, 3) and ids.dtype == mx.int32
    np.testing.assert_allclose(np.array(w), [[1.0, 1.2, 1.0]], rtol=1e-6)


def test_conditioning_without_t5_ids_names_the_loader():
    with pytest.raises(RuntimeError, match="ASDX_CLIPLoader"):
        bridge.conditioning_anima_to_mlx(_cond(with_ids=False), mx.bfloat16)


def test_conditioning_wrong_width_names_the_loader():
    with pytest.raises(RuntimeError, match="ASDX_CLIPLoader"):
        bridge.conditioning_anima_to_mlx(_cond(hidden=4096), mx.bfloat16)


def test_conditioning_multi_entry_rejected():
    combined = _cond() + _cond()
    with pytest.raises(RuntimeError, match="ConditioningCombine"):
        bridge.conditioning_anima_to_mlx(combined, mx.bfloat16)


def test_latent_out_dewhitens_wan21():
    z = mx.zeros((1, 16, 4, 4))
    out = bridge.mlx_to_comfy_latent_anima(z, {"samples": z})["samples"]
    assert tuple(out.shape) == (1, 16, 4, 4)
    assert abs(float(out[0, 0, 0, 0]) - (-0.7571)) < 1e-4  # WAN21_LATENTS_MEAN[0]


def test_noise_rejects_non_16_channel_latent():
    with pytest.raises(RuntimeError, match="16-channel"):
        bridge.prepare_noise_from_latent_anima({"samples": torch.zeros(1, 64, 8, 8)}, 0, mx.bfloat16)
