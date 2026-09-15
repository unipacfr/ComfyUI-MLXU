"""Tests for ASDX_MiniMaxH3EmptyLatentAV / ASDX_MiniMaxH3SigmaShift.

Runs in the bare project venv via ``comfy_stub.load_node_module``, same
pattern as ``test_depth_conditioning_node.py`` / ``test_vae_decode_audio.py``.
"""

from __future__ import annotations

import types
from unittest.mock import Mock

import pytest
import torch

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
nodes_module = load_node_module("minimax_h3_nodes")
ASDX_MiniMaxH3EmptyLatentAV = nodes_module.ASDX_MiniMaxH3EmptyLatentAV
ASDX_MiniMaxH3SigmaShift = nodes_module.ASDX_MiniMaxH3SigmaShift

# comfy_stub's comfy.model_management is an empty module -- tests touching a
# real symbol must set it themselves (see comfy_stub.py's _build_comfy docstring).
import comfy.model_management  # noqa: E402

comfy.model_management.intermediate_dtype = lambda: torch.float32


def test_temporal_shape_snaps_to_17k_plus_5_grid():
    # 100 is not on the grid (100 % 17 == 15, needs 5) -- must round up.
    frame_count, latent_t, audio_t = nodes_module._temporal_shape(100)
    assert frame_count % 17 == 5
    assert frame_count >= 100


def test_temporal_shape_matches_reference_formula_at_known_point():
    # length=124 is the node's own documented "~5s" default; cross-check the
    # exact reference formulas (align_frame_count/video_latent_t) by hand.
    frame_count, latent_t, audio_t = nodes_module._temporal_shape(124)
    assert frame_count == 124  # already on the 17k+5 grid (124 = 17*7 + 5)
    assert latent_t == ((124 - 5) // 17) * 5 + 2
    assert audio_t == round((124 / 24) * 40)


def test_empty_latent_av_shapes():
    result = ASDX_MiniMaxH3EmptyLatentAV.execute(width=1344, height=768, length=124)
    video, audio = result.values

    frame_count, latent_t, audio_t = nodes_module._temporal_shape(124)
    assert video["samples"].shape == (1, 24, latent_t, 768 // 16, 1344 // 16)
    assert audio["samples"].shape == (1, 32, 2, audio_t)
    assert bool(torch.all(video["samples"] == 0))
    assert bool(torch.all(audio["samples"] == 0))


def test_empty_latent_av_rejects_batch_size_other_than_one():
    with pytest.raises(RuntimeError, match="batch_size must be 1"):
        ASDX_MiniMaxH3EmptyLatentAV.execute(width=1344, height=768, length=124, batch_size=2)


def _minimax_model_dict(sigma_shift_video=12.0, sigma_shift_audio=3.0):
    config = types.SimpleNamespace(
        sigma_shift_video=sigma_shift_video,
        sigma_shift_audio=sigma_shift_audio,
        hidden_size=5376,
    )
    return {"type": "asdx_model", "transformer": Mock(), "config": config, "capability": Mock()}


def test_sigma_shift_rejects_non_asdx_model():
    with pytest.raises(RuntimeError, match="ASDX MODEL dict"):
        ASDX_MiniMaxH3SigmaShift.execute({"not_a_model": True}, 12.0, 3.0)


def test_sigma_shift_rejects_non_minimax_config():
    model = {"transformer": Mock(), "config": types.SimpleNamespace(hidden_size=4096)}
    with pytest.raises(RuntimeError, match="not a MiniMax H3 model"):
        ASDX_MiniMaxH3SigmaShift.execute(model, 12.0, 3.0)


def test_sigma_shift_overrides_config_values_without_mutating_input():
    model = _minimax_model_dict()
    result = ASDX_MiniMaxH3SigmaShift.execute(model, 20.0, 5.0)
    new_model = result.values[0]

    assert new_model["config"].sigma_shift_video == 20.0
    assert new_model["config"].sigma_shift_audio == 5.0
    # original model dict/config untouched
    assert model["config"].sigma_shift_video == 12.0
    assert model["config"].sigma_shift_audio == 3.0
    # other keys (transformer, capability) carried through unchanged
    assert new_model["transformer"] is model["transformer"]
