"""Tests for ASDX_MiniMaxH3Sampler.

Runs in the bare project venv via ``comfy_stub.load_node_module``.
``run_minimax_h3_sampling`` itself is monkeypatched out (it already has its
own real-formula-verified tests in ``test_sampling.py``); this file only
exercises the node-level plumbing: input validation, noise generation from
the LATENT shapes, and the mx<->torch bridge.
"""

from __future__ import annotations

import sys
import types

import mlx.core as mx
import pytest
import torch

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
nodes_module = load_node_module("minimax_h3_nodes")
ASDX_MiniMaxH3Sampler = nodes_module.ASDX_MiniMaxH3Sampler


def _minimax_model():
    return {"type": "asdx_model", "family": "minimax_h3", "transformer": object()}


def _minimax_conditioning():
    return {"type": "minimax_h3", "hidden_states": mx.zeros((3, 8))}


def test_rejects_wrong_model_type():
    with pytest.raises(RuntimeError, match="ASDX_MiniMaxH3ModelLoader"):
        ASDX_MiniMaxH3Sampler.execute(
            {"family": "flux2"}, _minimax_conditioning(),
            {"samples": torch.zeros(1, 4, 1, 2, 2)}, {"samples": torch.zeros(1, 4, 2, 2)},
            steps=4, seed=0,
        )


def test_rejects_wrong_conditioning_type():
    with pytest.raises(RuntimeError, match="ASDX_MiniMaxH3TextEncode.*ASDX_MiniMaxH3ImageToVideo.*ASDX_MiniMaxH3ReferenceToVideo"):
        ASDX_MiniMaxH3Sampler.execute(
            _minimax_model(), {"type": "clip"},
            {"samples": torch.zeros(1, 4, 1, 2, 2)}, {"samples": torch.zeros(1, 4, 2, 2)},
            steps=4, seed=0,
        )


def test_rejects_non_latent_input():
    with pytest.raises(RuntimeError, match="LATENT input for 'audio_latent'"):
        ASDX_MiniMaxH3Sampler.execute(
            _minimax_model(), _minimax_conditioning(),
            {"samples": torch.zeros(1, 4, 1, 2, 2)}, {"not_samples": None},
            steps=4, seed=0,
        )


def test_generates_noise_matching_latent_shapes_and_bridges_to_torch(monkeypatch):
    captured = {}

    def fake_run(model, video_noise, audio_noise, hidden_states, steps, payload=None):
        captured["video_shape"] = video_noise.shape
        captured["audio_shape"] = audio_noise.shape
        captured["steps"] = steps
        # return recognizable, non-zero output to check the torch bridge
        return video_noise + 1.0, audio_noise + 2.0

    sampling_stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.sampling")
    sampling_stub.run_minimax_h3_sampling = fake_run
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.sampling", sampling_stub)

    video_latent = {"samples": torch.zeros(1, 24, 2, 4, 4)}
    audio_latent = {"samples": torch.zeros(1, 32, 2, 5)}

    result = ASDX_MiniMaxH3Sampler.execute(
        _minimax_model(), _minimax_conditioning(), video_latent, audio_latent, steps=6, seed=42,
    )
    video_out, audio_out = result.values

    assert captured["video_shape"] == (1, 24, 2, 4, 4)
    assert captured["audio_shape"] == (1, 32, 2, 5)
    assert captured["steps"] == 6

    assert isinstance(video_out["samples"], torch.Tensor)
    assert isinstance(audio_out["samples"], torch.Tensor)
    assert tuple(video_out["samples"].shape) == (1, 24, 2, 4, 4)
    assert tuple(audio_out["samples"].shape) == (1, 32, 2, 5)


def test_same_seed_produces_same_starting_noise(monkeypatch):
    seen_noise = []

    def fake_run(model, video_noise, audio_noise, hidden_states, steps, payload=None):
        seen_noise.append(np_copy(video_noise))
        return video_noise, audio_noise

    sampling_stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.sampling")
    sampling_stub.run_minimax_h3_sampling = fake_run
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.sampling", sampling_stub)

    video_latent = {"samples": torch.zeros(1, 4, 1, 2, 2)}
    audio_latent = {"samples": torch.zeros(1, 4, 2, 2)}

    ASDX_MiniMaxH3Sampler.execute(_minimax_model(), _minimax_conditioning(), video_latent, audio_latent, steps=1, seed=7)
    ASDX_MiniMaxH3Sampler.execute(_minimax_model(), _minimax_conditioning(), video_latent, audio_latent, steps=1, seed=7)

    import numpy as np
    assert np.array_equal(seen_noise[0], seen_noise[1])


def np_copy(arr: mx.array):
    import numpy as np
    return np.array(arr).copy()


def _run_capturing_payload(monkeypatch, conditioning, seed):
    captured = {}

    def fake_run(model, video_noise, audio_noise, hidden_states, steps, payload=None):
        captured["payload"] = payload
        return video_noise, audio_noise

    sampling_stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.sampling")
    sampling_stub.run_minimax_h3_sampling = fake_run
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.sampling", sampling_stub)
    ASDX_MiniMaxH3Sampler.execute(
        _minimax_model(), conditioning,
        {"samples": torch.zeros(1, 4, 1, 2, 2)}, {"samples": torch.zeros(1, 4, 2, 2)}, steps=1, seed=seed,
    )
    return captured["payload"]


def test_sampler_passes_seed_and_token_tags_to_the_run(monkeypatch):
    conditioning = {**_minimax_conditioning(), "token_tags": mx.array([1, 1, 0])}
    payload = _run_capturing_payload(monkeypatch, conditioning, seed=1234)
    assert payload.seed == 1234
    assert payload.text_token_tags.tolist() == [1, 1, 0]


def test_sampler_passes_keyframes_and_refs_to_the_run(monkeypatch):
    keyframe, ref = object(), object()
    conditioning = {**_minimax_conditioning(), "keyframes": [keyframe], "refs": [ref]}
    payload = _run_capturing_payload(monkeypatch, conditioning, seed=5)
    assert payload.keyframes == (keyframe,) and payload.keyframes[0] is keyframe
    assert payload.refs == (ref,) and payload.refs[0] is ref
    assert payload.text_token_tags is None


def test_sampler_passes_no_payload_for_plain_conditioning(monkeypatch):
    assert _run_capturing_payload(monkeypatch, _minimax_conditioning(), seed=5) is None
