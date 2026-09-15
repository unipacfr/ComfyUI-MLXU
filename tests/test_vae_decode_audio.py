"""Tests for ``ASDX_VAEDecodeAudio``.

Runs in the bare project venv (no ComfyUI) via ``comfy_stub.load_node_module``,
same pattern as ``test_depth_conditioning_node.py``. Mirrors ComfyUI's own
generic ``comfy_extras.nodes_audio.vae_decode_audio`` contract (AUDIO =
``{"waveform": [B,C,T], "sample_rate": int}``), verified here against a mock
VAE rather than a real audio checkpoint.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
vae_module = load_node_module("vae")
ASDX_VAEDecodeAudio = vae_module.ASDX_VAEDecodeAudio


def _mock_audio_vae(sample_rate=32000, waveform_scale=0.1):
    # spec= restricts which attributes exist on the mock -- without it, Mock()
    # answers hasattr()/getattr() truthily for ANY name (e.g.
    # audio_sample_rate_output), defeating the node's getattr(...) fallback.
    vae = Mock(spec=["decode", "decode_tiled", "audio_sample_rate"])
    vae.audio_sample_rate = sample_rate

    def _decode(latent: torch.Tensor) -> torch.Tensor:
        b, c, t = latent.shape
        # vae's native output layout is [B, T, C] (movedim(-1,1) in the node
        # converts it to [B, C, T] afterward) -- return a plausibly-scaled
        # waveform, not unit-variance noise, so the std-normalization step
        # is actually exercised rather than being a no-op.
        return torch.randn(b, t, c) * waveform_scale

    vae.decode.side_effect = _decode
    return vae


def test_execute_rejects_non_latent_input():
    with pytest.raises(RuntimeError, match="expected LATENT input"):
        ASDX_VAEDecodeAudio.execute({"not_samples": None}, Mock())


def test_decodes_to_waveform_and_sample_rate_dict():
    vae = _mock_audio_vae(sample_rate=32000)
    latent = torch.randn(1, 32, 5)
    samples = {"samples": latent}

    result = ASDX_VAEDecodeAudio.execute(samples, vae)
    audio = result.values[0]

    assert set(audio.keys()) == {"waveform", "sample_rate"}
    assert audio["sample_rate"] == 32000
    # decode's mock output was [B=1, T=5, C=32]; movedim(-1,1) -> [1, 32, 5]
    assert audio["waveform"].shape == (1, 32, 5)


def test_samples_dict_sample_rate_overrides_vae_default():
    vae = _mock_audio_vae(sample_rate=32000)
    samples = {"samples": torch.randn(1, 32, 5), "sample_rate": 48000}
    result = ASDX_VAEDecodeAudio.execute(samples, vae)
    assert result.values[0]["sample_rate"] == 48000


def test_movedim_converts_vae_native_layout_to_channel_first():
    # vae.decode returns [B, T, C]; the node must movedim to [B, C, T].
    vae = Mock()
    vae.audio_sample_rate = 32000
    vae.decode.return_value = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)  # [B=2,T=3,C=4]
    samples = {"samples": torch.zeros(2, 4, 3)}

    result = ASDX_VAEDecodeAudio.execute(samples, vae)
    waveform = result.values[0]["waveform"]
    assert waveform.shape == (2, 4, 3)


def test_std_normalization_does_not_amplify_quiet_signal():
    # Per the reference: std < 1.0 is clamped to 1.0, so a genuinely quiet
    # signal (std well under 1) must NOT be amplified up to unit variance --
    # only loud signals (std > 1) get scaled down.
    vae = Mock()
    vae.audio_sample_rate = 32000
    quiet = torch.full((1, 5, 2), 0.01)  # [B,T,C], std ~0
    vae.decode.return_value = quiet
    samples = {"samples": torch.zeros(1, 2, 5)}

    result = ASDX_VAEDecodeAudio.execute(samples, vae)
    waveform = result.values[0]["waveform"]
    assert torch.allclose(waveform, quiet.movedim(-1, 1), atol=1e-5)


def test_nested_latent_unwraps_to_last_stream():
    # A packed audio+video NestedTensor latent (ComfyUI's own convention for
    # some pipelines) must unwrap to its audio (last) stream before decode.
    vae = _mock_audio_vae()
    video_stream = torch.zeros(1, 24, 2, 4, 4)
    audio_stream = torch.randn(1, 32, 5)

    class FakeNested:
        is_nested = True

        def unbind(self):
            return (video_stream, audio_stream)

    samples = {"samples": FakeNested()}
    result = ASDX_VAEDecodeAudio.execute(samples, vae)
    assert result.values[0]["waveform"].shape == (1, 32, 5)
