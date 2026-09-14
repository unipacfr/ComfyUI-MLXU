"""Tests for ``ASDX_LatentNoisePrep``.

Runs in the bare project venv (no ComfyUI) via ``comfy_stub.load_node_module``.
Covers input validation (inpaint requires a mask) and the model-dict
shallow-copy contract. The actual noise blend stays in _SamplerCore and is
covered by that module's own regression path, not here (this node never
touches the noise tensor).
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
latent_noise_prep = load_node_module("latent_noise_prep")
ASDX_LatentNoisePrep = latent_noise_prep.ASDX_LatentNoisePrep


def test_inpaint_requires_mask():
    model = {"transformer": Mock()}
    with pytest.raises(RuntimeError, match="requires a mask"):
        ASDX_LatentNoisePrep.execute(model, mode="inpaint", image=Mock(), mask=None)


def test_stores_latent_prep_without_mutating_input():
    model = {"transformer": Mock()}
    image = Mock()
    result = ASDX_LatentNoisePrep.execute(
        model, mode="img2img", image=image, mask=None, image_strength=0.6,
        mask_blur=4, mask_padding=64,
    )
    new_model = result.values[0]

    assert "latent_prep" not in model
    assert new_model["latent_prep"] == {
        "mode": "img2img", "image": image, "mask": None,
        "image_strength": 0.6, "mask_blur": 4, "mask_padding": 64,
    }
    assert new_model["transformer"] is model["transformer"]


def test_auto_mode_with_no_image_is_valid():
    model = {"transformer": Mock()}
    result = ASDX_LatentNoisePrep.execute(model, mode="auto")
    assert result.values[0]["latent_prep"]["mode"] == "auto"
