"""Tests for ``ASDX_DepthConditioning``.

Runs in the bare project venv (no ComfyUI) via ``comfy_stub.load_node_module``.
Only the input-validation guard and the model-dict shallow-copy contract are
exercised here -- see ``test_depth_conditioning.py`` for the structural
forward-pass test of the actual channel-concat mechanism this node feeds.
"""

from __future__ import annotations

import types
from unittest.mock import Mock

import pytest

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
depth_conditioning = load_node_module("depth_conditioning")
ASDX_DepthConditioning = depth_conditioning.ASDX_DepthConditioning


def _model_dict(family: str = "flux1_depth") -> dict:
    return {"transformer": Mock(), "capability": types.SimpleNamespace(family=family)}


def test_rejects_non_depth_model():
    model = _model_dict(family="flux1")
    with pytest.raises(RuntimeError, match="requires a flux1_depth model"):
        ASDX_DepthConditioning.execute(
            model, depth_image=Mock(), vae=Mock(), strength=1.0,
        )


def test_stores_depth_cond_without_mutating_input():
    model = _model_dict()
    depth_image = Mock()
    vae = Mock()
    result = ASDX_DepthConditioning.execute(model, depth_image=depth_image, vae=vae, strength=0.7)
    new_model = result.values[0]

    assert "depth_cond" not in model
    assert new_model["depth_cond"] == {"depth_image": depth_image, "vae": vae, "strength": 0.7}
    assert new_model["capability"] is model["capability"]
