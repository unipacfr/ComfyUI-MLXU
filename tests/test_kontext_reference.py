"""Tests for ``ASDX_KontextReference``.

Runs in the bare project venv (no ComfyUI) via ``comfy_stub.load_node_module``.
Only the model-dict shallow-copy contract is exercised -- this node has no
capability gate (Kontext reference conditioning applies to any FLUX-family
model, unlike Krea2 Identity Edit or flux1_depth).
"""

from __future__ import annotations

from unittest.mock import Mock

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
kontext_reference = load_node_module("kontext_reference")
ASDX_KontextReference = kontext_reference.ASDX_KontextReference


def test_stores_kontext_without_mutating_input():
    model = {"transformer": Mock()}
    reference_latent = {"samples": Mock()}
    result = ASDX_KontextReference.execute(model, reference_latent=reference_latent, strength=0.5)
    new_model = result.values[0]

    assert "kontext" not in model
    assert new_model["kontext"] == {"reference_latent": reference_latent, "strength": 0.5}
    assert new_model["transformer"] is model["transformer"]
