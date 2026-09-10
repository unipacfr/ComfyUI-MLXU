from __future__ import annotations

import pytest

from tests.support.krea2_module_loader import load_krea2_module


def test_plain_import_of_model_fails_outside_comfyui():
    """Documents why load_krea2_module exists: importing the real package
    path pulls in apple_silicon_nodes/__init__.py's `from comfy_api.latest
    import ...`, which is only available inside a running ComfyUI process.
    """
    with pytest.raises(ModuleNotFoundError, match="comfy_api"):
        import apple_silicon_nodes.native.krea2.model  # noqa: F401


def test_load_krea2_module_loads_model_standalone():
    model = load_krea2_module("model")
    assert hasattr(model, "SingleStreamDiT")
    assert hasattr(model, "Attention")
    assert hasattr(model, "SingleStreamBlock")


def test_load_krea2_module_is_cached():
    first = load_krea2_module("model")
    second = load_krea2_module("model")
    assert first is second
