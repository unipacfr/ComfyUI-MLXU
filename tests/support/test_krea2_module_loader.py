from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.support.krea2_module_loader import load_krea2_module

# Several tests elsewhere in the suite (tests/native/minimax_h3/test_curve_time_embedding.py,
# test_sampling.py, test_text_encoder_rope.py, tests/support/comfyui_reference_loader.py) put
# the real ComfyUI install permanently on sys.path -- with no restore -- so they can import its
# real `torch`/`comfy` packages. Once that has happened earlier in the session, a real `comfy_api`
# package becomes findable on disk, so a plain import here would resolve it for real instead of
# raising ModuleNotFoundError. Filter those paths out for the duration of this test too.
_COMFYUI_ROOT = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")


def test_plain_import_of_model_fails_outside_comfyui():
    """Documents why load_krea2_module exists: importing the real package
    path pulls in apple_silicon_nodes/__init__.py's `from comfy_api.latest
    import ...`, which is only available inside a running ComfyUI process.

    Other tests in the suite stub or execute `apple_silicon_nodes` /
    `apple_silicon_nodes.native` (e.g. tests/support/anima_module_loader.py),
    install fake `comfy_api`/`comfy` modules (tests/support/comfy_stub.py, used
    by node-level tests) into sys.modules, and/or leak the real ComfyUI
    install's path onto sys.path (see module docstring above) for their own
    standalone-import tricks. If any of that is still in effect when this test
    runs, the plain import below either reuses a cached `apple_silicon_nodes`
    package (never re-triggering its __init__.py), resolves `comfy_api`
    against a stub, or resolves it against the real on-disk package instead of
    failing -- either way the assertion stops testing anything. Evict every
    `apple_silicon_nodes`/`comfy`/`comfy_api` sys.modules entry and any sys.path
    entry under the real ComfyUI install first (snapshotting both so they can
    be restored) to make this test's outcome independent of collection/execution
    order.
    """
    _prefixes = ("apple_silicon_nodes", "comfy_api", "comfy")

    def _matches(name: str) -> bool:
        return any(name == p or name.startswith(p + ".") for p in _prefixes)

    modules_snapshot = {name: module for name, module in sys.modules.items() if _matches(name)}
    for name in modules_snapshot:
        del sys.modules[name]

    path_snapshot = list(sys.path)
    sys.path[:] = [p for p in sys.path if not Path(p).is_relative_to(_COMFYUI_ROOT)]
    try:
        with pytest.raises(ModuleNotFoundError, match="comfy_api"):
            import apple_silicon_nodes.native.krea2.model  # noqa: F401
    finally:
        sys.path[:] = path_snapshot
        for name in list(sys.modules):
            if _matches(name):
                del sys.modules[name]
        sys.modules.update(modules_snapshot)


def test_load_krea2_module_loads_model_standalone():
    model = load_krea2_module("model")
    assert hasattr(model, "SingleStreamDiT")
    assert hasattr(model, "Attention")
    assert hasattr(model, "SingleStreamBlock")


def test_load_krea2_module_is_cached():
    first = load_krea2_module("model")
    second = load_krea2_module("model")
    assert first is second
