"""Import the real `comfy.ldm.minimax.model` from the ComfyUI install on this
machine, for tests that verify a port against it directly.

`tests/support/comfy_stub.py` installs a fake, inert `comfy`/`comfy_api`
module into `sys.modules` for node-level tests that need to import
`apple_silicon_nodes` code outside a running ComfyUI process. If pytest
collects one of those test files before a test using this loader, the real
import below finds that stub already cached under the name `comfy` (not a
real package -- no `__path__`) and fails with "'comfy' is not a package".
Clearing any `comfy`/`comfy.*` entries first forces a genuinely fresh import
of the real package from the ComfyUI install, regardless of collection order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_COMFYUI_ROOT = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")
_COMFYUI_VENV_SITE_PACKAGES = _COMFYUI_ROOT / ".venv" / "lib" / "python3.13" / "site-packages"


def load_real_comfy_minimax_model():
    if not _COMFYUI_ROOT.exists() or not _COMFYUI_VENV_SITE_PACKAGES.exists():
        pytest.skip("ComfyUI install (with MiniMax H3 source + venv) not present on this machine")
    for name in list(sys.modules):
        if name == "comfy" or name.startswith("comfy."):
            del sys.modules[name]
    sys.path.insert(0, str(_COMFYUI_ROOT))
    sys.path.insert(0, str(_COMFYUI_VENV_SITE_PACKAGES))
    try:
        import comfy.ldm.minimax.model as mm
    except ImportError as e:
        pytest.skip(f"comfy.ldm.minimax.model not importable: {e}")
    return mm
