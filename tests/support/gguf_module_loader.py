"""Load apple_silicon_nodes/native/gguf/*.py standalone, bypassing
apple_silicon_nodes/__init__.py (which imports comfy_api and is only
importable inside a running ComfyUI process). Same trick as
`krea2_module_loader.py` -- see its docstring for the mechanism.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GGUF_DIR = _REPO_ROOT / "apple_silicon_nodes" / "native" / "gguf"


def _register_namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load_gguf_module(module_name: str) -> types.ModuleType:
    _register_namespace_package("apple_silicon_nodes", _REPO_ROOT / "apple_silicon_nodes")
    _register_namespace_package(
        "apple_silicon_nodes.native", _REPO_ROOT / "apple_silicon_nodes" / "native"
    )
    _register_namespace_package("apple_silicon_nodes.native.gguf", _GGUF_DIR)
    full_name = f"apple_silicon_nodes.native.gguf.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    return importlib.import_module(full_name)
