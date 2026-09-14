"""Load apple_silicon_nodes/native/{minimax_h3,gguf}/*.py and sibling
native/*.py modules standalone, bypassing apple_silicon_nodes/__init__.py
(which imports comfy_api and is only importable inside a running ComfyUI
process). Same trick as `krea2_module_loader.py` -- see its docstring for
the mechanism.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NATIVE_DIR = _REPO_ROOT / "apple_silicon_nodes" / "native"


def _register_namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load_native_module(dotted_name: str) -> types.ModuleType:
    """`dotted_name` is relative to `apple_silicon_nodes.native`, e.g.
    "safetensors_header" or "minimax_h3.config"."""
    _register_namespace_package("apple_silicon_nodes", _REPO_ROOT / "apple_silicon_nodes")
    _register_namespace_package("apple_silicon_nodes.native", _NATIVE_DIR)
    parts = dotted_name.split(".")
    prefix = "apple_silicon_nodes.native"
    for part in parts[:-1]:
        prefix = f"{prefix}.{part}"
        _register_namespace_package(prefix, _NATIVE_DIR / Path(*parts[: parts.index(part) + 1]))
    full_name = f"apple_silicon_nodes.native.{dotted_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    return importlib.import_module(full_name)
