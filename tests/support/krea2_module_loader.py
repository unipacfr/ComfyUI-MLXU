"""Load apple_silicon_nodes/native/krea2/*.py standalone, bypassing
apple_silicon_nodes/__init__.py (which imports comfy_api and is only
importable inside a running ComfyUI process).

Works by pre-registering the three parent packages
(apple_silicon_nodes, apple_silicon_nodes.native,
apple_silicon_nodes.native.krea2) in sys.modules as empty namespace
packages pointing at their real directories, before importing the target
submodule by dotted name. Python's import machinery only executes a
package's __init__.py the first time that exact name is imported; once
sys.modules already has an entry for it, that entry is reused as-is.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KREA2_DIR = _REPO_ROOT / "apple_silicon_nodes" / "native" / "krea2"


def _register_namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load_krea2_module(module_name: str) -> types.ModuleType:
    _register_namespace_package("apple_silicon_nodes", _REPO_ROOT / "apple_silicon_nodes")
    _register_namespace_package(
        "apple_silicon_nodes.native", _REPO_ROOT / "apple_silicon_nodes" / "native"
    )
    _register_namespace_package("apple_silicon_nodes.native.krea2", _KREA2_DIR)
    full_name = f"apple_silicon_nodes.native.krea2.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    return importlib.import_module(full_name)
