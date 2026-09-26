"""Load apple_silicon_nodes/native/anima/*.py standalone, bypassing
apple_silicon_nodes/__init__.py (which imports comfy_api).

Unlike the other families' module loaders, this one must also make
`apple_silicon_nodes.native._load_safetensors` real: native/anima/weight_map.py
does `from .. import _load_safetensors` inside load_anima_checkpoint(), and the
generic namespace-package trick used by qwen_image21_module_loader registers
`apple_silicon_nodes.native` as an EMPTY module (its __init__.py never runs),
so that lazy import would fail with ImportError under pytest. Instead we
execute the real apple_silicon_nodes/native/__init__.py by file spec (it does
not import comfy_api, only .krea2/.config/.weight_map/.common, all safe to
import standalone) and register IT as `apple_silicon_nodes.native`.
"""
from __future__ import annotations

import importlib
import importlib.util
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


def _ensure_real_native_package() -> None:
    _register_namespace_package("apple_silicon_nodes", _REPO_ROOT / "apple_silicon_nodes")
    name = "apple_silicon_nodes.native"
    existing = sys.modules.get(name)
    if existing is not None and hasattr(existing, "_load_safetensors"):
        return
    spec = importlib.util.spec_from_file_location(
        name, _NATIVE_DIR / "__init__.py", submodule_search_locations=[str(_NATIVE_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def load_native_module(dotted_name: str) -> types.ModuleType:
    """`dotted_name` is relative to `apple_silicon_nodes.native`, e.g.
    "anima.weight_map"."""
    _ensure_real_native_package()
    parts = dotted_name.split(".")
    prefix = "apple_silicon_nodes.native"
    for i, part in enumerate(parts[:-1]):
        prefix = f"{prefix}.{part}"
        _register_namespace_package(prefix, _NATIVE_DIR / Path(*parts[: i + 1]))
    full_name = f"apple_silicon_nodes.native.{dotted_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    return importlib.import_module(full_name)


__all__ = ["load_native_module"]
