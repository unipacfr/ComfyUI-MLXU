"""Minimal ``comfy_api`` / ``comfy`` stubs for running node-level tests in the
project venv, which has no ComfyUI install.

``apple_silicon_nodes/__init__.py`` imports ``comfy_api`` and is only importable
inside a running ComfyUI process (see ``krea2_module_loader``). Node modules
(``krea2_edit``, ``lora``, ``vae``, ...) import ``from comfy_api.latest import
io`` at module level, so they cannot be imported in the bare project venv
either. This stub installs just enough of ``comfy_api.latest.io`` and ``comfy``
into ``sys.modules`` for those modules to import and for their ``execute()``
methods to run under pytest — the real ComfyUI schema machinery is NOT
replicated, only the symbols the node code touches.

The stub is deliberately inert: ``ComfyNode`` is a plain base class, ``Schema``
and ``NodeOutput`` are dumb containers, and the input types are no-ops. Nothing
here validates or transforms data — it exists solely so the import and the
``execute()`` call graph resolve.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class _Slot:
    """A single input/output slot in a (stub) schema."""

    def __init__(self, kind: str, type_name: str, name: str | None, kwargs: dict):
        self.kind = kind
        self.type_name = type_name
        self.name = name
        self.kwargs = kwargs


def _make_input_type(type_name: str) -> type:
    """Build an input-type class whose ``.Input``/``.Output`` work both as
    classmethods (``io.Float.Input(...)``) and instance methods
    (``io.Custom("x").Input(...)``)."""

    class _Type:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        @classmethod
        def Input(cls, name: str, **kwargs: Any) -> _Slot:
            return _Slot("input", type_name, name, kwargs)

        @classmethod
        def Output(cls, name: str | None = None, **kwargs: Any) -> _Slot:
            return _Slot("output", type_name, name, kwargs)

    _Type.__name__ = type_name
    _Type.__qualname__ = type_name
    return _Type


class _ComfyNode:
    """Stand-in for ``comfy_api.latest.io.ComfyNode`` — a plain base class.

    Real nodes define ``define_schema``/``execute`` classmethods; the stub base
    adds nothing, so a node class is importable and its ``execute`` callable
    without any ComfyUI runtime.
    """


class _Schema:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _NodeOutput:
    """Holds the value(s) an ``execute()`` returns, one per output.

    Mirrors the real ``io.NodeOutput``'s positional contract: ``NodeOutput(a,
    b)`` for a two-output node. Exposes them as ``.values`` for test access.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.values: tuple[Any, ...] = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, i: int):
        return self.values[i]


def _build_io_module() -> types.ModuleType:
    io_mod = types.ModuleType("comfy_api.latest.io")
    io_mod.ComfyNode = _ComfyNode
    io_mod.Schema = _Schema
    io_mod.NodeOutput = _NodeOutput
    for name in (
        "Custom", "Float", "Int", "String", "Combo", "Boolean", "Image",
        "Latent", "Vae", "Mask", "Conditioning", "MultiType", "Hidden",
    ):
        setattr(io_mod, name, _make_input_type(name))
    # ``io.BytesIO`` is a real type in comfy_api (byte-buffer inputs); a plain
    # marker class is enough for import.
    io_mod.BytesIO = type("BytesIO", (), {})
    return io_mod


def _build_comfy_api() -> types.ModuleType:
    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = _build_io_module()
    latest.ComfyExtension = type("ComfyExtension", (), {})
    comfy_api.latest = latest
    return comfy_api


def _build_comfy() -> types.ModuleType:
    """Just enough of the ``comfy`` package for the node modules to import.

    Node modules reference ``comfy.sd`` / ``comfy.utils`` /
    ``comfy.text_encoders.krea2`` / ``comfy.model_management`` /
    ``comfy.latent_formats`` / ``comfy.sample`` — mostly lazily inside methods,
    but a few at module level (``conditioning.py``). Empty submodules satisfy
    the imports; tests that exercise a code path touching a real symbol must
    set the attribute they need on the relevant submodule.
    """
    comfy = types.ModuleType("comfy")
    for sub in (
        "sd", "utils", "sample", "model_management", "latent_formats",
        "text_encoders",
    ):
        mod = types.ModuleType(f"comfy.{sub}")
        setattr(comfy, sub, mod)
        sys.modules[f"comfy.{sub}"] = mod
    # ``comfy.text_encoders.krea2`` is imported by name in conditioning.py and
    # isinstance-checked against Krea2Tokenizer.
    krea2 = types.ModuleType("comfy.text_encoders.krea2")
    krea2.Krea2Tokenizer = type("Krea2Tokenizer", (), {})
    text_encoders = sys.modules["comfy.text_encoders"]
    text_encoders.krea2 = krea2
    sys.modules["comfy.text_encoders.krea2"] = krea2
    return comfy


_INSTALLED = False


def install_comfy_stubs() -> None:
    """Install the ``comfy_api``/``comfy`` stubs into ``sys.modules`` (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    if "comfy_api" not in sys.modules:
        sys.modules["comfy_api"] = _build_comfy_api()
        sys.modules["comfy_api.latest"] = sys.modules["comfy_api"].latest
        sys.modules["comfy_api.latest.io"] = sys.modules["comfy_api.latest"].io
    if "comfy" not in sys.modules:
        sys.modules["comfy"] = _build_comfy()
    _INSTALLED = True


def load_node_module(module_name: str) -> types.ModuleType:
    """Import ``apple_silicon_nodes.<module_name>`` in the bare project venv.

    Registers ``apple_silicon_nodes`` as an empty namespace package so its
    ``__init__.py`` (which imports ``comfy_api`` and the whole model zoo) is
    never executed, installs the comfy stubs, and imports the named leaf module
    by dotted path. Reuses the same namespace trick as ``krea2_module_loader``.

    ``apple_silicon_nodes.native`` is deliberately NOT pre-registered: the node
    modules (``lora.py``) do ``from .native import FluxTransformer`` at module
    level, and ``FluxTransformer`` is defined in ``native/__init__.py`` — so
    ``native`` must be a regular package whose ``__init__.py`` runs (it is
    MLX-only, no module-level ``comfy``). Pre-registering it as an empty
    namespace package would bypass that ``__init__.py`` and break the import.
    """
    import importlib
    from pathlib import Path

    from .krea2_module_loader import _register_namespace_package

    install_comfy_stubs()
    repo_root = Path(__file__).resolve().parents[2]
    _register_namespace_package("apple_silicon_nodes", repo_root / "apple_silicon_nodes")
    # A prior ``load_krea2_module()`` call (from the tests/native/krea2 suite,
    # which runs at module-import time) may have registered
    # ``apple_silicon_nodes.native`` as an empty namespace package in
    # ``sys.modules``. That would break ``lora.py``'s ``from .native import
    # FluxTransformer`` (defined in ``native/__init__.py``). Detect it (namespace
    # packages lack ``__file__``) and drop the poisoned entries so ``native``
    # re-imports as a regular package. The krea2 leaf modules (``native.krea2.*``)
    # are already in ``sys.modules`` and are reused, so nothing is orphaned.
    native_mod = sys.modules.get("apple_silicon_nodes.native")
    if native_mod is not None and getattr(native_mod, "__file__", None) is None:
        for _name in (
            "apple_silicon_nodes.native",
            "apple_silicon_nodes.native.krea2",
        ):
            sys.modules.pop(_name, None)
    # Same poisoning, but for `apple_silicon_nodes.native.anima` specifically:
    # `tests/support/anima_module_loader.py` (used by tests/native/anima/*)
    # registers it as an empty namespace-package stub (no __init__ run), which
    # can survive in sys.modules across the whole pytest session even when
    # `apple_silicon_nodes.native` itself is the real package -- breaking
    # `lora.py`'s `from .native.anima import AnimaTransformer`. Drop only the
    # poisoned "anima" package entry; its real leaf submodules (model/config/
    # weight_map, already imported with real __file__s) are reused when the
    # package re-imports.
    anima_mod = sys.modules.get("apple_silicon_nodes.native.anima")
    if anima_mod is not None and getattr(anima_mod, "__file__", None) is None:
        sys.modules.pop("apple_silicon_nodes.native.anima", None)
    full_name = f"apple_silicon_nodes.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    return importlib.import_module(full_name)
