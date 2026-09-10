"""
ComfyUI entry point.

ComfyUI imports `custom_nodes/<repo>/__init__.py` directly, but the actual
node package lives in apple_silicon_nodes/ (kept as its own importable
package name, used throughout the codebase and tests). This just re-exports
the V3 extension entry point so a plain `git clone` into custom_nodes/ works.
"""

from __future__ import annotations

__all__ = ["comfy_entrypoint"]


def __getattr__(name: str):
    """PEP 562 lazy attribute access: defer importing apple_silicon_nodes
    (which pulls in comfy_api, only available inside a running ComfyUI
    process) until comfy_entrypoint is actually accessed. Lets this file be
    imported standalone (e.g. by pytest) without failing."""
    if name == "comfy_entrypoint":
        from .apple_silicon_nodes import comfy_entrypoint
        return comfy_entrypoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
