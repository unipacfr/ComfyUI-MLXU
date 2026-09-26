"""Load apple_silicon_nodes/native/anima/*.py standalone, bypassing
apple_silicon_nodes/__init__.py (which imports comfy_api)."""
from __future__ import annotations

from support.qwen_image21_module_loader import load_native_module

__all__ = ["load_native_module"]
