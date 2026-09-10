"""Root-level pytest configuration.

This conftest prevents pytest from importing the root __init__.py, which
has ComfyUI-specific imports that fail outside a running ComfyUI process.
"""
from __future__ import annotations

from pathlib import Path

_root_dir = Path(__file__).parent
_init_file = _root_dir / "__init__.py"
_init_backup = _root_dir / "__init__.py.bak"


def pytest_configure(config):
    """Temporarily hide the root __init__.py before collection."""
    if _init_file.exists() and not _init_backup.exists():
        _init_file.rename(_init_backup)


def pytest_sessionfinish(session, exitstatus):
    """Restore __init__.py after tests finish."""
    if _init_backup.exists():
        try:
            _init_backup.rename(_init_file)
        except FileExistsError:
            # Already restored
            pass
