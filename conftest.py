"""Pytest configuration to enable test discovery.

Temporarily hides the root __init__.py during test collection since it
contains ComfyUI-specific imports that fail outside a running ComfyUI process.
"""
from __future__ import annotations

import atexit
import sys
from pathlib import Path

_repo_root = Path(__file__).parent
_init_file = _repo_root / "__init__.py"
_init_backup = _repo_root / "__init__.py.bak"

# Ensure repo root is in sys.path so tests can be imported as a package
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _restore_init_file():
    """Restore __init__.py if it was backed up. Called on exit."""
    if _init_backup.exists():
        try:
            _init_backup.rename(_init_file)
        except FileExistsError:
            pass  # Already restored


def pytest_configure(config):
    """Hide root __init__.py before collection to allow tests to run."""
    if _init_file.exists() and not _init_backup.exists():
        _init_file.rename(_init_backup)
        # Register cleanup handler in case pytest crashes
        atexit.register(_restore_init_file)


def pytest_sessionfinish(session, exitstatus):
    """Restore __init__.py after tests complete."""
    _restore_init_file()
