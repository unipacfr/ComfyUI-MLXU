"""Pytest configuration to enable test discovery.

NECESSARY WORKAROUND: Temporarily hides the repo root __init__.py (which
contains `from .apple_silicon_nodes import comfy_entrypoint` — a relative import
for ComfyUI custom-node integration) during pytest collection. Without this,
pytest's collection phase tries to import __init__.py directly as a top-level
module, causing "attempted relative import with no known parent package" errors.

This is a genuine pytest/repo-layout conflict: pytest will attempt this import
once ANY conftest.py exists in the rootdir, regardless of --ignore, import-mode,
or other configuration. The file rename is the only reliable solution.

Residual risk: SIGKILL or OOM-kill cannot be caught by any userspace mechanism
and may leave __init__.py.bak present with the real file missing. However, the
self-heal logic (below) automatically fixes this on the next pytest run.
"""
from __future__ import annotations

import atexit
import os
import signal
import sys
from pathlib import Path

_repo_root = Path(__file__).parent
_init_file = _repo_root / "__init__.py"
_init_backup = _repo_root / "__init__.py.bak"

# Self-heal: a prior crash between rename and restore leaves .bak present
# and the real file missing. Fix it before doing anything else.
if _init_backup.exists() and not _init_file.exists():
    _init_backup.rename(_init_file)

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


def _signal_restore(signum, frame):
    """Handle SIGTERM/SIGINT: restore file and re-raise."""
    _restore_init_file()
    # Re-raise default behavior after cleanup
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def pytest_configure(config):
    """Hide root __init__.py before collection to allow tests to run."""
    if _init_file.exists() and not _init_backup.exists():
        _init_file.rename(_init_backup)
        # Register cleanup handlers: atexit for normal exit, signals for termination
        atexit.register(_restore_init_file)
        signal.signal(signal.SIGTERM, _signal_restore)
        signal.signal(signal.SIGINT, _signal_restore)


def pytest_sessionfinish(session, exitstatus):
    """Restore __init__.py after tests complete."""
    _restore_init_file()
