"""Anima checkpoints must be rejected, not silently loaded as FLUX.1-dev."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
loader_mod = load_node_module("loader")

_ANIMA_DIR = Path("/Volumes/X10Pro/Images/models/diffusion_models/Anima")
_BF16_REAL = _ANIMA_DIR / "anime/WaiHassakuAnima.safetensors"
_INT8_REAL = _ANIMA_DIR / "anima/DaSiWa-ANIMA-ObsidianArchives-v2_int8_row-wise_convrot_runtime.safetensors"


@pytest.mark.parametrize("path", [_BF16_REAL, _INT8_REAL], ids=["bf16", "int8_convrot"])
def test_real_anima_checkpoint_raises(path: Path):
    if not path.exists():
        pytest.skip(f"no local {path.name}")
    with pytest.raises(RuntimeError, match="Anima"):
        loader_mod._detect_model_type(path)
