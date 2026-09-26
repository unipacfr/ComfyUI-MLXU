"""Anima checkpoints route to the native MLX loader via key detection."""

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
def test_real_anima_checkpoint_detected(path: Path):
    if not path.exists():
        pytest.skip(f"no local {path.name}")
    assert loader_mod._detect_model_type(path) == "anima"


def test_anima_hint_is_verified_by_keys(tmp_path):
    # A non-Anima file whose name contains "anima" (e.g. a Wan *animate* checkpoint)
    # must fall back to key detection, not route to the Anima loader.
    from safetensors.numpy import save_file
    import numpy as np
    p = tmp_path / "wan_animate_test.safetensors"
    save_file({"double_blocks.0.img_attn.qkv.weight": np.zeros((2, 2), np.float32)}, str(p))
    assert loader_mod._detect_model_type(p) == "dev"


def test_capability_entry():
    from apple_silicon_nodes.capability import CAPABILITY_PROFILES
    assert CAPABILITY_PROFILES[loader_mod._MODEL_TYPE_CAPABILITY["anima"]].latent_channels == 16
