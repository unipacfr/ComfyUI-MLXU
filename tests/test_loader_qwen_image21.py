"""Family detection for Qwen Image 2.1 against the real checkpoint files."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
loader_mod = load_node_module("loader")

_BF16_REAL = Path("/Volumes/X10Pro/Images/models/diffusion_models/Qwen 2/base model/qwen_image_2.1_bf16.safetensors")
_GGUF_REAL = Path("/Volumes/X10Pro/Images/models/diffusion_models/Qwen 2/gguf/qwen_image_2.1_Q8.gguf")


def test_qwen_image21_hints_present():
    assert any("qwen_image_2.1" in h or "qwen-image-2.1" in h or "qwen_image21" in h
                for h in loader_mod._QWEN_IMAGE21_HINTS)


@pytest.mark.skipif(not _BF16_REAL.exists(), reason="no local qwen_image_2.1_bf16.safetensors")
def test_detects_real_bf16_checkpoint():
    assert loader_mod._detect_model_type(_BF16_REAL) == "qwen_image21"


def test_structural_fallback_key_does_not_collide_with_other_families():
    # The distinctive key must not appear in any other family's own marker set.
    other_markers = ["input_blocks.", "noise_refiner.", "double_stream_modulation_img.", "txtfusion."]
    assert loader_mod._QWEN_IMAGE21_STRUCTURAL_KEY not in other_markers
    for marker in other_markers:
        assert marker not in loader_mod._QWEN_IMAGE21_STRUCTURAL_KEY
        assert loader_mod._QWEN_IMAGE21_STRUCTURAL_KEY not in marker


def test_capability_entry_present():
    assert "qwen_image21" in loader_mod._MODEL_TYPE_CAPABILITY
