"""Tests for ASDX_QwenImage21TextEncoderLoader / ASDX_QwenImage21TextEncode.

Loader-only tests run via comfy_stub (no real ComfyUI needed). The real
tokenize+encode path needs the local ComfyUI install (comfy.text_encoders.
qwen_image21) and is gated accordingly, matching this project's established
convention for real-reference tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module
from tests.support.comfyui_reference_loader import real_comfy_isolated

install_comfy_stubs()
nodes_module = load_node_module("qwen_image21_nodes")

ASDX_QwenImage21TextEncoderLoader = nodes_module.ASDX_QwenImage21TextEncoderLoader
ASDX_QwenImage21TextEncode = nodes_module.ASDX_QwenImage21TextEncode
encode_qwen_image21_prompt = nodes_module.encode_qwen_image21_prompt


def test_encode_rejects_wrong_loader_output():
    with pytest.raises(RuntimeError, match="ASDX_QwenImage21TextEncoderLoader"):
        encode_qwen_image21_prompt({"type": "wrong"}, "a cat")


_COMFYUI_ROOT = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")
_COMFYUI_VENV_SITE_PACKAGES = _COMFYUI_ROOT / ".venv" / "lib" / "python3.13" / "site-packages"
_TEXT_ENCODER_REAL = Path("/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_8b_bf16.safetensors")


@pytest.mark.skipif(
    not (_COMFYUI_ROOT.exists() and _COMFYUI_VENV_SITE_PACKAGES.exists() and _TEXT_ENCODER_REAL.exists()),
    reason="needs local ComfyUI install + real text encoder checkpoint",
)
def test_encode_real_prompt_end_to_end():
    # comfy_stub installed fake "comfy"/"comfy.text_encoders" modules above (needed to
    # import qwen_image21_nodes at all outside a running ComfyUI process); real_comfy_isolated
    # drops them for the duration of this test so `import comfy.text_encoders.qwen_image21`
    # resolves against the real package, then restores sys.modules/sys.path exactly on exit
    # (see its own docstring) -- fixes the leak a manual purge-with-no-teardown would cause.
    with real_comfy_isolated():
        from apple_silicon_nodes.native.qwen_image21.text_encoder_weight_map import (
            load_qwen_image21_text_encoder_checkpoint,
        )

        encoder = load_qwen_image21_text_encoder_checkpoint(_TEXT_ENCODER_REAL, dtype="float16")
        text_encoder = {"type": "asdx_qwen_image21_text_encoder", "encoder": encoder}
        out = encode_qwen_image21_prompt(text_encoder, "a red apple on a table")
    assert out["type"] == "qwen_image21"
    assert out["hidden_states"].ndim == 2
    assert out["hidden_states"].shape[1] == 4096
