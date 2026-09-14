"""Tests for ``ASDX_Krea2GroundedEncode``.

Runs in the bare project venv (no ComfyUI) via ``comfy_stub.load_node_module``.
Only the input-validation guard is exercised here -- encoding through a real
Krea2 CLIP vision tower needs a live checkpoint and is out of scope for this
suite (see ``test_krea2_edit.py`` for the project's convention on opt-in
full-checkpoint tests).
"""

from __future__ import annotations

import sys

import pytest

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
krea2_grounded_encode = load_node_module("krea2_grounded_encode")
ASDX_Krea2GroundedEncode = krea2_grounded_encode.ASDX_Krea2GroundedEncode

# The stub's ``comfy.sd`` module has no ``CLIP`` class (nothing needed it
# until now) -- add a marker type so ``isinstance(mlx_clip, comfy.sd.CLIP)``
# has something real to check against.
_comfy_sd = sys.modules["comfy.sd"]
if not hasattr(_comfy_sd, "CLIP"):
    _comfy_sd.CLIP = type("CLIP", (), {})


class _FakeClip(_comfy_sd.CLIP):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def test_rejects_non_krea2_clip():
    clip = _FakeClip(tokenizer=object())
    with pytest.raises(RuntimeError, match="requires a Krea2 mlx_clip"):
        ASDX_Krea2GroundedEncode.execute(
            mlx_clip=clip, text="a prompt", image=None, grounding_px=768,
        )


def test_rejects_non_clip_object():
    with pytest.raises(RuntimeError, match="must be a Comfy CLIP object"):
        ASDX_Krea2GroundedEncode.execute(
            mlx_clip=object(), text="a prompt", image=None, grounding_px=768,
        )
