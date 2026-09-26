"""Anima has no LoRA support (no Phase 1-3 forward-time residual, no merge
branch, and `detect_lora_family` has no Anima signature) -- applying a LoRA
must raise a clear error, not silently no-op or misfile under a generic
name-merge. The guard lives in `lora.py::_apply_lora_to_transformer`, the one
staticmethod every LoRA application path (ASDX_LoraLoader, ASDX_MultiLoraLoader,
and `sampler/core.py::_update_lora_schedule`'s per-step rescale) calls -- see
that method's docstring."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
lora_mod = load_node_module("lora")
anima_mod = load_node_module("native.anima.model")
anima_config_mod = load_node_module("native.anima.config")

ASDX_LoraLoader = lora_mod.ASDX_LoraLoader


def _tiny_anima_transformer():
    cfg = anima_config_mod.AnimaConfig(dtype="float32", model_channels=64, num_blocks=1, num_heads=2)
    return anima_mod.AnimaTransformer(cfg)


def test_apply_lora_to_anima_raises():
    transformer = _tiny_anima_transformer()
    lora = SimpleNamespace(deltas={}, factors={}, lokr_factors={}, loha_factors={})
    with pytest.raises(RuntimeError, match="LoRA is not supported for Anima"):
        ASDX_LoraLoader._apply_lora_to_transformer(transformer, lora, transformer.config)
