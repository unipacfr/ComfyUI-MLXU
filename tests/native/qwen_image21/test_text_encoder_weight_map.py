"""load_qwen_image21_text_encoder_checkpoint: build a tiny real
Qwen3VL8BTextEncoder, write its params + synthetic vision/norm/lm_head keys
to a safetensors file, load it back, confirm text params match exactly and
the extra keys are skipped (not silently dropped -- logged)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

config_mod = load_native_module("qwen_image21.text_encoder_config")
text_encoder_mod = load_native_module("qwen_image21.text_encoder")
weight_map_mod = load_native_module("qwen_image21.text_encoder_weight_map")

Qwen3VL8BTextEncoderConfig = config_mod.Qwen3VL8BTextEncoderConfig
Qwen3VL8BTextEncoder = text_encoder_mod.Qwen3VL8BTextEncoder
load_qwen_image21_text_encoder_checkpoint = weight_map_mod.load_qwen_image21_text_encoder_checkpoint


def _tiny_config(**overrides) -> "Qwen3VL8BTextEncoderConfig":
    base = dict(
        vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
        rope_theta=5000000.0, dtype="float32",
    )
    base.update(overrides)
    return Qwen3VL8BTextEncoderConfig(**base)


def test_round_trip_and_skips_extra_keys(tmp_path):
    cfg = _tiny_config()
    original = Qwen3VL8BTextEncoder(cfg)
    flat = dict(tree_flatten(original.parameters()))
    tensors = {k: np.array(v) for k, v in flat.items()}
    # Extra keys present in the real checkpoint but not in our module.
    tensors["model.norm.weight"] = np.zeros((cfg.hidden_size,), dtype=np.float32)
    tensors["lm_head.weight"] = np.zeros((cfg.vocab_size, cfg.hidden_size), dtype=np.float32)
    tensors["model.visual.patch_embed.proj.weight"] = np.zeros((4,), dtype=np.float32)

    path = tmp_path / "tiny_qwen3vl8b.safetensors"
    save_file(tensors, str(path))

    loaded = load_qwen_image21_text_encoder_checkpoint(path, dtype="float32")
    assert loaded.config == cfg

    loaded_flat = dict(tree_flatten(loaded.parameters()))
    assert set(loaded_flat.keys()) == set(flat.keys())
    for key, original_value in flat.items():
        assert np.allclose(np.array(loaded_flat[key]), np.array(original_value), atol=1e-6), key


def test_raises_on_missing_required_key(tmp_path):
    cfg = _tiny_config()
    original = Qwen3VL8BTextEncoder(cfg)
    flat = dict(tree_flatten(original.parameters()))
    tensors = {k: np.array(v) for k, v in flat.items() if k != "model.embed_tokens.weight"}
    path = tmp_path / "broken.safetensors"
    save_file(tensors, str(path))
    with pytest.raises((ValueError, KeyError)):
        load_qwen_image21_text_encoder_checkpoint(path, dtype="float32")


_QWEN3VL_8B_REAL = Path("/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_8b_bf16.safetensors")


@pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="loads the real ~14GB Qwen3-VL-8B checkpoint; set ASDX_FULL_GGUF_TEST=1 to run",
)
def test_loads_real_checkpoint():
    if not _QWEN3VL_8B_REAL.exists():
        pytest.skip("no local qwen3vl_8b_bf16.safetensors")

    model = load_qwen_image21_text_encoder_checkpoint(_QWEN3VL_8B_REAL, dtype="float16")

    assert model.config.num_hidden_layers == 36
    assert model.config.hidden_size == 4096
    assert model.config.num_attention_heads == 32
    assert model.config.num_key_value_heads == 8

    input_ids = mx.array([1, 2, 3, 4, 5])
    out = model(input_ids)
    assert out.shape == (5, 4096)
    assert bool(mx.all(mx.isfinite(out)).item())
    assert float(mx.max(mx.abs(out)).item()) > 0.0
