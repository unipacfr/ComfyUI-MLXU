"""load_qwen_image21_dit_checkpoint / load_qwen_image21_dit_from_gguf: synthetic round-trip
tests, then real checkpoint tests (bf16 and GGUF Q8) gated by ASDX_FULL_GGUF_TEST=1.

Key remapping needed: the checkpoint's `modulation.1.weight` (index 1 = the Linear inside
the reference's `nn.Sequential(SiLU(), Linear(...))`) must map to this module's own
`modulation.1.weight` -- MLX stores a plain Python list of layers under the attribute name
directly indexed (`self.modulation = [nn.SiLU(), nn.Linear(...)]`), so `tree_flatten` already
produces `modulation.1.weight` with no remapping needed (unlike flux2's `nn.Sequential`,
which needed a `.layers.` insertion -- confirmed empirically in the round-trip test below)."""

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

config_mod = load_native_module("qwen_image21.config")
model_mod = load_native_module("qwen_image21.model")
weight_map_mod = load_native_module("qwen_image21.weight_map")

QwenImage21Config = config_mod.QwenImage21Config
QwenImage21Transformer2DModel = model_mod.QwenImage21Transformer2DModel
load_qwen_image21_dit_checkpoint = weight_map_mod.load_qwen_image21_dit_checkpoint


def _tiny_config(**overrides):
    base = dict(
        in_channels=8, out_channels=8, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, context_in_dim=16, mlp_ratio=2,
        axes_dims_rope=(2, 3, 3), eps=1e-6, dtype="float32",
    )
    base.update(overrides)
    return QwenImage21Config(**base)


def test_round_trip_through_synthetic_safetensors(tmp_path):
    cfg = _tiny_config()
    original = QwenImage21Transformer2DModel(cfg)
    flat = dict(tree_flatten(original.parameters()))
    tensors = {k: np.array(v) for k, v in flat.items()}
    path = tmp_path / "tiny_dit.safetensors"
    save_file(tensors, str(path))

    loaded = load_qwen_image21_dit_checkpoint(path, dtype="float32")
    # axes_dims_rope is excluded: it's a RoPE per-axis split convention with no
    # corresponding weight in the checkpoint (no tensor's shape encodes it), so
    # detect_qwen_image21_config always returns the class default for it -- correct
    # for the real checkpoint, but not round-trippable for this tiny config's override.
    for field in ("in_channels", "out_channels", "num_layers", "attention_head_dim",
                  "num_attention_heads", "context_in_dim", "mlp_ratio", "eps", "dtype"):
        assert getattr(loaded.config, field) == getattr(cfg, field), field
    loaded_flat = dict(tree_flatten(loaded.parameters()))
    assert set(loaded_flat.keys()) == set(flat.keys())
    for key, original_value in flat.items():
        assert np.allclose(np.array(loaded_flat[key]), np.array(original_value), atol=1e-6), key


def test_raises_on_missing_required_key(tmp_path):
    cfg = _tiny_config()
    original = QwenImage21Transformer2DModel(cfg)
    flat = dict(tree_flatten(original.parameters()))
    tensors = {k: np.array(v) for k, v in flat.items() if k != "img_in.weight"}
    path = tmp_path / "broken.safetensors"
    save_file(tensors, str(path))
    with pytest.raises((ValueError, KeyError)):
        load_qwen_image21_dit_checkpoint(path, dtype="float32")


_DIT_BF16_REAL = Path("/Volumes/X10Pro/Images/models/diffusion_models/Qwen 2/base model/qwen_image_2.1_bf16.safetensors")
_DIT_GGUF_REAL = Path("/Volumes/X10Pro/Images/models/diffusion_models/Qwen 2/gguf/qwen_image_2.1_Q8.gguf")


@pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="loads the real ~14GB Qwen Image 2.1 DiT checkpoint; set ASDX_FULL_GGUF_TEST=1 to run",
)
def test_loads_real_bf16_checkpoint():
    if not _DIT_BF16_REAL.exists():
        pytest.skip("no local qwen_image_2.1_bf16.safetensors")
    model = load_qwen_image21_dit_checkpoint(_DIT_BF16_REAL, dtype="float16")
    assert model.config.num_layers == 32
    assert model.config.inner_dim == 4096
    x = mx.random.normal((1, 64, 8, 8))
    timestep = mx.array([0.5])
    context = mx.random.normal((1, 6, 4096))
    out = model(x, timestep, context)
    assert out.shape == (1, 64, 8, 8)
    assert bool(mx.all(mx.isfinite(out)).item())


@pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="loads the real ~7.7GB Qwen Image 2.1 DiT GGUF checkpoint; set ASDX_FULL_GGUF_TEST=1 to run",
)
def test_loads_real_gguf_checkpoint():
    if not _DIT_GGUF_REAL.exists():
        pytest.skip("no local qwen_image_2.1_Q8.gguf")
    load_qwen_image21_dit_from_gguf = weight_map_mod.load_qwen_image21_dit_from_gguf
    model = load_qwen_image21_dit_from_gguf(_DIT_GGUF_REAL, dtype="float16")
    assert model.config.num_layers == 32
    x = mx.random.normal((1, 64, 8, 8))
    timestep = mx.array([0.5])
    context = mx.random.normal((1, 6, 4096))
    out = model(x, timestep, context)
    assert out.shape == (1, 64, 8, 8)
    assert bool(mx.all(mx.isfinite(out)).item())


@pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="loads both real DiT checkpoints (~14GB + ~7.7GB); set ASDX_FULL_GGUF_TEST=1 to run",
)
def test_q8_0_dequant_matches_bf16_checkpoint():
    """The verification `native/gguf/dequant.py::_dequantize_q8_0`'s own docstring
    claims: dequantized Q8_0 tensors matched against the same tensor read from the
    real bf16 checkpoint, within Q8_0's quantization tolerance. That claim previously
    had no committed test -- this is it."""
    if not _DIT_BF16_REAL.exists() or not _DIT_GGUF_REAL.exists():
        pytest.skip("real bf16/GGUF DiT checkpoints not both present locally")

    reader_mod = load_native_module("gguf.reader")
    dequant_mod = load_native_module("gguf.dequant")
    read_gguf_header = reader_mod.read_gguf_header
    dequantize_tensor = dequant_mod.dequantize_tensor

    bf16_state = mx.load(str(_DIT_BF16_REAL))
    gguf_header = read_gguf_header(_DIT_GGUF_REAL)

    probe_keys = [
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.15.img_mlp.gate_up.weight",
        "transformer_blocks.31.attn.to_out.0.weight",
    ]
    for key in probe_keys:
        bf16_tensor = bf16_state[key].astype(mx.float32)
        q8_tensor = dequantize_tensor(_DIT_GGUF_REAL, gguf_header, key).astype(mx.float32)
        assert q8_tensor.shape == bf16_tensor.shape, key

        rel_err = float(mx.mean(mx.abs(q8_tensor - bf16_tensor)).item()) / float(
            mx.mean(mx.abs(bf16_tensor)).item()
        )
        # Q8_0 is a per-32-value-block 8-bit quantization (~1/127 relative step);
        # a few percent mean relative error is expected and healthy, not a bug.
        assert rel_err < 0.05, f"{key}: relative error {rel_err:.4f} exceeds Q8_0 tolerance"

        cos = float(
            mx.sum(q8_tensor.flatten() * bf16_tensor.flatten())
            / (mx.linalg.norm(q8_tensor.flatten()) * mx.linalg.norm(bf16_tensor.flatten()))
        )
        assert cos > 0.999, f"{key}: cosine similarity {cos:.5f} too low for a real Q8_0 match"
