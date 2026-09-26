from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.anima_module_loader import load_native_module

wm = load_native_module("anima.weight_map")

_DIR = Path("/Volumes/X10Pro/Images/models/diffusion_models/Anima")
_BF16 = _DIR / "anime/WaiHassakuAnima.safetensors"
_INT8 = _DIR / "anima/DaSiWa-ANIMA-ObsidianArchives-v2_int8_row-wise_convrot_runtime.safetensors"


def test_strip_prefix():
    assert wm.strip_anima_prefix("model.diffusion_model.blocks.0.mlp.layer1.weight") == "blocks.0.mlp.layer1.weight"
    assert wm.strip_anima_prefix("net.llm_adapter.embed.weight") == "llm_adapter.embed.weight"


@pytest.mark.parametrize("path", [_BF16, _INT8], ids=["bf16", "int8_convrot"])
def test_real_checkpoint_loads_strictly(path, capsys):
    if not path.exists():
        pytest.skip(f"no local {path.name}")
    model = wm.load_anima_checkpoint(path, dtype="bfloat16")
    assert "matched 685/685" in capsys.readouterr().out
    assert model.config.num_blocks == 28
    # Sanity: loaded weights are not random init (canon verify-checkpoint step 4).
    w = np.array(model.blocks[0].self_attn.q_proj.weight.astype(mx.float32))
    assert np.isfinite(w).all() and 1e-4 < w.std() < 1.0


def test_int8_dequant_close_to_bf16_sibling_shape():
    if not _INT8.exists():
        pytest.skip("no int8 file")
    model = wm.load_anima_checkpoint(_INT8, dtype="bfloat16")
    out = model(mx.random.normal((1, 16, 8, 8)), mx.array([0.5]), mx.random.normal((1, 512, 1024)))
    mx.eval(out)
    assert out.shape == (1, 16, 8, 8) and np.isfinite(np.array(out.astype(mx.float32))).all()
