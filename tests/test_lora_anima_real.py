"""Anima LoRA real-library coverage: apply every real *.safetensors file under
/Volumes/X10Pro/Images/models/loras/Anima/ to the real WaiHassakuAnima base
checkpoint and check the residual attach against the file's own key counts
(never hard-coded), plus a real-sized forward pass (per-file F16 overflow
risk is only visible at realistic activation magnitudes, not on a tiny
latent -- see the F16 note in the task report).

Skipped whole-file when the checkpoint or LoRA library is absent (CI/other
machines)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
lora_mod = load_node_module("lora")
weight_map_mod = load_node_module("native.anima.weight_map")

ASDX_LoraLoader = lora_mod.ASDX_LoraLoader

CHECKPOINT = Path("/Volumes/X10Pro/Images/models/diffusion_models/Anima/anime/WaiHassakuAnima.safetensors")
LORA_DIR = Path("/Volumes/X10Pro/Images/models/loras/Anima")
LORA_FILES = sorted(LORA_DIR.rglob("*.safetensors")) if LORA_DIR.is_dir() else []

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.is_file() or not LORA_FILES,
    reason="Anima checkpoint/LoRA library not present on this machine",
)


def _expected_target_count(path: Path) -> int:
    """Derive the number of distinct LoRA targets from the file's own keys:
    kohya/adaLN files carry one `.alpha` per target, PEFT files carry none
    but exactly one `.lora_A.*` per target."""
    with safe_open(path, framework="numpy") as sf:
        keys = list(sf.keys())
    alpha = sum(1 for k in keys if k.endswith(".alpha"))
    if alpha:
        return alpha
    return sum(1 for k in keys if ".lora_A." in k)


@pytest.fixture(scope="module")
def base():
    t0 = time.perf_counter()
    model = weight_map_mod.load_anima_checkpoint(CHECKPOINT, dtype="bfloat16")
    mx.eval(model.parameters())
    print(f"[test] base checkpoint loaded in {time.perf_counter() - t0:.1f}s")
    return model


def _inputs():
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((1, 16, 64, 64)).astype(np.float32))
    sigma = mx.array([0.5], dtype=mx.float32)
    qwen = mx.array(rng.standard_normal((1, 512, 1024)).astype(np.float32))
    ids = mx.array(rng.integers(0, 32128, (1, 12)).astype(np.int32))
    return x, sigma, qwen, ids


def _forward(model):
    x, sigma, qwen, ids = _inputs()
    ctx = model.encode_context(qwen, ids)
    out = model(x, sigma, ctx)
    out = out.astype(mx.float32)
    mx.eval(out)
    return np.array(out)


@pytest.mark.parametrize("lora_path", LORA_FILES, ids=[p.name for p in LORA_FILES])
def test_real_lora_file_attaches_and_forwards(base, lora_path):
    sig = lora_mod.detect_lora_family(lora_path)
    assert sig.family == "anima", (lora_path.name, sig.family)

    expected = _expected_target_count(lora_path)

    t0 = time.perf_counter()
    lora = ASDX_LoraLoader._load_lora_file(lora_path)
    lora.scale = lora_mod.base_lora_scale(lora.alpha, lora.rank) * 1.0
    new_model = ASDX_LoraLoader._apply_lora_to_transformer(base, lora, None)
    attached = len(list(lora_mod._iter_adaptable_leaves(new_model)))
    assert attached == expected, (lora_path.name, attached, expected)

    out = _forward(new_model)
    base_out = _forward(base)
    elapsed = time.perf_counter() - t0
    print(f"[test] {lora_path.name}: attached={attached} finite={np.isfinite(out).all()} "
          f"time={elapsed:.1f}s")

    assert np.isfinite(out).all(), f"{lora_path.name}: non-finite output"
    assert not np.allclose(out, base_out), f"{lora_path.name}: LoRA had no effect"

    del new_model, lora, out, base_out
    mx.clear_cache()
