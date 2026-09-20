"""Real fl2va / ref2va DiT weights with keyframe and reference conditions (small shapes)."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

wm = load_native_module("minimax_h3.weight_map")
cond = load_native_module("minimax_h3.condition")

BASE = Path("/Volumes/X10Pro/Images/models/diffusion_models/MiniMax H3/base model")
FILES = ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"]


def _mem(tag: str) -> str:
    """Format MLX active/cache/peak memory in GB."""
    return f"[mem {tag}] active {mx.get_active_memory() / 1e9:.2f} GB, cache {mx.get_cache_memory() / 1e9:.2f} GB, peak {mx.get_peak_memory() / 1e9:.2f} GB"


@pytest.mark.skipif(os.environ.get("ASDX_FULL_GGUF_TEST") != "1", reason="loads real 20GB DiTs; set ASDX_FULL_GGUF_TEST=1")
@pytest.mark.parametrize("fname", FILES)
def test_real_dit_runs_with_conditions(fname: str) -> None:
    path = BASE / fname
    if not path.exists():
        pytest.skip(f"{fname} not present")
    mx.clear_cache()
    mx.reset_peak_memory()
    model = wm.load_minimax_h3_checkpoint(path, dtype="float16")
    print(_mem(f"{fname} after load"))
    c = model.config
    rng = np.random.default_rng(0)
    video = mx.array(rng.standard_normal((1, c.latents_dim, 2, 8, 8)).astype(np.float32))
    audio = mx.array(rng.standard_normal((1, c.audio_latents_dim, 2, 4)).astype(np.float32))
    context = mx.array(rng.standard_normal((6, c.text_dim)).astype(np.float32))

    def run(payload):
        prepared = cond.prepare_condition(payload, c.patch_size) if payload is not None else None
        kw = {} if prepared is None else {"cond": prepared}
        v, a = model(video, audio, context, sigma_v=0.5, **kw)
        mx.eval(v, a)
        return np.array(v.astype(mx.float32)), np.array(a.astype(mx.float32))

    plain = run(None)
    kf = cond.ConditionPayload(keyframes=(cond.KeyframeCond(0, mx.array(rng.standard_normal((1, c.latents_dim, 1, 8, 8)).astype(np.float32))),), seed=1)
    ref = cond.ConditionPayload(refs=(cond.RefBlock(kind="image", latent=mx.array(rng.standard_normal((1, c.latents_dim, 1, 8, 8)).astype(np.float32)), latent_t=1, latent_h=8, latent_w=8),), seed=1)
    with_kf, with_ref = run(kf), run(ref)
    d_kf = np.abs(plain[0] - with_kf[0]).max()
    d_ref = np.abs(plain[0] - with_ref[0]).max()
    print(f"[real {fname}] delta keyframe {d_kf:.3f}, delta ref {d_ref:.3f}, peak {mx.get_peak_memory() / 1e9:.2f} GB")
    print(_mem(f"{fname} after forwards"))
    try:
        for out in (plain, with_kf, with_ref):
            assert out[0].shape == tuple(video.shape) and np.isfinite(out[0]).all() and np.isfinite(out[1]).all()
        assert d_kf > 1e-3 and d_ref > 1e-3
    finally:
        # Free this ~14 GB model before the next parametrized case loads its own.
        del model, c, run
        gc.collect()
        mx.clear_cache()
        print(_mem(f"{fname} after free"))
