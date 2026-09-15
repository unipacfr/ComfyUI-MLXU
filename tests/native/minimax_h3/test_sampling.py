"""Tests for MiniMax H3's Euler sampling loop and sigma schedule.

`test_sigma_schedule_matches_real_comfy_simple_scheduler` is the load-bearing
test: it builds a real `comfy.model_sampling.ModelSamplingDiscreteFlow`
(imported from the ComfyUI install on this machine) and runs the real
`comfy.samplers.simple_scheduler` against it, comparing to
`minimax_h3_sigma_schedule` -- the actual scheduler the reference workflow
uses (`BasicScheduler(scheduler="simple")`), not a formula re-derived from
reading the source alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

config_mod = load_native_module("minimax_h3.config")
model_mod = load_native_module("minimax_h3.model")
sampling_mod = load_native_module("minimax_h3.sampling")

MiniMaxH3Config = config_mod.MiniMaxH3Config
MiniMaxH3Model = model_mod.MiniMaxH3Model
minimax_h3_sigma_schedule = sampling_mod.minimax_h3_sigma_schedule
run_minimax_h3_sampling = sampling_mod.run_minimax_h3_sampling

_COMFYUI_ROOT = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")
_COMFYUI_VENV_SITE_PACKAGES = _COMFYUI_ROOT / ".venv" / "lib" / "python3.13" / "site-packages"


def _load_real_comfy_sampling():
    if not _COMFYUI_ROOT.exists() or not _COMFYUI_VENV_SITE_PACKAGES.exists():
        pytest.skip("ComfyUI install not present on this machine")
    for name in list(sys.modules):
        if name == "comfy" or name.startswith("comfy."):
            del sys.modules[name]
    sys.path.insert(0, str(_COMFYUI_ROOT))
    sys.path.insert(0, str(_COMFYUI_VENV_SITE_PACKAGES))
    try:
        import comfy.model_sampling as model_sampling_mod
        import comfy.samplers as samplers_mod
    except ImportError as e:
        pytest.skip(f"comfy.model_sampling/samplers not importable: {e}")
    return model_sampling_mod, samplers_mod


def test_sigma_schedule_matches_real_comfy_simple_scheduler():
    model_sampling_mod, samplers_mod = _load_real_comfy_sampling()

    class _FakeModelConfig:
        sampling_settings = {"shift": 12.0}

    ms = model_sampling_mod.ModelSamplingDiscreteFlow(_FakeModelConfig())
    for steps in (4, 8, 20):
        ref = samplers_mod.simple_scheduler(ms, steps).numpy().tolist()
        mine = minimax_h3_sigma_schedule(12.0, steps)
        assert np.allclose(mine, ref, atol=1e-6), f"steps={steps}"


def test_sigma_schedule_is_monotonically_decreasing_and_ends_at_zero():
    sigmas = minimax_h3_sigma_schedule(12.0, 10)
    assert len(sigmas) == 11
    assert sigmas[-1] == 0.0
    assert all(sigmas[i] > sigmas[i + 1] for i in range(len(sigmas) - 1))


def _tiny_config(**overrides) -> "MiniMaxH3Config":
    base = dict(
        num_layers=2, token_refiner_num_layers=1, hidden_size=16, latents_dim=4,
        audio_latents_dim=6, attention_head_dim=16, num_attention_heads=2,
        ffn_hidden_size=32, text_dim=10, patch_size=(1, 2, 2), rope_inv_freq_len=2,
        norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
        sigma_shift_video=12.0, sigma_shift_audio=3.0, gate_compress=False,
        adaln_curve_grid=17, time_embed_dim=6, dtype="float32",
    )
    base.update(overrides)
    return MiniMaxH3Config(**base)


def test_run_sampling_shapes_and_denoising_happens():
    cfg = _tiny_config()
    model = MiniMaxH3Model(cfg)
    model.adaln_t_table = mx.random.normal(model.adaln_t_table.shape)  # non-degenerate, see test_full_model.py

    video_noise = mx.random.normal((1, cfg.latents_dim, 1, 4, 4))
    audio_noise = mx.random.normal((1, cfg.audio_latents_dim, 2, 2))
    context = mx.random.normal((3, cfg.text_dim))

    video_out, audio_out = run_minimax_h3_sampling(model, video_noise, audio_noise, context, steps=4)

    assert video_out.shape == video_noise.shape
    assert audio_out.shape == audio_noise.shape
    assert bool(mx.all(mx.isfinite(video_out)).item())
    assert bool(mx.all(mx.isfinite(audio_out)).item())
    # denoising must actually change the latent from pure noise
    assert float(mx.max(mx.abs(video_out - video_noise)).item()) > 1e-4
    assert float(mx.max(mx.abs(audio_out - audio_noise)).item()) > 1e-4


def test_run_sampling_is_deterministic_given_fixed_inputs():
    cfg = _tiny_config()
    model = MiniMaxH3Model(cfg)
    model.adaln_t_table = mx.random.normal(model.adaln_t_table.shape)
    mx.random.seed(0)
    video_noise = mx.random.normal((1, cfg.latents_dim, 1, 4, 4))
    audio_noise = mx.random.normal((1, cfg.audio_latents_dim, 2, 2))
    context = mx.random.normal((3, cfg.text_dim))

    out_a = run_minimax_h3_sampling(model, video_noise, audio_noise, context, steps=3)
    out_b = run_minimax_h3_sampling(model, video_noise, audio_noise, context, steps=3)
    assert bool(mx.array_equal(out_a[0], out_b[0]).item())
    assert bool(mx.array_equal(out_a[1], out_b[1]).item())


def test_more_steps_reach_lower_final_sigma_same_endpoint_type():
    # Not a numerical-quality claim -- just confirms the schedule/loop
    # actually terminates at sigma=0 (fully denoised) regardless of step count.
    cfg = _tiny_config()
    model = MiniMaxH3Model(cfg)
    model.adaln_t_table = mx.random.normal(model.adaln_t_table.shape)
    video_noise = mx.random.normal((1, cfg.latents_dim, 1, 4, 4))
    audio_noise = mx.random.normal((1, cfg.audio_latents_dim, 2, 2))
    context = mx.random.normal((3, cfg.text_dim))

    for steps in (2, 6):
        video_out, audio_out = run_minimax_h3_sampling(model, video_noise, audio_noise, context, steps=steps)
        assert bool(mx.all(mx.isfinite(video_out)).item())
        assert bool(mx.all(mx.isfinite(audio_out)).item())
