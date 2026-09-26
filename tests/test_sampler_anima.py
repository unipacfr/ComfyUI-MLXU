"""Anima sampling loop: true two-pass CFG wiring, guards, and output format.

Uses a fake transformer (no real weights) so this stays fast -- only checks
that `_run_anima` calls the model the right number of times per step (CFG
on/off), raises the right guards, and that `self.guidance` (not a separate
node input) is the CFG scale, same pattern as `_run_sdxl` (core.py:~1511)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
core_mod = load_node_module("sampler.core")
_SamplerCore = core_mod._SamplerCore


class _FakeAnima:
    """Velocity = 0 for the positive context, 1 for the negative, so CFG is observable."""

    def __init__(self):
        self.config = SimpleNamespace(mlx_dtype=mx.float32, min_context_len=512)
        self.calls = []

    def encode_context(self, hidden, ids, weights):
        return mx.full((1, 512, 1024), float(hidden[0, 0, 0]))

    def __call__(self, x, t, ctx):
        self.calls.append(float(ctx[0, 0, 0]))
        return mx.zeros_like(x) if float(ctx[0, 0, 0]) > 0 else mx.ones_like(x)


def _cond(sign):
    return [[torch.full((1, 3, 1024), sign), {"t5xxl_ids": torch.tensor([1]), "t5xxl_weights": torch.tensor([1.0])}]]


def _core(guidance, negative=True, width=128, height=128, mode="auto", image=None, mask=None):
    positive = {"conditioning": _cond(1.0)}
    if negative:
        positive["_negative"] = _cond(-1.0)

    latent_h, latent_w = height // 8, width // 8
    noise = mx.zeros((1, 16, latent_h, latent_w), dtype=mx.float32)

    return _SamplerCore(
        transformer=_FakeAnima(),
        config=SimpleNamespace(mlx_dtype=mx.float32),
        positive=positive,
        noise=noise,
        height=height,
        width=width,
        output_shape=(latent_h, latent_w),
        model_type="anima",
        guidance=guidance,
        teacache=False,
        teacache_threshold=0.08,
        kontext=False,
        kontext_reference_latent=None,
        kontext_reference_strength=1.0,
        seacache=False,
        preview=False,
        lora_schedule=None,
        previewer=None,
        preview_device=None,
        capability=None,
        sampler_name="euler",
        scheduler_name="normal",
        memory_shape=None,
        mode=mode,
        image=image,
        mask=mask,
    )


def test_cfg_runs_two_passes_per_step():
    core = _core(guidance=4.5)
    core.run(steps=2, seed=0)
    assert len(core.transformer.calls) == 4


def test_cfg_output_value_matches_true_cfg_formula():
    """Call-count checks alone can't tell `v_neg + cfg*(v_pos-v_neg)` (correct)
    apart from a sign-swapped `v_pos + cfg*(v_neg-v_pos)` (wrong) -- both make
    exactly 2 calls/step. With this fake (v_pos=0, v_neg=1), the correct
    formula gives v=1-cfg; the swapped one gives v=cfg. Euler's update reduces
    to `x_next = x + v*(sigma_next-sigma)` (`_to_d` docstring: d==v exactly
    when denoised=x-v*sigma), which telescopes over a constant v from x_0=0 to
    `x_final = v*(0-sigma_0) = -v*sigma_0` (schedule always ends at sigma=0) --
    exact, no floating-point-schedule dependence beyond sigma_0 itself."""
    steps = 2
    cfg = 4.5
    core = _core(guidance=cfg)
    out = core.run(steps=steps, seed=0)

    sigma_0 = core_mod.calculate_sigmas("anima", "normal", steps, 128, 128)[0]
    expected_v = 1.0 - cfg  # v_neg(=1) + cfg*(v_pos(=0) - v_neg(=1))
    expected_x = -expected_v * sigma_0

    expected_latent = core_mod.bridge.mlx_to_comfy_latent_anima(
        mx.full((1, 16, 16, 16), expected_x, dtype=mx.float32), {"samples": None}
    )
    torch.testing.assert_close(out["samples"], expected_latent["samples"], atol=1e-4, rtol=1e-4)


def test_cfg_one_is_single_pass_and_needs_no_negative():
    core = _core(guidance=1.0, negative=False)
    core.run(steps=2, seed=0)
    assert len(core.transformer.calls) == 2


def test_cfg_without_negative_raises():
    with pytest.raises(RuntimeError, match="ASDX_ConditioningMerger"):
        _core(guidance=4.5, negative=False).run(steps=1, seed=0)


def test_odd_size_raises_before_compute():
    with pytest.raises(Exception, match="16"):
        _core(guidance=1.0, width=120, height=128).run(steps=1, seed=0)


def test_img2img_mode_raises():
    image = torch.zeros(1, 128, 128, 3)
    with pytest.raises(RuntimeError, match="text-to-image"):
        _core(guidance=1.0, image=image).run(steps=1, seed=0)


def test_explicit_inpaint_mode_raises():
    with pytest.raises(RuntimeError, match="text-to-image"):
        _core(guidance=1.0, mode="inpaint").run(steps=1, seed=0)


def test_txt2img_auto_mode_with_no_image_passes():
    core = _core(guidance=1.0, negative=False, mode="auto", image=None, mask=None)
    core.run(steps=1, seed=0)
