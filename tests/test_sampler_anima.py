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


def _core(guidance, negative=True, width=128, height=128):
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
    )


def test_cfg_runs_two_passes_per_step():
    core = _core(guidance=4.5)
    core.run(steps=2, seed=0)
    assert len(core.transformer.calls) == 4


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
