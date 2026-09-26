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
import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
core_mod = load_node_module("sampler.core")
lora_mod = load_node_module("lora")
anima_mod = load_node_module("native.anima.model")
anima_config_mod = load_node_module("native.anima.config")
_SamplerCore = core_mod._SamplerCore
ASDX_LoraLoader = lora_mod.ASDX_LoraLoader


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


# ── LoRA schedule vs. the LLM adapter context (real tiny AnimaTransformer) ──
#
# `_run_anima` used to call `self.transformer.encode_context(...)` once
# BEFORE the sampling loop, like ComfyUI's `Anima.extra_conds`. A LoRA
# schedule targeting `llm_adapter.*` therefore never showed up after step 0,
# and because `_rescale_attached_lora`'s fast path mutates the cached
# transformer IN PLACE, a cache-hit re-queue would even start the next run
# from the PREVIOUS run's last-step strength -- identical runs gave
# different images. Fixed: when a schedule is attached, re-encode context
# right after `_update_lora_schedule` inside the loop.

_ANIMA_CFG = anima_config_mod.AnimaConfig(dtype="float32", model_channels=64, num_blocks=1, num_heads=2)


def _anima_kohya_lora(tmp_path, name, out_dim, in_dim, rank, alpha, seed):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((rank, in_dim)) * 0.1).astype(np.float32)
    b = (rng.standard_normal((out_dim, rank)) * 0.1).astype(np.float32)
    stem = "lora_unet_" + name.replace(".", "_")
    tensors = {f"{stem}.lora_down.weight": a, f"{stem}.lora_up.weight": b,
               f"{stem}.alpha": np.array(alpha, dtype=np.float32)}
    path = tmp_path / "adapter_lora.safetensors"
    save_file(tensors, str(path))
    return path


def _real_anima_positive():
    g = torch.Generator().manual_seed(42)
    hidden = torch.randn(1, 5, 1024, generator=g)
    ids = torch.arange(4, dtype=torch.long)
    return {"conditioning": [[hidden, {"t5xxl_ids": ids, "t5xxl_weights": torch.ones(4)}]]}


def _schedule_core(transformer, schedule, seed_noise):
    return _SamplerCore(
        transformer=transformer,
        config=_ANIMA_CFG,
        positive=_real_anima_positive(),
        noise=mx.array(seed_noise),
        height=32,
        width=32,
        output_shape=(4, 4),
        model_type="anima",
        guidance=1.0,
        teacache=False,
        teacache_threshold=0.08,
        kontext=False,
        kontext_reference_latent=None,
        kontext_reference_strength=1.0,
        seacache=False,
        preview=False,
        lora_schedule=schedule,
        previewer=None,
        preview_device=None,
        capability=None,
        sampler_name="euler",
        scheduler_name="normal",
        memory_shape=None,
        mode="auto",
        image=None,
        mask=None,
    )


def _build_scheduled_transformer(tmp_path):
    """Real tiny AnimaTransformer + a synthetic LoRA targeting
    `llm_adapter.out_proj`, attached at `strength_start` the same way
    `ASDX_LoraSchedule.execute` does."""
    mx.random.seed(0)
    with mx.stream(mx.cpu):
        transformer = anima_mod.AnimaTransformer(_ANIMA_CFG)
        mx.eval(transformer.parameters())

    dim = _ANIMA_CFG.adapter_dim
    path = _anima_kohya_lora(tmp_path, "llm_adapter.out_proj", dim, dim, rank=4, alpha=2.0, seed=7)
    lora = ASDX_LoraLoader._load_lora_file(path)
    strength_start = 1.0
    lora.scale = lora_mod.base_lora_scale(lora.alpha, lora.rank) * strength_start
    with mx.stream(mx.cpu):
        new_transformer = ASDX_LoraLoader._apply_lora_to_transformer(transformer, lora, _ANIMA_CFG)

    schedule = {
        "name": "adapter_lora", "lora": lora,
        "strength_start": strength_start, "strength_end": 0.0,
        "strength_middle": 1.0, "strength_curve": "linear",
    }
    return new_transformer, schedule


def test_llm_adapter_schedule_reencodes_context_each_step(tmp_path, monkeypatch):
    transformer, schedule = _build_scheduled_transformer(tmp_path)

    orig_encode = anima_mod.AnimaTransformer.encode_context
    calls: list[int] = []

    def _spy(self, *a, **kw):
        calls.append(1)
        return orig_encode(self, *a, **kw)

    monkeypatch.setattr(anima_mod.AnimaTransformer, "encode_context", _spy)

    with mx.stream(mx.cpu):
        noise = np.zeros((1, 16, 4, 4), dtype=np.float32)
        _schedule_core(transformer, schedule, noise).run(steps=3, seed=0)
    assert len(calls) == 3, "context must be re-encoded on every step when a schedule is attached"

    calls.clear()
    with mx.stream(mx.cpu):
        _schedule_core(transformer, None, noise).run(steps=3, seed=0)
    assert len(calls) == 1, "without a schedule, context is still encoded once before the loop"


def test_llm_adapter_schedule_context_matches_manual_encode_at_scheduled_strength():
    """The context used at step t must reflect the schedule's strength at t,
    not the pre-loop (unscheduled) strength."""
    with mx.stream(mx.cpu):
        mx.random.seed(1)
        transformer = anima_mod.AnimaTransformer(_ANIMA_CFG)
        mx.eval(transformer.parameters())

        dim = _ANIMA_CFG.adapter_dim
        rng = np.random.default_rng(3)
        a = mx.array((rng.standard_normal((4, dim)) * 0.1).astype(np.float32))
        b = mx.array((rng.standard_normal((dim, 4)) * 0.1).astype(np.float32))

        lora = lora_mod.LoRAAdapter(name="t")
        lora.factors[("llm_adapter.out_proj.weight")] = (a, b)
        lora.rank = 4
        lora.alpha = None

        strength_start, strength_end = 1.0, 0.0
        lora.scale = lora_mod.base_lora_scale(lora.alpha, lora.rank) * strength_start
        attached = ASDX_LoraLoader._apply_lora_to_transformer(transformer, lora, _ANIMA_CFG)

        pos, ids, weights = (
            mx.array(np.random.default_rng(2).standard_normal((1, 5, 1024)).astype(np.float32)),
            mx.array(np.arange(4, dtype=np.int32))[None],
            mx.ones((1, 4), dtype=mx.float32),
        )

        # step 1 of 3: schedule strength at progress=1/3 (linear, first half).
        total_steps = 3
        step = 1
        progress = step / total_steps
        expected_strength = strength_start + (1.0 - strength_start) * (progress * 2)
        expected_scale = lora_mod.base_lora_scale(lora.alpha, lora.rank) * expected_strength

        schedule = {
            "name": "t", "lora": lora, "strength_start": strength_start,
            "strength_end": strength_end, "strength_middle": 1.0, "strength_curve": "linear",
        }
        scheduled_transformer = core_mod._SamplerCore._update_lora_schedule(
            attached, _ANIMA_CFG, schedule, step, total_steps
        )
        actual_ctx = scheduled_transformer.encode_context(pos, ids, weights)

        # Independently rebuild a transformer at exactly `expected_scale` and
        # compare its context to the one the schedule path produced.
        lora_manual = lora_mod.LoRAAdapter(name="t")
        lora_manual.factors[("llm_adapter.out_proj.weight")] = (a, b)
        lora_manual.rank = 4
        lora_manual.alpha = None
        lora_manual.scale = expected_scale
        manual_transformer = ASDX_LoraLoader._apply_lora_to_transformer(transformer, lora_manual, _ANIMA_CFG)
        expected_ctx = manual_transformer.encode_context(pos, ids, weights)

        np.testing.assert_allclose(np.array(actual_ctx), np.array(expected_ctx), atol=1e-5)


def test_llm_adapter_schedule_repeated_run_is_reproducible(tmp_path):
    """A cache-hit re-queue reuses the SAME (already-scheduled) transformer
    object across two `_run_anima` calls -- both must give identical output
    now that context is re-encoded inside the loop instead of frozen at
    whatever strength the transformer happened to be at before this run."""
    transformer, schedule = _build_scheduled_transformer(tmp_path)
    noise = np.random.default_rng(9).standard_normal((1, 16, 4, 4)).astype(np.float32)

    with mx.stream(mx.cpu):
        out1 = _schedule_core(transformer, schedule, noise).run(steps=3, seed=0)
    with mx.stream(mx.cpu):
        out2 = _schedule_core(transformer, schedule, noise).run(steps=3, seed=0)

    torch.testing.assert_close(out1["samples"], out2["samples"], atol=1e-6, rtol=1e-6)
