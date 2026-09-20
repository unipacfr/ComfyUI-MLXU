"""End-to-end DiT parity with the REAL ComfyUI MiniMaxH3Model (tiny, random weights),
including the plain t2v case that had no numerical oracle before."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support import minimax_h3_dit_reference as R
from support.minimax_h3_module_loader import load_native_module

cond_mod = load_native_module("minimax_h3.condition")
model_mod = load_native_module("minimax_h3.model")

LAT_T, LAT_H, LAT_W, AUDIO_T, SIGMA = 2, 4, 4, 3, 0.4
# 1e-4 puts t_video above the 0.999 condition pin, so `max(t_video, vis_aug)` binds the other way
SIGMAS = (SIGMA, 1e-4)


def _target(seed=3):
    rng = np.random.default_rng(seed)
    video = rng.standard_normal((1, R.TINY["latents_dim"], LAT_T, LAT_H, LAT_W)).astype(np.float32)
    audio = rng.standard_normal((1, R.TINY["audio_latents_dim"], 2, AUDIO_T)).astype(np.float32)
    context = rng.standard_normal((R.TEXT_LEN, R.TINY["text_dim"])).astype(np.float32)
    return video, audio, context


def _reference_forward(ref, theirs, video, audio, context, sigma=SIGMA):
    import torch

    with torch.no_grad():
        out = ref._forward(
            [torch.from_numpy(video), torch.from_numpy(audio)],
            torch.tensor([sigma * 1000.0]), torch.from_numpy(context)[None],
            transformer_options={}, minimax_payload=theirs,
        )
    return out[0].numpy(), out[1].numpy()


def _our_forward(model, ours, video, audio, context, sigma=SIGMA):
    prepared = cond_mod.prepare_condition(ours, model.config.patch_size, noise_fn=R.torch_noise)
    with mx.stream(mx.cpu):
        v, a = model(mx.array(video), mx.array(audio), mx.array(context), sigma_v=sigma, cond=prepared)
        mx.eval(v, a)
    return np.array(v), np.array(a)


# Measured baseline residual (ours vs real ComfyUI, CPU stream, weights x4, output scale ~1.8):
# max 4.8e-7 over every case; tolerance ~10x that. The condition noise (1e-3 of a row) is
# also covered at the rows level, the primary guard, by test_condition.py (seed, stream, aug).
PARITY_TOL = 5e-6


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("name", list(R.CASES))
def test_forward_matches_comfyui(name, sigma):
    ref = R.build_reference_model()
    model = R.ours_from_reference(ref)
    ours, theirs = R.payload_pair(name)
    video, audio, context = _target()
    want_v, want_a = _reference_forward(ref, theirs, video, audio, context, sigma)
    got_v, got_a = _our_forward(model, ours, video, audio, context, sigma)
    assert got_v.shape == want_v.shape and got_a.shape == want_a.shape
    assert np.abs(got_v - want_v).max() < PARITY_TOL
    assert np.abs(got_a - want_a).max() < PARITY_TOL


def test_conditions_change_the_output():
    """Null case: the parity metric must see a condition (else it measures nothing)."""
    ref = R.build_reference_model()
    model = R.ours_from_reference(ref)
    video, audio, context = _target()
    plain = _our_forward(model, R.payload_pair("t2v")[0], video, audio, context)
    with_kf = _our_forward(model, R.payload_pair("keyframe_first")[0], video, audio, context)
    with_ref = _our_forward(model, R.payload_pair("ref_image")[0], video, audio, context)
    # measured effect >= 8e-2 on both outputs; threshold 10x+ below
    for other in (with_kf, with_ref):
        assert np.abs(plain[0] - other[0]).max() > 5e-3 and np.abs(plain[1] - other[1]).max() > 5e-3


def test_cond_none_equals_empty_payload():
    ref = R.build_reference_model()
    model = R.ours_from_reference(ref)
    video, audio, context = _target()
    with mx.stream(mx.cpu):
        a = model(mx.array(video), mx.array(audio), mx.array(context), sigma_v=SIGMA)
        b = model(mx.array(video), mx.array(audio), mx.array(context), sigma_v=SIGMA,
                  cond=cond_mod.prepare_condition(cond_mod.ConditionPayload(), model.config.patch_size))
        mx.eval(*a, *b)
    assert np.array_equal(np.array(a[0]), np.array(b[0])) and np.array_equal(np.array(a[1]), np.array(b[1]))


def test_text_tags_of_wrong_length_are_rejected():
    ref = R.build_reference_model()
    model = R.ours_from_reference(ref)
    video, audio, context = _target()
    bad = cond_mod.ConditionPayload(text_token_tags=np.ones(R.TEXT_LEN + 1, dtype=np.int64))
    with pytest.raises(ValueError, match="text_token_tags"):
        model(mx.array(video), mx.array(audio), mx.array(context), sigma_v=SIGMA,
              cond=cond_mod.prepare_condition(bad, model.config.patch_size))


def test_gpu_stream_agrees_with_cpu_stream():
    """The production stream: GPU fp32 matmul noise (~7.5e-4 per matmul) must stay small vs the CPU result."""
    ref = R.build_reference_model()
    model = R.ours_from_reference(ref)
    ours, _ = R.payload_pair("combined")
    video, audio, context = _target()
    prepared = cond_mod.prepare_condition(ours, model.config.patch_size, noise_fn=R.torch_noise)
    gpu = model(mx.array(video), mx.array(audio), mx.array(context), sigma_v=SIGMA, cond=prepared)
    mx.eval(*gpu)
    with mx.stream(mx.cpu):
        cpu = model(mx.array(video), mx.array(audio), mx.array(context), sigma_v=SIGMA, cond=prepared)
        mx.eval(*cpu)
    scale = float(np.abs(np.array(cpu[0])).max())
    diff = float(np.abs(np.array(gpu[0]) - np.array(cpu[0])).max())
    assert diff < 5e-3 * scale  # measured 1.7e-3 * scale with the x4 weights (2e-3 was too tight); null below is 100x diff
    plain = _our_forward(model, R.payload_pair("t2v")[0], video, audio, context)
    assert np.abs(np.array(gpu[0]) - plain[0]).max() > 10 * diff  # null: the bound can fail


def _handbuilt_prepared(video_rows, audio_rows):
    """A text-only image ref (4 layout rows) and a text-only audio ref (2*2 rows), with the
    given condition rows: consistent iff video_rows has 4 and audio_rows 4 rows."""
    payload = cond_mod.ConditionPayload(refs=(
        cond_mod.RefBlock(kind="image", latent_t=1, latent_h=4, latent_w=4),
        cond_mod.RefBlock(kind="audio", ref_audio_t=2),
    ))
    return cond_mod.PreparedCondition(payload=payload, video_rows=video_rows, audio_rows=audio_rows)


@pytest.mark.parametrize("n_video, n_audio, field", [(6, 4, "video"), (4, 6, "audio")])
def test_unconsumed_condition_rows_are_rejected(n_video, n_audio, field):
    model = R.ours_from_reference(R.build_reference_model())
    video, audio, context = _target()
    rows_v = mx.zeros((n_video, R.TINY["latents_dim"] * 4))
    rows_a = mx.zeros((n_audio, R.TINY["audio_latents_dim"]))
    with pytest.raises(ValueError, match=rf"ASDX.*{field}.*rows"):
        model(mx.array(video), mx.array(audio), mx.array(context), sigma_v=SIGMA,
              cond=_handbuilt_prepared(rows_v, rows_a))


def test_exactly_consumed_condition_rows_are_accepted():
    model = R.ours_from_reference(R.build_reference_model())
    video, audio, context = _target()
    rows_v = mx.zeros((4, R.TINY["latents_dim"] * 4))
    rows_a = mx.zeros((4, R.TINY["audio_latents_dim"]))
    with mx.stream(mx.cpu):
        v, a = model(mx.array(video), mx.array(audio), mx.array(context), sigma_v=SIGMA,
                     cond=_handbuilt_prepared(rows_v, rows_a))
        mx.eval(v, a)
    assert v.shape == video.shape
