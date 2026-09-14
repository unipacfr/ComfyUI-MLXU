"""Wiring/integration tests for the assembled MiniMaxH3Model.

Every piece this model assembles (rope, attention, layout, patchify,
curve time embedding, adaln modulation) already has its own unit tests
verified numerically against the real reference (see the sibling test
files). This file checks the wiring: shapes, the output sign convention,
and that changing an input actually changes the output through the full
stack -- not a full numeric oracle re-derivation at this level, which
would require running the real 784-line PyTorch model end-to-end (heavy
ComfyUI runtime dependencies, out of scope for a unit test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

config_mod = load_native_module("minimax_h3.config")
model_mod = load_native_module("minimax_h3.model")

MiniMaxH3Config = config_mod.MiniMaxH3Config
MiniMaxH3Model = model_mod.MiniMaxH3Model


def _tiny_config(**overrides) -> "MiniMaxH3Config":
    base = dict(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=16,
        latents_dim=4,
        audio_latents_dim=6,
        attention_head_dim=16,
        num_attention_heads=2,
        ffn_hidden_size=32,
        text_dim=10,
        patch_size=(1, 2, 2),
        # rot_dim = rope_inv_freq_len*3*2 = 2*3*2 = 12, must stay < head_dim
        # (16) -- partial rope, matching the real checkpoint's 96-of-128 ratio.
        rope_inv_freq_len=2,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        final_norm_eps=1e-5,
        sigma_shift_video=12.0,
        sigma_shift_audio=3.0,
        gate_compress=False,
        adaln_curve_grid=17,
        time_embed_dim=6,
        dtype="float32",
    )
    base.update(overrides)
    return MiniMaxH3Config(**base)


def _inputs(cfg, text_len=3, latent_t=1, lat_h=4, lat_w=4, audio_t=2):
    video = mx.random.normal((1, cfg.latents_dim, latent_t, lat_h, lat_w))
    audio = mx.random.normal((1, cfg.audio_latents_dim, 2, audio_t))
    context = mx.random.normal((text_len, cfg.text_dim))
    return video, audio, context


def test_output_shapes_and_finite():
    cfg = _tiny_config()
    model = MiniMaxH3Model(cfg)
    video, audio, context = _inputs(cfg, latent_t=1, lat_h=4, lat_w=4, audio_t=2)

    video_out, audio_out = model(video, audio, context, sigma_v=0.5)

    assert video_out.shape == video.shape
    assert audio_out.shape == audio.shape
    assert bool(mx.all(mx.isfinite(video_out)).item())
    assert bool(mx.all(mx.isfinite(audio_out)).item())


def test_output_is_negated_relative_to_raw_final_layer():
    # Zero every block's adaln gates so the block stack is the identity
    # (verified independently in test_dit_block.py), then rebuild h via the
    # model's own submodules and confirm final_layer(h) == -model_output.
    layout_mod = load_native_module("minimax_h3.layout")
    patchify_mod = load_native_module("minimax_h3.patchify")

    cfg = _tiny_config(num_layers=1, token_refiner_num_layers=1)
    model = MiniMaxH3Model(cfg)
    for block in model.blocks:
        block.adaln_proj.linear.weight = mx.zeros_like(block.adaln_proj.linear.weight)
        block.adaln_proj.linear.bias = mx.zeros_like(block.adaln_proj.linear.bias)

    video, audio, context = _inputs(cfg, text_len=3, latent_t=1, lat_h=4, lat_w=4, audio_t=2)
    sigma_v = 0.3
    video_out, audio_out = model(video, audio, context, sigma_v=sigma_v)

    t_v = 1.0 - sigma_v
    t_a = 1.0 - model_mod.time_shift_sigma(sigma_v, cfg.sigma_shift_video, cfg.sigma_shift_audio)
    layout = layout_mod.PackedLayout(3, 1, 4, 4, 2)
    mod_segments, unique_t = model_mod.build_mod_segments(layout.segments, t_v, t_a)
    t_emb = model_mod.curve_time_embedding(model.adaln_t_table, mx.array(unique_t, dtype=mx.float32))

    video_rows = patchify_mod.patchify_video(video, cfg.patch_size)
    audio_rows = patchify_mod.pack_audio(audio)
    video_embed = model.video_patch_proj(video_rows)
    audio_embed = model.audio_patch_proj(audio_rows)
    text_states = model.token_refiner(model.condition_proj(context))
    h = mx.concatenate([text_states, audio_embed, video_embed], axis=0)  # blocks are identity, skip them

    va, vb, _ = next(s for s in layout.segments if s[2] == "video")
    aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")
    v, a = model.final_layer(h, t_emb, (va, vb, unique_t.index(t_v)), (aa, ab, unique_t.index(t_a)))
    expected_video = patchify_mod.unpatchify_video(v, 1, 2, 2, cfg.latents_dim, cfg.patch_size)
    expected_audio = patchify_mod.unpack_audio(a)

    assert bool(mx.allclose(video_out, -expected_video, atol=1e-4).item())
    assert bool(mx.allclose(audio_out, -expected_audio, atol=1e-4).item())


def test_different_sigma_changes_output():
    # adaln_t_table starts as all-zeros (a placeholder for real checkpoint
    # weights, not yet wired up) -- every row would embed to the same
    # all-zero t_emb regardless of sigma, making this test vacuous unless
    # the table actually varies by row.
    cfg = _tiny_config()
    model = MiniMaxH3Model(cfg)
    model.adaln_t_table = mx.random.normal(model.adaln_t_table.shape)
    video, audio, context = _inputs(cfg)

    out_a, _ = model(video, audio, context, sigma_v=0.2)
    out_b, _ = model(video, audio, context, sigma_v=0.8)
    assert float(mx.max(mx.abs(out_a - out_b)).item()) > 1e-4


def test_different_text_context_changes_output():
    cfg = _tiny_config()
    model = MiniMaxH3Model(cfg)
    video, audio, context_a = _inputs(cfg)
    context_b = context_a + mx.random.normal(context_a.shape)

    out_a, _ = model(video, audio, context_a, sigma_v=0.5)
    out_b, _ = model(video, audio, context_b, sigma_v=0.5)
    assert float(mx.max(mx.abs(out_a - out_b)).item()) > 1e-4


def test_video_and_audio_streams_do_not_leak_output_shape():
    # Different audio_t / spatial size must not crash and must reshape
    # correctly on the way back out (exercises patchify/unpatchify wired
    # through the full stack, not just their own unit tests).
    cfg = _tiny_config()
    model = MiniMaxH3Model(cfg)
    video, audio, context = _inputs(cfg, latent_t=2, lat_h=6, lat_w=4, audio_t=5)

    video_out, audio_out = model(video, audio, context, sigma_v=0.5)
    assert video_out.shape == (1, cfg.latents_dim, 2, 6, 4)
    assert audio_out.shape == (1, cfg.audio_latents_dim, 2, 5)
