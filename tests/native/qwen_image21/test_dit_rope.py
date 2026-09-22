"""Tests for qwen_image21.dit_rope -- ported verbatim from native/flux2/model.py's
rope_freqs/embed_nd/apply_rope/timestep_embedding (same FLUX-style RoPE math: the real
comfy reference for Qwen Image 2.1 imports comfy.ldm.flux.layers.EmbedND and
comfy.ldm.flux.math.apply_rope1/rope directly, unchanged). Cross-checked here against
flux2's own already-shipped, already-verified implementation rather than re-deriving from
the real ComfyUI install a second time -- flux2's version is a proven component of this
codebase, not a fresh claim."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

dit_rope_mod = load_native_module("qwen_image21.dit_rope")


def _load_flux2_rope():
    """flux2's model.py uses a relative import (`from .config import Flux2Config`), so a
    bare spec_from_file_location load fails with "attempted relative import with no known
    parent package". load_native_module already solves this (it registers proper
    apple_silicon_nodes.native.* namespace packages), so reuse it here instead of a
    separate ad hoc loader."""
    return load_native_module("flux2.model")


def test_rope_freqs_matches_flux2():
    flux2 = _load_flux2_rope()
    pos = mx.arange(5, dtype=mx.float32)
    ours = dit_rope_mod.rope_freqs(pos, 16, 10000.0)
    theirs = flux2.rope_freqs(pos, 16, 10000.0)
    assert np.allclose(np.array(ours), np.array(theirs), atol=1e-6)


def test_embed_nd_matches_flux2():
    flux2 = _load_flux2_rope()
    ids = mx.array(np.random.default_rng(0).integers(0, 10, size=(7, 3)).astype(np.float32))
    ours = dit_rope_mod.embed_nd(ids, (16, 56, 56), 10000.0)
    theirs = flux2.embed_nd(ids, (16, 56, 56), 10000.0)
    assert np.allclose(np.array(ours), np.array(theirs), atol=1e-6)


def test_apply_rope_matches_flux2():
    flux2 = _load_flux2_rope()
    rng = np.random.default_rng(0)
    ids = mx.array(rng.integers(0, 10, size=(5, 3)).astype(np.float32))
    freqs = dit_rope_mod.embed_nd(ids, (16, 56, 56), 10000.0)  # [3, 5, 64, 2, 2] -> concat axis=-3
    x = mx.array(rng.standard_normal((1, 4, 5, 128)).astype(np.float32))
    ours = dit_rope_mod.apply_rope(x, freqs)
    theirs = flux2.apply_rope(x, freqs)
    assert np.allclose(np.array(ours), np.array(theirs), atol=1e-5)


def test_timestep_embedding_matches_flux2():
    flux2 = _load_flux2_rope()
    t = mx.array([0.0, 0.5, 1.0])
    ours = dit_rope_mod.timestep_embedding(t, 256)
    theirs = flux2.timestep_embedding(t, 256)
    assert np.allclose(np.array(ours), np.array(theirs), atol=1e-6)


def test_apply_rope_output_finite():
    ids = mx.arange(6, dtype=mx.float32)[:, None] * mx.ones((1, 3))
    freqs = dit_rope_mod.embed_nd(ids, (16, 56, 56), 10000.0)
    x = mx.random.normal((1, 2, 6, 128))
    out = dit_rope_mod.apply_rope(x, freqs)
    assert out.shape == x.shape
    assert bool(mx.all(mx.isfinite(out)).item())


def test_apply_rope_preserves_input_dtype():
    # Regression test: apply_rope's multiply against `freqs` (always float32, from
    # rope_freqs) silently upconverted fp16/bf16 inputs to float32 with nothing
    # casting back -- the reference's _apply_rope1 ends with .type_as(x) for exactly
    # this reason.
    ids = mx.arange(6, dtype=mx.float32)[:, None] * mx.ones((1, 3))
    freqs = dit_rope_mod.embed_nd(ids, (16, 56, 56), 10000.0)
    for dtype in (mx.float16, mx.bfloat16, mx.float32):
        x = mx.random.normal((1, 2, 6, 128)).astype(dtype)
        out = dit_rope_mod.apply_rope(x, freqs)
        assert out.dtype == dtype
