# Qwen Image 2.1 brique 2 : DiT en MLX

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Le DiT `QwenImage21Transformer2DModel` en MLX (32 blocks single-stream, modulation "scale only, no shift", attention bloc-causale texte/image, RoPE 3 axes), numériquement fidèle à `comfy/ldm/qwen_image21/model.py`, chargé depuis `qwen_image_2.1_bf16.safetensors` et `qwen_image_2.1_Q8.gguf`. T2I seul (pas de références, pas de cache préfixe).

**Architecture:** Complète `apple_silicon_nodes/native/qwen_image21/` (brique 1 : text encoder, déjà fusionnée) avec `config.py`, `dit_rope.py` (porté depuis `native/flux2/model.py`), `model.py` (le DiT), `weight_map.py` (bf16 + GGUF).

**Tech Stack:** MLX (`mlx.core`, `mlx.nn`), numpy, safetensors, `native/gguf/` (déjà en place), pytest, `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-22-qwen-image-21-brick2-dit-design.md`. Référence : `comfy/ldm/qwen_image21/model.py` (353 lignes), `comfy/ldm/flux/math.py`/`layers.py` (RoPE partagé), `comfy/supported_models.py::QwenImage21` (`latent_formats.QwenImage21` : `latent_channels=64`, `latent_dimensions=2`, `spacial_downscale_ratio=16`) dans `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI`.

## Global Constraints

- Toujours `uv run pytest` / `uv run python` (jamais `python` nu).
- Style du dépôt : Python 3.10+, annotations complètes, docstrings en anglais, pas d'emoji ni de tiret cadratin.
- Fail-closed : tenseur manquant, forme inattendue, dtype inconnu -> exception, jamais de valeur par défaut plausible.
- **Checkpoint bf16 réel inspecté directement le 2026-09-22** (`qwen_image_2.1_bf16.safetensors`,
  265 tenseurs, PAS de préfixe `model.diffusion_model.` — clés nues `img_in.weight`,
  `transformer_blocks.N....weight`, etc., déjà 1:1 avec les noms d'attributs du module) :
  - `img_in.weight [4096, 64]`, `txt_in.{in_layer,out_layer}.weight [4096,4096]`,
    `txt_in.text_norm.weight [4096]`, `time_text_embed.timestep_embedder.linear_{1,2}.weight`,
    `modulation.1.weight [16384, 4096]` (index 1 = le Linear dans le Sequential SiLU+Linear),
    `norm_out.linear.weight [4096,4096]`, `proj_out.weight [64,4096]`.
  - Par block : `transformer_blocks.N.attn.{to_q,to_k,to_v}.weight [4096,4096]`,
    `attn.to_out.0.weight [4096,4096]`, `attn.norm_{q,k}.weight [128]`,
    `img_mlp.gate_up.weight [24576,4096]`, `img_mlp.out.weight [4096,12288]`.
  - **Aucun biais nulle part** (`bias=False` partout dans la référence, confirmé par absence
    totale de clés `.bias`).
- GGUF Q8 (`qwen_image_2.1_Q8.gguf`) : même 265 tenseurs (compté directement), même noms
  attendus (conversion GGUF standard préserve les noms).
- Cache préfixe désactivé (toujours recalcul), pas de `ref_latents`/`image_slots` (T2I seul) —
  décisions de la spec brique 2.
- Les tests importent via `load_native_module("qwen_image21.<nom>")` (déjà en place,
  `tests/support/qwen_image21_module_loader.py`, brique 1).
- Les tests sur checkpoint réel (~14GB bf16 / ~7.7GB GGUF) sont gardés par
  `ASDX_FULL_GGUF_TEST=1`, `pytest.skip` si le fichier est absent.
- **Aucun commit automatique** (règle utilisateur) : chaque étape « Commit » signifie « préparer
  le message et attendre l'ordre explicite de l'utilisateur ». Pas de ligne d'attribution.

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `apple_silicon_nodes/native/qwen_image21/config.py` (créer) | `QwenImage21Config`, `detect_qwen_image21_config` |
| `apple_silicon_nodes/native/qwen_image21/dit_rope.py` (créer) | `rope_freqs`, `embed_nd`, `apply_rope`, `timestep_embedding` |
| `apple_silicon_nodes/native/qwen_image21/model.py` (créer, bâti sur 3 tâches) | Tous les modules du DiT |
| `apple_silicon_nodes/native/qwen_image21/weight_map.py` (créer) | `load_qwen_image21_dit_checkpoint`, `load_qwen_image21_dit_from_gguf` |
| `tests/native/qwen_image21/test_config.py` (créer) | détection de config |
| `tests/native/qwen_image21/test_dit_rope.py` (créer) | parité RoPE contre `native/flux2/model.py` |
| `tests/native/qwen_image21/test_model_layers.py` (créer) | ZeroCenteredRMSNorm/TextProjection/TimestepProjEmbeddings/SwiGLUFeedForward/Attention |
| `tests/native/qwen_image21/test_model_block.py` (créer) | TransformerBlock/LastLayer/block_causal_attention |
| `tests/native/qwen_image21/test_model.py` (créer) | QwenImage21Transformer2DModel bout en bout (poids aléatoires) |
| `tests/native/qwen_image21/test_weight_map.py` (créer) | chargement bf16 + GGUF, `matched N/M` |

---

### Task 1: Config + détection

**Files:**
- Create: `apple_silicon_nodes/native/qwen_image21/config.py`
- Test: `tests/native/qwen_image21/test_config.py`

**Interfaces:**
- Consumes: rien.
- Produces: `QwenImage21Config` (frozen dataclass : `in_channels=64, out_channels=64,
  num_layers=32, attention_head_dim=128, num_attention_heads=32, context_in_dim=4096,
  mlp_ratio=3, axes_dims_rope=(16,56,56), eps=1e-6, dtype="float16"`, propriété `inner_dim`
  = `num_attention_heads * attention_head_dim`, `mlx_dtype`).
  `detect_qwen_image21_config(state_dict, dtype="float16") -> QwenImage21Config`. Consommé par
  les tâches 3, 4, 5, 6.

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""Tests for QwenImage21Config and its checkpoint detection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

config_mod = load_native_module("qwen_image21.config")
QwenImage21Config = config_mod.QwenImage21Config
detect_qwen_image21_config = config_mod.detect_qwen_image21_config


def test_defaults_match_real_checkpoint_header():
    cfg = QwenImage21Config()
    assert cfg.in_channels == 64
    assert cfg.out_channels == 64
    assert cfg.num_layers == 32
    assert cfg.attention_head_dim == 128
    assert cfg.num_attention_heads == 32
    assert cfg.context_in_dim == 4096
    assert cfg.mlp_ratio == 3
    assert cfg.axes_dims_rope == (16, 56, 56)
    assert cfg.inner_dim == 4096


def test_mlx_dtype_rejects_unknown():
    cfg = QwenImage21Config(dtype="int8")
    with pytest.raises(ValueError, match="dtype"):
        _ = cfg.mlx_dtype


def _fake_state_dict(num_layers=2, inner_dim=16, head_dim=4, context_dim=32, in_ch=8):
    heads = inner_dim // head_dim
    sd = {
        "img_in.weight": np.zeros((inner_dim, in_ch), dtype=np.float32),
        "txt_in.in_layer.weight": np.zeros((inner_dim, context_dim), dtype=np.float32),
        "proj_out.weight": np.zeros((in_ch, inner_dim), dtype=np.float32),
    }
    for i in range(num_layers):
        p = f"transformer_blocks.{i}."
        sd[p + "attn.to_q.weight"] = np.zeros((inner_dim, inner_dim), dtype=np.float32)
        sd[p + "attn.norm_q.weight"] = np.zeros((head_dim,), dtype=np.float32)
    return sd


def test_detect_from_real_shapes():
    sd = _fake_state_dict()
    cfg = detect_qwen_image21_config(sd)
    assert cfg.num_layers == 2
    assert cfg.inner_dim == 16
    assert cfg.attention_head_dim == 4
    assert cfg.num_attention_heads == 4
    assert cfg.context_in_dim == 32
    assert cfg.in_channels == 8
    assert cfg.out_channels == 8


def test_detect_raises_on_missing_keys():
    with pytest.raises(ValueError, match="cannot detect"):
        detect_qwen_image21_config({})
```

Run: `uv run pytest tests/native/qwen_image21/test_config.py -q` -> FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implémenter**

```python
"""Qwen Image 2.1 DiT (QwenImage21Transformer2DModel) configuration.

Ported from `comfy/ldm/qwen_image21/model.py::QwenImage21Transformer2DModel.__init__`'s
defaults -- verified against the real checkpoint's own header (`qwen_image_2.1_bf16.
safetensors`, 265 tensors, inspected directly on 2026-09-22): no `model.diffusion_model.`
prefix, bare keys already matching this module's own attribute names 1:1, no bias anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class QwenImage21Config:
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    eps: float = 1e-6
    dtype: str = "float16"

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def mlx_dtype(self) -> mx.Dtype:
        dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
        if self.dtype not in dtype_map:
            raise ValueError(f"ASDX: unknown Qwen Image 2.1 DiT dtype {self.dtype!r}")
        return dtype_map[self.dtype]


def detect_qwen_image21_config(state_dict: dict[str, Any], dtype: str = "float16") -> QwenImage21Config:
    """Derive config from a checkpoint's own tensor shapes."""
    img_in_w = state_dict.get("img_in.weight")
    txt_in_w = state_dict.get("txt_in.in_layer.weight")
    proj_out_w = state_dict.get("proj_out.weight")
    q_w = state_dict.get("transformer_blocks.0.attn.to_q.weight")
    q_norm_w = state_dict.get("transformer_blocks.0.attn.norm_q.weight")
    if img_in_w is None or txt_in_w is None or proj_out_w is None or q_w is None or q_norm_w is None:
        raise ValueError(
            "ASDX: cannot detect Qwen Image 2.1 DiT config -- checkpoint is missing "
            "img_in.weight / txt_in.in_layer.weight / proj_out.weight / "
            "transformer_blocks.0.attn.{to_q,norm_q}.weight after key normalization."
        )

    layer_indices = set()
    for key in state_dict:
        if key.startswith("transformer_blocks."):
            layer_indices.add(int(key.split(".")[1]))
    num_layers = max(layer_indices) + 1 if layer_indices else 0

    inner_dim, in_channels = img_in_w.shape
    context_in_dim = txt_in_w.shape[1]
    out_channels = proj_out_w.shape[0]
    attention_head_dim = q_norm_w.shape[0]
    num_attention_heads = inner_dim // attention_head_dim

    return QwenImage21Config(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        context_in_dim=context_in_dim,
        dtype=dtype,
    )
```

- [ ] **Step 3: Vérifier le succès**

Run: `uv run pytest tests/native/qwen_image21/test_config.py -q` -> PASS (4 tests).

- [ ] **Step 4: Commit**

```bash
git add apple_silicon_nodes/native/qwen_image21/config.py tests/native/qwen_image21/test_config.py
git commit -m "feat: Qwen Image 2.1 DiT config + checkpoint detection"
```

---

### Task 2: RoPE (dit_rope.py, porté depuis flux2)

**Files:**
- Create: `apple_silicon_nodes/native/qwen_image21/dit_rope.py`
- Test: `tests/native/qwen_image21/test_dit_rope.py`

**Interfaces:**
- Consumes: rien.
- Produces: `rope_freqs(pos, dim, theta) -> mx.array` (`[N, dim/2, 2, 2]`), `embed_nd(ids,
  axes_dim, theta) -> mx.array`, `apply_rope(x, freqs) -> mx.array`, `timestep_embedding(t, dim,
  max_period=10000.0, time_factor=1000.0) -> mx.array`. Consommé par la tâche 5.

- [ ] **Step 1: Écrire le test qui échoue**

```python
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
    """flux2's model.py is not importable standalone via load_native_module (it pulls in
    the rest of Flux2Transformer's dependencies); import its rope functions directly via
    the module's own file path instead."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "apple_silicon_nodes" / "native" / "flux2" / "model.py"
    spec = importlib.util.spec_from_file_location("flux2_model_for_rope_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
```

Run: `uv run pytest tests/native/qwen_image21/test_dit_rope.py -q` -> FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implémenter**

```python
"""FLUX-style N-axis RoPE, ported verbatim from `native/flux2/model.py`'s `rope_freqs`/
`embed_nd`/`apply_rope`/`timestep_embedding` -- the real ComfyUI reference for Qwen Image
2.1's DiT (`comfy/ldm/qwen_image21/model.py`) imports `comfy.ldm.flux.layers.EmbedND` and
`comfy.ldm.flux.math.apply_rope1`/`rope` directly, unchanged, so this is the same math,
just with axes_dims_rope=(16, 56, 56) instead of Flux2's own axis split. Duplicated per
this project's per-family file convention (see Krea2's own `rope.py`) rather than
cross-imported from `flux2`.
"""

from __future__ import annotations

import math

import mlx.core as mx


def rope_freqs(pos: mx.array, dim: int, theta: float) -> mx.array:
    """[N, dim/2, 2, 2] rotation-matrix RoPE table for one axis."""
    assert dim % 2 == 0
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta**scale)
    out = pos.astype(mx.float32)[:, None] * omega[None, :]
    cos, sin = mx.cos(out), mx.sin(out)
    return mx.stack([cos, -sin, sin, cos], axis=-1).reshape(*out.shape, 2, 2)


def embed_nd(ids: mx.array, axes_dim: tuple[int, ...], theta: float) -> mx.array:
    """N-axis RoPE embedding table. ids: [N, len(axes_dim)]."""
    parts = [rope_freqs(ids[:, i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return mx.concatenate(parts, axis=-3)


def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply the [...,2,2] rotation-matrix RoPE to Q or K. x: [B,H,N,D]."""
    B, H, N, D = x.shape
    x_pairs = x.reshape(B, H, N, D // 2, 1, 2)
    f = freqs[None, None]
    out = (f[..., 0] * x_pairs[..., 0]) + (f[..., 1] * x_pairs[..., 1])
    return out.reshape(B, H, N, D)


def timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0,
                        time_factor: float = 1000.0) -> mx.array:
    """Sinusoidal timestep embedding, matching comfy.ldm.flux.layers.timestep_embedding."""
    t = time_factor * t
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb
```

- [ ] **Step 3: Vérifier le succès**

Run: `uv run pytest tests/native/qwen_image21/test_dit_rope.py -q` -> PASS (5 tests).

- [ ] **Step 4: Commit**

```bash
git add apple_silicon_nodes/native/qwen_image21/dit_rope.py tests/native/qwen_image21/test_dit_rope.py
git commit -m "feat: Qwen Image 2.1 DiT RoPE (ported from flux2)"
```

---

### Task 3: Modèle — couches de base

**Files:**
- Create: `apple_silicon_nodes/native/qwen_image21/model.py`
- Test: `tests/native/qwen_image21/test_model_layers.py`

**Interfaces:**
- Consumes: `QwenImage21Config` (Task 1), `timestep_embedding` (Task 2).
- Produces: `ZeroCenteredRMSNorm(dim, eps)`, `TextProjection(in_dim, hidden_size, eps)`,
  `TimestepProjEmbeddings(embedding_dim)`, `SwiGLUFeedForward(dim, hidden_dim)`,
  `Attention(dim, heads, dim_head, eps)`. Consommé par les tâches 4 et 5.

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""Tests for the Qwen Image 2.1 DiT's base layers (no block/model assembly yet)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

model_mod = load_native_module("qwen_image21.model")
dit_rope_mod = load_native_module("qwen_image21.dit_rope")

ZeroCenteredRMSNorm = model_mod.ZeroCenteredRMSNorm
TextProjection = model_mod.TextProjection
TimestepProjEmbeddings = model_mod.TimestepProjEmbeddings
SwiGLUFeedForward = model_mod.SwiGLUFeedForward
Attention = model_mod.Attention


def test_zero_centered_rms_norm_matches_manual():
    dim = 16
    norm = ZeroCenteredRMSNorm(dim)
    norm.weight = mx.random.normal((dim,)) * 0.1
    x = mx.random.normal((3, dim))
    out = norm(x)

    w = np.array(norm.weight) + 1.0
    xn = np.array(x).astype(np.float32)
    variance = (xn * xn).mean(axis=-1, keepdims=True)
    expected = xn / np.sqrt(variance + norm.eps) * w
    assert np.allclose(np.array(out), expected, atol=1e-4)


def test_text_projection_shape_and_finite():
    proj = TextProjection(in_dim=32, hidden_size=16)
    x = mx.random.normal((2, 5, 32))
    out = proj(x)
    assert out.shape == (2, 5, 16)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_timestep_proj_embeddings_shape():
    emb = TimestepProjEmbeddings(embedding_dim=16)
    t = mx.array([0.0, 0.5, 1.0])
    out = emb(t)
    assert out.shape == (3, 16)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_swiglu_feed_forward_matches_manual():
    ff = SwiGLUFeedForward(dim=8, hidden_dim=16)
    x = mx.random.normal((3, 8))
    out = ff(x)

    gate_up = np.array(ff.gate_up(x))
    gate, up = gate_up[..., :16], gate_up[..., 16:]
    silu_gate = gate / (1.0 + np.exp(-gate))
    expected = np.array(ff.out(mx.array(silu_gate * up)))
    assert np.allclose(np.array(out), expected, atol=1e-4)


def test_attention_output_shape_and_finite():
    attn = Attention(dim=32, heads=4, dim_head=8)
    x = mx.random.normal((1, 6, 32))
    ids = mx.arange(6, dtype=mx.float32)[:, None] * mx.ones((1, 3))
    pe = dit_rope_mod.embed_nd(ids, (2, 3, 3), 10000.0)[None]  # [1, 3, 6, 4, 2, 2]
    pe = pe.transpose(0, 2, 1, 3, 4, 5)  # match attn's expected pe layout (see Step 2 docstring)
    out = attn(x, pe)
    assert out.shape == (1, 6, 32)
    assert bool(mx.all(mx.isfinite(out)).item())
```

Run: `uv run pytest tests/native/qwen_image21/test_model_layers.py -q` -> FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implémenter**

```python
"""Qwen Image 2.1 DiT (`QwenImage21Transformer2DModel`) -- ported from
`comfy/ldm/qwen_image21/model.py`. T2I only: no `ref_latents`/`image_slots` (no multi-image
editing/reference conditioning in this brick), no prefix KV cache across sampling steps
(recompute every step -- numerically identical to the cached path, just slower; no other
family in this project has that optimization either). Ports the reference's `in_training`
branch throughout (plain PyTorch-equivalent math), never the fused `comfy.quant_ops.ck.*`
kernel branch -- mathematically identical, ComfyUI itself falls back to this branch outside
its compiled-kernel fast path.

This file is built across three plan tasks (this one: base layers; next: TransformerBlock/
LastLayer/block_causal_attention; then: full model assembly) -- see the plan for the split.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .dit_rope import timestep_embedding


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    x32 = x.astype(mx.float32)
    variance = mx.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * mx.rsqrt(variance + eps)).astype(x.dtype) * weight


class ZeroCenteredRMSNorm(nn.Module):
    """Stored weight is scale - 1 (checkpoint convention); applied as (weight + 1) in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x.astype(mx.float32), self.weight.astype(mx.float32) + 1.0, self.eps).astype(x.dtype)


class RMSNorm(nn.Module):
    """Plain (non-zero-centered) RMSNorm, matching operations.RMSNorm -- used for
    Attention's per-head norm_q/norm_k, which the checkpoint stores as a plain scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x, self.weight, self.eps)


class TextProjection(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps)
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.gelu_approx(self.in_layer(self.text_norm(x))))


class TimestepProjEmbeddings(nn.Module):
    """`comfy.ldm.lightricks.model.TimestepEmbedding`: Linear -> SiLU -> Linear, bias-free
    (`sample_proj_bias=False`), matching the real checkpoint's `linear_1`/`linear_2` keys."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, embedding_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def __call__(self, timestep: mx.array) -> mx.array:
        emb = timestep_embedding(timestep.astype(mx.float32), 256)
        return self.linear_2(nn.silu(self.linear_1(emb.astype(timestep.dtype))))


class SwiGLUFeedForward(nn.Module):
    """Fused gate_up (checkpoint has `img_mlp.gate_up.weight [2*hidden_dim, dim]`,
    `img_mlp.out.weight [dim, hidden_dim]` -- always `fused=True` for the real checkpoint,
    the `fused=False` branch in the reference is dead for this model and not ported)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up(x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.out(nn.silu(gate) * up)


class Attention(nn.Module):
    """`(B, N, H, D)` throughout -- no transposes before RoPE, matching the reference's
    fused-kernel-friendly layout. Ports the `in_training` branch: RMSNorm on Q/K, then RoPE,
    then scaled-dot-product attention. `pe`: `[B, N, 1, head_dim//2, 2, 2]` (broadcastable
    against Q/K reshaped to `[B,H,N,D]` by `apply_rope`'s own broadcasting -- see
    `dit_rope.apply_rope`'s `f[None,None]` against `x_pairs` `[B,H,N,D/2,1,2]`)."""

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = [nn.Linear(inner_dim, dim, bias=False)]
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def __call__(self, x: mx.array, pe: mx.array, attn_fn=None) -> mx.array:
        """`attn_fn(q, k, v, heads) -> [B, N, inner_dim]` implements the block-causal
        attention pattern (Task 4); when omitted, plain full self-attention (used by this
        task's own unit tests only, never by the assembled model)."""
        B, N, _ = x.shape
        q = self.to_q(x).reshape(B, N, self.heads, self.dim_head)
        k = self.to_k(x).reshape(B, N, self.heads, self.dim_head)
        v = self.to_v(x).reshape(B, N, self.heads, self.dim_head)

        q, k = self.norm_q(q), self.norm_k(k)
        # apply_rope expects [B,H,N,D]; Attention's own layout is [B,N,H,D] (see docstring)
        from .dit_rope import apply_rope
        q = apply_rope(q.transpose(0, 2, 1, 3), pe).transpose(0, 2, 1, 3)
        k = apply_rope(k.transpose(0, 2, 1, 3), pe).transpose(0, 2, 1, 3)

        if attn_fn is None:
            qh, kh, vh = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
            scale = 1.0 / (self.dim_head**0.5)
            out = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=scale)
            out = out.transpose(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head)
        else:
            out = attn_fn(q, k, v, self.heads)
        return self.to_out[0](out)
```

- [ ] **Step 3: Vérifier le succès**

Run: `uv run pytest tests/native/qwen_image21/test_model_layers.py -q` -> PASS (5 tests). If
the `Attention` test's `pe` shape wiring doesn't line up with `apply_rope`'s broadcasting on
the first try, adjust the transpose/reshape in `Attention.__call__` to match -- the shape
contract that must hold is: `apply_rope(x, freqs)` requires `x: [B,H,N,D]` and
`freqs: [n_axes_concat, N, D//(2*n_axes)... ]` per `dit_rope.py`'s own shapes (verified in
Task 2's tests); get `pe` into a `[N, D/2, 2, 2]`-per-batch-row shape that broadcasts the
same way flux2's own `Attention`/`joint_attention` caller does (`native/flux2/model.py`'s
`joint_attention` is the reference for how `pe` is shaped when calling `apply_rope` in this
codebase -- read it if the shape doesn't line up empirically).

- [ ] **Step 4: Commit**

```bash
git add apple_silicon_nodes/native/qwen_image21/model.py tests/native/qwen_image21/test_model_layers.py
git commit -m "feat: Qwen Image 2.1 DiT base layers (norms, projections, attention)"
```

---

### Task 4: Modèle — TransformerBlock, LastLayer, attention bloc-causale

**Files:**
- Modify: `apple_silicon_nodes/native/qwen_image21/model.py`
- Test: `tests/native/qwen_image21/test_model_block.py`

**Interfaces:**
- Consumes: `Attention`, `SwiGLUFeedForward` (Task 3).
- Produces: `block_causal_attention(segments) -> attn_fn`, `QwenImage21TransformerBlock(dim,
  num_attention_heads, attention_head_dim, mlp_ratio, eps)`, `LastLayer(dim, eps)`. Consommé
  par la tâche 5.

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""Tests for QwenImage21TransformerBlock, LastLayer, block_causal_attention."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

model_mod = load_native_module("qwen_image21.model")
dit_rope_mod = load_native_module("qwen_image21.dit_rope")

block_causal_attention = model_mod.block_causal_attention
QwenImage21TransformerBlock = model_mod.QwenImage21TransformerBlock
LastLayer = model_mod.LastLayer


def _pe(seq_len, axes_dim=(2, 3, 3), theta=10000.0):
    ids = mx.arange(seq_len, dtype=mx.float32)[:, None] * mx.ones((1, 3))
    return dit_rope_mod.embed_nd(ids, axes_dim, theta)


def test_block_causal_attention_text_only_sees_itself_causally():
    # 2 segments: text [0,3) causal, image [3,5) full (sees everything incl. text).
    txt_len, total = 3, 5
    causal = mx.array(np.tril(np.ones((txt_len, txt_len), dtype=bool)))
    segments = [(0, txt_len, causal), (txt_len, total, None)]
    attn_fn = block_causal_attention(segments)

    heads, dim_head = 2, 4
    q = mx.random.normal((1, total, heads, dim_head))
    k = mx.random.normal((1, total, heads, dim_head))
    v = mx.random.normal((1, total, heads, dim_head))
    out = attn_fn(q, k, v, heads)
    assert out.shape == (1, total, heads * dim_head)
    assert bool(mx.all(mx.isfinite(out)).item())

    # Changing a later TEXT token's k/v must not change an earlier text token's output
    # (causal within the text segment), but changing an IMAGE token must not affect text
    # rows at all (text never attends to image).
    k2 = k.at[:, 2].add(10.0)  # perturb last text token
    out2 = attn_fn(q, k2, v, heads)
    assert bool(mx.allclose(out[:, :2], out2[:, :2], atol=1e-4).item())  # rows 0,1 unaffected
    assert not bool(mx.allclose(out[:, 2], out2[:, 2], atol=1e-4).item())  # row 2 sees itself

    k3 = k.at[:, 4].add(10.0)  # perturb an image token
    out3 = attn_fn(q, k3, v, heads)
    assert bool(mx.allclose(out[:, :3], out3[:, :3], atol=1e-4).item())  # text rows unaffected


def test_block_causal_attention_image_sees_prefix_and_itself():
    txt_len, total = 2, 4
    causal = mx.array(np.tril(np.ones((txt_len, txt_len), dtype=bool)))
    segments = [(0, txt_len, causal), (txt_len, total, None)]
    attn_fn = block_causal_attention(segments)
    heads, dim_head = 1, 4
    q = mx.random.normal((1, total, heads, dim_head))
    k = mx.random.normal((1, total, heads, dim_head))
    v = mx.random.normal((1, total, heads, dim_head))
    out = attn_fn(q, k, v, heads)

    k2 = k.at[:, 0].add(10.0)  # perturb a text token
    out2 = attn_fn(q, k2, v, heads)
    assert not bool(mx.allclose(out[:, 2:], out2[:, 2:], atol=1e-4).item())  # image rows see text


def test_transformer_block_output_shape_and_finite():
    dim, heads, head_dim = 32, 4, 8
    block = QwenImage21TransformerBlock(dim, heads, head_dim, mlp_ratio=2)
    seq = 6
    x = mx.random.normal((1, seq, dim))
    pe = _pe(seq, axes_dim=(head_dim // 4, head_dim // 4 * 3 // 2, head_dim // 4 * 3 // 2))
    # mod: (scale1, gate1, scale2, gate2, zero), each (prefix_row, target_rows)
    zero = mx.zeros((1, 1, dim))
    def split(t):
        return t[-1:][None], t[:-1][None]
    scale1 = split(mx.random.normal((seq, dim)))
    gate1 = split(mx.random.normal((seq, dim)))
    scale2 = split(mx.random.normal((seq, dim)))
    gate2 = split(mx.random.normal((seq, dim)))
    mod = (scale1, gate1, scale2, gate2, zero)
    attn_fn = block_causal_attention([(0, seq, None)])
    out = block(x, mod, pe, attn_fn, prefix_len=0)
    assert out.shape == x.shape
    assert bool(mx.all(mx.isfinite(out)).item())


def test_last_layer_output_shape():
    dim = 16
    layer = LastLayer(dim)
    x = mx.random.normal((1, 5, dim))
    temb = mx.random.normal((1, dim))
    out = layer(x, temb)
    assert out.shape == x.shape
```

Run: `uv run pytest tests/native/qwen_image21/test_model_block.py -q` -> FAIL.

- [ ] **Step 2: Ajouter à `model.py`**

```python
def _split_rows(p: mx.array) -> tuple[mx.array, mx.array]:
    """Shared modulation rows: (t=0 row for the text prefix, sampled-t rows for the
    target). `p`: [seq, dim] -> (prefix [1, 1, dim], target [1, seq-1, dim])."""
    return p[-1:][None], p[:-1][None]


def _modulated_norm(norm: nn.Module, x: mx.array, scale: tuple[mx.array, mx.array], prefix_len: int) -> mx.array:
    """LayerNorm (no affine) * (1 + scale), the target scale for every row, the prefix
    rows (if any) redone with the t=0 scale. Plain-math equivalent of the reference's
    `comfy.quant_ops.ck.adaln` fused-kernel branch."""
    s_prefix, s_target = scale
    out = norm(x)
    if prefix_len:
        prefix_out = out[:, :prefix_len] * (1 + s_prefix)
        target_out = out[:, prefix_len:] * (1 + s_target)
        return mx.concatenate([prefix_out, target_out], axis=1)
    return out * (1 + s_target)


def _gated_residual(x: mx.array, y: mx.array, gate: tuple[mx.array, mx.array], prefix_len: int) -> mx.array:
    g_prefix, g_target = gate
    if prefix_len:
        prefix_out = x[:, :prefix_len] + y[:, :prefix_len] * g_prefix
        target_out = x[:, prefix_len:] + y[:, prefix_len:] * g_target
        return mx.concatenate([prefix_out, target_out], axis=1)
    return x + y * g_target


class LayerNormNoAffine(nn.Module):
    """LayerNorm with elementwise_affine=False: normalizes, no learned scale/bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        mean = mx.mean(x32, axis=-1, keepdims=True)
        var = mx.var(x32, axis=-1, keepdims=True)
        return ((x32 - mean) * mx.rsqrt(var + self.eps)).astype(x.dtype)


def block_causal_attention(segments: list[tuple[int, int, mx.array | None]]):
    """`segments`: list of `(start, end, mask)`. Each segment's query rows `[start:end]`
    attend over key/value rows `[0:end]` (everything up to and including this segment),
    with `mask` applied if given (a `[n, end]` boolean mask, `True` = attend) or full
    attention over `[0:end]` if `mask` is `None`. Matches the reference's
    `block_causal_attention` exactly, minus the prefix-cache `cache.put` call (cache
    disabled in this brick -- see the brick 2 design's Risques section)."""

    def attn(q: mx.array, k: mx.array, v: mx.array, heads: int) -> mx.array:
        B, N, H, D = q.shape
        outs = []
        for start, end, mask in segments:
            qs = q[:, start:end].reshape(B, end - start, H, D).transpose(0, 2, 1, 3)
            ks = k[:, :end].reshape(B, end, H, D).transpose(0, 2, 1, 3)
            vs = v[:, :end].reshape(B, end, H, D).transpose(0, 2, 1, 3)
            scale = 1.0 / (D**0.5)
            attn_mask = None
            if mask is not None:
                attn_mask = mx.where(mask, mx.array(0.0), mx.array(-mx.inf)).astype(qs.dtype)
            out = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=scale, mask=attn_mask)
            outs.append(out.transpose(0, 2, 1, 3).reshape(B, end - start, H * D))
        return mx.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]

    return attn


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 mlp_ratio: int = 3, eps: float = 1e-6):
        super().__init__()
        self.img_norm1 = LayerNormNoAffine(dim, eps=eps)
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, eps=eps)
        self.img_norm2 = LayerNormNoAffine(dim, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio)

    def __call__(self, x: mx.array, mod, pe: mx.array, attn_fn, prefix_len: int) -> mx.array:
        scale1, gate1, scale2, gate2, _zero = mod
        x = _gated_residual(
            x, self.attn(_modulated_norm(self.img_norm1, x, scale1, prefix_len), pe, attn_fn), gate1, prefix_len
        )
        x = _gated_residual(x, self.img_mlp(_modulated_norm(self.img_norm2, x, scale2, prefix_len)), gate2, prefix_len)
        if x.dtype == mx.float16:
            x = mx.clip(x, -65504, 65504)
        return x


class LastLayer(nn.Module):
    """Scale only, no shift."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = LayerNormNoAffine(dim, eps=eps)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        scale = self.linear(nn.silu(temb))[:, None]
        return self.norm(x) * (1 + scale)
```

Also update `Attention.__call__` to accept the block's `attn_fn` (already does, per Task 3 --
no change needed there).

- [ ] **Step 3: Vérifier le succès**

Run: `uv run pytest tests/native/qwen_image21/test_model_block.py -q` -> PASS (4 tests).
Also re-run Task 3's tests to confirm no regression: `uv run pytest tests/native/qwen_image21/test_model_layers.py -q`.

- [ ] **Step 4: Commit**

```bash
git add apple_silicon_nodes/native/qwen_image21/model.py tests/native/qwen_image21/test_model_block.py
git commit -m "feat: Qwen Image 2.1 DiT transformer block + block-causal attention"
```

---

### Task 5: Modèle — assemblage complet (T2I)

**Files:**
- Modify: `apple_silicon_nodes/native/qwen_image21/model.py`
- Test: `tests/native/qwen_image21/test_model.py`

**Interfaces:**
- Consumes: tout ce qui précède + `QwenImage21Config` (Task 1) + `embed_nd`/`timestep_embedding`
  (Task 2).
- Produces: `QwenImage21Transformer2DModel(config)` avec `__call__(self, x: mx.array, timestep:
  mx.array, context: mx.array) -> mx.array` où `x: [B, C, H, W]`, `timestep: [B]`, `context:
  [B, seq, context_in_dim]`, retour `[B, out_channels, H, W]`. Consommé par la brique 3
  (intégration, hors de ce plan).

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""End-to-end test for QwenImage21Transformer2DModel: reduced random-weight config, checks
shapes flow through and output is NaN/Inf-free (verify-checkpoint step 2)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

config_mod = load_native_module("qwen_image21.config")
model_mod = load_native_module("qwen_image21.model")

QwenImage21Config = config_mod.QwenImage21Config
QwenImage21Transformer2DModel = model_mod.QwenImage21Transformer2DModel


def _tiny_config(**overrides):
    base = dict(
        in_channels=8, out_channels=8, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, context_in_dim=16, mlp_ratio=2,
        axes_dims_rope=(2, 3, 3), eps=1e-6, dtype="float32",
    )
    base.update(overrides)
    return QwenImage21Config(**base)


def test_forward_shape_and_finite():
    cfg = _tiny_config()
    model = QwenImage21Transformer2DModel(cfg)
    B, H, W = 1, 4, 4
    x = mx.random.normal((B, cfg.in_channels, H, W))
    timestep = mx.array([0.5])
    context = mx.random.normal((B, 6, cfg.context_in_dim))
    out = model(x, timestep, context)
    assert out.shape == (B, cfg.out_channels, H, W)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_different_prompt_changes_output():
    cfg = _tiny_config()
    model = QwenImage21Transformer2DModel(cfg)
    B, H, W = 1, 4, 4
    x = mx.random.normal((B, cfg.in_channels, H, W))
    timestep = mx.array([0.5])
    context_a = mx.random.normal((B, 6, cfg.context_in_dim))
    context_b = mx.random.normal((B, 6, cfg.context_in_dim))
    out_a = model(x, timestep, context_a)
    out_b = model(x, timestep, context_b)
    assert not bool(mx.allclose(out_a, out_b, atol=1e-4).item())


def test_different_timestep_changes_output():
    cfg = _tiny_config()
    model = QwenImage21Transformer2DModel(cfg)
    B, H, W = 1, 4, 4
    x = mx.random.normal((B, cfg.in_channels, H, W))
    context = mx.random.normal((B, 6, cfg.context_in_dim))
    out_a = model(x, mx.array([0.1]), context)
    out_b = model(x, mx.array([0.9]), context)
    assert not bool(mx.allclose(out_a, out_b, atol=1e-4).item())
```

Run: `uv run pytest tests/native/qwen_image21/test_model.py -q` -> FAIL.

- [ ] **Step 2: Ajouter à `model.py`**

```python
class QwenImage21Transformer2DModel(nn.Module):
    """T2I only: `build_sequence` below is the reference's ref-loop reduced to its
    single-iteration case (`ref_latents=[]`, so the loop over `ref_latents + [x]` runs
    exactly once, for the target image `x`) -- verified against the reference's general
    form by hand-tracing that reduction (see the brick 2 design doc's Risques section),
    not re-derived from scratch. No `image_slots`/multi-reference support."""

    def __init__(self, config: QwenImage21Config):
        super().__init__()
        self.config = config
        inner_dim = config.inner_dim
        self.time_text_embed = TimestepProjEmbeddings(inner_dim)
        self.txt_in = TextProjection(config.context_in_dim, inner_dim, eps=config.eps)
        self.img_in = nn.Linear(config.in_channels, inner_dim, bias=False)
        self.modulation = [nn.SiLU(), nn.Linear(inner_dim, 4 * inner_dim, bias=False)]
        self.transformer_blocks = [
            QwenImage21TransformerBlock(inner_dim, config.num_attention_heads, config.attention_head_dim,
                                         mlp_ratio=config.mlp_ratio, eps=config.eps)
            for _ in range(config.num_layers)
        ]
        self.norm_out = LastLayer(inner_dim, eps=config.eps)
        self.proj_out = nn.Linear(inner_dim, config.out_channels, bias=False)

    def _build_sequence(self, x: mx.array, context: mx.array) -> tuple[mx.array, mx.array, list]:
        """T2I reduction of the reference's `build_sequence`: one text segment (causal),
        one image segment (full attention over text + itself). Returns
        (hidden_states [B, txt_len+H*W, inner_dim], pe [1, txt_len+H*W, 1, ..., 2, 2] per
        dit_rope.embed_nd's layout, segments [(0, txt_len, causal_mask), (txt_len, total, None)])."""
        B, C, H, W = x.shape
        txt = self.txt_in(context)
        txt_len = txt.shape[1]

        causal = mx.tril(mx.ones((txt_len, txt_len), dtype=mx.bool_))
        segments = [(0, txt_len, causal), (txt_len, txt_len + H * W, None)]

        txt_ids = mx.arange(txt_len, dtype=mx.float32)[:, None] * mx.ones((1, 3))

        img = self.img_in(x.reshape(B, C, H * W).transpose(0, 2, 1))  # [B, H*W, inner_dim]
        hh = mx.arange(H, dtype=mx.float32) - (H - H // 2)
        ww = mx.arange(W, dtype=mx.float32) - (W - W // 2)
        t_axis = mx.full((H, W), float(txt_len))
        img_ids = mx.stack([t_axis, mx.broadcast_to(hh[:, None], (H, W)), mx.broadcast_to(ww[None, :], (H, W))], axis=-1)
        img_ids = img_ids.reshape(H * W, 3)

        ids = mx.concatenate([txt_ids, img_ids], axis=0)
        pe = embed_nd(ids, self.config.axes_dims_rope, 10000.0)[None]  # [1, n_axes*..., total, 2, 2] concat on axis -3

        hidden_states = mx.concatenate([txt, img], axis=1)
        return hidden_states, pe, segments

    def __call__(self, x: mx.array, timestep: mx.array, context: mx.array) -> mx.array:
        B, C, H, W = x.shape
        dtype = x.dtype

        hidden_states, pe, segments = self._build_sequence(x, context)
        prefix_len = hidden_states.shape[1] - H * W

        t = timestep.astype(dtype)
        temb = self.time_text_embed(mx.concatenate([t, mx.zeros((1,), dtype=dtype)]))
        mod_out = temb
        for layer in self.modulation:
            mod_out = layer(mod_out)
        scale1, gate1, scale2, gate2 = mx.split(mod_out, 4, axis=-1)
        mod = (
            _split_rows(scale1), _split_rows(mx.tanh(gate1)),
            _split_rows(scale2), _split_rows(mx.tanh(gate2)),
            mx.zeros((1, 1, scale1.shape[-1])),
        )

        attn_fn = block_causal_attention(segments)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, mod, pe, attn_fn, prefix_len)

        hidden_states = self.norm_out(hidden_states[:, prefix_len:], temb[:-1])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(0, 2, 1).reshape(B, self.config.out_channels, H, W)
```

Add `from .dit_rope import embed_nd` to `model.py`'s imports (alongside the existing
`timestep_embedding` import from Task 3).

- [ ] **Step 3: Vérifier le succès**

Run: `uv run pytest tests/native/qwen_image21/test_model.py -q` -> PASS (3 tests). If a shape
mismatch appears in `_build_sequence`'s `pe` construction or `Attention`'s consumption of it,
resolve it empirically against `dit_rope.py`'s own tested shapes (Task 2) -- do not guess a fix
blind; print intermediate shapes and compare against what `apply_rope`'s docstring/tests
established. Also re-run the full directory once: `uv run pytest tests/native/qwen_image21/ -q`
(all of brique 1 + this brick's Tasks 1-5) to confirm no collection errors and no regressions.

- [ ] **Step 4: Commit**

```bash
git add apple_silicon_nodes/native/qwen_image21/model.py tests/native/qwen_image21/test_model.py
git commit -m "feat: Qwen Image 2.1 DiT full model assembly (T2I)"
```

---

### Task 6: Weight map (bf16 + GGUF)

**Files:**
- Create: `apple_silicon_nodes/native/qwen_image21/weight_map.py`
- Test: `tests/native/qwen_image21/test_weight_map.py`

**Interfaces:**
- Consumes: `QwenImage21Config`/`detect_qwen_image21_config` (Task 1),
  `QwenImage21Transformer2DModel` (Task 5), `native/gguf/reader.py`/`dequant.py` (déjà en place).
- Produces: `load_qwen_image21_dit_checkpoint(path, dtype) -> QwenImage21Transformer2DModel`
  (bf16 safetensors), `load_qwen_image21_dit_from_gguf(path, dtype) ->
  QwenImage21Transformer2DModel` (GGUF Q8). Consommé par la brique 3.

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""load_qwen_image21_dit_checkpoint / load_qwen_image21_dit_from_gguf: synthetic round-trip
tests, then real checkpoint tests (bf16 and GGUF Q8) gated by ASDX_FULL_GGUF_TEST=1.

Key remapping needed: the checkpoint's `modulation.1.weight` (index 1 = the Linear inside
the reference's `nn.Sequential(SiLU(), Linear(...))`) must map to this module's own
`modulation.1.weight` -- MLX stores a plain Python list of layers under the attribute name
directly indexed (`self.modulation = [nn.SiLU(), nn.Linear(...)]`), so `tree_flatten` already
produces `modulation.1.weight` with no remapping needed (unlike flux2's `nn.Sequential`,
which needed a `.layers.` insertion -- confirm this empirically in Step 2, don't assume)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

config_mod = load_native_module("qwen_image21.config")
model_mod = load_native_module("qwen_image21.model")
weight_map_mod = load_native_module("qwen_image21.weight_map")

QwenImage21Config = config_mod.QwenImage21Config
QwenImage21Transformer2DModel = model_mod.QwenImage21Transformer2DModel
load_qwen_image21_dit_checkpoint = weight_map_mod.load_qwen_image21_dit_checkpoint


def _tiny_config(**overrides):
    base = dict(
        in_channels=8, out_channels=8, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, context_in_dim=16, mlp_ratio=2,
        axes_dims_rope=(2, 3, 3), eps=1e-6, dtype="float32",
    )
    base.update(overrides)
    return QwenImage21Config(**base)


def test_round_trip_through_synthetic_safetensors(tmp_path):
    cfg = _tiny_config()
    original = QwenImage21Transformer2DModel(cfg)
    flat = dict(tree_flatten(original.parameters()))
    tensors = {k: np.array(v) for k, v in flat.items()}
    path = tmp_path / "tiny_dit.safetensors"
    save_file(tensors, str(path))

    loaded = load_qwen_image21_dit_checkpoint(path, dtype="float32")
    assert loaded.config == cfg
    loaded_flat = dict(tree_flatten(loaded.parameters()))
    assert set(loaded_flat.keys()) == set(flat.keys())
    for key, original_value in flat.items():
        assert np.allclose(np.array(loaded_flat[key]), np.array(original_value), atol=1e-6), key


def test_raises_on_missing_required_key(tmp_path):
    cfg = _tiny_config()
    original = QwenImage21Transformer2DModel(cfg)
    flat = dict(tree_flatten(original.parameters()))
    tensors = {k: np.array(v) for k, v in flat.items() if k != "img_in.weight"}
    path = tmp_path / "broken.safetensors"
    save_file(tensors, str(path))
    with pytest.raises((ValueError, KeyError)):
        load_qwen_image21_dit_checkpoint(path, dtype="float32")


_DIT_BF16_REAL = Path("/Volumes/X10Pro/Images/models/diffusion_models/Qwen 2/base model/qwen_image_2.1_bf16.safetensors")
_DIT_GGUF_REAL = Path("/Volumes/X10Pro/Images/models/diffusion_models/Qwen 2/gguf/qwen_image_2.1_Q8.gguf")


@pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="loads the real ~14GB Qwen Image 2.1 DiT checkpoint; set ASDX_FULL_GGUF_TEST=1 to run",
)
def test_loads_real_bf16_checkpoint():
    if not _DIT_BF16_REAL.exists():
        pytest.skip("no local qwen_image_2.1_bf16.safetensors")
    model = load_qwen_image21_dit_checkpoint(_DIT_BF16_REAL, dtype="float16")
    assert model.config.num_layers == 32
    assert model.config.inner_dim == 4096
    x = mx.random.normal((1, 64, 8, 8))
    timestep = mx.array([0.5])
    context = mx.random.normal((1, 6, 4096))
    out = model(x, timestep, context)
    assert out.shape == (1, 64, 8, 8)
    assert bool(mx.all(mx.isfinite(out)).item())


@pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="loads the real ~7.7GB Qwen Image 2.1 DiT GGUF checkpoint; set ASDX_FULL_GGUF_TEST=1 to run",
)
def test_loads_real_gguf_checkpoint():
    if not _DIT_GGUF_REAL.exists():
        pytest.skip("no local qwen_image_2.1_Q8.gguf")
    load_qwen_image21_dit_from_gguf = weight_map_mod.load_qwen_image21_dit_from_gguf
    model = load_qwen_image21_dit_from_gguf(_DIT_GGUF_REAL, dtype="float16")
    assert model.config.num_layers == 32
    x = mx.random.normal((1, 64, 8, 8))
    timestep = mx.array([0.5])
    context = mx.random.normal((1, 6, 4096))
    out = model(x, timestep, context)
    assert out.shape == (1, 64, 8, 8)
    assert bool(mx.all(mx.isfinite(out)).item())
```

Run: `uv run pytest tests/native/qwen_image21/test_weight_map.py -q` -> FAIL.

- [ ] **Step 2: Implémenter**

```python
"""Checkpoint loading for QwenImage21Transformer2DModel: bf16 safetensors (mx.load, same
bfloat16-safe approach as the brique 1 text encoder loader) and GGUF Q8 (native/gguf/).

The real checkpoint (`qwen_image_2.1_bf16.safetensors`, 265 tensors, inspected directly on
2026-09-22) has NO `model.diffusion_model.` prefix and its keys already match this module's
own parameter names 1:1 -- no remapping needed (confirmed against `tree_flatten` output in
this file's own tests, not assumed)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .model import QwenImage21Transformer2DModel
from .config import detect_qwen_image21_config


def _flatten_module_params(model: QwenImage21Transformer2DModel) -> list[tuple[str, mx.array]]:
    return list(tree_flatten(model.parameters()))


def _assign_from_state_dict(model: QwenImage21Transformer2DModel, config, state_dict: dict[str, mx.array]) -> None:
    expected_keys = set(k for k, _ in _flatten_module_params(model))
    unexpected = [k for k in state_dict if k not in expected_keys]
    if unexpected:
        raise ValueError(
            f"ASDX: {len(unexpected)} unrecognized key(s) in checkpoint, e.g. {unexpected[:5]} -- "
            "not in QwenImage21Transformer2DModel's parameters."
        )

    weights = []
    for key, _ in _flatten_module_params(model):
        if key not in state_dict:
            raise ValueError(f"ASDX: checkpoint is missing required key {key!r} for QwenImage21Transformer2DModel.")
        weights.append((key, state_dict[key].astype(config.mlx_dtype)))

    print(f"[ASDX] Qwen Image 2.1 DiT: matched {len(weights)}/{len(expected_keys)} weights.")
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    mx.clear_cache()


def load_qwen_image21_dit_checkpoint(path: str | Path, dtype: str = "float16") -> QwenImage21Transformer2DModel:
    path = Path(path)
    state_dict: dict[str, mx.array] = mx.load(str(path))
    config = detect_qwen_image21_config(state_dict, dtype=dtype)
    model = QwenImage21Transformer2DModel(config)
    _assign_from_state_dict(model, config, state_dict)
    return model


def load_qwen_image21_dit_from_gguf(path: str | Path, dtype: str = "float16") -> QwenImage21Transformer2DModel:
    from ..gguf.reader import read_gguf_tensors
    from ..gguf.dequant import dequantize_tensor

    path = Path(path)
    raw_tensors = read_gguf_tensors(str(path))
    state_dict: dict[str, mx.array] = {
        name: dequantize_tensor(info, dtype=dtype) for name, info in raw_tensors.items()
    }
    config = detect_qwen_image21_config(state_dict, dtype=dtype)
    model = QwenImage21Transformer2DModel(config)
    _assign_from_state_dict(model, config, state_dict)
    return model
```

**Note pour l'implémenteur** : `native/gguf/reader.py`/`dequant.py` existent déjà (utilisés par
`native/minimax_h3/text_encoder_weight_map.py::load_qwen3_text_encoder_from_gguf`) mais leur
API exacte (noms de fonctions, signature de retour) n'a pas été vérifiée ici — lire
`apple_silicon_nodes/native/gguf/reader.py` et `dequant.py`, et l'usage réel dans
`minimax_h3/text_encoder_weight_map.py`, avant d'écrire `load_qwen_image21_dit_from_gguf` :
les noms `read_gguf_tensors`/`dequantize_tensor` ci-dessus sont un GABARIT à corriger contre
l'API réelle, pas une transcription vérifiée (contrairement au reste de ce plan).

- [ ] **Step 3: Vérifier le succès**

Run: `uv run pytest tests/native/qwen_image21/test_weight_map.py -q -k "not real"` -> PASS.
Then: `ASDX_FULL_GGUF_TEST=1 uv run pytest tests/native/qwen_image21/test_weight_map.py -k real -v`
-> PASS (both bf16 and GGUF real-checkpoint tests).

- [ ] **Step 4: Commit**

```bash
git add apple_silicon_nodes/native/qwen_image21/weight_map.py tests/native/qwen_image21/test_weight_map.py
git commit -m "feat: Qwen Image 2.1 DiT checkpoint loader (bf16 + GGUF Q8)"
```

---

### Task 7: Vérification finale

**Files:** aucun nouveau fichier.

- [ ] **Step 1: `verify-checkpoint` skill** sur `apple_silicon_nodes/native/qwen_image21/`
  (le DiT), checkpoint réel bf16 puis GGUF Q8 : py_compile, forward pass réduit (déjà couvert
  Task 5), matched N/M sur les deux formats, std chargé vs random-init sur 5-7 poids à
  profondeurs différentes.

- [ ] **Step 2: Agent `weight-map-reviewer`** sur `weight_map.py`.

- [ ] **Step 3: Agent `comfy-reference-diff`** — diff `model.py` contre
  `comfy/ldm/qwen_image21/model.py`, en particulier `_build_sequence`'s réduction T2I et
  `block_causal_attention`'s masquage, les points les plus délicats de ce portage.

- [ ] **Step 4: Corriger tout écart trouvé, relancer** `uv run pytest tests/native/qwen_image21/ -q`
  puis `ASDX_FULL_GGUF_TEST=1 uv run pytest tests/native/qwen_image21/ -q`.

- [ ] **Step 5: Commit (si corrections)**

```bash
git add apple_silicon_nodes/native/qwen_image21/ tests/native/qwen_image21/
git commit -m "fix: Qwen Image 2.1 DiT review fixes"
```
