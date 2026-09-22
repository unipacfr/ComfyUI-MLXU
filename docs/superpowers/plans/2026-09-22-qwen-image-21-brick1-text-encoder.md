# Qwen Image 2.1 brique 1 : text encoder Qwen3-VL-8B en MLX

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un encodeur texte Qwen3-VL-8B en MLX (`Qwen3VL8BTextEncoder`) qui transforme des tokens ids en hidden states `[S, 4096]` numériquement fidèle à la référence ComfyUI, chargé depuis les 750 tenseurs de `qwen3vl_8b_bf16.safetensors` (partie texte uniquement — pas de tour vision, hors périmètre de cette brique).

**Architecture:** 4 fichiers dans `apple_silicon_nodes/native/qwen_image21/` : `text_encoder_config.py` (config + détection depuis checkpoint), `rope.py` (RMSNorm + RoPE fusionnés, portés depuis `minimax_h3/rope.py`), `text_encoder.py` (Attention/MLP/TransformerBlock/backbone), `text_encoder_weight_map.py` (chargement bf16 réel, avec liste explicite des clés à ignorer : tour vision + norme finale + lm_head, absentes de notre module). Aucun nœud ComfyUI dans cette brique — c'est la brique 3 (intégration) qui câble `ASDX_CLIPLoader`.

**Tech Stack:** MLX (`mlx.core`, `mlx.nn`), numpy, safetensors, pytest, `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-22-qwen-image-21-design.md`. Référence ComfyUI : `comfy/text_encoders/llama.py` (`Qwen3_8BConfig`, `Qwen3VL_8BConfig`, `Llama2_`, `TransformerBlock`, `precompute_freqs_cis`), `comfy/text_encoders/qwen3vl.py`, `comfy/text_encoders/qwen_image21.py` (`T2I_TEMPLATE`, `layer_norm_hidden_state=False`) dans `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI`. Portage direct depuis `apple_silicon_nodes/native/minimax_h3/{text_encoder_config,rope,text_encoder}.py` (même famille de config Qwen3VL*Config, math RoPE déjà vérifiée contre `comfy_kitchen` dans ce projet).

## Global Constraints

- Toujours `uv run pytest` / `uv run python` (jamais `python` nu).
- Style du dépôt : Python 3.10+, annotations complètes, docstrings en anglais, pas d'emoji ni de tiret cadratin.
- Fail-closed : un tenseur manquant, une forme inattendue ou un dtype inconnu lève une exception, jamais de valeur par défaut plausible.
- Le checkpoint réel (`qwen3vl_8b_bf16.safetensors`, 750 tenseurs, vérifié par inspection directe du header le 2026-09-22) contient une tour vision complète (`model.visual.*`, 496 tenseurs) et une norme finale + lm_head (`model.norm.weight`, `lm_head.weight`) que notre module ne construit PAS — le chargeur doit les ignorer explicitement, jamais silencieusement (voir Tâche 4).
- Les tests importent les modules via `load_native_module("qwen_image21.<nom>")` (nouveau : `tests/support/qwen_image21_module_loader.py`, copie de `minimax_h3_module_loader.py` avec le nom de famille changé).
- Les tests sur le vrai checkpoint (~14GB bf16) sont gardés par `ASDX_FULL_GGUF_TEST=1` (convention déjà utilisée dans `tests/native/minimax_h3/`), et `pytest.skip` si le fichier est absent.
- **Aucun commit automatique** (règle utilisateur) : chaque étape « Commit » signifie « préparer le message et attendre l'ordre explicite de l'utilisateur ». Pas de ligne d'attribution dans les messages de commit.

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `apple_silicon_nodes/native/qwen_image21/__init__.py` (créer) | Namespace package, ré-exports |
| `apple_silicon_nodes/native/qwen_image21/text_encoder_config.py` (créer) | `Qwen3VL8BTextEncoderConfig`, `detect_qwen3vl_8b_text_encoder_config` |
| `apple_silicon_nodes/native/qwen_image21/rope.py` (créer) | `rms_norm`, `apply_rope_split_half`, `rms_norm_rope_split_half`, `qwen3_rope_cos_sin` |
| `apple_silicon_nodes/native/qwen_image21/text_encoder.py` (créer) | `RMSNorm`, `Attention`, `MLP`, `TransformerBlock`, `Qwen3VL8BTextEncoder` |
| `apple_silicon_nodes/native/qwen_image21/text_encoder_weight_map.py` (créer) | `load_qwen_image21_text_encoder_checkpoint` |
| `tests/support/qwen_image21_module_loader.py` (créer) | `load_native_module` pour `qwen_image21.*` |
| `tests/native/qwen_image21/test_text_encoder_config.py` (créer) | détection de config, cas d'erreur |
| `tests/native/qwen_image21/test_rope.py` (créer) | parité RoPE contre la vraie `comfy.text_encoders.llama.precompute_freqs_cis`/`apply_rope` |
| `tests/native/qwen_image21/test_text_encoder.py` (créer) | Attention/MLP/TransformerBlock/backbone : formes, GQA, causalité |
| `tests/native/qwen_image21/test_text_encoder_weight_map.py` (créer) | chargement synthétique + chargement du vrai checkpoint bf16 |

---

### Task 1: Config + détection

**Files:**
- Create: `apple_silicon_nodes/native/qwen_image21/text_encoder_config.py`
- Create: `tests/support/qwen_image21_module_loader.py`
- Test: `tests/native/qwen_image21/test_text_encoder_config.py`

**Interfaces:**
- Consumes: rien.
- Produces: `Qwen3VL8BTextEncoderConfig` (frozen dataclass avec `vocab_size=151936, hidden_size=4096, intermediate_size=12288, num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6, rope_theta=5000000.0, dtype="float16"`, propriétés `kv_groups`, `mlx_dtype`). `detect_qwen3vl_8b_text_encoder_config(state_dict: dict[str, Any], dtype: str = "float16") -> Qwen3VL8BTextEncoderConfig`. Consommé par les tâches 3 et 4.

- [ ] **Step 1: Créer le module loader**

`tests/support/qwen_image21_module_loader.py` (copie exacte de `minimax_h3_module_loader.py`, nom de famille changé) :

```python
"""Load apple_silicon_nodes/native/{qwen_image21,gguf}/*.py and sibling
native/*.py modules standalone, bypassing apple_silicon_nodes/__init__.py
(which imports comfy_api and is only importable inside a running ComfyUI
process). Same trick as `krea2_module_loader.py`/`minimax_h3_module_loader.py`.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NATIVE_DIR = _REPO_ROOT / "apple_silicon_nodes" / "native"


def _register_namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load_native_module(dotted_name: str) -> types.ModuleType:
    """`dotted_name` is relative to `apple_silicon_nodes.native`, e.g.
    "safetensors_header" or "qwen_image21.text_encoder_config"."""
    _register_namespace_package("apple_silicon_nodes", _REPO_ROOT / "apple_silicon_nodes")
    _register_namespace_package("apple_silicon_nodes.native", _NATIVE_DIR)
    parts = dotted_name.split(".")
    prefix = "apple_silicon_nodes.native"
    for part in parts[:-1]:
        prefix = f"{prefix}.{part}"
        _register_namespace_package(prefix, _NATIVE_DIR / Path(*parts[: parts.index(part) + 1]))
    full_name = f"apple_silicon_nodes.native.{dotted_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    return importlib.import_module(full_name)
```

- [ ] **Step 2: Écrire le test qui échoue**

`tests/native/qwen_image21/test_text_encoder_config.py` :

```python
"""Tests for Qwen3VL8BTextEncoderConfig and its checkpoint detection."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

config_mod = load_native_module("qwen_image21.text_encoder_config")
Qwen3VL8BTextEncoderConfig = config_mod.Qwen3VL8BTextEncoderConfig
detect_qwen3vl_8b_text_encoder_config = config_mod.detect_qwen3vl_8b_text_encoder_config


def test_default_matches_real_checkpoint_header():
    cfg = Qwen3VL8BTextEncoderConfig()
    assert cfg.vocab_size == 151936
    assert cfg.hidden_size == 4096
    assert cfg.intermediate_size == 12288
    assert cfg.num_hidden_layers == 36
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.kv_groups == 4


def test_rejects_non_divisible_heads():
    with pytest.raises(ValueError, match="kv"):
        Qwen3VL8BTextEncoderConfig(num_attention_heads=33, num_key_value_heads=8)


def test_mlx_dtype_rejects_unknown():
    cfg = Qwen3VL8BTextEncoderConfig(dtype="int8")
    with pytest.raises(ValueError, match="dtype"):
        _ = cfg.mlx_dtype


def _fake_state_dict(num_layers: int = 2, hidden: int = 16, heads: int = 4, kv_heads: int = 2, head_dim: int = 4, inter: int = 32, vocab: int = 10) -> dict:
    sd = {
        "model.embed_tokens.weight": np.zeros((vocab, hidden), dtype=np.float32),
        "model.norm.weight": np.zeros((hidden,), dtype=np.float32),
        "lm_head.weight": np.zeros((vocab, hidden), dtype=np.float32),
    }
    for i in range(num_layers):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = np.zeros((heads * head_dim, hidden), dtype=np.float32)
        sd[p + "self_attn.k_proj.weight"] = np.zeros((kv_heads * head_dim, hidden), dtype=np.float32)
        sd[p + "self_attn.q_norm.weight"] = np.zeros((head_dim,), dtype=np.float32)
        sd[p + "mlp.gate_proj.weight"] = np.zeros((inter, hidden), dtype=np.float32)
    return sd


def test_detect_from_real_shapes():
    sd = _fake_state_dict()
    cfg = detect_qwen3vl_8b_text_encoder_config(sd)
    assert cfg.num_hidden_layers == 2
    assert cfg.hidden_size == 16
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 4
    assert cfg.intermediate_size == 32
    assert cfg.vocab_size == 10


def test_detect_raises_on_missing_keys():
    with pytest.raises(ValueError, match="cannot detect"):
        detect_qwen3vl_8b_text_encoder_config({})
```

- [ ] **Step 3: Vérifier l'échec**

Run: `uv run pytest tests/native/qwen_image21/test_text_encoder_config.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implémenter**

`apple_silicon_nodes/native/qwen_image21/text_encoder_config.py` :

```python
"""Qwen3-VL-8B text-only backbone configuration -- Qwen Image 2.1's text
conditioning encoder.

Ported from `comfy/text_encoders/llama.py::Qwen3VL_8BConfig` (extends
`Qwen3_8BConfig`) -- verified against the real checkpoint's own header
(`qwen3vl_8b_bf16.safetensors`, 750 tensors, inspected directly on
2026-09-22): 36 text layers, hidden_size 4096, 32 attention heads / 8 KV
heads (GQA) at head_dim 128, intermediate_size 12288, rope_theta 5000000.0.

Unlike MiniMax H3's truncated 50-of-64-layer Qwen3-VL-32B checkpoint (which
has no `model.norm.weight`/`lm_head.weight`, `final_norm=False`/
`lm_head=False` in the reference config), this checkpoint is the FULL
Qwen3-VL-8B: it DOES contain `model.norm.weight` and `lm_head.weight`, plus a
complete `model.visual.*` vision tower (496 tensors). This module builds
neither the final norm, the lm_head, nor the vision tower -- Qwen Image
2.1's T2I path consumes the raw last-block hidden state
(`comfy/text_encoders/qwen_image21.py::QwenImage21Qwen3VLClipModel` sets
`layer_norm_hidden_state=False`, `layer="hidden"`, `layer_idx=-1`), and this
brick is text-only (no reference-image conditioning in scope). The weight
loader (`text_encoder_weight_map.py`) explicitly skips these extra keys
rather than silently accepting any unrecognized key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class Qwen3VL8BTextEncoderConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"ASDX: num_attention_heads={self.num_attention_heads} is not a multiple of "
                f"num_key_value_heads={self.num_key_value_heads} -- GQA requires an integer group size."
            )

    @property
    def kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def mlx_dtype(self) -> mx.Dtype:
        dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
        if self.dtype not in dtype_map:
            raise ValueError(f"ASDX: unknown Qwen3-VL-8B text encoder dtype {self.dtype!r}")
        return dtype_map[self.dtype]


def detect_qwen3vl_8b_text_encoder_config(
    state_dict: dict[str, Any], dtype: str = "float16"
) -> Qwen3VL8BTextEncoderConfig:
    """Derive config from a checkpoint's own tensor shapes, same rationale as
    MiniMax H3's `detect_qwen3_text_encoder_config`: read layer count and
    dims from the real weights rather than assuming the released 8B is
    exactly what the reference config class default says."""
    embed_w = state_dict.get("model.embed_tokens.weight")
    q_proj_w = state_dict.get("model.layers.0.self_attn.q_proj.weight")
    k_proj_w = state_dict.get("model.layers.0.self_attn.k_proj.weight")
    q_norm_w = state_dict.get("model.layers.0.self_attn.q_norm.weight")
    gate_proj_w = state_dict.get("model.layers.0.mlp.gate_proj.weight")
    if embed_w is None or q_proj_w is None or k_proj_w is None or q_norm_w is None or gate_proj_w is None:
        raise ValueError(
            "ASDX: cannot detect Qwen3-VL-8B text encoder config -- checkpoint is missing "
            "model.embed_tokens.weight / model.layers.0.self_attn.{q_proj,k_proj,q_norm}.weight / "
            "model.layers.0.mlp.gate_proj.weight after key normalization."
        )

    layer_indices = set()
    for key in state_dict:
        if key.startswith("model.layers."):
            layer_indices.add(int(key.split(".")[2]))
    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 0

    vocab_size, hidden_size = embed_w.shape
    head_dim = q_norm_w.shape[0]
    num_attention_heads = q_proj_w.shape[0] // head_dim
    num_key_value_heads = k_proj_w.shape[0] // head_dim
    intermediate_size = gate_proj_w.shape[0]

    return Qwen3VL8BTextEncoderConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
```

- [ ] **Step 5: Vérifier le succès**

Run: `uv run pytest tests/native/qwen_image21/test_text_encoder_config.py -q`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add apple_silicon_nodes/native/qwen_image21/text_encoder_config.py tests/support/qwen_image21_module_loader.py tests/native/qwen_image21/test_text_encoder_config.py
git commit -m "feat: Qwen Image 2.1 text encoder config + checkpoint detection"
```

---

### Task 2: RoPE + RMSNorm math

**Files:**
- Create: `apple_silicon_nodes/native/qwen_image21/rope.py`
- Test: `tests/native/qwen_image21/test_rope.py`

**Interfaces:**
- Consumes: rien.
- Produces: `rms_norm(x, weight, eps) -> mx.array`, `apply_rope_split_half(x, cos, sin) -> mx.array`, `rms_norm_rope_split_half(x, weight, eps, rot_dim, cos, sin) -> mx.array`, `qwen3_rope_cos_sin(seq_len, head_dim, theta) -> tuple[mx.array, mx.array]`. Consommé par la tâche 3.

- [ ] **Step 1: Écrire le test qui échoue**

`tests/native/qwen_image21/test_rope.py` :

```python
"""Parity of qwen_image21.rope against the real comfy Qwen3 RoPE path.

Reuses the same verification already done for MiniMax H3's identical
`qwen3_rope_cos_sin`/`rms_norm_rope_split_half` (this is generic Qwen3
text-only RoPE math, independent of model depth/width -- see
`native/minimax_h3/rope.py`'s module docstring), re-run here against the
real `comfy.text_encoders.llama` module directly rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

rope_mod = load_native_module("qwen_image21.rope")


def _load_real_comfy_llama():
    comfyui_root = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")
    venv_site_packages = comfyui_root / ".venv" / "lib" / "python3.13" / "site-packages"
    if not comfyui_root.exists() or not venv_site_packages.exists():
        pytest.skip("ComfyUI install not present on this machine")
    sys.path.insert(0, str(comfyui_root))
    sys.path.insert(0, str(venv_site_packages))
    try:
        import comfy.text_encoders.llama as llama
    except ImportError as e:
        pytest.skip(f"comfy.text_encoders.llama not importable: {e}")
    return llama


def test_cos_sin_matches_comfy_precompute_freqs_cis():
    llama = _load_real_comfy_llama()
    import torch

    seq_len, head_dim, theta = 7, 16, 5000000.0
    position_ids = torch.arange(seq_len).unsqueeze(0)  # [1, S], text-only path
    freqs = llama.precompute_freqs_cis(head_dim, position_ids, theta, None, None, interleaved_mrope=False, device="cpu")
    # freqs is [1, S, head_dim] cos/sin packed per comfy's rope_matrix convention;
    # compare against our own cos/sin by reconstructing rope_matrix's rotation on a probe vector.
    cos, sin = rope_mod.qwen3_rope_cos_sin(seq_len, head_dim, theta)
    x = torch.randn(1, seq_len, 1, head_dim)
    ref_rotated = llama.apply_rope(x, freqs)[0, :, 0, :].numpy()

    xm = mx.array(x[0, :, 0, :].numpy())
    got_rotated = np.array(rope_mod.apply_rope_split_half(xm, cos, sin))
    assert np.abs(got_rotated - ref_rotated).max() < 1e-4


def test_rms_norm_matches_torch_rms_norm():
    import torch
    import torch.nn.functional as F

    x = np.random.randn(3, 16).astype(np.float32)
    weight = np.random.randn(16).astype(np.float32)
    eps = 1e-6
    expected = F.rms_norm(torch.from_numpy(x), (16,), weight=torch.from_numpy(weight), eps=eps).numpy()
    got = np.array(rope_mod.rms_norm(mx.array(x), mx.array(weight), eps))
    assert np.allclose(got, expected, atol=1e-5)


def test_rms_norm_rope_split_half_rotates_only_rot_dim():
    cos, sin = rope_mod.qwen3_rope_cos_sin(4, 8, 5000000.0)
    x = mx.random.normal((4, 2, 8))
    weight = mx.ones(8)
    out = rope_mod.rms_norm_rope_split_half(x, weight, 1e-6, 8, cos[:, None, :], sin[:, None, :])
    assert out.shape == x.shape
    assert bool(mx.all(mx.isfinite(out)).item())
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/qwen_image21/test_rope.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implémenter**

`apple_silicon_nodes/native/qwen_image21/rope.py` (ported from `minimax_h3/rope.py`'s generic helpers plus `minimax_h3/text_encoder_rope.py`'s `qwen3_rope_cos_sin` — same math, this config class family (`Qwen3VL_8BConfig`/`Qwen3VL_32BConfig` both extend `Qwen3_8BConfig`) shares the identical text-only RoPE branch, verified previously against `comfy_kitchen`'s eager reference for the 32B variant; re-verified above against `comfy.text_encoders.llama` directly for this checkpoint's own dims) :

```python
"""Standard (non-multimodal) RoPE + fused per-head RMSNorm for Qwen3-VL-8B's
text-only path (Qwen Image 2.1's T2I encoder -- no image reference
conditioning in this brick).

For text-only input, `comfy/text_encoders/llama.py`'s `precompute_freqs_cis`
builds `position_ids` as `[1, S]` (single sequential axis), which fails its
own `position_ids.shape[0] > 1` gate for Qwen3-VL's interleaved M-RoPE -- so
it always takes the plain single-axis branch:

    inv_freq[i] = 1 / theta^(2i/head_dim), i in [0, head_dim/2)
    angle[s, i] = s * inv_freq[i]
    cos/sin = cat(angle, angle).cos()/.sin()

and `apply_rope`'s rotation is the "split in half, not interleaved pairs"
convention. Ported and re-verified here against real `comfy.text_encoders.
llama.precompute_freqs_cis`/`apply_rope` (see `test_rope.py`) -- this is the
same math already verified for MiniMax H3's Qwen3-VL-32B (`native/minimax_h3/
rope.py`, `text_encoder_rope.py`), which shares the same `Qwen3_8BConfig`
base class and text-only RoPE branch; duplicated here per this project's
per-family file convention (see Krea2's own `rope.py`) rather than
cross-imported from `minimax_h3`.
"""

from __future__ import annotations

import mlx.core as mx


def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Standard RMSNorm over the last axis, matching
    `torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)`."""
    x32 = x.astype(mx.float32)
    variance = mx.mean(x32 * x32, axis=-1, keepdims=True)
    normed = x32 * mx.rsqrt(variance + eps)
    return (normed.astype(x.dtype)) * weight


def apply_rope_split_half(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """`x`: `[..., S, 2*half]`. `cos`/`sin`: `[S, half]` (or broadcastable,
    e.g. `[S, 1, half]` against `x` shaped `[S, heads, head_dim]`). Splits
    `x` into two contiguous halves (not interleaved pairs) and rotates pair
    `i = (x1[i], x2[i])` by angle `i`."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return mx.concatenate([out1, out2], axis=-1)


def rms_norm_rope_split_half(
    x: mx.array, weight: mx.array, eps: float, rot_dim: int, cos: mx.array, sin: mx.array
) -> mx.array:
    """Fused RMSNorm + partial split-half RoPE. `x`: `[..., S, head_dim]`.
    `cos`/`sin`: `[S, rot_dim//2]`, broadcastable against `x`'s leading axes."""
    x_norm = rms_norm(x, weight, eps)
    if rot_dim == 0 or rot_dim == x.shape[-1]:
        return apply_rope_split_half(x_norm, cos, sin) if rot_dim else x_norm
    rotated = apply_rope_split_half(x_norm[..., :rot_dim], cos, sin)
    return mx.concatenate([rotated, x_norm[..., rot_dim:]], axis=-1)


def qwen3_rope_cos_sin(seq_len: int, head_dim: int, theta: float) -> tuple[mx.array, mx.array]:
    """`[S, head_dim // 2]` cos/sin for sequential positions `0..seq_len-1`,
    single axis, full `head_dim` rotated -- matches
    `precompute_freqs_cis(head_dim, torch.arange(seq_len).unsqueeze(0), theta)`'s
    plain (non-interleaved-MRoPE) branch."""
    exponent = mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim
    inv_freq = 1.0 / (theta**exponent)  # [half]
    positions = mx.arange(seq_len, dtype=mx.float32)  # [S]
    angles = positions[:, None] * inv_freq[None, :]  # [S, half]
    return mx.cos(angles), mx.sin(angles)
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/native/qwen_image21/test_rope.py -q`
Expected: PASS (3 tests) if the local ComfyUI venv is present; SKIP with a clear reason otherwise (never a silent pass).

- [ ] **Step 5: Commit**

```bash
git add apple_silicon_nodes/native/qwen_image21/rope.py tests/native/qwen_image21/test_rope.py
git commit -m "feat: Qwen Image 2.1 text encoder RoPE + RMSNorm math"
```

---

### Task 3: Model (Attention/MLP/TransformerBlock/backbone)

**Files:**
- Create: `apple_silicon_nodes/native/qwen_image21/text_encoder.py`
- Test: `tests/native/qwen_image21/test_text_encoder.py`

**Interfaces:**
- Consumes: `Qwen3VL8BTextEncoderConfig` (Task 1), `rms_norm_rope_split_half`/`qwen3_rope_cos_sin` (Task 2).
- Produces: `RMSNorm(dim, eps)`, `Attention(config)`, `MLP(config)`, `TransformerBlock(config)`, `Qwen3VL8BTextEncoder(config)` with `__call__(self, input_ids: mx.array) -> mx.array` returning `[S, hidden_size]`. Consommé par la tâche 4 et la brique 3 (intégration).

- [ ] **Step 1: Écrire le test qui échoue**

`tests/native/qwen_image21/test_text_encoder.py` :

```python
"""Tests for the Qwen3-VL-8B text-only encoder (Attention/MLP/
TransformerBlock/Qwen3VL8BTextEncoder). Text-only: no vision tower, no
reference-image conditioning -- out of scope for this brick."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.qwen_image21_module_loader import load_native_module

config_mod = load_native_module("qwen_image21.text_encoder_config")
rope_mod = load_native_module("qwen_image21.rope")
text_encoder_mod = load_native_module("qwen_image21.text_encoder")

Qwen3VL8BTextEncoderConfig = config_mod.Qwen3VL8BTextEncoderConfig
Attention = text_encoder_mod.Attention
MLP = text_encoder_mod.MLP
TransformerBlock = text_encoder_mod.TransformerBlock
Qwen3VL8BTextEncoder = text_encoder_mod.Qwen3VL8BTextEncoder


def _tiny_config(**overrides) -> "Qwen3VL8BTextEncoderConfig":
    base = dict(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
        rope_theta=5000000.0, dtype="float32",
    )
    base.update(overrides)
    return Qwen3VL8BTextEncoderConfig(**base)


def test_attention_output_shape_and_finite():
    cfg = _tiny_config()
    attn = Attention(cfg)
    seq_len = 5
    x = mx.random.normal((seq_len, cfg.hidden_size))
    cos, sin = rope_mod.qwen3_rope_cos_sin(seq_len, cfg.head_dim, cfg.rope_theta)
    out = attn(x, cos, sin)
    assert out.shape == (seq_len, cfg.hidden_size)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_attention_gqa_matches_manual_repeat():
    cfg = _tiny_config(num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    attn = Attention(cfg)
    seq_len = 4
    mx.random.seed(0)
    x = mx.random.normal((seq_len, cfg.hidden_size))
    cos, sin = rope_mod.qwen3_rope_cos_sin(seq_len, cfg.head_dim, cfg.rope_theta)

    out = attn(x, cos, sin)

    q = attn.q_proj(x).reshape(seq_len, cfg.num_attention_heads, cfg.head_dim)
    k = attn.k_proj(x).reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim)
    v = attn.v_proj(x).reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim)

    cos_b, sin_b = cos[:, None, :], sin[:, None, :]
    q = rope_mod.rms_norm_rope_split_half(q, attn.q_norm.weight, attn.eps, cfg.head_dim, cos_b, sin_b)
    k = rope_mod.rms_norm_rope_split_half(k, attn.k_norm.weight, attn.eps, cfg.head_dim, cos_b, sin_b)

    qn = np.array(q).transpose(1, 0, 2)
    kn = np.array(k).transpose(1, 0, 2)
    vn = np.array(v).transpose(1, 0, 2)
    group = cfg.kv_groups
    kn = np.repeat(kn, group, axis=0)
    vn = np.repeat(vn, group, axis=0)

    scale = 1.0 / np.sqrt(cfg.head_dim)
    scores = np.einsum("hsd,htd->hst", qn, kn) * scale
    causal = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    scores = scores + causal
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    manual = np.einsum("hst,htd->hsd", weights, vn).transpose(1, 0, 2).reshape(seq_len, -1)
    expected = np.array(attn.o_proj(mx.array(manual)))

    assert np.abs(np.array(out) - expected).max() < 2e-3


def test_mlp_matches_standard_swiglu():
    cfg = _tiny_config()
    mlp = MLP(cfg)
    x = mx.random.normal((3, cfg.hidden_size))
    out = mlp(x)
    gate = np.array(mlp.gate_proj(x))
    up = np.array(mlp.up_proj(x))
    silu_gate = gate / (1.0 + np.exp(-gate))
    expected = np.array(mlp.down_proj(mx.array(silu_gate * up)))
    assert np.allclose(np.array(out), expected, atol=1e-5)


def test_transformer_block_is_residual():
    cfg = _tiny_config()
    block = TransformerBlock(cfg)
    block.self_attn.o_proj.weight = mx.zeros_like(block.self_attn.o_proj.weight)
    block.mlp.down_proj.weight = mx.zeros_like(block.mlp.down_proj.weight)
    x = mx.random.normal((4, cfg.hidden_size))
    cos, sin = rope_mod.qwen3_rope_cos_sin(4, cfg.head_dim, cfg.rope_theta)
    out = block(x, cos, sin)
    assert bool(mx.allclose(out, x, atol=1e-5).item())


def test_encoder_output_shape():
    cfg = _tiny_config()
    encoder = Qwen3VL8BTextEncoder(cfg)
    input_ids = mx.array([1, 2, 3, 4, 5])
    out = encoder(input_ids)
    assert out.shape == (5, cfg.hidden_size)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_encoder_is_causal():
    cfg = _tiny_config()
    encoder = Qwen3VL8BTextEncoder(cfg)
    ids_a = mx.array([1, 2, 3, 4, 5])
    ids_b = mx.array([1, 2, 3, 4, 99])
    out_a = encoder(ids_a)
    out_b = encoder(ids_b)
    assert bool(mx.allclose(out_a[:4], out_b[:4], atol=1e-5).item())
    assert float(mx.max(mx.abs(out_a[4] - out_b[4])).item()) > 1e-4


def test_encoder_has_no_final_norm_attribute():
    # Deliberate divergence from the real checkpoint (which has model.norm.weight
    # and lm_head.weight): this encoder consumes the raw last-block hidden state.
    cfg = _tiny_config()
    encoder = Qwen3VL8BTextEncoder(cfg)
    assert not hasattr(encoder.model, "norm")
    assert not hasattr(encoder, "lm_head")
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/qwen_image21/test_text_encoder.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implémenter**

`apple_silicon_nodes/native/qwen_image21/text_encoder.py` :

```python
"""Qwen3-VL-8B text-only backbone -- Qwen Image 2.1's T2I text conditioning
encoder (text-only scope; no vision tower, no reference-image conditioning
in this brick -- see text_encoder_config.py's module docstring).

Ported from `comfy/text_encoders/llama.py`'s shared `Attention`/`MLP`/
`TransformerBlock`/`Llama2_` classes (Qwen3-VL uses this same generic
backbone, configured via `Qwen3VL_8BConfig` -- there is no Qwen3-VL-specific
transformer block class). The real checkpoint's `model.norm`/`lm_head` are
deliberately not built here: Qwen Image 2.1's T2I path consumes the raw
last-block hidden state (`comfy/text_encoders/qwen_image21.py`:
`layer_norm_hidden_state=False`, `layer="hidden"`, `layer_idx=-1`).

Causal masking: `Llama2_.forward` always builds a causal mask for
`seq_len > 1`; this port does too (`mx.fast.scaled_dot_product_attention(...,
mask="causal")`).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .rope import qwen3_rope_cos_sin, rms_norm, rms_norm_rope_split_half
from .text_encoder_config import Qwen3VL8BTextEncoderConfig


class RMSNorm(nn.Module):
    """Plain RMSNorm (`x / sqrt(mean(x^2) + eps) * weight`), matching
    `torch.nn.functional.rms_norm`'s direct-multiply convention."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    """GQA self-attention with per-head RMSNorm + full-rotation RoPE
    (`rot_dim == head_dim`). `mx.fast.scaled_dot_product_attention` handles
    GQA natively -- `k`/`v` are kept at their own (smaller) head count."""

    def __init__(self, config: Qwen3VL8BTextEncoderConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.eps = config.rms_norm_eps
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = RMSNorm(self.head_dim, eps=self.eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        s = x.shape[0]
        q = self.q_proj(x).reshape(s, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(s, self.kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(s, self.kv_heads, self.head_dim)

        cos_b, sin_b = cos[:, None, :], sin[:, None, :]
        q = rms_norm_rope_split_half(q, self.q_norm.weight, self.eps, self.head_dim, cos_b, sin_b)
        k = rms_norm_rope_split_half(k, self.k_norm.weight, self.eps, self.head_dim, cos_b, sin_b)

        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        scale = 1.0 / (self.head_dim**0.5)
        mask = "causal" if s > 1 else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out = out[0].transpose(1, 0, 2).reshape(s, self.heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """Standard SwiGLU: separate `gate_proj`/`up_proj` (confirmed from the
    real checkpoint header -- not fused)."""

    def __init__(self, config: Qwen3VL8BTextEncoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Standard pre-norm block: `x += attn(norm1(x))`, `x += mlp(norm2(x))`."""

    def __init__(self, config: Qwen3VL8BTextEncoderConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class _Qwen3VL8BBackbone(nn.Module):
    """Embedding + `num_hidden_layers` `TransformerBlock`s. No final norm, no
    lm_head: the output is the raw hidden state after the last block."""

    def __init__(self, config: Qwen3VL8BTextEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.embed_tokens(input_ids)
        cos, sin = qwen3_rope_cos_sin(x.shape[0], self.config.head_dim, self.config.rope_theta)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


class Qwen3VL8BTextEncoder(nn.Module):
    """Wraps `_Qwen3VL8BBackbone` under a `model` attribute, matching the
    checkpoint's own key prefix (`model.embed_tokens.weight`,
    `model.layers.N....weight`) -- checkpoint-key-compatible attribute names
    throughout this project, so weight loading needs no translation table."""

    def __init__(self, config: Qwen3VL8BTextEncoderConfig):
        super().__init__()
        self.config = config
        self.model = _Qwen3VL8BBackbone(config)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """`input_ids`: `[S]` int32 token ids (batch already squeezed).
        Returns `[S, hidden_size]`."""
        return self.model(input_ids)
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/native/qwen_image21/test_text_encoder.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add apple_silicon_nodes/native/qwen_image21/text_encoder.py tests/native/qwen_image21/test_text_encoder.py
git commit -m "feat: Qwen Image 2.1 text encoder model (Attention/MLP/TransformerBlock/backbone)"
```

---

### Task 4: Weight map + loader (checkpoint réel)

**Files:**
- Create: `apple_silicon_nodes/native/qwen_image21/text_encoder_weight_map.py`
- Test: `tests/native/qwen_image21/test_text_encoder_weight_map.py`

**Interfaces:**
- Consumes: `Qwen3VL8BTextEncoderConfig`/`detect_qwen3vl_8b_text_encoder_config` (Task 1), `Qwen3VL8BTextEncoder` (Task 3).
- Produces: `load_qwen_image21_text_encoder_checkpoint(path: str | Path, dtype: str = "float16") -> Qwen3VL8BTextEncoder`. Consommé par la brique 3 (intégration).

- [ ] **Step 1: Écrire le test qui échoue**

`tests/native/qwen_image21/test_text_encoder_weight_map.py` :

```python
"""load_qwen_image21_text_encoder_checkpoint: build a tiny real
Qwen3VL8BTextEncoder, write its params + synthetic vision/norm/lm_head keys
to a safetensors file, load it back, confirm text params match exactly and
the extra keys are skipped (not silently dropped -- logged)."""

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

config_mod = load_native_module("qwen_image21.text_encoder_config")
text_encoder_mod = load_native_module("qwen_image21.text_encoder")
weight_map_mod = load_native_module("qwen_image21.text_encoder_weight_map")

Qwen3VL8BTextEncoderConfig = config_mod.Qwen3VL8BTextEncoderConfig
Qwen3VL8BTextEncoder = text_encoder_mod.Qwen3VL8BTextEncoder
load_qwen_image21_text_encoder_checkpoint = weight_map_mod.load_qwen_image21_text_encoder_checkpoint


def _tiny_config(**overrides) -> "Qwen3VL8BTextEncoderConfig":
    base = dict(
        vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
        rope_theta=5000000.0, dtype="float32",
    )
    base.update(overrides)
    return Qwen3VL8BTextEncoderConfig(**base)


def test_round_trip_and_skips_extra_keys(tmp_path):
    cfg = _tiny_config()
    original = Qwen3VL8BTextEncoder(cfg)
    flat = dict(tree_flatten(original.parameters()))
    tensors = {k: np.array(v) for k, v in flat.items()}
    # Extra keys present in the real checkpoint but not in our module.
    tensors["model.norm.weight"] = np.zeros((cfg.hidden_size,), dtype=np.float32)
    tensors["lm_head.weight"] = np.zeros((cfg.vocab_size, cfg.hidden_size), dtype=np.float32)
    tensors["model.visual.patch_embed.proj.weight"] = np.zeros((4,), dtype=np.float32)

    path = tmp_path / "tiny_qwen3vl8b.safetensors"
    save_file(tensors, str(path))

    loaded = load_qwen_image21_text_encoder_checkpoint(path, dtype="float32")
    assert loaded.config == cfg

    loaded_flat = dict(tree_flatten(loaded.parameters()))
    assert set(loaded_flat.keys()) == set(flat.keys())
    for key, original_value in flat.items():
        assert np.allclose(np.array(loaded_flat[key]), np.array(original_value), atol=1e-6), key


def test_raises_on_missing_required_key(tmp_path):
    cfg = _tiny_config()
    original = Qwen3VL8BTextEncoder(cfg)
    flat = dict(tree_flatten(original.parameters()))
    tensors = {k: np.array(v) for k, v in flat.items() if k != "model.embed_tokens.weight"}
    path = tmp_path / "broken.safetensors"
    save_file(tensors, str(path))
    with pytest.raises((ValueError, KeyError)):
        load_qwen_image21_text_encoder_checkpoint(path, dtype="float32")


_QWEN3VL_8B_REAL = Path("/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_8b_bf16.safetensors")


@pytest.mark.skipif(
    os.environ.get("ASDX_FULL_GGUF_TEST") != "1",
    reason="loads the real ~14GB Qwen3-VL-8B checkpoint; set ASDX_FULL_GGUF_TEST=1 to run",
)
def test_loads_real_checkpoint():
    if not _QWEN3VL_8B_REAL.exists():
        pytest.skip("no local qwen3vl_8b_bf16.safetensors")

    model = load_qwen_image21_text_encoder_checkpoint(_QWEN3VL_8B_REAL, dtype="float16")

    assert model.config.num_hidden_layers == 36
    assert model.config.hidden_size == 4096
    assert model.config.num_attention_heads == 32
    assert model.config.num_key_value_heads == 8

    input_ids = mx.array([1, 2, 3, 4, 5])
    out = model(input_ids)
    assert out.shape == (5, 4096)
    assert bool(mx.all(mx.isfinite(out)).item())
    assert float(mx.max(mx.abs(out)).item()) > 0.0
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/qwen_image21/test_text_encoder_weight_map.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implémenter**

`apple_silicon_nodes/native/qwen_image21/text_encoder_weight_map.py` :

```python
"""Checkpoint loading for Qwen3VL8BTextEncoder.

The real checkpoint (`qwen3vl_8b_bf16.safetensors`, 750 tensors, inspected
directly on 2026-09-22) is the FULL Qwen3-VL-8B: it carries a complete
`model.visual.*` vision tower (496 tensors) plus `model.norm.weight` and
`lm_head.weight`, none of which `Qwen3VL8BTextEncoder` builds (see
`text_encoder_config.py`'s module docstring for why). Those keys are
explicitly recognized and skipped -- any OTHER unrecognized key still
raises, so a genuinely unexpected checkpoint shape is never silently
accepted (fail-closed, this project's convention throughout).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten
from safetensors import safe_open

from .text_encoder import Qwen3VL8BTextEncoder
from .text_encoder_config import detect_qwen3vl_8b_text_encoder_config


def _is_skippable_extra_key(key: str) -> bool:
    """Keys present in the real checkpoint that this text-only, no-final-norm
    encoder deliberately does not build."""
    return key.startswith("model.visual.") or key in ("model.norm.weight", "lm_head.weight")


def load_qwen_image21_text_encoder_checkpoint(path: str | Path, dtype: str = "float16") -> Qwen3VL8BTextEncoder:
    path = Path(path)
    state_dict: dict[str, mx.array] = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            state_dict[key] = mx.array(f.get_tensor(key))

    config = detect_qwen3vl_8b_text_encoder_config(state_dict, dtype=dtype)
    model = Qwen3VL8BTextEncoder(config)

    expected_keys = set(k for k, _ in _flatten_module_params(model))
    skipped = [k for k in state_dict if _is_skippable_extra_key(k)]
    unexpected = [k for k in state_dict if k not in expected_keys and not _is_skippable_extra_key(k)]
    if unexpected:
        raise ValueError(
            f"ASDX: {len(unexpected)} unrecognized key(s) in {path.name}, e.g. {unexpected[:5]} -- "
            "not in Qwen3VL8BTextEncoder's parameters and not a known skippable extra "
            "(model.visual.*, model.norm.weight, lm_head.weight)."
        )
    print(f"[ASDX] Qwen Image 2.1 text encoder: skipping {len(skipped)} extra key(s) "
          "(vision tower + final norm + lm_head, not used by the T2I text path).")

    weights = []
    for key, _ in _flatten_module_params(model):
        if key not in state_dict:
            raise ValueError(f"ASDX: {path.name} is missing required key {key!r} for Qwen3VL8BTextEncoder.")
        weights.append((key, state_dict[key].astype(config.mlx_dtype)))

    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    mx.clear_cache()
    return model


def _flatten_module_params(model: Qwen3VL8BTextEncoder) -> list[tuple[str, mx.array]]:
    from mlx.utils import tree_flatten
    return list(tree_flatten(model.parameters()))
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/native/qwen_image21/test_text_encoder_weight_map.py -q`
Expected: PASS (2 tests always; the real-checkpoint test PASSes when run with `ASDX_FULL_GGUF_TEST=1 uv run pytest tests/native/qwen_image21/test_text_encoder_weight_map.py -q`, otherwise SKIP).

- [ ] **Step 5: Run the real-checkpoint test**

Run: `ASDX_FULL_GGUF_TEST=1 uv run pytest tests/native/qwen_image21/test_text_encoder_weight_map.py::test_loads_real_checkpoint -v`
Expected: PASS. If it fails on an unexpected key, inspect the real header (`python -c "..."` dump, as done during design) rather than widening `_is_skippable_extra_key` blindly.

- [ ] **Step 6: Commit**

```bash
git add apple_silicon_nodes/native/qwen_image21/text_encoder_weight_map.py tests/native/qwen_image21/test_text_encoder_weight_map.py
git commit -m "feat: Qwen Image 2.1 text encoder checkpoint loader"
```

---

### Task 5: Vérification finale (verify-checkpoint + weight-map-reviewer)

**Files:** aucun nouveau fichier — vérification des tâches 1-4.

**Interfaces:**
- Consumes: tout ce qui précède.
- Produces: confirmation que la brique 1 est terminée (aucune API neuve).

- [ ] **Step 1: Lancer le skill `verify-checkpoint`**

Invoquer le skill `verify-checkpoint` sur `apple_silicon_nodes/native/qwen_image21/` : py_compile, forward pass à poids aléatoires (déjà couvert par les tests de la tâche 3), chargement N/M matché sur `qwen3vl_8b_bf16.safetensors` réel (tâche 4, `ASDX_FULL_GGUF_TEST=1`), sanity std-vs-random-init (comparer `mx.std(out)` entre `Qwen3VL8BTextEncoder(cfg)` fraîchement initialisé et le modèle chargé sur poids réels — les deux ne doivent pas être numériquement indiscernables).

- [ ] **Step 2: Lancer l'agent `weight-map-reviewer`**

Sur `apple_silicon_nodes/native/qwen_image21/text_encoder_weight_map.py` et la fonction `load_qwen_image21_text_encoder_checkpoint` — pour la classe de bugs silencieux de chargement déjà rencontrée dans ce projet (voir canon `[[lora-family-dispatch]]` et l'historique MiniMax H3).

- [ ] **Step 3: Corriger tout écart trouvé, puis relancer la suite complète**

Run: `uv run pytest tests/native/qwen_image21/ -q` puis `ASDX_FULL_GGUF_TEST=1 uv run pytest tests/native/qwen_image21/ -q`
Expected: PASS intégral dans les deux cas.

- [ ] **Step 4: Commit (si des corrections ont été faites)**

```bash
git add apple_silicon_nodes/native/qwen_image21/ tests/native/qwen_image21/
git commit -m "fix: Qwen Image 2.1 text encoder review fixes"
```
