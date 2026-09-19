# MiniMax H3 brique 1 : tour vision Qwen3-VL en MLX

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Une tour vision Qwen3-VL-32B en MLX (`VisionTower`) qui transforme une image en embeddings fusionnés + 3 tenseurs DeepStack, numériquement identique à ComfyUI, chargée depuis les 351 tenseurs `visual.*` des fichiers TE (safetensors et GGUF).

**Architecture:** 3 fichiers focalisés dans `apple_silicon_nodes/native/minimax_h3/` : `vision_preprocess.py` (image -> patchs, torch CPU pour la parité du bilinéaire), `vision_tower.py` (modèle MLX), `vision_weight_map.py` (vérification des formes, assignation, chargement par `checkpoint_source`). Le raccordement à l'encodeur de texte est la brique 2 (hors de ce plan).

**Tech Stack:** MLX (`mlx.core`, `mlx.nn`), numpy, torch (prétraitement seulement), pytest, `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-19-minimax-h3-i2v-ref-design.md` (brique 1). Référence : `comfy/text_encoders/qwen3vl.py`, `qwen35.py` (`Qwen35VisionModel`), `qwen_vl.py` (`process_qwen2vl_images`) dans `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI`.

## Global Constraints

- Toujours `uv run pytest` / `uv run python` (jamais `python` nu).
- Style du dépôt : Python 3.10+, annotations complètes, docstrings en anglais, pas d'emoji ni de tiret cadratin.
- Fail-closed : un tenseur manquant, une forme inattendue ou un dtype inconnu lève une exception, jamais de valeur par défaut plausible.
- La tour tourne en **float32** (comme ComfyUI : `image.to(dtype=torch.float32)`), poids bf16 convertis à l'assignation (~2,4 Go ; la spec disait ~1,2 Go en bf16, corrigé ici).
- Torch n'est utilisé que pour le prétraitement d'image (bilinéaire identique à la référence) ; aucun torch dans le chemin de calcul.
- Les tests de parité contre ComfyUI sont ignorés (`pytest.skip`) si l'installation ComfyUI est absente ; les tests sur vrais poids exigent `ASDX_FULL_GGUF_TEST=1`.
- Les tests importent les modules via `load_native_module("minimax_h3.<nom>")` (`tests/support/minimax_h3_module_loader.py`).
- **Aucun commit automatique** (règle utilisateur) : chaque étape « Commit » signifie « préparer le message et attendre l'ordre explicite de l'utilisateur ». Pas de ligne d'attribution dans les messages de commit.

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `apple_silicon_nodes/native/minimax_h3/vision_preprocess.py` (créer) | `preprocess_image` : image [H,W,3] -> patchs aplatis + grille |
| `apple_silicon_nodes/native/minimax_h3/vision_tower.py` (créer) | `VisionConfig`, `VisionTower` |
| `apple_silicon_nodes/native/minimax_h3/vision_weight_map.py` (créer) | `verify_vision_shapes`, `assign_vision_weights`, `load_vision_tower` |
| `tests/support/comfyui_reference_loader.py` (modifier) | ajoute `load_real_comfy_qwen3vl()` |
| `tests/native/minimax_h3/test_vision_preprocess.py` (créer) | parité du prétraitement |
| `tests/native/minimax_h3/test_vision_tower.py` (créer) | modèle : parité poids aléatoires, cas nul |
| `tests/native/minimax_h3/test_vision_weight_map.py` (créer) | assignation, formes, vrais poids |

---

### Task 1: Prétraitement d'image

**Files:**
- Create: `apple_silicon_nodes/native/minimax_h3/vision_preprocess.py`
- Modify: `tests/support/comfyui_reference_loader.py` (ajout en fin de fichier)
- Test: `tests/native/minimax_h3/test_vision_preprocess.py`

**Interfaces:**
- Consumes: rien.
- Produces: `preprocess_image(image: np.ndarray, *, patch_size: int = 16, temporal_patch_size: int = 2, merge_size: int = 2, min_pixels: int = 3136, max_pixels: int = 12845056, image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5), image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> tuple[np.ndarray, tuple[int, int, int]]` : renvoie `(patchs float32 [gh*gw, 3*T*P*P], (1, gh, gw))`. Consommé par la tâche 3 et la brique 2.
  `load_real_comfy_qwen3vl() -> tuple[module, module, module]` : `(comfy.text_encoders.qwen3vl, comfy.text_encoders.qwen_vl, comfy.ops)`.

- [ ] **Step 1: Ajouter le chargeur de la vraie référence**

Ajouter à la fin de `tests/support/comfyui_reference_loader.py` :

```python
def load_real_comfy_qwen3vl():
    """Real `comfy.text_encoders.qwen3vl`, `qwen_vl` and `comfy.ops`, for
    numerical parity tests of the MLX vision tower."""
    if not _COMFYUI_ROOT.exists() or not _COMFYUI_VENV_SITE_PACKAGES.exists():
        pytest.skip("ComfyUI install not present on this machine")
    for name in list(sys.modules):
        if name == "comfy" or name.startswith("comfy."):
            del sys.modules[name]
    sys.path.insert(0, str(_COMFYUI_ROOT))
    sys.path.insert(0, str(_COMFYUI_VENV_SITE_PACKAGES))
    try:
        import comfy.ops as ops
        import comfy.text_encoders.qwen3vl as qwen3vl
        import comfy.text_encoders.qwen_vl as qwen_vl
    except ImportError as e:
        pytest.skip(f"comfy Qwen3-VL modules not importable: {e}")
    return qwen3vl, qwen_vl, ops
```

- [ ] **Step 2: Écrire le test qui échoue**

`tests/native/minimax_h3/test_vision_preprocess.py` :

```python
"""Parity of vision_preprocess.preprocess_image against the real
comfy.text_encoders.qwen_vl.process_qwen2vl_images."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_qwen3vl
from support.minimax_h3_module_loader import load_native_module

pre = load_native_module("minimax_h3.vision_preprocess")


@pytest.mark.parametrize("shape", [(224, 336), (250, 333), (48, 64), (1100, 900)])
def test_matches_comfyui(shape):
    _, qwen_vl, _ = load_real_comfy_qwen3vl()
    import torch

    rng = np.random.default_rng(0)
    image = rng.random((shape[0], shape[1], 3), dtype=np.float32)
    ref_patches, ref_grid = qwen_vl.process_qwen2vl_images(
        torch.from_numpy(image)[None], patch_size=16, image_mean=[0.5] * 3, image_std=[0.5] * 3
    )
    patches, grid = pre.preprocess_image(image)
    assert grid == tuple(int(v) for v in ref_grid[0].tolist())
    assert patches.shape == tuple(ref_patches.shape)
    assert np.abs(patches - ref_patches.numpy()).max() < 1e-6


def test_min_pixels_upscales_tiny_images():
    patches, grid = pre.preprocess_image(np.zeros((10, 10, 3), dtype=np.float32))
    assert grid[1] * grid[2] * 16 * 16 >= 3136


def test_rejects_non_rgb():
    with pytest.raises(ValueError, match="3 channels"):
        pre.preprocess_image(np.zeros((64, 64, 4), dtype=np.float32))
```

- [ ] **Step 3: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_vision_preprocess.py -q`
Expected: FAIL (`ModuleNotFoundError: ... vision_preprocess`).

- [ ] **Step 4: Implémenter**

`apple_silicon_nodes/native/minimax_h3/vision_preprocess.py` :

```python
"""Image -> Qwen3-VL vision patches, ported 1:1 from
`comfy/text_encoders/qwen_vl.py::process_qwen2vl_images` (image path,
temporal_patch_size frames = the same frame repeated).

Runs on the host with torch: the bilinear `F.interpolate(align_corners=False)`
is what the reference (and the released model) uses, and a single small image
per call makes a device round trip irrelevant next to matching its numerics.
This is the only torch use in the vision path.
"""

from __future__ import annotations

import math

import numpy as np


def preprocess_image(
    image: np.ndarray,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """`image`: float32 `[H, W, 3]` in `[0, 1]`. Returns `(patches, grid)`:
    `patches` float32 `[gh*gw, 3*temporal_patch_size*patch_size**2]` in
    merge-block order, `grid = (1, gh, gw)`."""
    import torch
    import torch.nn.functional as F

    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"ASDX: vision preprocessing needs 3 channels, got {channels}")

    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    img = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1)
    resized = F.interpolate(img.unsqueeze(0), size=(h_bar, w_bar), mode="bilinear", align_corners=False).squeeze(0)
    mean = torch.tensor(image_mean, dtype=resized.dtype).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=resized.dtype).view(3, 1, 1)
    normalized = (resized - mean) / std

    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    frames = normalized.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
    patches = frames.reshape(
        1, temporal_patch_size, 3,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    return flat.numpy(), (1, grid_h, grid_w)
```

- [ ] **Step 5: Vérifier le succès**

Run: `uv run pytest tests/native/minimax_h3/test_vision_preprocess.py -q`
Expected: PASS (les 4 cas de parité passent, ou sont ignorés si ComfyUI est absent ; dans ce cas le dire dans le compte rendu).

- [ ] **Step 6: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/vision_preprocess.py tests/support/comfyui_reference_loader.py tests/native/minimax_h3/test_vision_preprocess.py
git commit -m "feat: add Qwen3-VL image preprocessing with ComfyUI parity test"
```

---

### Task 2: Modèle `VisionTower`

**Files:**
- Create: `apple_silicon_nodes/native/minimax_h3/vision_tower.py`
- Test: `tests/native/minimax_h3/test_vision_tower.py`

**Interfaces:**
- Consumes: rien de la tâche 1 (les tests génèrent leurs patchs).
- Produces: `VisionConfig` (dataclass figée, champs : `hidden_size=1152, intermediate_size=4304, depth=27, num_heads=16, patch_size=16, temporal_patch_size=2, in_channels=3, spatial_merge_size=2, num_position_embeddings=2304, deepstack_visual_indexes=(8, 16, 24), out_hidden_size=5120`, propriétés `head_dim`, `patch_dim`, `merge_dim`) ; `VisionTower(config)` avec `__call__(patches: mx.array, grids: list[tuple[int, int, int]]) -> tuple[mx.array, list[mx.array]]` : `(merged [N/4, out_hidden], [deepstack_i [N/4, out_hidden]])`. Noms de paramètres identiques aux clés du checkpoint sans le préfixe `visual.` (sauf `patch_embed.proj.weight`, `Linear` 2D `[hidden, patch_dim]`).

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/native/minimax_h3/test_vision_tower.py` :

```python
"""VisionTower parity vs the real ComfyUI Qwen3VLVisionModel on a tiny
random-weight config (architecture, not weights), plus a metric sanity check."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_qwen3vl
from support.minimax_h3_module_loader import load_native_module

tower_mod = load_native_module("minimax_h3.vision_tower")
map_mod = load_native_module("minimax_h3.vision_weight_map")

TINY = dict(
    hidden_size=64, intermediate_size=128, depth=4, num_heads=4, patch_size=4,
    temporal_patch_size=2, in_channels=3, spatial_merge_size=2,
    num_position_embeddings=16, deepstack_visual_indexes=(0, 1, 2), out_hidden_size=32,
)
GRIDS = [(1, 4, 6), (1, 2, 4)]  # two images with different grids: exercises per-image spans


def _reference(seed: int = 0):
    import torch

    qwen3vl, _, ops = load_real_comfy_qwen3vl()
    cfg = dict(TINY, deepstack_visual_indexes=list(TINY["deepstack_visual_indexes"]))
    ref = qwen3vl.Qwen3VLVisionModel(cfg, device="cpu", dtype=torch.float32, ops=ops.disable_weight_init)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ref.parameters():  # disable_weight_init leaves weights uninitialized
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
    return ref


def _patches(seed: int = 1):
    n = sum(t * h * w for t, h, w in GRIDS)
    cfg = tower_mod.VisionConfig(**TINY)
    return np.random.default_rng(seed).standard_normal((n, cfg.patch_dim)).astype(np.float32)


def _mlx_from_ref(ref):
    model = tower_mod.VisionTower(tower_mod.VisionConfig(**TINY))
    tensors = {k: mx.array(v.detach().numpy()) for k, v in ref.state_dict().items()}
    map_mod.assign_vision_weights(model, tensors)
    return model


def test_matches_comfyui_on_random_weights():
    import torch

    ref = _reference()
    patches = _patches()
    with torch.no_grad():
        ref_merged, ref_ds = ref(torch.from_numpy(patches), torch.tensor(GRIDS))
    model = _mlx_from_ref(ref)
    merged, ds = model(mx.array(patches), GRIDS)
    mx.eval(merged, ds)

    assert merged.shape == tuple(ref_merged.shape)
    assert len(ds) == len(ref_ds) == 3
    assert np.abs(np.array(merged) - ref_merged.numpy()).max() < 1e-3
    for got, want in zip(ds, ref_ds):
        assert np.abs(np.array(got) - want.numpy()).max() < 1e-3


def test_metric_separates_different_weights():
    """Null case: the parity metric must see a changed weight (else it measures nothing)."""
    import torch

    ref = _reference()
    patches = _patches()
    with torch.no_grad():
        ref_merged, _ = ref(torch.from_numpy(patches), torch.tensor(GRIDS))
    other = _mlx_from_ref(_reference(seed=99))
    merged, _ = other(mx.array(patches), GRIDS)
    assert np.abs(np.array(merged) - ref_merged.numpy()).max() > 1e-2


def test_output_row_counts_follow_merge():
    model = tower_mod.VisionTower(tower_mod.VisionConfig(**TINY))
    merged, ds = model(mx.array(_patches()), GRIDS)
    total = sum(t * h * w for t, h, w in GRIDS) // 4
    assert merged.shape[0] == total and all(d.shape[0] == total for d in ds)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_vision_tower.py -q`
Expected: FAIL (`ModuleNotFoundError: ... vision_tower`).

- [ ] **Step 3: Implémenter le modèle**

`apple_silicon_nodes/native/minimax_h3/vision_tower.py` :

```python
"""Qwen3-VL vision tower (ViT + DeepStack) in MLX, for MiniMax H3's
vision-grounded text conditioning (fl2va keyframes, ref2va references).

Ported from `comfy/text_encoders/qwen35.py::Qwen35VisionModel` and
`qwen3vl.py::Qwen3VLVisionModel`. Parameter names match the checkpoint's
`visual.*` keys (prefix stripped) so `vision_weight_map.py` assigns by name.
The one deliberate difference: the reference's `Conv3d` patch embed has
kernel == stride, so it is exactly a Linear over the flattened
`[C, T, P, P]` patch; `patch_embed.proj` is that Linear (weight `[hidden,
patch_dim]`) and the loader reshapes the 5-D / GGUF 4-D conv weight into it.

Everything runs in float32, like the reference (`image.to(float32)`).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    depth: int = 27
    num_heads: int = 16
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)
    out_hidden_size: int = 5120

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

    @property
    def merge_dim(self) -> int:
        return self.hidden_size * self.spatial_merge_size**2


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Split-half rotary embedding on `[N, heads, head_dim]`."""
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


class VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden, bias=True)
        self.linear_fc2 = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, spans: list[tuple[int, int]]) -> mx.array:
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        q = _apply_rope(qkv[:, 0], cos, sin)
        k = _apply_rope(qkv[:, 1], cos, sin)
        v = qkv[:, 2]
        outs = []
        for a, b in spans:  # attention never crosses a frame boundary
            qs, ks, vs = (t[a:b].transpose(1, 0, 2)[None] for t in (q, k, v))
            o = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=self.head_dim**-0.5)
            outs.append(o[0].transpose(1, 0, 2).reshape(b - a, -1))
        return self.proj(mx.concatenate(outs, axis=0))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config.hidden_size, config.num_heads)
        self.mlp = VisionMLP(config.hidden_size, config.intermediate_size)

    def __call__(self, x, cos, sin, spans):
        x = x + self.attn(self.norm1(x), cos, sin, spans)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.proj = nn.Linear(config.patch_dim, config.hidden_size, bias=True)


class PatchMerger(nn.Module):
    """Main merger: LayerNorm on the per-patch hidden size, THEN spatial merge."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.merge_dim = config.merge_dim
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(config.merge_dim, config.merge_dim, bias=True)
        self.linear_fc2 = nn.Linear(config.merge_dim, config.out_hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x).reshape(-1, self.merge_dim)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class DeepstackMerger(nn.Module):
    """DeepStack merger: spatial merge FIRST, then LayerNorm on the merged dim."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.merge_dim = config.merge_dim
        self.norm = nn.LayerNorm(config.merge_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(config.merge_dim, config.merge_dim, bias=True)
        self.linear_fc2 = nn.Linear(config.merge_dim, config.out_hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x.reshape(-1, self.merge_dim))
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class VisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config)
        self.deepstack_merger_list = [DeepstackMerger(config) for _ in config.deepstack_visual_indexes]

    # ---- host-side index/table construction (small, once per call) ----

    def _rotary_tables(self, grids: list[tuple[int, int, int]]) -> tuple[mx.array, mx.array]:
        cfg = self.config
        merge = cfg.spatial_merge_size
        dim = cfg.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        max_hw = max(max(h, w) for _, h, w in grids)
        table = np.outer(np.arange(max_hw, dtype=np.float32), inv_freq)  # [max_hw, dim/2]
        rows = []
        for t, h, w in grids:
            r = np.arange(h // merge)[:, None, None, None] * merge + np.arange(merge)[None, None, :, None]
            c = np.arange(w // merge)[None, :, None, None] * merge + np.arange(merge)[None, None, None, :]
            r = np.broadcast_to(r, (h // merge, w // merge, merge, merge)).reshape(-1)
            c = np.broadcast_to(c, (h // merge, w // merge, merge, merge)).reshape(-1)
            coords = np.stack([r, c], axis=-1)
            rows.append(np.tile(coords, (t, 1)) if t > 1 else coords)
        pos = np.concatenate(rows)  # [N, 2]
        freqs = table[pos].reshape(pos.shape[0], -1)  # [N, head_dim/2]
        emb = np.concatenate([freqs, freqs], axis=-1)  # [N, head_dim]
        return mx.array(np.cos(emb)[:, None, :]), mx.array(np.sin(emb)[:, None, :])

    def _pos_embeddings(self, grids: list[tuple[int, int, int]]) -> mx.array:
        cfg = self.config
        side = int(cfg.num_position_embeddings**0.5)
        merge = cfg.spatial_merge_size
        pieces = []
        for t, h, w in grids:
            h_idx = np.linspace(0, side - 1, h, dtype=np.float32)
            w_idx = np.linspace(0, side - 1, w, dtype=np.float32)
            h0, w0 = h_idx.astype(np.int64), w_idx.astype(np.int64)
            h1, w1 = np.minimum(h0 + 1, side - 1), np.minimum(w0 + 1, side - 1)
            dh, dw = (h_idx - h0)[:, None], (w_idx - w0)[None, :]
            idx = [(h0[:, None] * side + w0[None]), (h0[:, None] * side + w1[None]),
                   (h1[:, None] * side + w0[None]), (h1[:, None] * side + w1[None])]
            wts = [(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw]
            emb = sum(
                self.pos_embed(mx.array(i.reshape(-1))) * mx.array(wt.reshape(-1, 1).astype(np.float32))
                for i, wt in zip(idx, wts)
            )  # [h*w, C]
            emb = mx.tile(emb, (t, 1)) if t > 1 else emb
            emb = emb.reshape(t, h // merge, merge, w // merge, merge, -1)
            pieces.append(emb.transpose(0, 1, 3, 2, 4, 5).reshape(-1, emb.shape[-1]))
        return mx.concatenate(pieces, axis=0)

    # ---- forward ----

    def __call__(self, patches: mx.array, grids: list[tuple[int, int, int]]) -> tuple[mx.array, list[mx.array]]:
        """`patches`: `[N, patch_dim]` (see `vision_preprocess`), `grids`: one
        `(t, h, w)` per image. Returns `(merged, deepstack)`, each `[N/4,
        out_hidden]` (spatial_merge_size**2 patches per output row)."""
        x = self.patch_embed.proj(patches.astype(mx.float32))
        x = x + self._pos_embeddings(grids)
        cos, sin = self._rotary_tables(grids)
        spans, offset = [], 0
        for t, h, w in grids:
            for _ in range(t):  # one attention segment per frame
                spans.append((offset, offset + h * w))
                offset += h * w
        deepstack: list[mx.array] = []
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, spans)
            if i in self.config.deepstack_visual_indexes:
                deepstack.append(self.deepstack_merger_list[self.config.deepstack_visual_indexes.index(i)](x))
        return self.merger(x), deepstack
```

- [ ] **Step 4: Écrire un `assign_vision_weights` provisoire pour lancer les tests**

Les tests de cette tâche appellent `map_mod.assign_vision_weights` : la tâche 3 en donne la version finale. Pour exécuter la tâche 2 seule, créer `apple_silicon_nodes/native/minimax_h3/vision_weight_map.py` avec **exactement** le contenu de la tâche 3, étape 3 (fonction `assign_vision_weights` et `verify_vision_shapes` incluses). La tâche 3 ajoute ensuite ses tests.

- [ ] **Step 5: Vérifier le succès**

Run: `uv run pytest tests/native/minimax_h3/test_vision_tower.py -q`
Expected: PASS. Si `test_matches_comfyui_on_random_weights` échoue avec une erreur > 1e-3, les suspects par ordre de probabilité : (1) ordre des champs dans `qkv.reshape(n, 3, heads, dim)`, (2) `_apply_rope` (moitié inversée), (3) ordre `transpose(0, 1, 3, 2, 4, 5)` de l'interpolation de position, (4) gelu exact contre tanh (`gelu_approx` seulement dans `VisionMLP`).

- [ ] **Step 6: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/vision_tower.py apple_silicon_nodes/native/minimax_h3/vision_weight_map.py tests/native/minimax_h3/test_vision_tower.py
git commit -m "feat: add Qwen3-VL vision tower in MLX with ComfyUI parity tests"
```

---

### Task 3: Chargement des poids `visual.*` (safetensors et GGUF)

**Files:**
- Create: `apple_silicon_nodes/native/minimax_h3/vision_weight_map.py` (si déjà créé à la tâche 2, le compléter avec `load_vision_tower`)
- Test: `tests/native/minimax_h3/test_vision_weight_map.py`

**Interfaces:**
- Consumes: `VisionConfig`, `VisionTower` (tâche 2) ; `open_checkpoint(path) -> TensorSource` avec `.shapes()` et `.get(name)` (`checkpoint_source.py`).
- Produces: `verify_vision_shapes(shapes: dict[str, tuple[int, ...]], config: VisionConfig) -> None` (lève `ValueError`) ; `assign_vision_weights(model: VisionTower, tensors: dict[str, mx.array]) -> int` (nombre de paramètres assignés ; lève `KeyError` si manquant, `ValueError` si taille incohérente) ; `load_vision_tower(path: str | Path) -> VisionTower`.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/native/minimax_h3/test_vision_weight_map.py` :

```python
"""assign_vision_weights / verify_vision_shapes / load_vision_tower."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_qwen3vl
from support.minimax_h3_module_loader import load_native_module

tower_mod = load_native_module("minimax_h3.vision_tower")
map_mod = load_native_module("minimax_h3.vision_weight_map")
source_mod = load_native_module("minimax_h3.checkpoint_source")
pre_mod = load_native_module("minimax_h3.vision_preprocess")

TINY = tower_mod.VisionConfig(
    hidden_size=64, intermediate_size=128, depth=2, num_heads=4, patch_size=4,
    num_position_embeddings=16, deepstack_visual_indexes=(0,), out_hidden_size=32,
)


def _tensors_for(model, patch_5d: bool = False):
    flat = dict(tree_flatten(model.parameters()))
    out = {k: mx.array(np.random.default_rng(0).standard_normal(v.shape).astype(np.float32)) for k, v in flat.items()}
    if patch_5d:  # checkpoint stores the Conv3d weight 5-D
        c = model.config
        out["patch_embed.proj.weight"] = out["patch_embed.proj.weight"].reshape(
            c.hidden_size, c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size
        )
    return out


def test_assign_reshapes_5d_and_gguf_4d_patch_embed():
    c = TINY
    for shape in [
        (c.hidden_size, c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size),
        (c.hidden_size * c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size),  # GGUF layout
    ]:
        model = tower_mod.VisionTower(c)
        tensors = _tensors_for(model)
        tensors["patch_embed.proj.weight"] = tensors["patch_embed.proj.weight"].reshape(shape)
        assert map_mod.assign_vision_weights(model, tensors) == len(dict(tree_flatten(model.parameters())))
        got = dict(tree_flatten(model.parameters()))["patch_embed.proj.weight"]
        assert got.shape == (c.hidden_size, c.patch_dim)


def test_assign_raises_on_missing_key():
    model = tower_mod.VisionTower(TINY)
    tensors = _tensors_for(model)
    del tensors["blocks.1.attn.qkv.bias"]
    with pytest.raises(KeyError, match="blocks.1.attn.qkv.bias"):
        map_mod.assign_vision_weights(model, tensors)


def test_assign_raises_on_wrong_size():
    model = tower_mod.VisionTower(TINY)
    tensors = _tensors_for(model)
    tensors["pos_embed.weight"] = mx.zeros((3, 3))
    with pytest.raises(ValueError, match="pos_embed.weight"):
        map_mod.assign_vision_weights(model, tensors)


def test_verify_shapes_accepts_real_dims_and_rejects_other():
    cfg = tower_mod.VisionConfig()
    real = {
        "patch_embed.proj.weight": (1152, 3, 2, 16, 16), "pos_embed.weight": (2304, 1152),
        "merger.linear_fc2.weight": (5120, 4608), "blocks.0.mlp.linear_fc1.weight": (4304, 1152),
        **{f"blocks.{i}.norm1.weight": (1152,) for i in range(27)},
        **{f"deepstack_merger_list.{i}.norm.weight": (4608,) for i in range(3)},
    }
    map_mod.verify_vision_shapes(real, cfg)
    with pytest.raises(ValueError, match="blocks"):
        map_mod.verify_vision_shapes({k: v for k, v in real.items() if k != "blocks.26.norm1.weight"}, cfg)


_ST = Path("/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
_GGUF = Path("/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf")
_FULL = pytest.mark.skipif(os.environ.get("ASDX_FULL_GGUF_TEST") != "1", reason="reads real checkpoints; set ASDX_FULL_GGUF_TEST=1")


@_FULL
def test_gguf_and_safetensors_visual_weights_are_identical():
    if not (_ST.exists() and _GGUF.exists()):
        pytest.skip("real TE checkpoints not present")
    st, gg = source_mod.open_checkpoint(_ST), source_mod.open_checkpoint(_GGUF)
    names = [n for n in st.shapes() if n.startswith("visual.")]
    assert len(names) == 351 and set(names) == {n for n in gg.shapes() if n.startswith("visual.")}
    for name in names:
        a, b = st.get(name), gg.get(name)
        assert a.size == b.size, name
        assert bool(mx.array_equal(a.reshape(-1), b.reshape(-1))), name


@_FULL
def test_real_weights_match_comfyui_on_a_real_image():
    if not _ST.exists():
        pytest.skip("real TE checkpoint not present")
    import torch
    from safetensors import safe_open

    qwen3vl, _, ops = load_real_comfy_qwen3vl()
    cfg = {**qwen3vl.QWEN3VL_VISION_COMMON, **qwen3vl.QWEN3VL_VISION["qwen3vl_32b"], "out_hidden_size": 5120}
    ref = qwen3vl.Qwen3VLVisionModel(cfg, device="cpu", dtype=torch.float32, ops=ops.disable_weight_init)
    with safe_open(str(_ST), framework="pt") as f:
        state = {k[len("visual."):]: f.get_tensor(k).float() for k in f.keys() if k.startswith("visual.")}
    ref.load_state_dict(state, strict=True)

    image = np.random.default_rng(3).random((224, 336, 3), dtype=np.float32)
    patches, grid = pre_mod.preprocess_image(image)
    with torch.no_grad():
        ref_merged, ref_ds = ref(torch.from_numpy(patches), torch.tensor([grid]))
    model = map_mod.load_vision_tower(_ST)
    merged, ds = model(mx.array(patches), [grid])
    mx.eval(merged, ds)

    def cos(a, b):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cos(merged, ref_merged.numpy()) > 0.9999
    for got, want in zip(ds, ref_ds):
        assert cos(got, want.numpy()) > 0.9999
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_vision_weight_map.py -q`
Expected: FAIL (module absent ou `load_vision_tower` manquant).

- [ ] **Step 3: Implémenter**

`apple_silicon_nodes/native/minimax_h3/vision_weight_map.py` :

```python
"""Loads the `visual.*` weights of MiniMax H3's Qwen3-VL-32B text-encoder
checkpoint (safetensors or GGUF, both dense BF16 here) into a `VisionTower`.

Format facts (inspected on the real files, 351 `visual.*` tensors each, same
names in both): the Conv3d `patch_embed.proj.weight` is `(1152, 3, 2, 16, 16)`
in safetensors but `(3456, 2, 16, 16)` in GGUF (the RGB axis folded into the
first dimension); both flatten row-major to `(1152, 1536)`, which is the
Linear the tower uses. Only the Qwen3-VL-32B geometry exists, so shapes are
verified against `VisionConfig()` rather than auto-detected.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .checkpoint_source import open_checkpoint
from .vision_tower import VisionConfig, VisionTower

_PREFIX = "visual."


def verify_vision_shapes(shapes: dict[str, tuple[int, ...]], config: VisionConfig) -> None:
    """Raise `ValueError` unless `shapes` (names without the `visual.` prefix)
    describe exactly `config`'s geometry."""
    depth = len([k for k in shapes if re.fullmatch(r"blocks\.\d+\.norm1\.weight", k)])
    n_ds = len([k for k in shapes if re.fullmatch(r"deepstack_merger_list\.\d+\.norm\.weight", k)])
    expected = {
        "pos_embed.weight": (config.num_position_embeddings, config.hidden_size),
        "merger.linear_fc2.weight": (config.out_hidden_size, config.merge_dim),
        "blocks.0.mlp.linear_fc1.weight": (config.intermediate_size, config.hidden_size),
    }
    problems = []
    if depth != config.depth:
        problems.append(f"blocks: found {depth}, expected {config.depth}")
    if n_ds != len(config.deepstack_visual_indexes):
        problems.append(f"deepstack mergers: found {n_ds}, expected {len(config.deepstack_visual_indexes)}")
    for name, shape in expected.items():
        if tuple(shapes.get(name, ())) != shape:
            problems.append(f"{name}: found {shapes.get(name)}, expected {shape}")
    patch = shapes.get("patch_embed.proj.weight")
    if patch is None or math.prod(patch) != config.hidden_size * config.patch_dim:
        problems.append(f"patch_embed.proj.weight: found {patch}, expected {config.hidden_size * config.patch_dim} elements")
    if problems:
        raise ValueError("ASDX: visual tower checkpoint does not match Qwen3-VL-32B: " + "; ".join(problems))


def assign_vision_weights(model: VisionTower, tensors: dict[str, mx.array]) -> int:
    """Assign `tensors` (names without the `visual.` prefix) to `model` by
    name, converting to float32. Any missing name or size mismatch raises."""
    flat = dict(tree_flatten(model.parameters()))
    missing = sorted(set(flat) - set(tensors))
    if missing:
        raise KeyError(f"ASDX: vision tower weights missing from checkpoint: {missing[:5]} ({len(missing)} total)")
    new: dict[str, mx.array] = {}
    for name, current in flat.items():
        tensor = tensors[name]
        if tensor.size != current.size:
            raise ValueError(f"ASDX: vision weight '{name}': checkpoint shape {tuple(tensor.shape)} does not fit {tuple(current.shape)}")
        new[name] = tensor.reshape(current.shape).astype(mx.float32)
    model.update(tree_unflatten(list(new.items())))
    mx.eval(model.parameters())
    return len(new)


def load_vision_tower(path: str | Path) -> VisionTower:
    """Build the Qwen3-VL-32B `VisionTower` and load its `visual.*` weights."""
    source = open_checkpoint(path)
    shapes = {n[len(_PREFIX):]: s for n, s in source.shapes().items() if n.startswith(_PREFIX)}
    config = VisionConfig()
    verify_vision_shapes(shapes, config)
    tensors = {name: source.get(_PREFIX + name) for name in shapes}
    model = VisionTower(config)
    assigned = assign_vision_weights(model, tensors)
    print(f"[ASDX] MiniMax H3 vision tower ({Path(path).suffix.lstrip('.').lower()}): assigned {assigned} params")
    return model
```

- [ ] **Step 4: Vérifier le succès (tests synthétiques)**

Run: `uv run pytest tests/native/minimax_h3/test_vision_weight_map.py tests/native/minimax_h3/test_vision_tower.py -q`
Expected: PASS (les 2 tests `_FULL` sont ignorés).

- [ ] **Step 5: Vérifier sur les vrais fichiers**

Run: `ASDX_FULL_GGUF_TEST=1 uv run pytest tests/native/minimax_h3/test_vision_weight_map.py -q`
Expected: PASS. Si `test_gguf_and_safetensors_visual_weights_are_identical` échoue sur `patch_embed.proj.weight`, la fusion GGUF `(3456, 2, 16, 16)` n'est pas un simple `reshape` : ne pas deviner, inspecter les valeurs (`a.reshape(-1)` contre `b.reshape(-1)`) et corriger `assign_vision_weights` pour le GGUF.

- [ ] **Step 6: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/vision_weight_map.py tests/native/minimax_h3/test_vision_weight_map.py
git commit -m "feat: load Qwen3-VL vision tower weights from H3 text encoder checkpoints"
```

---

### Task 4: Mesure mémoire, régression, trace

**Files:**
- Modify: `docs/superpowers/specs/2026-09-19-minimax-h3-i2v-ref-design.md` (ligne « ~1,2 Go » de la brique 1)
- Test: suite complète

**Interfaces:**
- Consumes: `load_vision_tower` (tâche 3).
- Produces: chiffres mesurés (pic, actif) consignés dans la spec ; suite verte hors échec Krea2 préexistant.

- [ ] **Step 1: Mesurer le chargement réel**

Run :
```bash
uv run python - <<'EOF'
import sys, time; sys.path.insert(0, "tests")
from support.minimax_h3_module_loader import load_native_module
import mlx.core as mx
m = load_native_module("minimax_h3.vision_weight_map")
for p in ["/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
          "/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"]:
    mx.reset_peak_memory(); t = time.time()
    tower = m.load_vision_tower(p)
    print(p.rsplit(".", 1)[-1], "load s", round(time.time() - t), "peak GB", round(mx.get_peak_memory() / 1e9, 2),
          "active GB", round(mx.get_active_memory() / 1e9, 2))
    del tower; mx.clear_cache()
EOF
```
Expected: pic et actif de l'ordre de 2 à 5 Go. Noter les vrais chiffres.

- [ ] **Step 2: Consigner dans la spec**

Dans `docs/superpowers/specs/2026-09-19-minimax-h3-i2v-ref-design.md`, brique 1, remplacer « tour maintenue en bf16 (~0,6 Md de paramètres, ~1,2 Go) » par « tour en float32 comme la référence (~0,6 Md de paramètres) ; mesuré : pic X Go, actif Y Go (safetensors), X' / Y' (GGUF) » avec les chiffres de l'étape 1.

- [ ] **Step 3: Suite complète**

Run: `uv run pytest tests -q`
Expected: tout passe, sauf `tests/support/test_krea2_module_loader.py::test_plain_import_of_model_fails_outside_comfyui` (échec préexistant, déjà présent sans nos changements ; le nommer dans le compte rendu).

- [ ] **Step 4: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add docs/superpowers/specs/2026-09-19-minimax-h3-i2v-ref-design.md
git commit -m "docs: record measured vision tower memory in the H3 i2v/ref spec"
```

---

## Auto-revue du plan

- **Couverture de la spec, brique 1** : tour vision (tâche 2), chargement `visual.*` des deux formats (tâche 3), parité ComfyUI (tâches 1-3), chargement optionnel dans le loader de l'encodeur : **volontairement reporté à la brique 2** (le raccordement au loader `Qwen3TextEncoder` dépend de la présentation du prompt).
- **Placeholders** : aucun ; le seul pointeur croisé (tâche 2 étape 4) renvoie à du code complet en tâche 3 étape 3.
- **Cohérence des types** : `VisionConfig.patch_dim/merge_dim`, `VisionTower.__call__(patches, grids)`, `assign_vision_weights(model, tensors) -> int`, `load_vision_tower(path)` sont utilisés à l'identique dans les 3 tâches.
- **Risque connu** : les tolérances de parité (1e-3 sur poids aléatoires, cosinus > 0,9999 sur vrais poids) sont des choix initiaux ; les ajuster seulement après avoir vérifié le cas nul (`test_metric_separates_different_weights`).
