# MiniMax H3 brique 2 : présentation du prompt et encodeur avec vision

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire passer des images (keyframes i2v, références ref2va) et des blocs vidéo par la tour vision (brique 1), puis par l'encodeur de texte Qwen3-VL-32B en MLX, avec injection DeepStack et M-RoPE entrelacé, en produisant le contexte `[S, 5120]` et les `token_tags` que consommera le DiT.

**Architecture:** Le tokenizer de ComfyUI (`MiniMaxH3Tokenizer`, sans poids) reste la source de la présentation du prompt (`<Picture i>: `, blocs vision, libellés audio/vidéo). Un nouveau module `vision_conditioning.py` déplie ses entrées en séquence de ids + lignes vision (équivalent de `process_tokens` de ComfyUI) ; `vision_rope.py` fournit les position ids et le RoPE entrelacé ; `text_encoder.py` accepte un `VisionInputs` optionnel (le chemin texte seul reste inchangé bit à bit).

**Tech Stack:** MLX, numpy, torch (prétraitement et tests de parité seulement), pytest, `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-19-minimax-h3-i2v-ref-design.md` (brique 2 et « Notes de passage vers la brique 2 »). Références : `comfy/text_encoders/minimax.py`, `qwen3vl.py`, `qwen_vl.py`, `llama.py` (`precompute_freqs_cis`, `Llama2_.forward`), `sd1_clip.py::process_tokens` dans `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI`.

## Global Constraints

- Toujours `uv run pytest` / `uv run python` (jamais `python` nu). Python 3.10+, annotations complètes, docstrings en anglais, pas d'emoji ni de tiret cadratin.
- Fail-closed : forme, entrée ou dépendance manquante lève une exception explicite, jamais de valeur plausible par défaut.
- **Précision GPU (canon de la brique 1)** : le matmul fp32 GPU de MLX a ~7,5e-4 d'erreur relative. Les tests de parité numérique contre ComfyUI tournent sous `mx.stream(mx.cpu)` (tolérance 1e-3) ; en plus, chaque brique porte un test GPU contre CPU sans ComfyUI (borne relative 1e-2 avec cas nul). Les seuils se règlent sur le cosinus par ligne et l'erreur relative L2, jamais sur un cosinus aplati.
- Les tests de parité contre ComfyUI sont ignorés (`pytest.skip`) si l'installation est absente ; les tests sur vrais poids exigent `ASDX_FULL_GGUF_TEST=1`. Un test ignoré n'est pas une preuve : le dire.
- Les tests importent via `load_native_module("minimax_h3.<nom>")` (`tests/support/minimax_h3_module_loader.py`) ; les tests de nœuds via `tests/support/comfy_stub.py` comme `tests/test_minimax_h3_loaders.py`.
- Le chemin texte seul de l'encodeur (`vision=None`) doit rester identique : les tests existants (`test_text_encoder*.py`) doivent passer sans modification.
- **Aucun commit automatique** (règle utilisateur) : chaque étape « Commit » signifie « mettre les fichiers dans l'index et attendre l'ordre explicite ». Pas de ligne d'attribution.

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `apple_silicon_nodes/native/minimax_h3/vision_preprocess.py` (modifier) | ajoute `preprocess_video_block`, entrée 4-D, helper de taille partagé |
| `apple_silicon_nodes/native/minimax_h3/vision_rope.py` (créer) | `mrope_position_ids`, `interleaved_mrope_cos_sin` |
| `apple_silicon_nodes/native/minimax_h3/text_encoder.py` (modifier) | `VisionInputs`, `vision=` dans `Qwen3TextEncoder.__call__` |
| `apple_silicon_nodes/native/minimax_h3/vision_conditioning.py` (créer) | `token_tags`, `build_vision_conditioning`, `encode_with_vision` |
| `apple_silicon_nodes/minimax_h3_nodes.py` (modifier) | option `load_vision` du loader, helper `encode_minimax_h3_prompt` |
| `tests/support/comfyui_reference_loader.py` (modifier) | ajoute `load_real_comfy_text_encoders()` |
| `tests/native/minimax_h3/test_vision_preprocess.py` (modifier) | cas vidéo et 4-D |
| `tests/native/minimax_h3/test_vision_rope.py` (créer) | parité positions et RoPE |
| `tests/native/minimax_h3/test_text_encoder_vision.py` (créer) | parité encodeur, DeepStack, GPU contre CPU |
| `tests/native/minimax_h3/test_vision_conditioning.py` (créer) | présentation, tags, réel |
| `tests/test_minimax_h3_loaders.py` (modifier) | option `load_vision`, helper de prompt |

---

### Task 1: Prétraitement vidéo et entrée 4-D

**Files:**
- Modify: `apple_silicon_nodes/native/minimax_h3/vision_preprocess.py`
- Modify: `tests/support/comfyui_reference_loader.py` (ajout en fin de fichier)
- Test: `tests/native/minimax_h3/test_vision_preprocess.py` (ajouts)

**Interfaces:**
- Consumes: `preprocess_image` existant (brique 1).
- Produces: `preprocess_image(image: np.ndarray, ...)` accepte désormais `[H, W, 3]` **ou** `[1, H, W, 3]` (sinon `ValueError` contenant « 3 channels ») ; `preprocess_video_block(frames: np.ndarray, *, patch_size: int = 16, temporal_patch_size: int = 2, merge_size: int = 2, min_pixels: int = 3136, max_pixels: int = 12845056, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5)) -> tuple[np.ndarray, tuple[int, int, int]]` : `frames` float32 `[2, H, W, 3]` en `[0, 1]`, renvoie `(patchs float32 [gh*gw, 3*2*P*P], (1, gh, gw))` avec les **deux images distinctes** dans le patch temporel ; `load_real_comfy_text_encoders() -> tuple[module, module, module, module]` = `(comfy.text_encoders.minimax, qwen_vl, llama, comfy.ops)`.

- [ ] **Step 1: Ajouter le chargeur de la référence**

Ajouter à la fin de `tests/support/comfyui_reference_loader.py` :

```python
def load_real_comfy_text_encoders():
    """Real `comfy.text_encoders.minimax`, `qwen_vl`, `llama` and `comfy.ops`,
    for parity tests of the MiniMax H3 vision-grounded text encoder."""
    if not _COMFYUI_ROOT.exists() or not _COMFYUI_VENV_SITE_PACKAGES.exists():
        pytest.skip("ComfyUI install not present on this machine")
    for name in list(sys.modules):
        if name == "comfy" or name.startswith("comfy."):
            del sys.modules[name]
    sys.path.insert(0, str(_COMFYUI_ROOT))
    sys.path.insert(0, str(_COMFYUI_VENV_SITE_PACKAGES))
    try:
        import comfy.ops as ops
        import comfy.text_encoders.llama as llama
        import comfy.text_encoders.minimax as minimax
        import comfy.text_encoders.qwen_vl as qwen_vl
    except ImportError as e:
        pytest.skip(f"comfy text-encoder modules not importable: {e}")
    return minimax, qwen_vl, llama, ops
```

- [ ] **Step 2: Écrire les tests qui échouent**

Ajouter à `tests/native/minimax_h3/test_vision_preprocess.py` (compléter l'import : `from support.comfyui_reference_loader import load_real_comfy_qwen3vl, load_real_comfy_text_encoders`) :

```python
@pytest.mark.parametrize("shape", [(224, 336), (250, 333), (48, 64)])
def test_video_block_matches_comfyui(shape):
    minimax, _, _, _ = load_real_comfy_text_encoders()
    import torch

    rng = np.random.default_rng(1)
    frames = rng.random((2, shape[0], shape[1], 3), dtype=np.float32)
    ref_patches, ref_grid = minimax.process_video_block(torch.from_numpy(frames))
    patches, grid = pre.preprocess_video_block(frames)
    assert grid == tuple(int(v) for v in ref_grid[0].tolist())
    assert patches.shape == tuple(ref_patches.shape)
    assert np.abs(patches - ref_patches.numpy()).max() < 1e-6


def test_video_block_uses_two_distinct_frames():
    rng = np.random.default_rng(2)
    a = rng.random((1, 64, 64, 3), dtype=np.float32)
    b = rng.random((1, 64, 64, 3), dtype=np.float32)
    distinct, _ = pre.preprocess_video_block(np.concatenate([a, b]))
    repeated, _ = pre.preprocess_video_block(np.concatenate([a, a]))
    assert np.abs(distinct - repeated).max() > 0.1  # null case: the frames must matter


def test_video_block_rejects_wrong_frame_count():
    with pytest.raises(ValueError, match="2 frames"):
        pre.preprocess_video_block(np.zeros((3, 64, 64, 3), dtype=np.float32))


def test_image_accepts_4d_comfy_layout():
    rng = np.random.default_rng(3)
    image = rng.random((96, 128, 3), dtype=np.float32)
    a, ga = pre.preprocess_image(image)
    b, gb = pre.preprocess_image(image[None])
    assert ga == gb and np.array_equal(a, b)


def test_image_rejects_batches_and_other_ranks():
    with pytest.raises(ValueError, match="3 channels"):
        pre.preprocess_image(np.zeros((2, 64, 64, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="3 channels"):
        pre.preprocess_image(np.zeros((64, 64), dtype=np.float32))
```

- [ ] **Step 3: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_vision_preprocess.py -q`
Expected: FAIL (`AttributeError: ... preprocess_video_block`, et les cas 4-D/rang).

- [ ] **Step 4: Remplacer le module par la version complète**

`apple_silicon_nodes/native/minimax_h3/vision_preprocess.py` (contenu intégral ; la logique de `preprocess_image` est inchangée, la taille cible est extraite dans `_target_size` pour ne pas la dupliquer) :

```python
"""Image / video block -> Qwen3-VL vision patches, ported 1:1 from
`comfy/text_encoders/qwen_vl.py::process_qwen2vl_images` (image path: one
frame repeated `temporal_patch_size` times) and
`comfy/text_encoders/minimax.py::process_video_block` (video path: two
DISTINCT frames fill the temporal patch).

Runs on the host with torch: the bilinear `F.interpolate(align_corners=False)`
is what the reference (and the released model) uses, and a handful of small
images per call makes a device round trip irrelevant next to matching its
numerics. This is the only torch use in the vision path.
"""

from __future__ import annotations

import math

import numpy as np


def _target_size(
    height: int, width: int, *, patch_size: int, merge_size: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Smart-resize target `(h_bar, w_bar)`, identical in both reference paths."""
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
    return h_bar, w_bar


def _as_hwc(image: np.ndarray) -> np.ndarray:
    """Accept `[H, W, 3]` or a single ComfyUI IMAGE `[1, H, W, 3]`."""
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"ASDX: vision preprocessing needs [H, W, 3] or [1, H, W, 3] (3 channels), got shape {image.shape}"
        )
    return image


def _patchify(
    frames_chw: "torch.Tensor",  # [T, 3, h_bar, w_bar] normalized
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    h_bar, w_bar = frames_chw.shape[-2:]
    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    patches = frames_chw.reshape(
        1, temporal_patch_size, 3,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    return flat.numpy(), (1, grid_h, grid_w)


def _resize_normalize(
    frames: np.ndarray, h_bar: int, w_bar: int, image_mean, image_std
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    chw = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32)).permute(0, 3, 1, 2)
    resized = F.interpolate(chw, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(image_mean, dtype=resized.dtype).view(1, 3, 1, 1)
    std = torch.tensor(image_std, dtype=resized.dtype).view(1, 3, 1, 1)
    return (resized - mean) / std


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
    """`image`: float32 `[H, W, 3]` (or `[1, H, W, 3]`) in `[0, 1]`. Returns
    `(patches, grid)`: `patches` float32 `[gh*gw, 3*temporal_patch_size*
    patch_size**2]` in merge-block order, `grid = (1, gh, gw)`."""
    image = _as_hwc(image)
    h_bar, w_bar = _target_size(
        image.shape[0], image.shape[1],
        patch_size=patch_size, merge_size=merge_size, min_pixels=min_pixels, max_pixels=max_pixels,
    )
    normalized = _resize_normalize(image[None], h_bar, w_bar, image_mean, image_std)  # [1, 3, h, w]
    frames = normalized.repeat(temporal_patch_size, 1, 1, 1)
    return _patchify(frames, patch_size=patch_size, temporal_patch_size=temporal_patch_size, merge_size=merge_size)


def preprocess_video_block(
    frames: np.ndarray,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """`frames`: float32 `[temporal_patch_size, H, W, 3]` in `[0, 1]` (a 2-frame
    video block, sampled at 2 fps). Same resize/normalize policy as
    `preprocess_image`, but the frames fill the temporal patch (`grid_t = 1`)."""
    if frames.ndim != 4 or frames.shape[0] != temporal_patch_size or frames.shape[-1] != 3:
        raise ValueError(
            f"ASDX: a video block needs exactly {temporal_patch_size} frames of [H, W, 3] (3 channels), "
            f"got shape {frames.shape}"
        )
    h_bar, w_bar = _target_size(
        frames.shape[1], frames.shape[2],
        patch_size=patch_size, merge_size=merge_size, min_pixels=min_pixels, max_pixels=max_pixels,
    )
    normalized = _resize_normalize(frames, h_bar, w_bar, image_mean, image_std)
    return _patchify(normalized, patch_size=patch_size, temporal_patch_size=temporal_patch_size, merge_size=merge_size)
```

- [ ] **Step 5: Vérifier le succès**

Run: `uv run pytest tests/native/minimax_h3/test_vision_preprocess.py -q -rs`
Expected: PASS, 0 ignoré (les tests de la brique 1 restent verts : `test_min_pixels_upscales_tiny_images`, `test_rejects_non_rgb`, la parité image). Si un test de parité est ignoré, dire pourquoi.

- [ ] **Step 6: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/vision_preprocess.py tests/support/comfyui_reference_loader.py tests/native/minimax_h3/test_vision_preprocess.py
git commit -m "feat: add video block preprocessing and 4-D image input for the vision tower"
```

---

### Task 2: Position ids M-RoPE et RoPE entrelacé

**Files:**
- Create: `apple_silicon_nodes/native/minimax_h3/vision_rope.py`
- Test: `tests/native/minimax_h3/test_vision_rope.py`

**Interfaces:**
- Consumes: rien de la tâche 1.
- Produces: `mrope_position_ids(infos: list[dict], seq_len: int) -> np.ndarray | None` : `infos` = liste de `{"index": int, "size": int, "grid": (t, h, w)}` dans l'ordre de la séquence ; renvoie float32 `[3, seq_len]` ou `None` si `infos` est vide ; `interleaved_mrope_cos_sin(position_ids: np.ndarray, head_dim: int, theta: float, rope_dims: tuple[int, int, int] = (24, 20, 20)) -> tuple[mx.array, mx.array]` : `(cos, sin)` float32 `[S, head_dim // 2]` (angles non dupliqués, comme `qwen3_rope_cos_sin`) ; lève `ValueError` si `sum(rope_dims) != head_dim // 2`.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/native/minimax_h3/test_vision_rope.py` :

```python
"""Parity of vision_rope against comfy.text_encoders.qwen_vl.qwen2vl_mrope_position_ids
and llama.precompute_freqs_cis (Qwen3-VL interleaved M-RoPE)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_text_encoders
from support.minimax_h3_module_loader import load_native_module

rope = load_native_module("minimax_h3.vision_rope")

# (index, size, grid): sizes = t*h*w/4 for the merged tokens
CASES = [
    [dict(index=5, size=6, grid=(1, 4, 6))],
    [dict(index=3, size=6, grid=(1, 4, 6)), dict(index=20, size=2, grid=(1, 2, 4))],
    [dict(index=0, size=4, grid=(1, 4, 4)), dict(index=9, size=4, grid=(1, 4, 4)), dict(index=30, size=6, grid=(1, 4, 6))],
]


def _ref_infos(cases, torch):
    return [
        {"type": "image", "index": c["index"], "size": c["size"], "extra": {"grid": torch.tensor([list(c["grid"])])}}
        for c in cases
    ]


@pytest.mark.parametrize("infos", CASES)
def test_position_ids_match_comfyui(infos):
    import torch

    _, qwen_vl, _, _ = load_real_comfy_text_encoders()
    seq_len = infos[-1]["index"] + infos[-1]["size"] + 7
    ref = qwen_vl.qwen2vl_mrope_position_ids(_ref_infos(infos, torch), seq_len, "cpu")
    got = rope.mrope_position_ids(infos, seq_len)
    assert got.shape == tuple(ref.shape) == (3, seq_len)
    assert np.array_equal(got, ref.numpy())


def test_position_ids_empty_is_none():
    assert rope.mrope_position_ids([], 10) is None


@pytest.mark.parametrize("infos", CASES)
def test_interleaved_cos_sin_match_comfyui(infos):
    import torch

    _, qwen_vl, llama, _ = load_real_comfy_text_encoders()
    seq_len = infos[-1]["index"] + infos[-1]["size"] + 7
    pos = qwen_vl.qwen2vl_mrope_position_ids(_ref_infos(infos, torch), seq_len, "cpu")
    ref_cos, ref_sin, _ = llama.precompute_freqs_cis(128, pos, 5000000.0, rope_dims=[24, 20, 20], interleaved_mrope=True)
    with mx.stream(mx.cpu):
        cos, sin = rope.interleaved_mrope_cos_sin(pos.numpy(), 128, 5000000.0, (24, 20, 20))
        mx.eval(cos, sin)
    assert cos.shape == (seq_len, 64)
    assert np.abs(np.array(cos) - ref_cos[0, :, :64].numpy()).max() < 1e-4
    assert np.abs(np.array(sin) - ref_sin[0].numpy()).max() < 1e-4


def test_interleaving_differs_from_plain_rope():
    """Null case: with H/W positions that differ from T, the interleaved table must
    differ from a plain single-axis table."""
    pos = np.stack([np.arange(8), np.arange(8) * 3, np.arange(8) * 5]).astype(np.float32)
    cos, _ = rope.interleaved_mrope_cos_sin(pos, 128, 5000000.0)
    plain, _ = rope.interleaved_mrope_cos_sin(np.stack([pos[0]] * 3), 128, 5000000.0)
    assert np.abs(np.array(cos) - np.array(plain)).max() > 1e-2


def test_rope_dims_must_cover_half_head_dim():
    with pytest.raises(ValueError, match="rope_dims"):
        rope.interleaved_mrope_cos_sin(np.zeros((3, 4), dtype=np.float32), 128, 1.0, (24, 20, 10))
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_vision_rope.py -q`
Expected: FAIL (`ModuleNotFoundError: ... vision_rope`).

- [ ] **Step 3: Implémenter**

`apple_silicon_nodes/native/minimax_h3/vision_rope.py` :

```python
"""M-RoPE for the vision-grounded Qwen3-VL text encoder.

`mrope_position_ids` is a literal port of
`comfy/text_encoders/qwen_vl.py::qwen2vl_mrope_position_ids` (text runs are
sequential, each vision span gets its own T/H/W grid positions, and every
later token shifts by the span's compression). `interleaved_mrope_cos_sin`
ports the `interleaved_mrope=True` branch of
`comfy/text_encoders/llama.py::precompute_freqs_cis`: T-frequencies by
default, with the H and W frequencies replacing every 3rd dimension.
Text-only prompts never reach here (their positions stay `[1, S]` and the
plain single-axis RoPE in `text_encoder_rope.py` applies).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def mrope_position_ids(infos: list[dict], seq_len: int) -> np.ndarray | None:
    """`infos`: one `{"index", "size", "grid": (t, h, w)}` per vision span, in
    sequence order (`index` = first row of the span, `size` = its row count).
    Returns float32 `[3, seq_len]`, or `None` when there is no vision span."""
    if not infos:
        return None
    position_ids = np.zeros((3, seq_len), dtype=np.float32)
    offset = 0
    for n, info in enumerate(infos):
        t, h, w = info["grid"]
        start, size = info["index"], info["size"]
        if n == 0:
            position_ids[:, :start] = np.arange(start, dtype=np.float32)
        end = start + size
        len_max = max(t, h, w) // 2
        start_next = len_max + start
        position_ids[:, end:] = np.arange(
            start_next + offset, start_next + (seq_len - end) + offset, dtype=np.float32
        )
        position_ids[0, start:end] = start + offset
        max_h = h // 2
        position_ids[1, start:end] = np.repeat(
            np.arange(start + offset, start + max_h + offset), math.ceil(size / max_h)
        )[:size]
        max_w = w // 2
        position_ids[2, start:end] = np.tile(
            np.arange(start + offset, start + max_w + offset), math.ceil(size / max_w)
        )[:size]
        offset += len_max - size
    return position_ids


def interleaved_mrope_cos_sin(
    position_ids: np.ndarray,
    head_dim: int,
    theta: float,
    rope_dims: tuple[int, int, int] = (24, 20, 20),
) -> tuple[mx.array, mx.array]:
    """`position_ids`: `[3, S]`. Returns `(cos, sin)`, float32 `[S, head_dim//2]`
    (angles are not duplicated, same convention as `qwen3_rope_cos_sin`)."""
    half = head_dim // 2
    if sum(rope_dims) != half:
        raise ValueError(f"ASDX: rope_dims {rope_dims} must sum to head_dim // 2 = {half}")
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = position_ids[:, :, None].astype(np.float32) * inv_freq[None, None, :]  # [3, S, half]
    inter = freqs[0].copy()
    for axis, offset in ((1, 1), (2, 2)):
        idx = slice(offset, rope_dims[axis] * 3, 3)
        inter[..., idx] = freqs[axis][..., idx]
    return mx.array(np.cos(inter)), mx.array(np.sin(inter))
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/native/minimax_h3/test_vision_rope.py -q -rs`
Expected: PASS, 0 ignoré. Si la parité des positions échoue, ne pas deviner : imprimer les deux tableaux sur le cas 2 (deux images) et comparer ligne par ligne (le suspect est `offset` et la région `[:, end:]` réécrite par chaque image).

- [ ] **Step 5: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/vision_rope.py tests/native/minimax_h3/test_vision_rope.py
git commit -m "feat: add Qwen3-VL M-RoPE position ids and interleaved cos/sin with ComfyUI parity"
```

---

### Task 3: Encodeur de texte avec lignes vision et DeepStack

**Files:**
- Modify: `apple_silicon_nodes/native/minimax_h3/text_encoder.py` (imports en tête ; `_Qwen3Backbone.__call__` ; `Qwen3TextEncoder.__call__` ; nouvelle dataclass `VisionInputs`)
- Test: `tests/native/minimax_h3/test_text_encoder_vision.py`

**Interfaces:**
- Consumes: `interleaved_mrope_cos_sin`, `mrope_position_ids` (tâche 2) ; `qwen3_rope_cos_sin` (existant).
- Produces: `VisionInputs` (dataclass figée) avec `rows: mx.array` `[Nv, hidden]` (lignes vision fusionnées, remplacent les embeddings texte), `row_indices: np.ndarray` int `[Nv]` (positions dans la séquence dépliée, croissantes), `deepstack: list[mx.array]` (chacun `[Nv, hidden]`, ajouté après la couche `i` pour `i < len(deepstack)`), `position_ids: np.ndarray` `[3, S]`, `rope_dims: tuple[int, int, int] = (24, 20, 20)` ; `Qwen3TextEncoder.__call__(input_ids: mx.array, vision: VisionInputs | None = None) -> mx.array` ; avec `vision=None` le comportement est **identique** à avant.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/native/minimax_h3/test_text_encoder_vision.py` :

```python
"""Qwen3TextEncoder with vision rows: parity vs the real ComfyUI Llama2_ (interleaved
M-RoPE + DeepStack) on a tiny random-weight config, and a GPU-vs-CPU self-consistency test."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_text_encoders
from support.minimax_h3_module_loader import load_native_module

config_mod = load_native_module("minimax_h3.text_encoder_config")
enc_mod = load_native_module("minimax_h3.text_encoder")
rope_mod = load_native_module("minimax_h3.vision_rope")

HIDDEN, HEAD_DIM, ROPE_DIMS = 32, 8, (2, 1, 1)  # sum == HEAD_DIM // 2
SEQ = 16
INFOS = [dict(index=3, size=4, grid=(1, 4, 4)), dict(index=10, size=2, grid=(1, 2, 4))]


def _cfg():
    return config_mod.Qwen3TextEncoderConfig(
        vocab_size=100, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=HEAD_DIM,
        rms_norm_eps=1e-6, rope_theta=5000000.0, dtype="float32",
    )


def _vision(seed: int = 0) -> "enc_mod.VisionInputs":
    rng = np.random.default_rng(seed)
    idx = np.concatenate([np.arange(i["index"], i["index"] + i["size"]) for i in INFOS])
    n = len(idx)
    return enc_mod.VisionInputs(
        rows=mx.array(rng.standard_normal((n, HIDDEN)).astype(np.float32)),
        row_indices=idx,
        deepstack=[mx.array(rng.standard_normal((n, HIDDEN)).astype(np.float32)) for _ in range(3)],
        position_ids=rope_mod.mrope_position_ids(INFOS, SEQ),
        rope_dims=ROPE_DIMS,
    )


def _seeded_encoder(seed: int = 0):
    enc = enc_mod.Qwen3TextEncoder(_cfg())
    rng = np.random.default_rng(seed)
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(enc.parameters()))
    enc.update(tree_unflatten([(k, mx.array((rng.standard_normal(v.shape) * 0.1).astype(np.float32))) for k, v in flat.items()]))
    return enc


IDS = mx.array(np.random.default_rng(5).integers(0, 100, SEQ).astype(np.int32))


def test_text_only_path_is_unchanged():
    enc = _seeded_encoder()
    a = np.array(enc(IDS))
    b = np.array(enc(IDS, vision=None))
    assert np.array_equal(a, b)


def test_vision_rows_change_the_output():
    enc = _seeded_encoder()
    text = np.array(enc(IDS))
    with_vision = np.array(enc(IDS, vision=_vision()))
    assert np.abs(text - with_vision).max() > 1e-2  # null case: the vision input must matter
    other = np.array(enc(IDS, vision=_vision(seed=9)))
    assert np.abs(other - with_vision).max() > 1e-2


def test_matches_comfyui_llama_with_deepstack_and_mrope():
    import torch

    _, _, llama, ops = load_real_comfy_text_encoders()
    cfg = llama.Qwen3VL_32BConfig(
        vocab_size=100, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=HEAD_DIM,
    )
    cfg.rope_dims = list(ROPE_DIMS)
    ref = llama.Llama2_(cfg, device="cpu", dtype=torch.float32, ops=ops.disable_weight_init)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in ref.parameters():  # disable_weight_init leaves weights uninitialized
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)

    vision = _vision()
    ids = torch.from_numpy(np.array(IDS)).long()[None]
    embeds = ref.embed_tokens(ids).clone()
    rows = torch.from_numpy(np.array(vision.rows))
    mask = torch.zeros((1, SEQ), dtype=torch.bool)
    mask[0, torch.from_numpy(vision.row_indices)] = True
    embeds[0, torch.from_numpy(vision.row_indices)] = rows
    deepstack = [torch.from_numpy(np.array(d)) for d in vision.deepstack]
    with torch.no_grad():
        out = ref(ids, embeds=embeds, position_ids=torch.from_numpy(vision.position_ids),
                  visual_pos_masks=mask, deepstack_embeds=deepstack)
    ref_hidden = (out[0] if isinstance(out, tuple) else out)[0].numpy()

    with mx.stream(mx.cpu):
        enc = enc_mod.Qwen3TextEncoder(_cfg())
        enc.update(tree_unflatten([("model." + k, mx.array(v.numpy())) for k, v in ref.state_dict().items()]))
        got = enc(IDS, vision=vision)
        mx.eval(got)
    assert got.shape == ref_hidden.shape
    assert np.abs(np.array(got) - ref_hidden).max() < 1e-3


def test_gpu_stream_agrees_with_cpu_stream():
    enc = _seeded_encoder()
    vision = _vision()
    gpu = np.array(enc(IDS, vision=vision))
    with mx.stream(mx.cpu):
        cpu = np.array(enc(IDS, vision=vision))
    bound = 1e-2 * np.abs(cpu).max()
    assert np.abs(gpu - cpu).max() < bound
    other = _seeded_encoder(seed=7)
    with mx.stream(mx.cpu):
        far = np.array(other(IDS, vision=vision))
    assert np.abs(gpu - far).max() > 10 * bound  # null case: the bound can fail
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_text_encoder_vision.py -q`
Expected: FAIL (`AttributeError: ... VisionInputs`).

- [ ] **Step 3: Implémenter dans `text_encoder.py`**

Remplacer le bloc d'imports du fichier par :

```python
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import RMSNorm
from .rope import rms_norm_rope_split_half
from .text_encoder_config import Qwen3TextEncoderConfig
from .text_encoder_rope import qwen3_rope_cos_sin
from .vision_rope import interleaved_mrope_cos_sin
```

Ajouter, juste avant `class _Qwen3Backbone` :

```python
@dataclass(frozen=True)
class VisionInputs:
    """Vision-grounded conditioning for one prompt (fl2va / ref2va): rows that
    replace text embeddings, DeepStack features added after the first layers,
    and the M-RoPE position ids. See `vision_conditioning.py`."""

    rows: mx.array                  # [Nv, hidden], merged vision embeddings
    row_indices: np.ndarray         # [Nv] int, positions in the unfolded sequence
    deepstack: list[mx.array]       # each [Nv, hidden]; layer i gets deepstack[i] for i < len
    position_ids: np.ndarray        # [3, S] M-RoPE ids (float32)
    rope_dims: tuple[int, int, int] = (24, 20, 20)
```

Remplacer `_Qwen3Backbone.__call__` par :

```python
    def __call__(self, input_ids: mx.array, vision: VisionInputs | None = None) -> mx.array:
        x = self.embed_tokens(input_ids)
        if vision is None:
            cos, sin = qwen3_rope_cos_sin(x.shape[0], self.config.head_dim, self.config.rope_theta)
        else:
            idx = mx.array(np.asarray(vision.row_indices, dtype=np.int32))
            x[idx] = vision.rows.astype(x.dtype)
            cos, sin = interleaved_mrope_cos_sin(
                vision.position_ids, self.config.head_dim, self.config.rope_theta, vision.rope_dims
            )
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            if vision is not None and i < len(vision.deepstack):
                x = x.at[idx].add(vision.deepstack[i].astype(x.dtype))
        return x
```

Remplacer `Qwen3TextEncoder.__call__` par :

```python
    def __call__(self, input_ids: mx.array, vision: VisionInputs | None = None) -> mx.array:
        """`input_ids`: `[S]` int32 token ids (batch already squeezed, this
        port's convention throughout); with `vision`, the ids at
        `vision.row_indices` are placeholders that get replaced. Returns
        `[S, hidden_size]`."""
        return self.model(input_ids, vision)
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/native/minimax_h3/test_text_encoder_vision.py tests/native/minimax_h3/test_text_encoder.py tests/native/minimax_h3/test_text_encoder_weight_map.py -q -rs`
Expected: PASS (les tests existants de l'encodeur passent sans modification). Le test de parité construit `Qwen3VL_32BConfig` : si la dataclass n'accepte pas un de ces champs ou si la valeur de retour de `Llama2_` diffère (tuple ou tenseur), adapter la construction/lecture de la référence en lisant `comfy/text_encoders/llama.py`, sans changer la tolérance ni la logique testée. Si la parité échoue, suspects dans l'ordre : (1) l'ordre embeddings-puis-remplacement (les lignes vision remplacent, ne s'ajoutent pas), (2) l'index `i < len(deepstack)` (couches 0 à 2), (3) le découpage des sections dans `interleaved_mrope_cos_sin`.

- [ ] **Step 5: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/text_encoder.py tests/native/minimax_h3/test_text_encoder_vision.py
git commit -m "feat: let the MiniMax H3 text encoder take vision rows, DeepStack and M-RoPE"
```

---

### Task 4: Déplier la présentation et encoder avec vision

**Files:**
- Create: `apple_silicon_nodes/native/minimax_h3/vision_conditioning.py`
- Test: `tests/native/minimax_h3/test_vision_conditioning.py`

**Interfaces:**
- Consumes: `preprocess_image`, `preprocess_video_block` (tâche 1) ; `mrope_position_ids` (tâche 2) ; `VisionInputs`, `Qwen3TextEncoder` (tâche 3) ; `VisionTower.__call__(patches, grids) -> (merged, deepstack)` (brique 1).
- Produces: `PLACEHOLDER_ID = 151643` ; `token_tags(seq_len: int, infos: list[dict]) -> np.ndarray` (int64, 1 pour le texte, 0 pour tout bloc vision, marqueurs `<|vision_start|>`/`<|vision_end|>` compris) ; `build_vision_conditioning(entries, tower, *, rope_dims=(24, 20, 20)) -> tuple[np.ndarray, VisionInputs | None, np.ndarray]` = `(input_ids int32 [S], vision, tags)` ; `encode_with_vision(encoder, tower, entries, *, rope_dims=(24, 20, 20)) -> tuple[mx.array, np.ndarray]` = `(hidden [S, hidden], tags)` (`rope_dims` doit sommer à `head_dim // 2` de l'encodeur ; les tests à `head_dim=8` passent `(2, 1, 1)`). `entries` est la liste renvoyée par `MiniMaxH3Tokenizer.tokenize_with_weights(...)["qwen3vl_32b"][0]` : éléments `(id | dict, poids)` ; un dict vision porte `"data"` (tenseur ou tableau `[1|2, H, W, C]`) et éventuellement `"minimax_video_block": True`.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/native/minimax_h3/test_vision_conditioning.py` :

```python
"""Unfolding the MiniMax H3 presentation into ids + vision rows, token tags, and the
end-to-end encode (tiny random-weight tower + encoder; real weights behind ASDX_FULL_GGUF_TEST=1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_text_encoders
from support.minimax_h3_module_loader import load_native_module

cond = load_native_module("minimax_h3.vision_conditioning")
enc_mod = load_native_module("minimax_h3.text_encoder")
config_mod = load_native_module("minimax_h3.text_encoder_config")
tower_mod = load_native_module("minimax_h3.vision_tower")

VISION_START, VISION_END = 151652, 151653
TINY_TOWER = dict(
    hidden_size=64, intermediate_size=128, depth=4, num_heads=4, patch_size=16,
    temporal_patch_size=2, in_channels=3, spatial_merge_size=2,
    num_position_embeddings=16, deepstack_visual_indexes=(0, 1, 2), out_hidden_size=32,
)


def _seed(module, seed):
    rng = np.random.default_rng(seed)
    flat = dict(tree_flatten(module.parameters()))
    module.update(tree_unflatten([(k, mx.array((rng.standard_normal(v.shape) * 0.1).astype(np.float32))) for k, v in flat.items()]))
    return module


def _tower(seed=0):
    return _seed(tower_mod.VisionTower(tower_mod.VisionConfig(**TINY_TOWER)), seed)


def _encoder(seed=1):
    cfg = config_mod.Qwen3TextEncoderConfig(
        vocab_size=200000, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, dtype="float32",
    )
    return _seed(enc_mod.Qwen3TextEncoder(cfg), seed)


def _image(seed, h=64, w=96):
    return np.random.default_rng(seed).random((1, h, w, 3), dtype=np.float32)


def _entries(image=None, video=None):
    e = [(9707, 1.0), (11, 1.0)]
    if image is not None:
        e += [(VISION_START, 1.0), ({"type": "image", "data": image, "original_type": "image"}, 1.0), (VISION_END, 1.0)]
    if video is not None:
        e += [(VISION_START, 1.0),
              ({"type": "image", "data": video, "original_type": "image", "minimax_video_block": True}, 1.0),
              (VISION_END, 1.0)]
    return e + [(1879, 1.0), (0, 1.0)]


def test_text_only_entries_give_no_vision():
    ids, vision, tags = cond.build_vision_conditioning(_entries(), _tower())
    assert vision is None
    assert ids.tolist() == [9707, 11, 1879, 0] and tags.tolist() == [1, 1, 1, 1]


def test_image_is_unfolded_between_the_markers():
    ids, vision, tags = cond.build_vision_conditioning(_entries(image=_image(0)), _tower())
    size = (64 // 16) * (96 // 16) // 4  # merged rows: gh*gw/4
    assert ids.shape[0] == 2 + 1 + size + 1 + 2
    assert ids[2] == VISION_START and ids[2 + 1 + size] == VISION_END
    assert vision.row_indices.tolist() == list(range(3, 3 + size))
    assert vision.rows.shape == (size, 32) and len(vision.deepstack) == 3
    assert vision.position_ids.shape == (3, ids.shape[0])
    # whole block incl. the flanking markers carries the video tag 0
    assert tags.tolist() == [1, 1] + [0] * (1 + size + 1) + [1, 1]


def test_token_tags_match_comfyui():
    minimax, _, _, _ = load_real_comfy_text_encoders()
    infos = [dict(index=3, size=4, grid=(1, 4, 4)), dict(index=12, size=2, grid=(1, 2, 4)), dict(index=0, size=1, grid=(1, 2, 2))]
    ref = minimax.token_tags_from_embeds_info(
        20, [{"type": "image", "index": i["index"], "size": i["size"]} for i in infos]
    )
    assert np.array_equal(cond.token_tags(20, infos), ref.numpy())


def test_video_block_and_image_rows_are_ordered():
    ids, vision, _ = cond.build_vision_conditioning(
        _entries(image=_image(0), video=np.concatenate([_image(1, 48, 48), _image(2, 48, 48)])), _tower()
    )
    assert np.all(np.diff(vision.row_indices) > 0)
    assert vision.rows.shape[0] == len(vision.row_indices) == vision.deepstack[0].shape[0]


def test_encode_is_finite_and_vision_matters():
    tower, enc = _tower(), _encoder()
    with_img, tags = cond.encode_with_vision(enc, tower, _entries(image=_image(0)), rope_dims=(2, 1, 1))
    other, _ = cond.encode_with_vision(enc, tower, _entries(image=_image(5)), rope_dims=(2, 1, 1))
    mx.eval(with_img, other)
    assert bool(mx.all(mx.isfinite(with_img)).item())
    assert with_img.shape[0] == tags.shape[0]
    assert np.abs(np.array(with_img) - np.array(other)).max() > 1e-3  # null case: a different image must change it


def test_image_without_a_tower_is_an_error():
    with pytest.raises(ValueError, match="vision tower"):
        cond.build_vision_conditioning(_entries(image=_image(0)), None)


def test_real_tokenizer_presentation_is_unfolded():
    minimax, _, _, _ = load_real_comfy_text_encoders()
    import torch

    try:
        tokenizer = minimax.MiniMaxH3Tokenizer()
    except Exception as e:  # tokenizer files unavailable
        pytest.skip(f"MiniMaxH3Tokenizer not constructible here: {e}")
    img = torch.from_numpy(_image(0))
    entries = tokenizer.tokenize_with_weights("a cat", images=[img])["qwen3vl_32b"][0]
    ids, vision, tags = cond.build_vision_conditioning(entries, _tower())
    assert vision is not None and int(tags.sum()) < len(tags)
    assert ids[np.where(tags == 0)[0][0]] == VISION_START


_ST = Path("/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")


@pytest.mark.skipif(os.environ.get("ASDX_FULL_GGUF_TEST") != "1", reason="loads the real 26GB text encoder; set ASDX_FULL_GGUF_TEST=1")
def test_real_text_encoder_with_a_real_image():
    if not _ST.exists():
        pytest.skip("real TE checkpoint not present")
    minimax, _, _, _ = load_real_comfy_text_encoders()
    import torch

    enc_wm = load_native_module("minimax_h3.text_encoder_weight_map")
    vis_wm = load_native_module("minimax_h3.vision_weight_map")
    tokenizer = minimax.MiniMaxH3Tokenizer()
    encoder = enc_wm.load_qwen3_text_encoder_checkpoint(_ST, dtype="float16")
    tower = vis_wm.load_vision_tower(_ST)
    img = torch.from_numpy(np.random.default_rng(0).random((1, 224, 336, 3), dtype=np.float32))
    mx.reset_peak_memory()
    with_img, tags = cond.encode_with_vision(encoder, tower, tokenizer.tokenize_with_weights("a cat", images=[img])["qwen3vl_32b"][0])
    text_only, _ = cond.encode_with_vision(encoder, tower, tokenizer.tokenize_with_weights("a cat")["qwen3vl_32b"][0])
    mx.eval(with_img, text_only)
    print(f"[real] with image: {with_img.shape}, text only: {text_only.shape}, peak {mx.get_peak_memory() / 1e9:.2f} GB")
    assert bool(mx.all(mx.isfinite(with_img)).item()) and with_img.shape[1] == 5120
    assert with_img.shape[0] == tags.shape[0] > text_only.shape[0]
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_vision_conditioning.py -q`
Expected: FAIL (`ModuleNotFoundError: ... vision_conditioning`).

- [ ] **Step 3: Implémenter**

`apple_silicon_nodes/native/minimax_h3/vision_conditioning.py` :

```python
"""Unfolds MiniMax H3's tokenizer presentation into what the vision-grounded
text encoder consumes: the same construction as
`comfy/sd1_clip.py::SDClipModel.process_tokens` (each vision entry becomes
`size` merged rows spliced between its `<|vision_start|>`/`<|vision_end|>`
ids) plus `comfy/text_encoders/minimax.py::token_tags_from_embeds_info`.
"""

from __future__ import annotations

import numbers

import mlx.core as mx
import numpy as np

from .text_encoder import Qwen3TextEncoder, VisionInputs
from .vision_preprocess import preprocess_image, preprocess_video_block
from .vision_rope import mrope_position_ids
from .vision_tower import VisionTower

# Any id works: rows at vision positions are replaced by the tower's output.
PLACEHOLDER_ID = 151643


def token_tags(seq_len: int, infos: list[dict]) -> np.ndarray:
    """adaLN token tags: 1 for text, 0 (video modality) for every vision block,
    including the flanking `<|vision_start|>`/`<|vision_end|>` tokens."""
    tags = np.ones(seq_len, dtype=np.int64)
    for info in infos:
        tags[max(0, info["index"] - 1): info["index"] + info["size"] + 1] = 0
    return tags


def _to_numpy(data) -> np.ndarray:
    if hasattr(data, "detach"):  # torch tensor (ComfyUI IMAGE)
        data = data.detach().cpu().float().numpy()
    return np.asarray(data, dtype=np.float32)


def build_vision_conditioning(
    entries: list, tower: VisionTower | None, *, rope_dims: tuple[int, int, int] = (24, 20, 20)
) -> tuple[np.ndarray, VisionInputs | None, np.ndarray]:
    """`entries`: `MiniMaxH3Tokenizer.tokenize_with_weights(...)["qwen3vl_32b"][0]`.
    Returns `(input_ids int32 [S], vision or None, token_tags [S])`."""
    ids: list[int] = []
    rows: list[mx.array] = []
    deepstack_parts: list[list[mx.array]] = []
    infos: list[dict] = []
    for item, _weight in entries:
        if isinstance(item, numbers.Integral):
            ids.append(int(item))
            continue
        if tower is None:
            raise ValueError("ASDX: the prompt contains images but no vision tower is loaded (enable load_vision on the text encoder loader)")
        data = _to_numpy(item["data"])
        if item.get("minimax_video_block", False):
            patches, grid = preprocess_video_block(data)
        else:
            patches, grid = preprocess_image(data)
        merged, deepstack = tower(mx.array(patches), [grid])
        mx.eval(merged, *deepstack)
        size = merged.shape[0]
        infos.append({"type": "image", "index": len(ids), "size": size, "grid": grid})
        ids.extend([PLACEHOLDER_ID] * size)
        rows.append(merged)
        deepstack_parts.append(deepstack)

    input_ids = np.asarray(ids, dtype=np.int32)
    tags = token_tags(len(ids), infos)
    if not infos:
        return input_ids, None, tags
    vision = VisionInputs(
        rows=mx.concatenate(rows, axis=0),
        row_indices=np.concatenate([np.arange(i["index"], i["index"] + i["size"]) for i in infos]),
        deepstack=[mx.concatenate([part[k] for part in deepstack_parts], axis=0) for k in range(len(deepstack_parts[0]))],
        position_ids=mrope_position_ids(infos, len(ids)),
        rope_dims=rope_dims,
    )
    return input_ids, vision, tags


def encode_with_vision(
    encoder: Qwen3TextEncoder,
    tower: VisionTower | None,
    entries: list,
    *,
    rope_dims: tuple[int, int, int] = (24, 20, 20),
) -> tuple[mx.array, np.ndarray]:
    """Run the presentation through the encoder. Returns `(hidden [S, hidden], token_tags [S])`."""
    input_ids, vision, tags = build_vision_conditioning(entries, tower, rope_dims=rope_dims)
    hidden = encoder(mx.array(input_ids), vision=vision)
    mx.eval(hidden)
    return hidden, tags
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/native/minimax_h3/test_vision_conditioning.py -q -rs`
Expected: PASS, un seul ignoré (le test réel sans drapeau). Si `test_real_tokenizer_presentation_is_unfolded` est ignoré pour tokenizer indisponible, le dire.

- [ ] **Step 5: Vérifier sur les vrais fichiers**

Run: `ASDX_FULL_GGUF_TEST=1 uv run pytest tests/native/minimax_h3/test_vision_conditioning.py -q -rs -s -k real_text_encoder`
Expected: PASS. Relever la ligne `[real] ...` (formes et pic mémoire) pour le compte rendu. Si le pic dépasse ~25 Go, ne pas conclure : rapporter le chiffre.

- [ ] **Step 6: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/vision_conditioning.py tests/native/minimax_h3/test_vision_conditioning.py
git commit -m "feat: unfold the MiniMax H3 presentation into vision rows and encode with the vision tower"
```

---

### Task 5: Option `load_vision` du loader et helper de prompt

**Files:**
- Modify: `apple_silicon_nodes/minimax_h3_nodes.py` (`ASDX_MiniMaxH3TextEncoderLoader.define_schema/execute` ; nouvelle fonction module `encode_minimax_h3_prompt`)
- Test: `tests/test_minimax_h3_loaders.py` (ajouts)

**Interfaces:**
- Consumes: `load_vision_tower(path)` (brique 1) ; `encode_with_vision(encoder, tower, entries) -> (hidden, tags)` (tâche 4).
- Produces: le dict du loader gagne `"vision_tower"` (`VisionTower` ou `None`) ; le loader accepte `load_vision: bool = False` (entrée `io.Boolean`, clé de cache incluant le drapeau) ; `encode_minimax_h3_prompt(text_encoder: dict, prompt: str, *, images: list | None = None, ref_items: list | None = None) -> dict` renvoie `{"type": "minimax_h3", "hidden_states": mx.array, "token_tags": mx.array (int), "text": prompt}` (consommé par les nœuds de la brique 4) ; lève `RuntimeError` si l'entrée n'est pas la sortie du loader ou si l'embedding est non fini. Le nœud `ASDX_MiniMaxH3TextEncode` existant n'est PAS modifié.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à `tests/test_minimax_h3_loaders.py` (utiliser `_fake_folder_paths`, `Mock`, `types`, `sys`, `pytest` déjà importés ; `nodes_module` est déjà défini en tête) :

```python
def test_text_encoder_loader_loads_vision_tower_only_on_request(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "folder_paths", _fake_folder_paths({"qwen.safetensors": "/models/text_encoders/qwen.safetensors"})
    )
    nodes_module._TEXT_ENCODER_CACHE.clear()
    encoder, tower = Mock(), Mock()
    tower_calls = []

    enc_stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.text_encoder_weight_map")
    enc_stub.load_qwen3_text_encoder_checkpoint = lambda path, dtype: encoder
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.text_encoder_weight_map", enc_stub)
    vis_stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.vision_weight_map")
    vis_stub.load_vision_tower = lambda path: tower_calls.append(str(path)) or tower
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.vision_weight_map", vis_stub)

    plain = ASDX_MiniMaxH3TextEncoderLoader.execute("qwen.safetensors").values[0]
    assert plain["vision_tower"] is None and tower_calls == []

    with_vision = ASDX_MiniMaxH3TextEncoderLoader.execute("qwen.safetensors", load_vision=True).values[0]
    assert with_vision["vision_tower"] is tower and tower_calls == ["/models/text_encoders/qwen.safetensors"]
    assert with_vision is not plain  # the flag is part of the cache key


def test_encode_prompt_helper_rejects_wrong_input_type():
    with pytest.raises(RuntimeError, match="ASDX_MiniMaxH3TextEncoderLoader"):
        nodes_module.encode_minimax_h3_prompt({"type": "something_else"}, "a prompt")


def test_encode_prompt_helper_returns_hidden_states_and_tags(monkeypatch):
    import mlx.core as mx
    import numpy as np

    _install_fake_minimax_tokenizer(monkeypatch, [11, 22, 33])
    hidden = mx.ones((3, 8))
    stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.vision_conditioning")
    stub.encode_with_vision = lambda enc, tower, entries: (hidden, np.array([1, 1, 1]))
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.vision_conditioning", stub)

    desc = {"type": "asdx_minimax_h3_text_encoder", "encoder": Mock(), "vision_tower": None}
    out = nodes_module.encode_minimax_h3_prompt(desc, "a prompt")
    assert out["type"] == "minimax_h3" and out["text"] == "a prompt"
    assert out["hidden_states"] is hidden and out["token_tags"].tolist() == [1, 1, 1]


def test_encode_prompt_helper_aborts_on_non_finite(monkeypatch):
    import mlx.core as mx
    import numpy as np

    _install_fake_minimax_tokenizer(monkeypatch, [11])
    stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.vision_conditioning")
    stub.encode_with_vision = lambda enc, tower, entries: (mx.array([[float("nan")]]), np.array([1]))
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.vision_conditioning", stub)
    desc = {"type": "asdx_minimax_h3_text_encoder", "encoder": Mock(), "vision_tower": None}
    with pytest.raises(RuntimeError, match="non-finite"):
        nodes_module.encode_minimax_h3_prompt(desc, "x")
```

Note : `_install_fake_minimax_tokenizer` existe déjà dans ce fichier ; son `FakeTokenizer.tokenize_with_weights(self, text)` n'accepte pas `images=`/`minimax_ref_items=`. Ajouter les paramètres au faux tokenizer : `def tokenize_with_weights(self, text, images=(), minimax_ref_items=None):` (modification minimale du helper de test, sans changer son résultat).

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_minimax_h3_loaders.py -q`
Expected: FAIL (`load_vision` inconnu, `encode_minimax_h3_prompt` absent).

- [ ] **Step 3: Implémenter**

Dans `ASDX_MiniMaxH3TextEncoderLoader.define_schema`, ajouter à `inputs` après `precision` :

```python
                io.Boolean.Input("load_vision", default=False,
                                 tooltip="Also load the Qwen3-VL vision tower (needed for image/video references; ~2.4GB more)."),
```

Remplacer la signature et le corps de `execute` du loader par :

```python
    @classmethod
    def execute(cls, encoder_name: str, precision: str = "float16", load_vision: bool = False) -> io.NodeOutput:
        import folder_paths

        found = folder_paths.get_full_path("text_encoders", encoder_name)
        if not found:
            raise RuntimeError(f"ASDX MiniMax H3 Text Encoder Loader: could not find '{encoder_name}'.")
        path = Path(found)

        cache_key = f"{path}:{precision}:vision={int(load_vision)}"
        if cache_key in _TEXT_ENCODER_CACHE:
            print(f"[ASDX] MiniMax H3 text encoder cache hit: {encoder_name}")
            return io.NodeOutput(_TEXT_ENCODER_CACHE[cache_key])

        from .native.minimax_h3.text_encoder_weight_map import load_qwen3_text_encoder_checkpoint

        estimate = _gate_minimax_h3_component("text_encoder", path, precision, _DIT_CACHE)

        _TEXT_ENCODER_CACHE.clear()
        encoder = load_qwen3_text_encoder_checkpoint(path, dtype=precision)
        tower = None
        if load_vision:
            from .native.minimax_h3.vision_weight_map import load_vision_tower

            tower = load_vision_tower(path)  # separate open of the same file: TensorSource.get() pops tensors
        # measured peak of the tower load alone: 3.6GB (brick 1)
        peak = (estimate.predicted_peak_bytes if estimate else 0) + (int(3.6e9) if load_vision else 0)
        result = {
            "type": "asdx_minimax_h3_text_encoder",
            "name": encoder_name,
            "path": str(path),
            "encoder": encoder,
            "vision_tower": tower,
            "precision": precision,
            "_predicted_peak_bytes": peak,
        }
        _TEXT_ENCODER_CACHE[cache_key] = result
        return io.NodeOutput(result)
```

Ajouter au niveau module, juste avant `class ASDX_MiniMaxH3TextEncode` :

```python
def encode_minimax_h3_prompt(
    text_encoder: dict, prompt: str, *, images: list | None = None, ref_items: list | None = None
) -> dict:
    """Tokenize (ComfyUI's weight-free `MiniMaxH3Tokenizer`), unfold vision
    blocks, and encode with the native encoder (+ vision tower when the
    prompt has images). `images`: fl2va keyframes (ComfyUI IMAGE tensors);
    `ref_items`: ref2va items in request order (see
    `comfy_extras/nodes_minimax_h3.py::MiniMaxH3ReferenceToVideo`). Returns the
    conditioning dict the sampler consumes, plus `token_tags`."""
    if not isinstance(text_encoder, dict) or text_encoder.get("type") != "asdx_minimax_h3_text_encoder":
        raise RuntimeError("ASDX MiniMax H3 Text Encode: expected the output of ASDX_MiniMaxH3TextEncoderLoader.")

    import comfy.text_encoders.minimax

    from .native.minimax_h3.vision_conditioning import encode_with_vision

    tokenizer = comfy.text_encoders.minimax.MiniMaxH3Tokenizer()
    entries = tokenizer.tokenize_with_weights(
        prompt, images=images or [], minimax_ref_items=ref_items
    )["qwen3vl_32b"][0]
    hidden_states, tags = encode_with_vision(text_encoder["encoder"], text_encoder.get("vision_tower"), entries)
    if not bool(mx.all(mx.isfinite(hidden_states)).item()):
        raise RuntimeError(
            "ASDX MiniMax H3 Text Encode: produced a non-finite (NaN/Inf) "
            "embedding -- aborting before the expensive sampling pass."
        )
    print(f"[ASDX] MiniMax H3 Text Encode: {len(prompt)} chars, {hidden_states.shape[0]} rows")
    return {
        "type": "minimax_h3",
        "hidden_states": hidden_states,
        "token_tags": mx.array(tags),
        "text": prompt,
    }
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_minimax_h3_loaders.py -q`
Expected: PASS (les anciens tests du fichier passent : le nœud `TextEncode` et l'appel `load_calls == [(path, dtype)]` du loader sont inchangés).

- [ ] **Step 5: Suite complète**

Run: `uv run pytest tests -q`
Expected: tout passe sauf `tests/support/test_krea2_module_loader.py::test_plain_import_of_model_fails_outside_comfyui` (préexistant). Rapporter le résumé.

- [ ] **Step 6: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/minimax_h3_nodes.py tests/test_minimax_h3_loaders.py
git commit -m "feat: optional vision tower on the H3 text encoder loader and a vision-aware prompt encode helper"
```

---

## Auto-revue du plan

- **Couverture de la spec, brique 2** : tokenisation/présentation (tâche 4, via le tokenizer ComfyUI, parité des tags), encodeur avec insertion des lignes vision, DeepStack et M-RoPE (tâches 2-3, parité ComfyUI), `token_tags` (tâche 4), chargement optionnel (tâche 5), et les 5 « Notes de passage » : bloc vidéo (T1), entrée 4-D (T1), cast float32 -> dtype de l'encodeur (T3 : `.astype(x.dtype)`), source non partageable (T5 : ouverture séparée), `t>1` déjà vérifié (brique 1).
- **Placeholders** : aucun ; l'incertitude connue (champs de `Qwen3VL_32BConfig`, valeur de retour de `Llama2_`) est nommée à la tâche 3 étape 4 avec la conduite à tenir.
- **Cohérence des types** : `VisionInputs(rows, row_indices, deepstack, position_ids, rope_dims)`, `mrope_position_ids(infos, seq_len)`, `interleaved_mrope_cos_sin(position_ids, head_dim, theta, rope_dims)`, `build_vision_conditioning(entries, tower)` et `encode_with_vision(encoder, tower, entries)` sont utilisés à l'identique dans les tâches 3, 4 et 5.
- **Risque connu** : la parité de l'encodeur (tâche 3) dépend de la construction d'une référence ComfyUI minuscule ; les seuils restent 1e-3 sur CPU et 1e-2 relatif pour GPU contre CPU, à ne pas assouplir sans avoir vérifié les cas nuls.
- **Hors périmètre de ce plan** : les nœuds i2v/ref (brique 4), les lignes de condition du DiT (brique 3).
