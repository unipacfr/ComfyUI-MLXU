# MiniMax H3 brique 4 : nœuds i2v et ref (keyframes, références images, vidéos, audios)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deux nœuds ComfyUI, `ASDX_MiniMaxH3ImageToVideo` (t2v + première/dernière image) et `ASDX_MiniMaxH3ReferenceToVideo` (images, vidéos, audios), qui produisent le conditioning complet (`hidden_states`, `token_tags`, `keyframes`, `refs`) et les latents vidéo et audio vides que consomme `ASDX_MiniMaxH3Sampler`, en suivant fidèlement `comfy_extras/nodes_minimax_h3.py`.

**Architecture:** Un module `apple_silicon_nodes/minimax_h3_conditioning.py` (torch autorisé, comfy importé au besoin) porte la géométrie (canevas, tailles de références, échantillonnage vidéo), l'encodage VAE par le pont `comfy.sd.VAE` (`vae.encode`, latents déjà normalisés par le VAE, moyenne du posterior), la construction des `KeyframeCond`/`RefBlock` et l'estimation des lignes. Deux constructeurs de haut niveau (`build_i2v_conditioning`, `build_reference_conditioning`) appellent `encode_minimax_h3_prompt` (brique 2). Les classes de nœuds restent minces.

**Tech Stack:** torch (CPU, images et VAE), MLX (conversion des latents), `comfy_api.latest` (`io`, `io.Autogrow`), pytest, `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-19-minimax-h3-i2v-ref-design.md` (brique 4 et « Notes de passage vers la brique 4 »). Références : `comfy_extras/nodes_minimax_h3.py` (lignes 29-365), `comfy/ldm/minimax/vae.py::encode`, `audio_vae.py::encode`, `comfy/sd.py::VAE.encode` dans `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI`.

## Global Constraints

- Toujours `uv run pytest` / `uv run python` (jamais `python` nu). Python 3.10+, annotations complètes, docstrings en anglais, pas d'emoji ni de tiret cadratin (les noms d'affichage des nœuds gardent le préfixe existant du projet).
- Fail-closed : plafonds 9 images, 3 vidéos, 3 audios ; vidéo de référence < 5 images refusée ; forme de latent inattendue renvoyée par un VAE refusée avec un message `ASDX` ; jamais de réordonnancement des références.
- **Comportement ComfyUI, pas Rust** : `vae.encode` de ComfyUI renvoie les latents normalisés (moyenne du posterior, sans échantillonnage, sans aller-retour fp16). Le Rust échantillonne le posterior à graine fixe 42 et fait un aller-retour fp16 : écart connu et accepté ici (moyenne = échantillon sans bruit), à consigner dans le compte rendu.
- Contrat du DiT (brique 3) : ne créer un `RefBlock` que lorsque le VAE est présent ; `latent_h`/`latent_w` en unités latentes (pixels / 16) ; latents de keyframe sur la grille spatiale de la cible ; `resolved_frame_index` en image pixel sur la grille 17k+5 (dernière image = `frame_count - 1`). Le layout lève `ValueError` si les déclarations contredisent les latents : ne jamais contourner.
- Sans VAE, les références ne conditionnent que le texte (comportement ComfyUI) : le message est explicite et aucun `RefBlock` n'est créé.
- Le `token_tags` et les `hidden_states` viennent de `encode_minimax_h3_prompt` (brique 2, point d'intégration unique). Les images passées au tokenizer sont des tenseurs ComfyUI `[1, H, W, 3]` en `[0, 1]` sur CPU.
- Tests de nœuds via `tests/support/comfy_stub.py` (`install_comfy_stubs`, `load_node_module`) ; tests dépendant du vrai ComfyUI via `tests/support/comfyui_reference_loader.py` (skip propre si absent) ; tests sur vrais fichiers derrière `ASDX_FULL_GGUF_TEST=1`.
- **Toute vérification est prouvée par mutation** (canon) : casser volontairement l'élément testé, montrer l'échec, restaurer, prouver `git diff` inchangé ; un survivant = un trou de test à combler.
- **Aucun commit automatique** (règle utilisateur) : « Commit » = mettre dans l'index et attendre l'ordre. Pas de ligne d'attribution.
- Mettre à jour `README.md` (règle du projet) avec la section MiniMax H3 (tâche 4).

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `apple_silicon_nodes/minimax_h3_conditioning.py` (créer) | géométrie, encodage, constructeurs, estimation des lignes |
| `apple_silicon_nodes/minimax_h3_nodes.py` (modifier) | 2 classes de nœuds + `NODE_LIST` |
| `tests/test_minimax_h3_conditioning_geometry.py` (créer) | géométrie (tâche 1) |
| `tests/test_minimax_h3_conditioning_encode.py` (créer) | encodage et constructeurs avec faux VAE (tâches 2 et 3) |
| `tests/test_minimax_h3_i2v_ref_nodes.py` (créer) | nœuds, schémas (tâche 4) |
| `tests/native/minimax_h3/test_real_pipeline_cond.py` (créer) | essai de bout en bout réel (tâche 5) |
| `README.md` (modifier) | section MiniMax H3 |

---

### Task 1: Géométrie et estimation des lignes

**Files:**
- Create: `apple_silicon_nodes/minimax_h3_conditioning.py`
- Test: `tests/test_minimax_h3_conditioning_geometry.py`

**Interfaces:**
- Consumes: rien.
- Produces (dans `minimax_h3_conditioning.py`) : constantes `CANVAS_MULTIPLE = 32`, `BASE_SHORT_EDGE = 768`, `MAX_PIXELS = 768 * 1344`, `REF_IMAGE_SHORT_EDGE = 2048`, `FPS = 24`, `MAX_REF_IMAGES = 9`, `MAX_REF_VIDEOS = 3`, `MAX_REF_AUDIOS = 3`, `ROW_WARN_THRESHOLD = 74_898` ; `adapt_canvas(width: int, height: int) -> tuple[int, int]` ; `resize_image(image: torch.Tensor, width: int, height: int, crop: str) -> torch.Tensor` (`[B, H, W, C]` -> `[B, height, width, 3]`, `crop` dans `{"disabled", "center"}`, filtre lanczos de `comfy.utils.common_upscale`) ; `ref_image_size(img_h: int, img_w: int, gen_width: int, gen_height: int, mode: str) -> tuple[int, int]` (`(tw, th)`, `mode` dans `{"match", "max"}`) ; `prepare_ref_video(frames: torch.Tensor, frame_count: int) -> tuple[torch.Tensor, int, int]` (`(frames tronquées à 17k+5, cw, ch)`) ; `qwen_video_frames(frames: torch.Tensor) -> tuple[torch.Tensor, list[float]]` (images à 2 fps et leurs horodatages) ; `estimate_packed_rows(width: int, height: int, latent_t: int, audio_t: int, keyframe_frames: int = 0, ref_image_sizes: tuple[tuple[int, int], ...] = (), ref_videos: tuple[tuple[int, int, int, int], ...] = (), ref_audio_t: int = 0) -> int` (lignes hors texte : cible, keyframes, références ; `ref_videos` = `(latent_t, cw, ch, audio_t)`).

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_minimax_h3_conditioning_geometry.py` :

```python
"""Geometry helpers of minimax_h3_conditioning, checked against the real ComfyUI
`comfy_extras.nodes_minimax_h3` values (skipped cleanly when it is not importable)
and against hand-computed cases from that source."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
geo = load_node_module("minimax_h3_conditioning")


@pytest.fixture(autouse=True)
def _resize_without_comfy(monkeypatch):
    """`resize_image` uses comfy.utils.common_upscale (real lanczos, ComfyUI's own numerics);
    the geometry checks here only need the output shape, so use torch bilinear."""
    import torch.nn.functional as F

    def fake_resize(image, width, height, crop):
        x = F.interpolate(image[..., :3].movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False)
        return x.movedim(1, -1)

    monkeypatch.setattr(geo, "resize_image", fake_resize)


_COMFYUI = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")
_SITE = _COMFYUI / ".venv" / "lib" / "python3.13" / "site-packages"


def _reference_module():
    if not _COMFYUI.exists() or not _SITE.exists():
        pytest.skip("ComfyUI install not present")
    for name in list(sys.modules):
        if name == "comfy" or name.startswith("comfy.") or name in ("nodes", "comfy_extras", "comfy_api") or name.startswith(("comfy_extras.", "comfy_api.")):
            del sys.modules[name]
    sys.path.insert(0, str(_COMFYUI))
    sys.path.insert(0, str(_SITE))
    try:
        return importlib.import_module("comfy_extras.nodes_minimax_h3")
    except Exception as e:  # heavy runtime imports may fail outside a ComfyUI process
        pytest.skip(f"comfy_extras.nodes_minimax_h3 not importable here: {e}")


@pytest.mark.parametrize("wh", [(1344, 768), (768, 1344), (1920, 1080), (640, 640), (100, 4000), (32, 32)])
def test_adapt_canvas_hand_computed_and_reference(wh):
    w, h = wh
    got = geo.adapt_canvas(w, h)
    assert got[0] % 32 == 0 and got[1] % 32 == 0 and got[0] >= 32 and got[1] >= 32
    assert got[0] * got[1] <= 768 * 1344 * 1.03  # rounding to 32 may exceed the cap slightly
    assert geo.adapt_canvas(1344, 768) == (1344, 768)
    ref = _reference_module().adapt_canvas
    assert got == ref(w, h)


@pytest.mark.parametrize("mode", ["match", "max"])
@pytest.mark.parametrize("hw", [(768, 1344), (4000, 3000), (256, 256), (1080, 1920)])
def test_ref_image_size(mode, hw):
    h, w = hw
    tw, th = geo.ref_image_size(h, w, 1344, 768, mode)
    assert tw % 32 == 0 and th % 32 == 0 and tw >= 32 and th >= 32
    if mode == "match" and hw == (768, 1344):
        assert (tw, th) == (1344, 768)  # already the generation area: unchanged
    if hw == (256, 256):
        assert (tw, th) == (256, 256)  # never upscaled, in either mode
    if mode == "max" and hw == (4000, 3000):
        assert min(tw, th) <= 2048 + 32  # short edge capped near 2048


def test_ref_image_size_match_scales_down_to_generation_area():
    tw, th = geo.ref_image_size(3000, 4000, 1344, 768, "match")
    assert tw * th <= 1344 * 768 * 1.06 and tw > th


def test_ref_image_size_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        geo.ref_image_size(100, 100, 1344, 768, "huge")


def test_prepare_ref_video_grid_and_limits():
    frames = torch.rand(30, 64, 96, 3)
    out, cw, ch = geo.prepare_ref_video(frames, frame_count=124)
    assert out.shape[0] == 22 and out.shape[0] % 17 == 5  # 30 -> 22 (17k+5)
    assert out.shape[1:] == (ch, cw, 3) and cw % 32 == 0 and ch % 32 == 0
    short, _, _ = geo.prepare_ref_video(torch.rand(200, 64, 96, 3), frame_count=39)
    assert short.shape[0] == 39  # truncated to the target's frame count, already on the grid
    with pytest.raises(ValueError, match="at least 5"):
        geo.prepare_ref_video(torch.rand(4, 64, 96, 3), frame_count=124)


def test_qwen_video_frames_two_fps():
    frames = torch.arange(39, dtype=torch.float32).view(39, 1, 1, 1).expand(39, 4, 4, 3)
    sampled, stamps = geo.qwen_video_frames(frames)
    assert [int(v) for v in sampled[:, 0, 0, 0].tolist()] == [0, 12, 24, 36]
    assert stamps == [0.0, 0.5, 1.0, 1.5]


def test_estimate_packed_rows_matches_layout_arithmetic():
    # 1344x768: 42*24 = 1008 rows per latent frame; latent_t 37 (124 frames); audio 2*T
    base = geo.estimate_packed_rows(1344, 768, 37, 207)
    assert base == 37 * 1008 + 2 * 207
    with_kf = geo.estimate_packed_rows(1344, 768, 37, 207, keyframe_frames=1)
    assert with_kf == base + 1008
    with_ref = geo.estimate_packed_rows(1344, 768, 37, 207, ref_image_sizes=((1344, 768),))
    assert with_ref == base + 1008
    with_video = geo.estimate_packed_rows(1344, 768, 37, 207, ref_videos=((2, 640, 352, 3),))
    assert with_video == base + 2 * (20 * 11) + 2 * 3
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_minimax_h3_conditioning_geometry.py -q`
Expected: FAIL (module `minimax_h3_conditioning` absent).

- [ ] **Step 3: Implémenter**

`apple_silicon_nodes/minimax_h3_conditioning.py` :

```python
"""Conditioning builders for the MiniMax H3 i2v (fl2va) and ref2va nodes.

Ports the geometry and reference-preparation logic of ComfyUI's
`comfy_extras/nodes_minimax_h3.py` (the runtime this project mirrors):
canvas adaptation, reference sizing, video-reference preparation, and the
encoding of keyframes/references through the ComfyUI `VAE` bridge (whose
`encode` already returns normalized latents, posterior mean).
"""

from __future__ import annotations

import math

import torch

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
# The fused SwiGLU projection [1, S, 2*14336] crosses i32::MAX elements above this many rows
# (mlx-gen-minimax-h3 cost.rs); matmul was measured exact on both sides there, so this only warns.
ROW_WARN_THRESHOLD = 74_898


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    """768-short-edge canvas with a 768*1344 area cap, per-axis round to 32
    (`nodes_minimax_h3.adapt_canvas`)."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def resize_image(image: torch.Tensor, width: int, height: int, crop: str) -> torch.Tensor:
    """`[B, H, W, C]` -> `[B, height, width, 3]` with ComfyUI's lanczos
    (`crop`: "disabled" = plain stretch, "center" = aspect-preserving cover)."""
    import comfy.utils

    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _round32(value: float) -> int:
    return max(CANVAS_MULTIPLE, round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def ref_image_size(img_h: int, img_w: int, gen_width: int, gen_height: int, mode: str) -> tuple[int, int]:
    """Target `(tw, th)` of a reference image: "match" scales it (down only,
    keeping the aspect) to the generation's pixel area; "max" uses the
    reference pipeline's 2048 short edge. Multiples of 32."""
    if mode == "match":
        scale = min(1.0, math.sqrt((gen_width * gen_height) / (img_w * img_h)))
    elif mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(img_w, img_h))
    else:
        raise ValueError(f"ASDX MiniMax H3: ref_image_size mode must be 'match' or 'max', got {mode!r}")
    return _round32(img_w * scale), _round32(img_h * scale)


def prepare_ref_video(frames: torch.Tensor, frame_count: int) -> tuple[torch.Tensor, int, int]:
    """Resize a reference video onto its adapted canvas, cap it at the target's
    frame count and crop it to the model's 17k+5 clip grid. Returns
    `(frames, canvas_w, canvas_h)`."""
    vh, vw = frames.shape[1], frames.shape[2]
    cw, ch = adapt_canvas(vw, vh)
    if vw * vh < cw * ch:  # never upscale a small reference
        cw, ch = _round32(vw), _round32(vh)
    frames = resize_image(frames, cw, ch, "disabled")
    if frames.shape[0] > frame_count:
        frames = frames[:frame_count]
    n = frames.shape[0]
    if n < 5:
        raise ValueError("ASDX MiniMax H3: reference videos need at least 5 frames (~0.2s at 24 fps)")
    while n % 17 != 5:
        n -= 1
    return frames[:n], cw, ch


def qwen_video_frames(frames: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    """The text encoder sees a reference video at 2 fps with timestamps."""
    idx = list(range(0, frames.shape[0], FPS // 2))
    return frames[idx], [i / 2.0 for i in range(len(idx))]


def estimate_packed_rows(
    width: int,
    height: int,
    latent_t: int,
    audio_t: int,
    keyframe_frames: int = 0,
    ref_image_sizes: tuple[tuple[int, int], ...] = (),
    ref_videos: tuple[tuple[int, int, int, int], ...] = (),
    ref_audio_t: int = 0,
) -> int:
    """Rows of the packed DiT sequence EXCLUDING the text rows: target video and
    audio, keyframe rows (one latent frame each), reference images `(tw, th)`,
    reference videos `(latent_t, cw, ch, audio_t)` and standalone reference audio."""
    def frame_rows(w: int, h: int) -> int:
        return (h // 16 // 2) * (w // 16 // 2)

    rows = latent_t * frame_rows(width, height) + 2 * audio_t
    rows += keyframe_frames * frame_rows(width, height)
    rows += sum(frame_rows(tw, th) for tw, th in ref_image_sizes)
    rows += sum(lt * frame_rows(cw, ch) + 2 * at for lt, cw, ch, at in ref_videos)
    rows += 2 * ref_audio_t
    return rows
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_minimax_h3_conditioning_geometry.py -q -rs`
Expected: PASS ; le test de parité `adapt_canvas` peut être ignoré si `comfy_extras.nodes_minimax_h3` n'est pas importable hors d'un processus ComfyUI : dire pourquoi, et vérifier alors `adapt_canvas` contre les valeurs calculées à la main dans le test (`(1344, 768) -> (1344, 768)`) et contre la source. Les tests `resize_image` (lanczos) sont couverts par la tâche 2. Preuves par mutation (montrer l'échec, restaurer) : (a) `min(1.0, ...)` -> `min(2.0, ...)` dans `ref_image_size` ; (b) retirer la boucle `while n % 17 != 5` ; (c) `range(0, n, FPS // 2)` -> `range(0, n, FPS)` ; (d) omettre `2 * at` dans `estimate_packed_rows`.

- [ ] **Step 5: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/minimax_h3_conditioning.py tests/test_minimax_h3_conditioning_geometry.py
git commit -m "feat: add MiniMax H3 canvas, reference sizing and row estimation helpers"
```

---

### Task 2: Encodage VAE des keyframes et des références

**Files:**
- Modify: `apple_silicon_nodes/minimax_h3_conditioning.py` (ajouts)
- Test: `tests/test_minimax_h3_conditioning_encode.py` (partie encodage)

**Interfaces:**
- Consumes: `resize_image`, `ref_image_size`, `prepare_ref_video`, `qwen_video_frames`, `adapt_canvas`, constantes (tâche 1) ; `KeyframeCond`, `RefBlock` (`native/minimax_h3/condition.py`).
- Produces : `to_mx(t: torch.Tensor) -> mx.array` (float32 CPU) ; `encode_video_latent(vae, frames: torch.Tensor) -> mx.array` (`vae.encode(frames)` `[T, H, W, 3]` -> latent MLX `[1, 24, t, H/16, W/16]` ; `ValueError` avec message `ASDX` si rang != 5 ou canaux != 24) ; `encode_audio_latent(audio_vae, audio: dict) -> tuple[mx.array, int]` (`audio["waveform"]` `[B, C, L]`, `audio["sample_rate"]` ; rééchantillonne à `getattr(audio_vae, "audio_sample_rate", 32000)` par `torchaudio` si besoin ; `z = audio_vae.encode(waveform[:1].movedim(1, -1))` ; renvoie `(z_mlx, z.shape[-1])` ; `ValueError` si rang != 4) ; `build_keyframes(vae, width: int, height: int, frame_count: int, first_frame, last_frame) -> tuple[list[torch.Tensor], list[KeyframeCond]]` (images pour le tokenizer, keyframes ; première image étirée `crop="disabled"` à l'index 0, dernière recadrée `crop="center"` à l'index `frame_count - 1`) ; `build_references(vae, audio_vae, width: int, height: int, frame_count: int, ref_image_size_mode: str, ref_images: dict, ref_videos: dict, ref_video_audios: dict, ref_audios: dict) -> tuple[list[dict], list[RefBlock]]` (`ref_items` dans l'ordre : images, puis pour chaque vidéo son `{"type": "audio"}` éventuel avant `{"type": "video", "data": images à 2 fps, "timestamps": [...]}`, puis audios ; `refs` dans le même ordre, uniquement si `vae` n'est pas `None`).

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_minimax_h3_conditioning_encode.py` (première partie) :

```python
"""Encoding of keyframes and references through fake VAEs (shape/contract level),
then the same paths through real lanczos resizing when ComfyUI is available."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
cond = load_node_module("minimax_h3_conditioning")


def _lat_t(frames: int) -> int:
    return max(1, (frames - 1) // 17 * 5 + 2) if frames > 1 else 1


class FakeVAE:
    """Mimics comfy.sd.VAE.encode on [T, H, W, 3] for the MiniMax H3 video VAE."""

    def __init__(self):
        self.calls = []

    def encode(self, pixels):
        self.calls.append(tuple(pixels.shape))
        t, h, w, _ = pixels.shape
        return torch.full((1, 24, _lat_t(t), h // 16, w // 16), 0.5)


class FakeAudioVAE:
    audio_sample_rate = 32000

    def __init__(self):
        self.calls = []

    def encode(self, wave):  # wave: [1, L, C]
        self.calls.append(tuple(wave.shape))
        return torch.full((1, 32, 2, wave.shape[1] // 800), 0.25)


@pytest.fixture(autouse=True)
def _resize_without_comfy(monkeypatch):
    """`resize_image` uses comfy.utils.common_upscale; use torch bilinear here (numerics of
    the lanczos are ComfyUI's own; this file checks the contracts)."""
    import torch.nn.functional as F

    def fake_resize(image, width, height, crop):
        x = image[..., :3].movedim(-1, 1)
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return x.movedim(1, -1)

    monkeypatch.setattr(cond, "resize_image", fake_resize)


def test_encode_video_latent_returns_mlx_latent():
    z = cond.encode_video_latent(FakeVAE(), torch.rand(5, 64, 96, 3))
    assert isinstance(z, mx.array) and z.shape == (1, 24, 2, 4, 6) and z.dtype == mx.float32


def test_encode_video_latent_rejects_wrong_shape():
    class Bad:
        def encode(self, pixels):
            return torch.zeros(1, 16, 2, 4, 6)

    with pytest.raises(ValueError, match="ASDX"):
        cond.encode_video_latent(Bad(), torch.rand(5, 64, 96, 3))


def test_encode_audio_latent_no_resample():
    audio = {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
    z, t = cond.encode_audio_latent(FakeAudioVAE(), audio)
    assert z.shape == (1, 32, 2, 40) and t == 40


def test_encode_audio_latent_resamples_to_vae_rate():
    pytest.importorskip("torchaudio")
    vae = FakeAudioVAE()
    cond.encode_audio_latent(vae, {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000})
    assert vae.calls[0][1] == 32000  # 1 s resampled from 48 kHz to 32 kHz


def test_keyframes_first_stretched_last_at_final_frame():
    vae = FakeVAE()
    first, last = torch.rand(1, 100, 200, 3), torch.rand(1, 300, 200, 3)
    images, kfs = cond.build_keyframes(vae, 96, 64, 124, first, last)
    assert [k.resolved_frame_index for k in kfs] == [0, 123]
    assert len(images) == 2 and images[0].shape == (1, 64, 96, 3)
    assert vae.calls == [(1, 64, 96, 3), (1, 64, 96, 3)]
    assert kfs[0].latent.shape == (1, 24, 1, 4, 6)


def test_keyframes_last_only_and_none():
    images, kfs = cond.build_keyframes(FakeVAE(), 96, 64, 124, None, torch.rand(1, 64, 96, 3))
    assert [k.resolved_frame_index for k in kfs] == [123] and len(images) == 1
    assert cond.build_keyframes(FakeVAE(), 96, 64, 124, None, None) == ([], [])


def test_references_order_labels_and_blocks():
    vae, avae = FakeVAE(), FakeAudioVAE()
    img = torch.rand(1, 64, 96, 3)
    vid = torch.rand(22, 64, 96, 3)
    snd = {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
    items, refs = cond.build_references(
        vae, avae, 96, 64, 124, "match",
        ref_images={"ref_image_1": img}, ref_videos={"ref_video_1": vid},
        ref_video_audios={"ref_video_audio_1": snd}, ref_audios={"ref_audio_1": snd},
    )
    assert [i["type"] for i in items] == ["image", "audio", "video", "audio"]  # audio label BEFORE its video
    assert [r.kind for r in refs] == ["image", "video_audio", "audio"]
    assert refs[0].latent_h == 4 and refs[0].latent_w == 6 and refs[0].latent.shape[2] == refs[0].latent_t == 1
    assert refs[1].ref_audio_t == 40 and refs[1].latent_t == refs[1].latent.shape[2]
    assert refs[2].latent is None and refs[2].ref_audio_t == 40


def test_references_without_vae_are_text_only():
    items, refs = cond.build_references(
        None, None, 96, 64, 124, "match",
        ref_images={"ref_image_1": torch.rand(1, 64, 96, 3)}, ref_videos={}, ref_video_audios={},
        ref_audios={"ref_audio_1": {"waveform": torch.zeros(1, 2, 100), "sample_rate": 32000}},
    )
    assert [i["type"] for i in items] == ["image", "audio"] and refs == []


def test_reference_caps_and_short_video():
    with pytest.raises(ValueError, match="at most 9"):
        cond.build_references(None, None, 96, 64, 124, "match",
                              ref_images={f"ref_image_{i}": torch.rand(1, 32, 32, 3) for i in range(10)},
                              ref_videos={}, ref_video_audios={}, ref_audios={})
    with pytest.raises(ValueError, match="at least 5"):
        cond.build_references(FakeVAE(), None, 96, 64, 124, "match", ref_images={},
                              ref_videos={"ref_video_1": torch.rand(4, 64, 96, 3)}, ref_video_audios={}, ref_audios={})


def test_video_audio_pairing_is_by_index():
    items, refs = cond.build_references(
        FakeVAE(), FakeAudioVAE(), 96, 64, 124, "match", ref_images={},
        ref_videos={"ref_video_1": torch.rand(22, 64, 96, 3), "ref_video_2": torch.rand(22, 64, 96, 3)},
        ref_video_audios={"ref_video_audio_2": {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}},
        ref_audios={},
    )
    assert [i["type"] for i in items] == ["video", "audio", "video"]  # only video 2 has a soundtrack
    assert [r.kind for r in refs] == ["video", "video_audio"]
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_minimax_h3_conditioning_encode.py -q`
Expected: FAIL (`AttributeError: ... encode_video_latent`).

- [ ] **Step 3: Implémenter**

Ajouter à `minimax_h3_conditioning.py` (imports en tête : `import mlx.core as mx`, `import numpy as np`, `from .native.minimax_h3.condition import KeyframeCond, RefBlock`) :

```python
def to_mx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().to("cpu", torch.float32).numpy())


def encode_video_latent(vae, frames: torch.Tensor) -> mx.array:
    """`vae.encode` of `[T, H, W, 3]` frames -> MLX latent `[1, 24, t, H/16, W/16]`
    (already normalized by the VAE)."""
    z = vae.encode(frames)
    if z.ndim != 5 or z.shape[1] != 24:
        raise ValueError(f"ASDX MiniMax H3: the video VAE returned a latent of shape {tuple(z.shape)}, expected [1, 24, T, h, w]")
    return to_mx(z)


def encode_audio_latent(audio_vae, audio: dict) -> tuple[mx.array, int]:
    """Encode an AUDIO input through the audio VAE (`nodes_minimax_h3._encode_ref_audio`):
    resample to the VAE's rate, first batch item, `[1, C, L]` -> `[1, L, C]`."""
    waveform, sr = audio["waveform"], audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    if z.ndim != 4:
        raise ValueError(f"ASDX MiniMax H3: the audio VAE returned a latent of shape {tuple(z.shape)}, expected [1, 32, 2, T]")
    return to_mx(z), int(z.shape[-1])


def build_keyframes(vae, width: int, height: int, frame_count: int, first_frame, last_frame):
    """fl2va anchors. The first frame is the geometry anchor (plain stretch to the canvas);
    the last frame is a follower (aspect-preserving cover crop) at the final pixel frame."""
    images: list[torch.Tensor] = []
    keyframes: list[KeyframeCond] = []
    for frame, crop, index in ((first_frame, "disabled", 0), (last_frame, "center", frame_count - 1)):
        if frame is None:
            continue
        image = resize_image(frame[:1], width, height, crop)
        images.append(image)
        keyframes.append(KeyframeCond(resolved_frame_index=index, latent=encode_video_latent(vae, image)))
    return images, keyframes


def _check_cap(name: str, count: int, cap: int) -> None:
    if count > cap:
        raise ValueError(f"ASDX MiniMax H3: at most {cap} {name} references are supported, got {count}")


def build_references(vae, audio_vae, width: int, height: int, frame_count: int, ref_image_size_mode: str,
                     ref_images: dict, ref_videos: dict, ref_video_audios: dict, ref_audios: dict):
    """ref2va references in request order: images, then videos (a video's soundtrack label
    right before it), then standalone audios. Returns `(ref_items for the tokenizer,
    refs for the DiT)`; without a VAE only `ref_items` (text conditioning) is produced."""
    images = {k: v for k, v in (ref_images or {}).items() if v is not None}
    videos = {k: v for k, v in (ref_videos or {}).items() if v is not None}
    audios = {k: v for k, v in (ref_audios or {}).items() if v is not None}
    _check_cap("image", len(images), MAX_REF_IMAGES)
    _check_cap("video", len(videos), MAX_REF_VIDEOS)
    _check_cap("audio", len(audios), MAX_REF_AUDIOS)

    ref_items: list[dict] = []
    refs: list[RefBlock] = []

    for img in images.values():
        tw, th = ref_image_size(img.shape[1], img.shape[2], width, height, ref_image_size_mode)
        resized = resize_image(img[:1], tw, th, "disabled")
        ref_items.append({"type": "image", "data": resized})
        if vae is not None:
            z = encode_video_latent(vae, resized)
            refs.append(RefBlock(kind="image", latent=z, latent_t=int(z.shape[2]), latent_h=th // 16, latent_w=tw // 16))

    soundtracks = ref_video_audios or {}
    for name, video in videos.items():
        soundtrack = soundtracks.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        frames, cw, ch = prepare_ref_video(video, frame_count)
        if soundtrack is not None:
            ref_items.append({"type": "audio"})  # its <Audio j> label comes before <Video k>
        sampled, stamps = qwen_video_frames(frames)
        ref_items.append({"type": "video", "data": sampled, "timestamps": stamps})
        if vae is None:
            continue
        z = encode_video_latent(vae, frames)
        audio_latent, audio_t = (None, 0)
        if soundtrack is not None and audio_vae is not None:
            audio_latent, audio_t = encode_audio_latent(audio_vae, soundtrack)
        refs.append(RefBlock(kind="video_audio" if audio_t else "video", latent=z, audio_latent=audio_latent,
                             latent_t=int(z.shape[2]), latent_h=ch // 16, latent_w=cw // 16, ref_audio_t=audio_t))

    for audio in audios.values():
        ref_items.append({"type": "audio"})
        if audio_vae is not None:
            audio_latent, audio_t = encode_audio_latent(audio_vae, audio)
            refs.append(RefBlock(kind="audio", audio_latent=audio_latent, ref_audio_t=audio_t))

    return ref_items, refs
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_minimax_h3_conditioning_encode.py tests/test_minimax_h3_conditioning_geometry.py -q -rs`
Expected: PASS. Preuves par mutation : (a) première et dernière image échangées (`crop`/`index`) ; (b) placer le libellé audio d'une vidéo APRÈS la vidéo ; (c) apparier la bande son par position au lieu de l'index du nom ; (d) créer un `RefBlock` sans VAE ; (e) retirer un plafond ; montrer chaque échec.

- [ ] **Step 5: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/minimax_h3_conditioning.py tests/test_minimax_h3_conditioning_encode.py
git commit -m "feat: encode MiniMax H3 keyframes and references through the ComfyUI VAE bridge"
```

---

### Task 3: Constructeurs de conditioning de haut niveau

**Files:**
- Modify: `apple_silicon_nodes/minimax_h3_conditioning.py` (ajouts)
- Test: `tests/test_minimax_h3_conditioning_encode.py` (ajouts)

**Interfaces:**
- Consumes: `build_keyframes`, `build_references`, `estimate_packed_rows`, `ROW_WARN_THRESHOLD`, `to_mx` (tâches 1-2) ; `encode_minimax_h3_prompt(text_encoder, prompt, *, images, ref_items)` de `minimax_h3_nodes` (brique 2, importé dans la fonction pour éviter l'import circulaire) ; `_temporal_shape(length) -> (frame_count, latent_t, audio_t)` de `minimax_h3_nodes`.
- Produces : `empty_av_latents(width: int, height: int, length: int) -> tuple[dict, dict, int, int, int]` (`(video_latent, audio_latent, frame_count, latent_t, audio_t)` ; latents `{"samples": tensor zéros}` CPU, formes `[1, 24, latent_t, H/16, W/16]` et `[1, 32, 2, audio_t]`) ; `build_i2v_conditioning(text_encoder: dict, vae, prompt: str, width: int, height: int, length: int, first_frame, last_frame) -> tuple[dict, dict, dict]` `(conditioning, video_latent, audio_latent)` ; `build_reference_conditioning(text_encoder: dict, vae, audio_vae, prompt: str, width: int, height: int, length: int, ref_image_size_mode: str, ref_images, ref_videos, ref_video_audios, ref_audios) -> tuple[dict, dict, dict]`. Le conditioning renvoyé est celui de `encode_minimax_h3_prompt` plus `"keyframes"` (liste de `KeyframeCond`) et/ou `"refs"` (liste de `RefBlock`) quand non vides. Une ligne `[ASDX] ... rows` est imprimée et un avertissement si le total (texte compris) dépasse `ROW_WARN_THRESHOLD`.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à `tests/test_minimax_h3_conditioning_encode.py` :

```python
def _stub_prompt(monkeypatch, rows=10):
    """Replace encode_minimax_h3_prompt so the builders can be tested without an encoder."""
    import sys
    import types

    seen = {}
    nodes = types.ModuleType("apple_silicon_nodes.minimax_h3_nodes")

    def fake_encode(text_encoder, prompt, *, images=None, ref_items=None):
        seen.update(prompt=prompt, images=images, ref_items=ref_items)
        return {"type": "minimax_h3", "hidden_states": mx.zeros((rows, 4)), "token_tags": mx.ones((rows,), dtype=mx.int32), "text": prompt}

    nodes.encode_minimax_h3_prompt = fake_encode
    nodes._temporal_shape = lambda length: (124, 37, 207)
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.minimax_h3_nodes", nodes)
    return seen


def test_empty_av_latents_shapes(monkeypatch):
    _stub_prompt(monkeypatch)
    video, audio, frame_count, latent_t, audio_t = cond.empty_av_latents(1344, 768, 124)
    assert tuple(video["samples"].shape) == (1, 24, 37, 48, 84) and tuple(audio["samples"].shape) == (1, 32, 2, 207)
    assert (frame_count, latent_t, audio_t) == (124, 37, 207)


def test_i2v_conditioning_carries_keyframes_and_images(monkeypatch):
    seen = _stub_prompt(monkeypatch)
    conditioning, video, audio = cond.build_i2v_conditioning(
        {"type": "asdx_minimax_h3_text_encoder"}, FakeVAE(), "a cat", 96, 64, 124, torch.rand(1, 64, 96, 3), None)
    assert [k.resolved_frame_index for k in conditioning["keyframes"]] == [0] and "refs" not in conditioning
    assert len(seen["images"]) == 1 and seen["ref_items"] is None
    assert tuple(video["samples"].shape)[3:] == (4, 6)


def test_i2v_without_frames_is_plain_t2v(monkeypatch):
    seen = _stub_prompt(monkeypatch)
    conditioning, _, _ = cond.build_i2v_conditioning({}, FakeVAE(), "a cat", 96, 64, 124, None, None)
    assert "keyframes" not in conditioning and "refs" not in conditioning and not seen["images"]


def test_i2v_needs_a_vae_when_frames_are_given(monkeypatch):
    _stub_prompt(monkeypatch)
    with pytest.raises(ValueError, match="vae"):
        cond.build_i2v_conditioning({}, None, "a cat", 96, 64, 124, torch.rand(1, 64, 96, 3), None)


def test_reference_conditioning_forwards_ref_items_in_order(monkeypatch):
    seen = _stub_prompt(monkeypatch)
    snd = {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
    conditioning, _, _ = cond.build_reference_conditioning(
        {}, FakeVAE(), FakeAudioVAE(), "a cat", 96, 64, 124, "match",
        ref_images={"ref_image_1": torch.rand(1, 64, 96, 3)}, ref_videos={}, ref_video_audios={}, ref_audios={"ref_audio_1": snd})
    assert [i["type"] for i in seen["ref_items"]] == ["image", "audio"]
    assert [r.kind for r in conditioning["refs"]] == ["image", "audio"] and "keyframes" not in conditioning


def test_row_warning_only_above_threshold(monkeypatch, capsys):
    _stub_prompt(monkeypatch, rows=10)
    args = ({}, None, None, "a cat", 1344, 768, 124, "match")
    kwargs = dict(ref_images={}, ref_videos={}, ref_video_audios={}, ref_audios={})
    cond.build_reference_conditioning(*args, **kwargs)
    quiet = capsys.readouterr().out
    assert "packed sequence" in quiet and "WARNING" not in quiet  # 10 text + 37296 + 414 rows: under the threshold
    monkeypatch.setattr(cond, "ROW_WARN_THRESHOLD", 1000)
    cond.build_reference_conditioning(*args, **kwargs)
    assert "WARNING" in capsys.readouterr().out
```

Note pour l'exécutant : le faux `_temporal_shape` renvoie une forme fixe ; les tests n'exercent que les contrats.

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_minimax_h3_conditioning_encode.py -q`
Expected: FAIL (`AttributeError: ... empty_av_latents`).

- [ ] **Step 3: Implémenter**

Ajouter à `minimax_h3_conditioning.py` :

```python
def empty_av_latents(width: int, height: int, length: int):
    """Zero video/audio latents on the model's 17k+5 grid (the sampler draws its own noise)."""
    from .minimax_h3_nodes import _temporal_shape

    frame_count, latent_t, audio_t = _temporal_shape(length)
    video = torch.zeros([1, 24, latent_t, height // 16, width // 16])
    audio = torch.zeros([1, 32, 2, audio_t])
    return {"samples": video}, {"samples": audio}, frame_count, latent_t, audio_t


def _report_rows(text_rows: int, rows_without_text: int) -> None:
    total = text_rows + rows_without_text
    print(f"[ASDX] MiniMax H3 packed sequence: {total} rows ({text_rows} text + {rows_without_text} video/audio/conditions)")
    if total > ROW_WARN_THRESHOLD:
        print(
            f"[ASDX] MiniMax H3 WARNING: {total} rows exceeds {ROW_WARN_THRESHOLD}: the packed sequence is long "
            f"(memory and time grow with it; reference tokens ride through all 50 blocks at every step). "
            f"Consider ref_image_size='match', fewer references or a shorter/smaller generation."
        )


def build_i2v_conditioning(text_encoder: dict, vae, prompt: str, width: int, height: int, length: int,
                           first_frame, last_frame):
    """t2v / fl2va: prompt (+ optional first/last frame) -> (conditioning, video_latent, audio_latent)."""
    from .minimax_h3_nodes import encode_minimax_h3_prompt

    if (first_frame is not None or last_frame is not None) and vae is None:
        raise ValueError("ASDX MiniMax H3: first_frame/last_frame need the vae input to be encoded")
    video_latent, audio_latent, frame_count, latent_t, audio_t = empty_av_latents(width, height, length)
    images, keyframes = build_keyframes(vae, width, height, frame_count, first_frame, last_frame)
    conditioning = encode_minimax_h3_prompt(text_encoder, prompt, images=images)
    if keyframes:
        conditioning["keyframes"] = keyframes
    rows = estimate_packed_rows(width, height, latent_t, audio_t, keyframe_frames=len(keyframes))
    _report_rows(int(conditioning["hidden_states"].shape[0]), rows)
    return conditioning, video_latent, audio_latent


def build_reference_conditioning(text_encoder: dict, vae, audio_vae, prompt: str, width: int, height: int,
                                 length: int, ref_image_size_mode: str, ref_images, ref_videos,
                                 ref_video_audios, ref_audios):
    """ref2va: prompt + references -> (conditioning, video_latent, audio_latent)."""
    from .minimax_h3_nodes import encode_minimax_h3_prompt

    video_latent, audio_latent, frame_count, latent_t, audio_t = empty_av_latents(width, height, length)
    ref_items, refs = build_references(vae, audio_vae, width, height, frame_count, ref_image_size_mode,
                                       ref_images, ref_videos, ref_video_audios, ref_audios)
    if (ref_images or ref_videos or ref_audios) and vae is None:
        print("[ASDX] MiniMax H3: no vae connected, the references only condition the text encoder")
    conditioning = encode_minimax_h3_prompt(text_encoder, prompt, ref_items=ref_items or None)
    if refs:
        conditioning["refs"] = refs
    image_sizes = tuple((r.latent_w * 16, r.latent_h * 16) for r in refs if r.kind == "image")
    videos = tuple((r.latent_t, r.latent_w * 16, r.latent_h * 16, r.ref_audio_t) for r in refs if r.kind in ("video", "video_audio"))
    standalone_audio = sum(r.ref_audio_t for r in refs if r.kind == "audio")
    rows = estimate_packed_rows(width, height, latent_t, audio_t, ref_image_sizes=image_sizes,
                                ref_videos=videos, ref_audio_t=standalone_audio)
    _report_rows(int(conditioning["hidden_states"].shape[0]), rows)
    return conditioning, video_latent, audio_latent
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_minimax_h3_conditioning_encode.py tests/test_minimax_h3_conditioning_geometry.py -q -rs`
Expected: PASS. Preuves par mutation : (a) n'ajouter jamais `"keyframes"` au conditioning ; (b) passer `ref_items` au tokenizer sous `images=` ; (c) supprimer le contrôle « vae absent » ; (d) supprimer l'avertissement de seuil.

- [ ] **Step 5: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/minimax_h3_conditioning.py tests/test_minimax_h3_conditioning_encode.py
git commit -m "feat: build MiniMax H3 i2v and ref2va conditioning with row accounting"
```

---

### Task 4: Nœuds ComfyUI, enregistrement et README

**Files:**
- Modify: `apple_silicon_nodes/minimax_h3_nodes.py` (deux classes, `NODE_LIST`)
- Modify: `README.md`
- Test: `tests/test_minimax_h3_i2v_ref_nodes.py`

**Interfaces:**
- Consumes: `build_i2v_conditioning`, `build_reference_conditioning` (tâche 3) ; `MAX_REF_IMAGES`, `MAX_REF_VIDEOS`, `MAX_REF_AUDIOS`.
- Produces : `ASDX_MiniMaxH3ImageToVideo` (entrées : `text_encoder` `io.Custom("asdx_minimax_h3_text_encoder")`, `vae` `io.Vae` optionnel, `prompt`, `width` 1344, `height` 768, `length` 124, `first_frame`/`last_frame` `io.Image` optionnels ; sorties : `conditioning` `io.Custom("asdx_minimax_h3_conditioning")`, `video_latent`, `audio_latent` `io.Latent`) et `ASDX_MiniMaxH3ReferenceToVideo` (mêmes + `audio_vae` `io.Vae` optionnel, `ref_image_size` `io.Combo` `["match", "max"]` défaut `"match"`, `io.Autogrow` `ref_images` (préfixe `ref_image_`, max 9), `ref_videos` (`ref_video_`, max 3), `ref_video_audios` (`ref_video_audio_`, max 3), `ref_audios` (`ref_audio_`, max 3)), les deux dans `NODE_LIST`.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_minimax_h3_i2v_ref_nodes.py` :

```python
"""The two new nodes: registration, schema (against the real comfy_api when available) and
that execute() delegates to the conditioning builders."""

from __future__ import annotations

import sys
import types

import pytest

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
nodes = load_node_module("minimax_h3_nodes")


def test_nodes_are_registered():
    ids = [n.__name__ for n in nodes.NODE_LIST]
    assert "ASDX_MiniMaxH3ImageToVideo" in ids and "ASDX_MiniMaxH3ReferenceToVideo" in ids


def _install_fake_builders(monkeypatch):
    calls = {}
    mod = types.ModuleType("apple_silicon_nodes.minimax_h3_conditioning")
    mod.MAX_REF_IMAGES, mod.MAX_REF_VIDEOS, mod.MAX_REF_AUDIOS = 9, 3, 3

    def i2v(*args, **kwargs):
        calls["i2v"] = (args, kwargs)
        return {"type": "minimax_h3"}, {"samples": 1}, {"samples": 2}

    def ref(*args, **kwargs):
        calls["ref"] = (args, kwargs)
        return {"type": "minimax_h3"}, {"samples": 1}, {"samples": 2}

    mod.build_i2v_conditioning, mod.build_reference_conditioning = i2v, ref
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.minimax_h3_conditioning", mod)
    return calls


def test_i2v_execute_delegates(monkeypatch):
    calls = _install_fake_builders(monkeypatch)
    enc = {"type": "asdx_minimax_h3_text_encoder"}
    out = nodes.ASDX_MiniMaxH3ImageToVideo.execute(enc, "a cat", 96, 64, 124, vae="VAE", first_frame="F", last_frame="L")
    assert out.values[0] == {"type": "minimax_h3"} and out.values[1] == {"samples": 1} and out.values[2] == {"samples": 2}
    args, kwargs = calls["i2v"]
    flat = list(args) + list(kwargs.values())
    assert enc in flat and all(v in flat for v in ("VAE", "a cat", 96, 64, 124, "F", "L"))


def test_i2v_rejects_wrong_encoder_type():
    with pytest.raises(RuntimeError, match="ASDX_MiniMaxH3TextEncoderLoader"):
        nodes.ASDX_MiniMaxH3ImageToVideo.execute({"type": "other"}, "a cat", 96, 64, 124)


def test_ref_execute_delegates_with_all_inputs(monkeypatch):
    calls = _install_fake_builders(monkeypatch)
    enc = {"type": "asdx_minimax_h3_text_encoder"}
    nodes.ASDX_MiniMaxH3ReferenceToVideo.execute(
        enc, "a cat", 96, 64, 124, ref_image_size="max", vae="V", audio_vae="A",
        ref_images={"ref_image_1": "I"}, ref_videos={"ref_video_1": "VID"},
        ref_video_audios={"ref_video_audio_1": "S"}, ref_audios={"ref_audio_1": "AU"})
    args, kwargs = calls["ref"]
    flat = list(args) + list(kwargs.values())
    for expected in ("max", "V", "A", {"ref_image_1": "I"}, {"ref_video_1": "VID"}, {"ref_video_audio_1": "S"}, {"ref_audio_1": "AU"}):
        assert expected in flat


def test_schemas_build_with_the_real_comfy_api():
    """The Autogrow / Combo / Custom schema needs the real comfy_api: skipped where unavailable."""
    from tests.support.comfyui_reference_loader import load_real_comfy_api_io

    io = load_real_comfy_api_io()
    for cls_name in ("ASDX_MiniMaxH3ImageToVideo", "ASDX_MiniMaxH3ReferenceToVideo"):
        src = getattr(nodes, cls_name)
        schema = src.define_schema()
        assert schema.node_id == cls_name
```

Note pour l'exécutant : `load_real_comfy_api_io` n'existe pas encore : l'ajouter à `tests/support/comfyui_reference_loader.py` (même schéma que `load_real_comfy_qwen3vl` : purge de `comfy*`, insertion des chemins ComfyUI, `import comfy_api.latest`, skip propre si absent). Le test `test_schemas_build_with_the_real_comfy_api` doit CONSTRUIRE le schéma avec le vrai `comfy_api` : comme `minimax_h3_nodes` est chargé avec les stubs dans ce fichier, charger une seconde copie du module de nœuds avec le vrai `comfy_api` (ou construire le schéma par le vrai `io` en substituant `nodes.io`) ; si cela s'avère impossible sans lancer ComfyUI, remplacer par un test qui vérifie la structure des entrées (`define_schema` avec le `io` factice) et l'écrire clairement comme tel, en signalant la limite. 

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_minimax_h3_i2v_ref_nodes.py -q`
Expected: FAIL (`AttributeError: ... ASDX_MiniMaxH3ImageToVideo`).

- [ ] **Step 3: Implémenter**

Dans `minimax_h3_nodes.py`, avant `NODE_LIST`, ajouter :

```python
def _check_encoder(text_encoder: dict, who: str) -> None:
    if not isinstance(text_encoder, dict) or text_encoder.get("type") != "asdx_minimax_h3_text_encoder":
        raise RuntimeError(f"ASDX {who}: expected the output of ASDX_MiniMaxH3TextEncoderLoader.")


class ASDX_MiniMaxH3ImageToVideo(io.ComfyNode):
    """t2v and fl2va: prompt (+ optional first/last frame) -> conditioning + empty AV latents.

    Mirrors ComfyUI's `MiniMaxH3ImageToVideo`. The first frame is stretched onto the canvas
    and becomes frame 0; the last frame is cover-cropped and anchors the final frame. The
    text encoder must have been loaded with `load_vision` when frames are connected."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MiniMaxH3ImageToVideo",
            display_name="🍏 ASDX MiniMax H3 Image to Video",
            category="ASDX/Conditioning",
            inputs=[
                io.Custom("asdx_minimax_h3_text_encoder").Input("text_encoder"),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE, needed when a frame is connected."),
                io.String.Input("prompt", multiline=True, default=""),
                io.Int.Input("width", default=1344, min=32, max=4096, step=32),
                io.Int.Input("height", default=768, min=32, max=4096, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s)."),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[
                io.Custom("asdx_minimax_h3_conditioning").Output(display_name="conditioning"),
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder: dict, prompt: str, width: int, height: int, length: int,
                vae=None, first_frame=None, last_frame=None) -> io.NodeOutput:
        _check_encoder(text_encoder, "MiniMax H3 Image to Video")
        from .minimax_h3_conditioning import build_i2v_conditioning

        conditioning, video_latent, audio_latent = build_i2v_conditioning(
            text_encoder, vae, prompt, width, height, length, first_frame, last_frame)
        return io.NodeOutput(conditioning, video_latent, audio_latent)


class ASDX_MiniMaxH3ReferenceToVideo(io.ComfyNode):
    """ref2va: prompt + reference images / videos / audio -> conditioning + empty AV latents.

    Mirrors ComfyUI's `MiniMaxH3ReferenceToVideo`. References enter the presentation in fixed
    order (images, then videos with their soundtrack label first, then standalone audio),
    ordinals are 1-based per type: use <Picture i> / <Video k> / <Audio j> in the prompt.
    Reference tokens ride through every sampling step: 'max' can be several times slower.
    Requires the ref2va DiT checkpoint and a text encoder loaded with `load_vision`."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        from .minimax_h3_conditioning import MAX_REF_AUDIOS, MAX_REF_IMAGES, MAX_REF_VIDEOS

        def grow(name: str, prefix: str, item, cap: int):
            return io.Autogrow.Input(name, optional=True, template=io.Autogrow.TemplatePrefix(
                input=item, prefix=prefix, min=0, max=cap))

        return io.Schema(
            node_id="ASDX_MiniMaxH3ReferenceToVideo",
            display_name="🍏 ASDX MiniMax H3 Reference to Video",
            category="ASDX/Conditioning",
            description="<Picture i> / <Video k> / <Audio j> reference conditioning for MiniMax H3. Use the same tags when prompting.",
            inputs=[
                io.Custom("asdx_minimax_h3_text_encoder").Input("text_encoder"),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE. Without it the reference images/videos only condition the text encoder."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Audio VAE. Without it the reference audio only conditions the text encoder."),
                io.String.Input("prompt", multiline=True, default=""),
                io.Int.Input("width", default=1344, min=32, max=4096, step=32),
                io.Int.Input("height", default=768, min=32, max=4096, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s)."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="'match' scales each reference (down only, keeping the aspect) to the generation's pixel area; 'max' uses a 2048px short edge for the best identity fidelity but adds many rows."),
                grow("ref_images", "ref_image_", io.Image.Input("ref_image"), MAX_REF_IMAGES),
                grow("ref_videos", "ref_video_", io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (>= 5 frames)"), MAX_REF_VIDEOS),
                grow("ref_video_audios", "ref_video_audio_", io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"), MAX_REF_VIDEOS),
                grow("ref_audios", "ref_audio_", io.Audio.Input("ref_audio"), MAX_REF_AUDIOS),
            ],
            outputs=[
                io.Custom("asdx_minimax_h3_conditioning").Output(display_name="conditioning"),
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder: dict, prompt: str, width: int, height: int, length: int,
                ref_image_size: str = "match", vae=None, audio_vae=None, ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None) -> io.NodeOutput:
        _check_encoder(text_encoder, "MiniMax H3 Reference to Video")
        from .minimax_h3_conditioning import build_reference_conditioning

        conditioning, video_latent, audio_latent = build_reference_conditioning(
            text_encoder, vae, audio_vae, prompt, width, height, length, ref_image_size,
            ref_images or {}, ref_videos or {}, ref_video_audios or {}, ref_audios or {})
        return io.NodeOutput(conditioning, video_latent, audio_latent)
```

Ajouter les deux classes à `NODE_LIST` (après `ASDX_MiniMaxH3TextEncode`, avant le sampler).

Dans `README.md`, ajouter une section « MiniMax H3 (Apple Silicon natif) » qui liste les nœuds (`Empty Latent (AV)`, `Sigma Shift`, `Model Loader`, `Text Encoder Loader` avec l'option `load_vision`, `Text Encode`, `Image to Video`, `Reference to Video`, `Sampler`), les formats de checkpoint acceptés (GGUF, safetensors INT8 ConvRot ; le W4A8 est refusé), le graphe minimal t2v / i2v / ref, et les limites connues (pas de PDD, pas d'AddGuide, le fichier `ref2va` pour les références, les références allongent la séquence). Lire d'abord la structure actuelle du README et suivre son style ; ne rien supprimer.

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_minimax_h3_i2v_ref_nodes.py tests/test_minimax_h3_loaders.py tests/test_minimax_h3_sampler_node.py -q -rs` puis `uv run pytest tests -q` (seul échec permis : `tests/support/test_krea2_module_loader.py::test_plain_import_of_model_fails_outside_comfyui`). Preuves par mutation : `execute` ne transmet pas `first_frame` ; `ref_audios` non transmis ; type d'encodeur non vérifié ; nœud absent de `NODE_LIST`.

- [ ] **Step 5: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/minimax_h3_nodes.py tests/test_minimax_h3_i2v_ref_nodes.py tests/support/comfyui_reference_loader.py README.md
git commit -m "feat: add MiniMax H3 image-to-video and reference-to-video nodes"
```

---

### Task 5: Essai de bout en bout sur les vrais fichiers

**Files:**
- Test: `tests/native/minimax_h3/test_real_pipeline_cond.py` (créer)

**Interfaces:**
- Consumes: `build_i2v_conditioning`, `build_reference_conditioning` (tâche 3), chargeurs (`load_qwen3_text_encoder_checkpoint`, `load_vision_tower`, `load_minimax_h3_checkpoint`), `payload_from_conditioning`, `run_minimax_h3_sampling`.
- Produces : un test réel (`ASDX_FULL_GGUF_TEST=1`) qui enchaîne : (1) charger le TE (safetensors) et la tour, encoder une image de test `256x384` avec un vrai `comfy.sd.VAE` vidéo (`/Volumes/X10Pro/Images/models/vae/minimax_h3_video_vae_fp16.safetensors`) pour construire un conditioning i2v ; (2) libérer TE, tour et VAE ; (3) charger le DiT `fl2va` (safetensors INT8) et exécuter 1 pas de `run_minimax_h3_sampling` sur `256x384`, `length=5`, avec `payload_from_conditioning` ; (4) sortie finie et différente du même run sans keyframe.

- [ ] **Step 1: Écrire le test**

```python
"""End to end on real files (i2v): TE+tower+video VAE -> conditioning -> real fl2va DiT, 1 step, tiny canvas."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_text_encoders
from support.minimax_h3_module_loader import load_native_module

MODELS = Path("/Volumes/X10Pro/Images/models")
TE = MODELS / "text_encoders" / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE = MODELS / "vae" / "minimax_h3_video_vae_fp16.safetensors"
DIT = MODELS / "diffusion_models" / "MiniMax H3" / "base model" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"


@pytest.mark.skipif(os.environ.get("ASDX_FULL_GGUF_TEST") != "1", reason="loads real TE, VAE and DiT; set ASDX_FULL_GGUF_TEST=1")
def test_i2v_pipeline_on_real_files():
    for p in (TE, VAE, DIT):
        if not p.exists():
            pytest.skip(f"{p.name} not present")
    load_real_comfy_text_encoders()  # puts the real ComfyUI on sys.path (tokenizer, VAE class)
    import comfy.sd
    import comfy.utils
    import torch

    te_wm = load_native_module("minimax_h3.text_encoder_weight_map")
    vis_wm = load_native_module("minimax_h3.vision_weight_map")
    dit_wm = load_native_module("minimax_h3.weight_map")
    sampling = load_native_module("minimax_h3.sampling")

    import importlib
    nodes = importlib.import_module("apple_silicon_nodes.minimax_h3_nodes")  # requires the package import path of the project
    cond_builders = importlib.import_module("apple_silicon_nodes.minimax_h3_conditioning")

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(VAE)))
    text_encoder = {"type": "asdx_minimax_h3_text_encoder", "encoder": te_wm.load_qwen3_text_encoder_checkpoint(TE, dtype="float16"),
                    "vision_tower": vis_wm.load_vision_tower(TE)}
    frame = torch.from_numpy(np.random.default_rng(0).random((1, 256, 384, 3), dtype=np.float32))
    conditioning, video_latent, audio_latent = cond_builders.build_i2v_conditioning(
        text_encoder, vae, "a cat walks", 384, 256, 5, frame, None)
    assert [k.resolved_frame_index for k in conditioning["keyframes"]] == [0]
    hidden = conditioning["hidden_states"]
    assert hidden.shape[1] == 5120 and conditioning["token_tags"].shape[0] == hidden.shape[0]

    del text_encoder, vae
    gc.collect()
    mx.clear_cache()

    model = dit_wm.load_minimax_h3_checkpoint(DIT, dtype="float16")
    video = mx.random.normal(tuple(video_latent["samples"].shape))
    audio = mx.random.normal(tuple(audio_latent["samples"].shape))
    payload = nodes.payload_from_conditioning(conditioning, seed=1)
    with_kf = sampling.run_minimax_h3_sampling(model, video, audio, hidden, 1, payload=payload)
    plain = sampling.run_minimax_h3_sampling(model, video, audio, hidden, 1)
    mx.eval(*with_kf, *plain)
    print(f"[real i2v] peak {mx.get_peak_memory() / 1e9:.2f} GB, keyframe delta {float(mx.max(mx.abs(with_kf[0] - plain[0])).item()):.3f}")
    assert bool(mx.all(mx.isfinite(with_kf[0])).item()) and bool(mx.all(mx.isfinite(with_kf[1])).item())
    assert float(mx.max(mx.abs(with_kf[0] - plain[0])).item()) > 1e-3
```

Note pour l'exécutant : `importlib.import_module("apple_silicon_nodes.minimax_h3_nodes")` importe `comfy.model_management` et `comfy_api` au chargement du module ; utiliser le vrai ComfyUI déjà mis sur `sys.path` par `load_real_comfy_text_encoders()`. Si le package `apple_silicon_nodes` ne s'importe pas ainsi hors d'un processus ComfyUI (son `__init__` importe `comfy_api`), charger `minimax_h3_nodes.py` et `minimax_h3_conditioning.py` par le mécanisme de `tests/support/comfy_stub.py::load_node_module` en gardant le VAE réel : adapter uniquement l'import, pas la logique. `comfy.sd.VAE(sd=...)` peut demander un `device`/`dtype` : lire `comfy/sd.py` (constructeur `VAE`) et adapter pour un CPU. Si la construction du VAE réel n'est pas réalisable hors d'un processus ComfyUI complet, ne pas la contourner : remplacer UNIQUEMENT l'étape VAE par un faux VAE qui renvoie un latent aléatoire de la bonne forme, l'écrire clairement dans le test et le rapport, et garder le reste (TE + tour + DiT réels).

- [ ] **Step 2: Exécuter une seule fois**

Run: `ASDX_FULL_GGUF_TEST=1 uv run pytest tests/native/minimax_h3/test_real_pipeline_cond.py -q -rs -s`
Expected: 1 passé. Relever verbatim la ligne `[real i2v] ...` et la mémoire après chaque étape. Si un échec survient sur les vrais fichiers alors que les tests synthétiques passent, ne pas le contourner : rapporter l'erreur complète.

- [ ] **Step 3: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add tests/native/minimax_h3/test_real_pipeline_cond.py
git commit -m "test: run the MiniMax H3 i2v pipeline end to end on real files"
```

---

## Auto-revue du plan

- **Couverture de la spec, brique 4** : nœud i2v (`first_frame` étirée, `last_frame` recadrée, latents encodés par le pont VAE, largeur/hauteur explicites) T1-T4 ; nœud ref avec `ref_image_size` `match`/`max`, images <= 9, vidéos <= 3 (>= 5 images, 17k+5, son apparié par index, 2 fps pour l'encodeur), audios <= 3 rééchantillonnés à 32 kHz et encodés par la VAE audio (pont, aucun encodeur à porter) T1-T4 ; « sans VAE : texte seulement, message explicite » T2-T3 ; gate/avertissement en lignes T1, T3 ; notes de passage de la brique 3 (pas de `RefBlock` sans VAE, unités latentes, index de la dernière image) T2 ; essai réel T5 ; README T4.
- **Placeholders** : aucun ; les incertitudes assumées sont nommées avec la conduite à tenir : import du vrai `comfy_extras.nodes_minimax_h3` hors ComfyUI (T1), construction du schéma avec le vrai `comfy_api` (T4), construction du vrai `comfy.sd.VAE` en test (T5, avec un repli explicite et signalé).
- **Cohérence des types** : `KeyframeCond`/`RefBlock` (brique 3), `build_keyframes`, `build_references`, `build_i2v_conditioning`, `build_reference_conditioning`, `estimate_packed_rows`, `encode_minimax_h3_prompt(text_encoder, prompt, *, images, ref_items)` sont utilisés à l'identique dans les tâches 2 à 5.
- **Écarts connus assumés** : moyenne du posterior sans aller-retour fp16 (ComfyUI) contre échantillon à graine 42 + fp16 (Rust) ; `PDD` et `AddGuide` hors périmètre ; le nœud ne libère pas lui-même l'encodeur (le gate du loader de DiT s'en charge).
