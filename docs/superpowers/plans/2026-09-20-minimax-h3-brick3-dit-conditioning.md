# MiniMax H3 brique 3 : DiT avec lignes de condition (keyframes et références)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Le DiT MLX accepte des lignes de condition (`cond`, `cond_audio`, `ref_img`, `ref_audio`) placées dans la séquence packée, avec leurs positions, leurs timesteps par segment (0,999 / 1,0 par défaut), leur bruit d'augmentation et les tags de modulation du texte, et le sampler les fournit à chaque pas. Critère : parité numérique de bout en bout avec le vrai `MiniMaxH3Model` de ComfyUI sur un petit modèle à poids aléatoires, **y compris le t2v seul, qui n'a aucun oracle numérique aujourd'hui**.

**Architecture:** Un module `condition.py` porte les dataclasses (`KeyframeCond`, `RefBlock`, `ConditionPayload`) et `prepare_condition` (patchification + bruit d'augmentation, calculé une fois par run). `layout.py` étend `PackedLayout`. `model.py` assemble la séquence par segments et donne à chaque segment son timestep et son tag. `sampling.py` prépare la condition une seule fois et la passe à chaque pas ; le nœud sampler la construit depuis le dict de conditioning.

**Tech Stack:** MLX, numpy, torch (tests de parité seulement), pytest, `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-19-minimax-h3-i2v-ref-design.md` (brique 3). Références : `comfy/ldm/minimax/model.py` (`PackedLayout`, `_cond_video_rows`, `_cond_audio_rows`, `_forward`) dans `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI` ; Rust `mlx-gen-minimax-h3/src/conditioning.rs` pour la justification des constantes.

## Global Constraints

- Toujours `uv run pytest` / `uv run python` (jamais `python` nu). Python 3.10+, annotations complètes, docstrings en anglais, pas d'emoji ni de tiret cadratin.
- Fail-closed : forme ou valeur incohérente (tags de longueur inattendue, latent de mauvaise forme) lève une exception explicite.
- **Précision GPU (canon)** : le matmul fp32 GPU de MLX a ~7,5e-4 d'erreur relative. Les tests de parité contre ComfyUI tournent sous `mx.stream(mx.cpu)` avec tolérance serrée ; un test GPU contre CPU (borne relative avec cas nul) couvre la production.
- **Toute parité est prouvée par mutation** (canon) : chaque tâche de parité exige que l'exécutant casse volontairement l'élément testé (par ex. retire la condition, change un timestep, ignore les tags), montre que le test échoue, puis restaure le code (`git diff` du source inchangé).
- Bruit d'augmentation : le flux de torch ne peut pas être reproduit par MLX. Le code prend une fonction `noise_fn(shape, seed) -> mx.array` (par défaut `mx.random.normal` à clé fixe) ; les tests injectent le bruit de torch (`torch.randn(shape, generator=torch.Generator().manual_seed(seed))`) des deux côtés.
- **Comportement t2v inchangé** : sans condition (`cond=None`), la sortie du modèle doit rester identique à l'actuelle ; les tests existants (`test_full_model.py`, `test_layout.py`, `test_sampling.py`, `test_dit_block.py`) passent sans modification d'attentes.
- Tests importés via `load_native_module("minimax_h3.<nom>")` ; tests de nœuds via `tests/support/comfy_stub.py`.
- **Aucun commit automatique** (règle utilisateur) : chaque étape « Commit » = mettre dans l'index et attendre l'ordre. Pas de ligne d'attribution.

## Décisions de conception (tranchées ici)

- Le bruit d'augmentation suit ComfyUI : graine = graine de la requête (`payload.seed`), **le même flux redémarré pour chaque condition**, audio à `seed + 1`. Comme il est identique à chaque pas, il est calculé **une fois** par run (`prepare_condition`), pas à chaque forward. (Le Rust utilise un flux dédié à graine fixe 42 pour l'échantillon du posterior VAE : c'est l'encodage des keyframes, sujet de la brique 4, pas cette augmentation.)
- Pas de masque de débruitage (inpainting) ni de PDD : hors périmètre.
- Les latents de condition arrivent déjà normalisés (encodage et `fp16` aller-retour = brique 4).

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `apple_silicon_nodes/native/minimax_h3/condition.py` (créer) | dataclasses, `default_noise`, `prepare_condition`, `PreparedCondition` |
| `apple_silicon_nodes/native/minimax_h3/layout.py` (modifier) | `ref_t_span`, `PackedLayout(..., keyframes, refs)` |
| `apple_silicon_nodes/native/minimax_h3/model.py` (modifier) | `build_mod_segments` étendu, `MiniMaxH3Model.__call__(..., cond=None)` |
| `apple_silicon_nodes/native/minimax_h3/sampling.py` (modifier) | `run_minimax_h3_sampling(..., payload=None, noise_fn=None)` |
| `apple_silicon_nodes/minimax_h3_nodes.py` (modifier) | le nœud sampler construit le payload depuis le conditioning |
| `tests/support/minimax_h3_dit_reference.py` (créer) | petit modèle ComfyUI de référence, copie des poids, cas de test partagés |
| `tests/native/minimax_h3/test_condition.py`, `test_layout_cond.py`, `test_full_model_cond.py` (créer) | parité |
| `tests/native/minimax_h3/test_sampling.py`, `tests/test_minimax_h3_loaders.py` (modifier) | sampler et nœud |

---

### Task 1: Dataclasses de condition, lignes de condition et référence de test partagée

**Files:**
- Create: `apple_silicon_nodes/native/minimax_h3/condition.py`
- Create: `tests/support/minimax_h3_dit_reference.py`
- Test: `tests/native/minimax_h3/test_condition.py`

**Interfaces:**
- Consumes: `patchify_video(video, patch_size)` et `pack_audio(audio)` de `patchify.py` (existants, batch 1, entrée `[1, C, T, H, W]` / `[1, C, 2, T]`).
- Produces (dans `condition.py`) :
  - `VISUAL_COND_TIMESTEP = 0.999`, `AUDIO_COND_TIMESTEP = 1.0`
  - `KeyframeCond(resolved_frame_index: int, latent: mx.array | None = None, audio_latent: mx.array | None = None)` (dataclass figée ; `latent` `[1, C, T, h, w]`, `audio_latent` `[1, 32.., 2, T]`)
  - `RefBlock(kind: str, latent: mx.array | None = None, audio_latent: mx.array | None = None, latent_t: int = 0, latent_h: int = 0, latent_w: int = 0, ref_audio_t: int = 0)` avec `kind` dans `{"image", "video", "video_audio", "audio"}`
  - `ConditionPayload(text_token_tags: np.ndarray | None = None, keyframes: tuple[KeyframeCond, ...] = (), refs: tuple[RefBlock, ...] = (), seed: int = 0, visual_cond_noise_aug: float = 0.999, audio_cond_noise_aug: float = 1.0)`
  - `default_noise(shape: tuple[int, ...], seed: int) -> mx.array`
  - `PreparedCondition(payload: ConditionPayload, video_rows: mx.array | None, audio_rows: mx.array | None)`
  - `prepare_condition(payload: ConditionPayload, patch_size: tuple[int, int, int], noise_fn: Callable[[tuple[int, ...], int], mx.array] = default_noise) -> PreparedCondition`
- Produces (dans `tests/support/minimax_h3_dit_reference.py`) : `TINY: dict` (config), `build_reference_model(seed: int = 0)` (vrai `MiniMaxH3Model` ComfyUI, float32, CPU, poids et buffers initialisés), `ours_from_reference(ref) -> MiniMaxH3Model`, `torch_noise(shape, seed) -> mx.array`, `CASES: dict[str, dict]` et `payload_pair(name: str, seed: int = 7) -> tuple[ConditionPayload, dict]` (payload nôtre, payload ComfyUI équivalent).

- [ ] **Step 1: Créer la référence de test partagée**

`tests/support/minimax_h3_dit_reference.py` :

```python
"""A tiny real ComfyUI `MiniMaxH3Model` (random weights, fp32, CPU) and the
matching MLX model + payloads, so parity tests compare the SAME forward."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support.comfyui_reference_loader import load_real_comfy_minimax_model
from support.minimax_h3_module_loader import load_native_module

condition = load_native_module("minimax_h3.condition")
config_mod = load_native_module("minimax_h3.config")
model_mod = load_native_module("minimax_h3.model")

TINY = dict(
    num_layers=2, token_refiner_num_layers=1, hidden_size=16, latents_dim=4, audio_latents_dim=6,
    attention_head_dim=16, num_attention_heads=2, ffn_hidden_size=32, text_dim=10,
    patch_size=(1, 2, 2), rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
    sigma_shift_video=12.0, sigma_shift_audio=3.0, gate_compress=False,
    adaln_curve_grid=17, time_embed_dim=6,
)


def build_reference_model(seed: int = 0):
    """Real `comfy.ldm.minimax.model.MiniMaxH3Model`, tiny, initialized (the
    `disable_weight_init` ops leave weights and buffers uninitialized)."""
    mm = load_real_comfy_minimax_model()
    import comfy.ops
    import torch

    ref = mm.MiniMaxH3Model(dtype=torch.float32, operations=comfy.ops.disable_weight_init, **TINY)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
        ref.adaln_t_table.copy_(torch.randn(ref.adaln_t_table.shape, generator=gen))
        n = ref.rope.inv_freq.shape[0]
        ref.rope.inv_freq.copy_(1.0 / (10000.0 ** (torch.arange(n, dtype=torch.float32) / n)))
    return ref


def ours_from_reference(ref):
    """Our MLX model with the reference's exact weights (every parameter must be covered)."""
    model = model_mod.MiniMaxH3Model(config_mod.MiniMaxH3Config(dtype="float32", **TINY))
    state = {k: mx.array(v.detach().numpy()) for k, v in ref.state_dict().items()}
    ours = dict(tree_flatten(model.parameters()))
    missing = sorted(set(ours) - set(state))
    assert not missing, f"reference state_dict lacks {missing[:5]}"
    model.update(tree_unflatten([(k, state[k].reshape(ours[k].shape)) for k in ours]))
    mx.eval(model.parameters())
    return model


def torch_noise(shape, seed: int) -> mx.array:
    """The reference's own noise draw (fresh CPU generator per condition)."""
    import torch

    gen = torch.Generator("cpu").manual_seed(int(seed))
    return mx.array(torch.randn(tuple(shape), generator=gen, dtype=torch.float32).numpy())


def _rand(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# name -> spec. Latent shapes: video [1, 4, T, h, w] (h, w even), audio [1, 6, 2, T].
CASES: dict[str, dict] = {
    "t2v": {},
    "keyframe_first": {"keyframes": [dict(idx=0, video=(1, 4, 4))]},
    "keyframes_first_last": {"keyframes": [dict(idx=0, video=(1, 4, 4)), dict(idx=7, video=(1, 4, 4))]},
    "keyframe_with_audio": {"keyframes": [dict(idx=0, video=(1, 4, 4), audio=3)]},
    "ref_image": {"refs": [dict(kind="image", video=(1, 4, 4))]},
    "ref_video_audio": {"refs": [dict(kind="video_audio", video=(2, 4, 4), audio=3)]},
    "ref_audio": {"refs": [dict(kind="audio", audio=2)]},
    "tags": {"tags": [1, 1, 0, 0, 0, 1]},
    "combined": {
        "keyframes": [dict(idx=0, video=(1, 4, 4), audio=2)],
        "refs": [dict(kind="image", video=(1, 4, 4)), dict(kind="video_audio", video=(2, 4, 4), audio=3)],
        "tags": [1, 1, 0, 0, 0, 1],
    },
}
TEXT_LEN = 6  # matches the tags above; every case uses a 6-row context


def payload_pair(name: str, seed: int = 7):
    """`(our ConditionPayload, ComfyUI minimax_payload dict)` for a named case."""
    import torch

    spec = CASES[name]
    kfs, refs = [], []
    ref_kfs, ref_refs = [], []
    n = 0
    for kf in spec.get("keyframes", []):
        n += 1
        video = _rand((1, TINY["latents_dim"], *kf["video"][:1], *kf["video"][1:]), n) if "video" in kf else None
        audio = _rand((1, TINY["audio_latents_dim"], 2, kf["audio"]), n + 100) if "audio" in kf else None
        kfs.append(condition.KeyframeCond(kf["idx"], None if video is None else mx.array(video), None if audio is None else mx.array(audio)))
        ref_kfs.append({"resolved_frame_index": kf["idx"],
                        **({"latent": torch.from_numpy(video)} if video is not None else {}),
                        **({"audio_latent": torch.from_numpy(audio)} if audio is not None else {})})
    for r in spec.get("refs", []):
        n += 1
        video = _rand((1, TINY["latents_dim"], *r["video"]), n) if "video" in r else None
        audio = _rand((1, TINY["audio_latents_dim"], 2, r["audio"]), n + 100) if "audio" in r else None
        t, h, w = r.get("video", (0, 0, 0))
        refs.append(condition.RefBlock(kind=r["kind"], latent=None if video is None else mx.array(video),
                                       audio_latent=None if audio is None else mx.array(audio),
                                       latent_t=t, latent_h=h, latent_w=w, ref_audio_t=r.get("audio", 0)))
        blk = {"kind": r["kind"], "latent_t": t, "latent_h": h, "latent_w": w, "ref_audio_t": r.get("audio", 0)}
        if video is not None:
            blk["latent"] = torch.from_numpy(video)
        if audio is not None:
            blk["audio_latent"] = torch.from_numpy(audio)
        ref_refs.append(blk)
    tags = spec.get("tags")
    ours = condition.ConditionPayload(
        text_token_tags=None if tags is None else np.asarray(tags, dtype=np.int64),
        keyframes=tuple(kfs), refs=tuple(refs), seed=seed,
    )
    theirs: dict = {"seed": seed}
    if ref_kfs:
        theirs["keyframes"] = ref_kfs
        theirs["cond_video_latents"] = [k["latent"] for k in ref_kfs if "latent" in k]
        theirs["cond_audio_latents"] = [k["audio_latent"] for k in ref_kfs if "audio_latent" in k]
    if ref_refs:
        theirs["refs"] = ref_refs
        theirs["cond_video_latents"] = theirs.get("cond_video_latents", []) + [r["latent"] for r in ref_refs if "latent" in r]
        theirs["cond_audio_latents"] = theirs.get("cond_audio_latents", []) + [r["audio_latent"] for r in ref_refs if "audio_latent" in r]
    if tags is not None:
        theirs["text_token_tags"] = torch.tensor(tags, dtype=torch.long)[None]
    return ours, theirs
```

Note pour l'exécutant : `video` d'un `keyframe`/`ref` est un triplet `(T, h, w)` de latent (h, w pairs) ; la forme du latent est `[1, latents_dim, T, h, w]`. La ligne `_rand((1, TINY["latents_dim"], *kf["video"][:1], *kf["video"][1:]), n)` est équivalente à `(1, C, *kf["video"])` : la simplifier ainsi si plus lisible. Si `build_reference_model` échoue parce que le `MiniMaxH3Model` de ComfyUI exige d'autres arguments, lire le constructeur (`comfy/ldm/minimax/model.py` lignes 473-511) et ne changer que la construction.

- [ ] **Step 2: Écrire les tests qui échouent**

`tests/native/minimax_h3/test_condition.py` :

```python
"""condition.prepare_condition vs the real ComfyUI `_cond_video_rows` / `_cond_audio_rows`."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support import minimax_h3_dit_reference as R
from support.minimax_h3_module_loader import load_native_module

cond = load_native_module("minimax_h3.condition")
PATCH = (1, 2, 2)


@pytest.mark.parametrize("name", ["keyframe_first", "keyframes_first_last", "keyframe_with_audio", "ref_image", "ref_video_audio", "ref_audio", "combined"])
def test_rows_match_comfyui(name):
    ref = R.build_reference_model()
    ours, theirs = R.payload_pair(name, seed=7)
    prepared = cond.prepare_condition(ours, PATCH, noise_fn=R.torch_noise)
    want_v = ref._cond_video_rows(theirs, "cpu")
    want_a = ref._cond_audio_rows(theirs, "cpu")
    if want_v is None:
        assert prepared.video_rows is None
    else:
        assert np.abs(np.array(prepared.video_rows) - want_v.numpy()).max() < 1e-6
    if want_a is None:
        assert prepared.audio_rows is None
    else:
        assert np.abs(np.array(prepared.audio_rows) - want_a.numpy()).max() < 1e-6


def test_no_conditions_gives_no_rows():
    prepared = cond.prepare_condition(cond.ConditionPayload(), PATCH)
    assert prepared.video_rows is None and prepared.audio_rows is None


def test_default_noise_is_deterministic_and_seed_dependent():
    a, b, c = cond.default_noise((4, 3), 7), cond.default_noise((4, 3), 7), cond.default_noise((4, 3), 8)
    assert np.array_equal(np.array(a), np.array(b)) and np.abs(np.array(a) - np.array(c)).max() > 0.1


def test_noise_actually_augments_the_rows():
    """Null case: with aug 0.999 the rows differ slightly from the raw patchified latent; with aug 1.0 they equal it."""
    ours, _ = R.payload_pair("keyframe_first", seed=7)
    noisy = cond.prepare_condition(ours, PATCH, noise_fn=R.torch_noise).video_rows
    clean = cond.prepare_condition(
        cond.ConditionPayload(keyframes=ours.keyframes, seed=7, visual_cond_noise_aug=1.0), PATCH, noise_fn=R.torch_noise
    ).video_rows
    delta = float(mx.max(mx.abs(noisy - clean)).item())
    assert 1e-5 < delta < 0.1


def test_same_stream_restarts_for_every_condition():
    ours, _ = R.payload_pair("keyframes_first_last", seed=7)
    calls = []
    cond.prepare_condition(ours, PATCH, noise_fn=lambda shape, seed: calls.append((tuple(shape), seed)) or R.torch_noise(shape, seed))
    assert [s for _, s in calls] == [7, 7]  # every visual condition draws from the same seed
```

- [ ] **Step 3: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_condition.py -q`
Expected: FAIL (`ModuleNotFoundError: ... condition`).

- [ ] **Step 4: Implémenter**

`apple_silicon_nodes/native/minimax_h3/condition.py` :

```python
"""Condition inputs for MiniMax H3's packed DiT (fl2va keyframes, ref2va
references): the payload types, and the once-per-run preparation of the
condition rows (patchify + noise augmentation).

Ported from `comfy/ldm/minimax/model.py::MiniMaxH3Model._cond_video_rows` /
`_cond_audio_rows`. The augmented anchor is `aug * z + (1 - aug) * noise`
(aug 0.999 for video, 1.0 = clean for audio), and every condition restarts
the SAME noise stream (audio at seed + 1). The noise is identical at every
sampling step, so `prepare_condition` draws it once per run. The stream
cannot match torch's; `noise_fn` lets tests inject the reference's draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import numpy as np

from .patchify import pack_audio, patchify_video

VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


@dataclass(frozen=True)
class KeyframeCond:
    """An fl2va anchor: `latent` `[1, C, T, h, w]` and/or `audio_latent`
    `[1, A, 2, T]` (normalized), pinned at pixel frame `resolved_frame_index`."""

    resolved_frame_index: int
    latent: mx.array | None = None
    audio_latent: mx.array | None = None


@dataclass(frozen=True)
class RefBlock:
    """A ref2va reference. `kind`: "image" | "video" | "video_audio" | "audio".
    `latent_t/h/w` are the video latent dims (h, w before the 2x2 patching);
    `ref_audio_t` the reference audio length in latent frames."""

    kind: str
    latent: mx.array | None = None
    audio_latent: mx.array | None = None
    latent_t: int = 0
    latent_h: int = 0
    latent_w: int = 0
    ref_audio_t: int = 0


@dataclass(frozen=True)
class ConditionPayload:
    text_token_tags: np.ndarray | None = None  # [text_len], 0 = vision row (video modality), 1 = text
    keyframes: tuple[KeyframeCond, ...] = ()
    refs: tuple[RefBlock, ...] = ()
    seed: int = 0
    visual_cond_noise_aug: float = VISUAL_COND_TIMESTEP
    audio_cond_noise_aug: float = AUDIO_COND_TIMESTEP


@dataclass(frozen=True)
class PreparedCondition:
    payload: ConditionPayload
    video_rows: mx.array | None  # [Nc, patch_dim], keyframes' rows then refs' rows
    audio_rows: mx.array | None  # [Na, audio_dim]


def default_noise(shape: tuple[int, ...], seed: int) -> mx.array:
    return mx.random.normal(tuple(shape), key=mx.random.key(int(seed)))


def _augment(rows: mx.array, aug: float, seed: int, noise_fn: Callable) -> mx.array:
    if aug >= 1.0:
        return rows
    noise = noise_fn(tuple(rows.shape), seed)
    return aug * rows + (1.0 - aug) * noise


def prepare_condition(
    payload: ConditionPayload,
    patch_size: tuple[int, int, int],
    noise_fn: Callable[[tuple[int, ...], int], mx.array] = default_noise,
) -> PreparedCondition:
    video = [k.latent for k in payload.keyframes if k.latent is not None]
    video += [r.latent for r in payload.refs if r.latent is not None]
    audio = [k.audio_latent for k in payload.keyframes if k.audio_latent is not None]
    audio += [r.audio_latent for r in payload.refs if r.audio_latent is not None]

    video_rows = [
        _augment(patchify_video(z.astype(mx.float32), patch_size), payload.visual_cond_noise_aug, payload.seed, noise_fn)
        for z in video
    ]
    audio_rows = [
        _augment(pack_audio(z.astype(mx.float32)), payload.audio_cond_noise_aug, payload.seed + 1, noise_fn)
        for z in audio
    ]
    return PreparedCondition(
        payload=payload,
        video_rows=mx.concatenate(video_rows, axis=0) if video_rows else None,
        audio_rows=mx.concatenate(audio_rows, axis=0) if audio_rows else None,
    )
```

- [ ] **Step 5: Vérifier le succès**

Run: `uv run pytest tests/native/minimax_h3/test_condition.py -q -rs`
Expected: PASS, 0 ignoré (si ignoré : dire pourquoi, un test ignoré n'est pas une preuve). Preuve par mutation : remplacer temporairement `seed + 1` par `seed` dans le calcul du bruit audio, montrer que `test_rows_match_comfyui[keyframe_with_audio]` échoue, restaurer.

- [ ] **Step 6: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/condition.py tests/support/minimax_h3_dit_reference.py tests/native/minimax_h3/test_condition.py
git commit -m "feat: add MiniMax H3 condition payload types and augmented condition rows"
```

---

### Task 2: `PackedLayout` avec segments de condition

**Files:**
- Modify: `apple_silicon_nodes/native/minimax_h3/layout.py` (fonction `ref_t_span` ; `PackedLayout`)
- Test: `tests/native/minimax_h3/test_layout_cond.py`

**Interfaces:**
- Consumes: `KeyframeCond`, `RefBlock` (tâche 1, accès par attributs) ; helpers existants `frame_grid`, `video_t_spans`, `video_grid`, `audio_grid`, `FRAME_RESCALE` de `layout.py`.
- Produces: `ref_t_span(blk) -> float` ; `PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t, keyframes=(), refs=())` avec `segments: list[tuple[int, int, str]]` (kinds : `text`, `cond`, `cond_audio`, `ref_img`, `ref_audio`, `audio`, `video`, dans cet ordre de séquence, audio puis vidéo cibles en dernier), `position_ids: mx.array` `[S, 3]` float64, `seq_len`, `signature`. Sans `keyframes`/`refs` le résultat est identique à l'actuel (`test_layout.py` passe sans modification).

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/native/minimax_h3/test_layout_cond.py` :

```python
"""PackedLayout with keyframe/reference segments vs the real ComfyUI PackedLayout."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support import minimax_h3_dit_reference as R
from support.comfyui_reference_loader import load_real_comfy_minimax_model
from support.minimax_h3_module_loader import load_native_module

layout_mod = load_native_module("minimax_h3.layout")

TARGET = dict(latent_t=2, latent_h=4, latent_w=4, audio_t=3)


@pytest.mark.parametrize("name", list(R.CASES))
def test_layout_matches_comfyui(name):
    mm = load_real_comfy_minimax_model()
    ours, theirs = R.payload_pair(name)
    ref = mm.PackedLayout(
        R.TEXT_LEN, TARGET["latent_t"], TARGET["latent_h"], TARGET["latent_w"], TARGET["audio_t"],
        keyframes=theirs.get("keyframes"), refs=theirs.get("refs"),
    )
    mine = layout_mod.PackedLayout(
        R.TEXT_LEN, TARGET["latent_t"], TARGET["latent_h"], TARGET["latent_w"], TARGET["audio_t"],
        keyframes=ours.keyframes, refs=ours.refs,
    )
    assert mine.seq_len == ref.seq_len
    assert mine.segments == [(a, b, k) for a, b, k in ref.segments]
    assert np.abs(np.array(mine.position_ids) - ref.position_ids.numpy()).max() < 1e-9


def test_target_streams_stay_last():
    ours, _ = R.payload_pair("combined")
    mine = layout_mod.PackedLayout(R.TEXT_LEN, 2, 4, 4, 3, keyframes=ours.keyframes, refs=ours.refs)
    kinds = [k for _, _, k in mine.segments]
    assert kinds[:1] == ["text"] and kinds[-2:] == ["audio", "video"]
    assert set(kinds) >= {"cond", "cond_audio", "ref_img", "ref_audio"}


def test_ref_t_span():
    RB = load_native_module("minimax_h3.condition").RefBlock
    assert layout_mod.ref_t_span(RB(kind="image")) == 1.0
    assert layout_mod.ref_t_span(RB(kind="audio", ref_audio_t=4)) == 4.0
    video = layout_mod.ref_t_span(RB(kind="video_audio", latent_t=2, ref_audio_t=3))
    assert video == max(3.0, sum(layout_mod.video_t_spans(2)))
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_layout_cond.py -q`
Expected: FAIL (`TypeError: ... unexpected keyword 'keyframes'`).

- [ ] **Step 3: Implémenter**

Dans `apple_silicon_nodes/native/minimax_h3/layout.py`, mettre à jour l'en-tête du module (retirer la phrase « restricted to the minimal t2va path ... deliberately not accepted as parameters ») en la remplaçant par : `Supports keyframe (fl2va) and reference (ref2va) condition segments; see PackedLayout. No denoise masks.` puis remplacer la classe `PackedLayout` par :

```python
def ref_t_span(blk) -> float:
    """Time-axis span a reference block occupies ahead of the target streams
    (`comfy/ldm/minimax/model.py::_ref_t_span`)."""
    if blk.kind == "image":
        return 1.0
    if blk.kind == "audio":
        return float(blk.ref_audio_t)
    if blk.kind in ("video", "video_audio"):
        return max(float(blk.ref_audio_t), sum(video_t_spans(blk.latent_t)))
    return 0.0


class PackedLayout:
    """Static packed-sequence structure for one shape/conditioning signature:
    `[text | (keyframe cond rows) | (reference rows) | audio | video]`.

    Ported from `comfy/ldm/minimax/model.py::PackedLayout`. `keyframes` and
    `refs` are the objects of `condition.py` (attribute access). Segment
    kinds: text, cond, cond_audio, ref_img, ref_audio, audio, video. The
    target audio then video are always the last two segments. Positions are
    float64 and built on the CPU stream (float64 is CPU-only in MLX)."""

    def __init__(
        self,
        text_len: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        keyframes=(),
        refs=(),
    ):
        with mx.stream(mx.cpu):
            frame, w_axis = frame_grid(latent_h, latent_w)
            frame_rows = frame.shape[0]
            target_audio_w = (float(w_axis[0].item()), float(w_axis[-1].item()))

            text_pos = mx.zeros((text_len, 3), dtype=mx.float64)
            text_pos[:, 0] = mx.arange(text_len, dtype=mx.float64)
            pos = [text_pos]
            sizes: list[tuple[str, int]] = [("text", text_len)]

            # references pack between text and the targets: the target timeline starts after their spans
            cursor = float(text_len) + sum(ref_t_span(b) for b in refs)

            for kf in keyframes:
                cond_t = cursor + FRAME_RESCALE * kf.resolved_frame_index
                if kf.latent is not None:
                    vt = kf.latent.shape[2]
                    sizes.append(("cond", vt * frame_rows))
                    pos.append(video_grid(vt, frame, cond_t))
                if kf.audio_latent is not None:
                    rt = kf.audio_latent.shape[-1]
                    sizes.append(("cond_audio", rt * 2))
                    pos.append(audio_grid(cond_t, rt, *target_audio_w))

            ref_cursor = float(text_len)
            for blk in refs:
                if blk.kind == "image":
                    r_frame, _ = frame_grid(blk.latent_h, blk.latent_w)
                    n = r_frame.shape[0]
                    g = mx.concatenate([mx.full((n, 1), ref_cursor, dtype=mx.float64), r_frame], axis=-1)
                    sizes.append(("ref_img", n))
                    pos.append(g)
                    ref_cursor += 1.0
                elif blk.kind == "audio":
                    rt = blk.ref_audio_t
                    if rt > 0:
                        sizes.append(("ref_audio", rt * 2))
                        pos.append(audio_grid(ref_cursor, rt, *target_audio_w))
                    ref_cursor += float(rt)
                elif blk.kind in ("video", "video_audio"):
                    # the block's audio rows pack immediately before its video rows, sharing the cursor origin
                    rt, vt = blk.ref_audio_t, blk.latent_t
                    r_frame, r_w_axis = frame_grid(blk.latent_h, blk.latent_w)
                    if rt > 0:
                        sizes.append(("ref_audio", rt * 2))
                        pos.append(audio_grid(ref_cursor, rt, float(r_w_axis[0].item()), float(r_w_axis[-1].item())))
                    sizes.append(("ref_img", vt * r_frame.shape[0]))
                    pos.append(video_grid(vt, r_frame, ref_cursor))
                    ref_cursor += max(float(rt), sum(video_t_spans(vt)))

            sizes.append(("audio", audio_t * 2))
            pos.append(audio_grid(cursor, audio_t, *target_audio_w))
            sizes.append(("video", latent_t * frame_rows))
            pos.append(video_grid(latent_t, frame, cursor))

            self.position_ids = mx.concatenate(pos, axis=0)

        row = 0
        segments: list[tuple[int, int, str]] = []
        for kind, n in sizes:
            segments.append((row, row + n, kind))
            row += n
        self.segments = segments
        self.seq_len = row
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/native/minimax_h3/test_layout_cond.py tests/native/minimax_h3/test_layout.py -q -rs`
Expected: PASS, 0 ignoré (`test_layout.py` inchangé passe : le t2v est identique). Preuve par mutation : dans le curseur d'une référence image, remplacer `ref_cursor += 1.0` par `ref_cursor += 2.0`, montrer que `test_layout_matches_comfyui[combined]` échoue, restaurer.

- [ ] **Step 5: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/layout.py tests/native/minimax_h3/test_layout_cond.py
git commit -m "feat: extend the MiniMax H3 packed layout with keyframe and reference segments"
```

---

### Task 3: Forward du DiT avec lignes de condition et parité de bout en bout

**Files:**
- Modify: `apple_silicon_nodes/native/minimax_h3/model.py` (`build_mod_segments` ; `MiniMaxH3Model.__call__`)
- Test: `tests/native/minimax_h3/test_full_model_cond.py`

**Interfaces:**
- Consumes: `PackedLayout` (tâche 2), `PreparedCondition`, `VISUAL_COND_TIMESTEP`, `AUDIO_COND_TIMESTEP` (tâche 1), `R.build_reference_model`, `R.ours_from_reference`, `R.payload_pair`, `R.torch_noise` (tâche 1).
- Produces: `build_mod_segments(segments, t_video, t_audio, *, vis_aug: float = 0.999, aud_aug: float = 1.0, text_tags: np.ndarray | None = None) -> tuple[list[tuple[int, int, int]], list[float]]` (appel positionnel existant inchangé) ; `MiniMaxH3Model.__call__(video_latent, audio_latent, context, sigma_v, cond: PreparedCondition | None = None) -> tuple[mx.array, mx.array]`. Avec `cond=None` la sortie est identique à l'actuelle.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/native/minimax_h3/test_full_model_cond.py` :

```python
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


def _target(seed=3):
    rng = np.random.default_rng(seed)
    video = rng.standard_normal((1, R.TINY["latents_dim"], LAT_T, LAT_H, LAT_W)).astype(np.float32)
    audio = rng.standard_normal((1, R.TINY["audio_latents_dim"], 2, AUDIO_T)).astype(np.float32)
    context = rng.standard_normal((R.TEXT_LEN, R.TINY["text_dim"])).astype(np.float32)
    return video, audio, context


def _reference_forward(ref, theirs, video, audio, context):
    import torch

    with torch.no_grad():
        out = ref._forward(
            [torch.from_numpy(video), torch.from_numpy(audio)],
            torch.tensor([SIGMA * 1000.0]), torch.from_numpy(context)[None],
            transformer_options={}, minimax_payload=theirs,
        )
    return out[0].numpy(), out[1].numpy()


def _our_forward(model, ours, video, audio, context):
    prepared = cond_mod.prepare_condition(ours, model.config.patch_size, noise_fn=R.torch_noise)
    with mx.stream(mx.cpu):
        v, a = model(mx.array(video), mx.array(audio), mx.array(context), sigma_v=SIGMA, cond=prepared)
        mx.eval(v, a)
    return np.array(v), np.array(a)


@pytest.mark.parametrize("name", list(R.CASES))
def test_forward_matches_comfyui(name):
    ref = R.build_reference_model()
    model = R.ours_from_reference(ref)
    ours, theirs = R.payload_pair(name)
    video, audio, context = _target()
    want_v, want_a = _reference_forward(ref, theirs, video, audio, context)
    got_v, got_a = _our_forward(model, ours, video, audio, context)
    assert got_v.shape == want_v.shape and got_a.shape == want_a.shape
    assert np.abs(got_v - want_v).max() < 1e-4
    assert np.abs(got_a - want_a).max() < 1e-4


def test_conditions_change_the_output():
    """Null case: the parity metric must see a condition (else it measures nothing)."""
    ref = R.build_reference_model()
    model = R.ours_from_reference(ref)
    video, audio, context = _target()
    plain = _our_forward(model, R.payload_pair("t2v")[0], video, audio, context)
    with_kf = _our_forward(model, R.payload_pair("keyframe_first")[0], video, audio, context)
    with_ref = _our_forward(model, R.payload_pair("ref_image")[0], video, audio, context)
    assert np.abs(plain[0] - with_kf[0]).max() > 1e-3 and np.abs(plain[0] - with_ref[0]).max() > 1e-3


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
    assert diff < 2e-3 * scale
    plain = _our_forward(model, R.payload_pair("t2v")[0], video, audio, context)
    assert np.abs(np.array(gpu[0]) - plain[0]).max() > 10 * diff  # null: the bound can fail
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_full_model_cond.py -q`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'cond'`). Si l'échec vient plutôt de `build_reference_model` ou de `ref._forward` (le modèle ComfyUI n'a pas été exécuté sur CPU ici), le corriger d'abord dans `tests/support/minimax_h3_dit_reference.py` : si `_forward` échoue uniquement sur des aides d'exécution (`comfy.model_prefetch`, `comfy.model_management`, attention optimisée), les neutraliser par un monkeypatch dans le harnais de test, **sans jamais toucher aux mathématiques du modèle**.

- [ ] **Step 3: Implémenter `build_mod_segments`**

Dans `model.py`, ajouter en tête de fichier `import numpy as np` s'il manque, et remplacer `build_mod_segments` par (garder la docstring d'origine pour la première partie, ajouter la mention des segments de condition) :

```python
_SEG_TAG = {"video": 0, "text": 1, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}


def build_mod_segments(
    segments: list[tuple[int, int, str]],
    t_video: float,
    t_audio: float,
    *,
    vis_aug: float = 0.999,
    aud_aug: float = 1.0,
    text_tags: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, int]], list[float]]:
    """Assigns each packed-sequence segment a modulation row `t_row[t] * 3 +
    tag` and returns the sorted unique timestep values those rows index into
    (feed to `curve_time_embedding`/`AdalnProj` as `t_emb`'s `M` rows).

    Timestep per kind (`comfy/ldm/minimax/model.py::_forward`): text and
    video follow `t_video`, audio `t_audio`; keyframe/reference video rows
    are pinned at `max(t_video, vis_aug)` and their audio rows at
    `max(t_audio, aud_aug)`. Tags: video 0, text 1, audio 2 (condition
    rows carry their modality's tag). `text_tags` (`[text_len]`, 0 = vision
    row) splits the text segment into runs, vision rows taking tag 0.
    No denoise masks."""
    seg_t = {
        "text": t_video, "video": t_video, "audio": t_audio,
        "cond": max(t_video, vis_aug), "ref_img": max(t_video, vis_aug),
        "cond_audio": max(t_audio, aud_aug), "ref_audio": max(t_audio, aud_aug),
    }
    unique_t = sorted({t_video, t_audio} | {seg_t[k] for _, _, k in segments})
    t_row = {t: i for i, t in enumerate(unique_t)}
    mod_segments: list[tuple[int, int, int]] = []
    for a, b, kind in segments:
        base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = np.asarray(text_tags).reshape(-1).tolist()
            if len(tags) != b - a:
                raise ValueError(f"ASDX: text_token_tags has {len(tags)} entries, the text segment has {b - a} rows")
            run = 0
            for i in range(1, len(tags) + 1):
                if i == len(tags) or tags[i] != tags[run]:
                    mod_segments.append((a + run, a + i, base + int(tags[run])))
                    run = i
        else:
            mod_segments.append((a, b, base + _SEG_TAG[kind]))
    return mod_segments, unique_t
```

- [ ] **Step 4: Implémenter le forward**

Dans `model.py`, ajouter les imports `from .condition import AUDIO_COND_TIMESTEP, VISUAL_COND_TIMESTEP, PreparedCondition` ; remplacer `MiniMaxH3Model.__call__` par :

```python
    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        context: mx.array,
        sigma_v: float,
        cond: PreparedCondition | None = None,
    ) -> tuple[mx.array, mx.array]:
        """`video_latent`: `[1, latents_dim, T, H, W]`. `audio_latent`:
        `[1, audio_latents_dim, 2, T_audio]`. `context`: `[L, text_dim]`.
        `sigma_v`: the video stream's flow-matching sigma in `[0, 1]`.
        `cond`: prepared keyframe/reference rows (`condition.prepare_condition`,
        computed once per sampling run); `None` = plain t2va. Returns the
        NEGATED velocities of the target streams (the reference's outer
        `forward()` sign flip)."""
        cfg = self.config
        shift_v, shift_a = cfg.sigma_shift_video, cfg.sigma_shift_audio
        t_v = 1.0 - sigma_v
        t_a = 1.0 - time_shift_sigma(sigma_v, shift_v, shift_a)

        payload = cond.payload if cond is not None else None
        text_len = context.shape[0]
        latent_t, lat_h, lat_w = video_latent.shape[2], video_latent.shape[3], video_latent.shape[4]
        audio_t = audio_latent.shape[-1]

        layout = PackedLayout(
            text_len, latent_t, lat_h, lat_w, audio_t,
            keyframes=payload.keyframes if payload else (), refs=payload.refs if payload else (),
        )
        mod_segments, unique_t = build_mod_segments(
            layout.segments, t_v, t_a,
            vis_aug=payload.visual_cond_noise_aug if payload else VISUAL_COND_TIMESTEP,
            aud_aug=payload.audio_cond_noise_aug if payload else AUDIO_COND_TIMESTEP,
            text_tags=payload.text_token_tags if payload else None,
        )
        t_emb = curve_time_embedding(self.adaln_t_table, mx.array(unique_t, dtype=mx.float32))

        video_embed = self.video_patch_proj(patchify_video(video_latent, cfg.patch_size))
        audio_embed = self.audio_patch_proj(pack_audio(audio_latent))
        cond_video = self.video_patch_proj(cond.video_rows) if cond is not None and cond.video_rows is not None else None
        cond_audio = self.audio_patch_proj(cond.audio_rows) if cond is not None and cond.audio_rows is not None else None

        text_states = context
        if text_states.shape[-1] != cfg.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states))

        pieces: list[mx.array] = []
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                pieces.append(text_states)
            elif kind in ("cond", "ref_img"):
                pieces.append(cond_video[voff:voff + n])
                voff += n
            elif kind in ("cond_audio", "ref_audio"):
                pieces.append(cond_audio[aoff:aoff + n])
                aoff += n
            elif kind == "audio":
                pieces.append(audio_embed)
            else:
                pieces.append(video_embed)
        h = mx.concatenate(pieces, axis=0)

        angles = _rope_freqs(layout.position_ids, self.rope.inv_freq)
        cos, sin = rope_cos_sin(angles)

        for block in self.blocks:
            h = block(h, t_emb, mod_segments, cos, sin)

        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")
        v, a = self.final_layer(
            h, t_emb, (va, vb, unique_t.index(t_v)), (aa, ab, unique_t.index(t_a))
        )

        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, cfg.latents_dim, cfg.patch_size)
        audio_out = unpack_audio(a)
        return -video_out, -audio_out
```

Vérifier au préalable que les noms importés (`patchify_video`, `pack_audio`, `unpatchify_video`, `unpack_audio`, `_rope_freqs`, `rope_cos_sin`, `PackedLayout`, `curve_time_embedding`) existent déjà dans `model.py` avec ces noms (ils sont utilisés par l'implémentation actuelle) ; ne rien renommer.

- [ ] **Step 5: Vérifier le succès**

Run: `uv run pytest tests/native/minimax_h3/test_full_model_cond.py tests/native/minimax_h3/test_full_model.py tests/native/minimax_h3/test_layout.py tests/native/minimax_h3/test_dit_block.py -q -rs`
Expected: PASS, 0 ignoré. **Si `test_forward_matches_comfyui[t2v]` échoue** : c'est potentiellement un vrai écart du t2v actuel (aucun oracle n'existait). Ne pas assouplir : diagnostiquer par bissection contre `ref._forward` (embeddings, positions, t_emb, chaque bloc, couche finale), rapporter l'écart et la cause avant toute correction. Preuves par mutation (une par ligne, montrer l'échec puis restaurer, `git diff` du source inchangé) : (a) `max(t_video, vis_aug)` -> `t_video` dans `seg_t["cond"]` ; (b) ignorer `text_tags` dans `build_mod_segments` ; (c) ne pas ajouter `cond_audio` dans `pieces`.

- [ ] **Step 6: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/model.py tests/native/minimax_h3/test_full_model_cond.py
git commit -m "feat: give the MiniMax H3 DiT keyframe and reference condition rows with ComfyUI parity"
```

---

### Task 4: Le sampler et le nœud fournissent la condition

**Files:**
- Modify: `apple_silicon_nodes/native/minimax_h3/sampling.py` (`run_minimax_h3_sampling`)
- Modify: `apple_silicon_nodes/minimax_h3_nodes.py` (`ASDX_MiniMaxH3Sampler.execute` ; nouvelle fonction module `payload_from_conditioning`)
- Test: `tests/native/minimax_h3/test_sampling.py` (ajouts), `tests/test_minimax_h3_loaders.py` (ajouts)

**Interfaces:**
- Consumes: `ConditionPayload`, `prepare_condition`, `default_noise` (tâche 1) ; `MiniMaxH3Model.__call__(..., cond=)` (tâche 3).
- Produces: `run_minimax_h3_sampling(model, video_latent, audio_latent, context, steps, payload: ConditionPayload | None = None, noise_fn=None) -> tuple[mx.array, mx.array]` : quand `payload` est fourni, `prepare_condition(payload, model.config.patch_size, noise_fn or default_noise)` est appelé **une seule fois** et le résultat passé en `cond=` à chaque appel du modèle ; sans `payload` l'appel du modèle reste `model(video, audio, context, sigma_v=...)` (sans `cond`). `payload_from_conditioning(conditioning: dict, seed: int) -> ConditionPayload | None` (module `minimax_h3_nodes`) : lit `token_tags` (mx.array ou None), `keyframes` (liste de `KeyframeCond`), `refs` (liste de `RefBlock`) du dict de conditioning ; renvoie `None` si aucun des trois n'est présent ; sinon un `ConditionPayload` avec `seed`.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à `tests/native/minimax_h3/test_sampling.py` (réutiliser ses imports et son schéma de modèle factice ; lire d'abord le fichier pour en suivre le style) :

```python
def test_payload_is_prepared_once_and_passed_every_step(monkeypatch):
    cond_mod = load_native_module("minimax_h3.condition")
    seen = []

    class FakeModel:
        def __init__(self, cfg):
            self.config = cfg

        def __call__(self, video, audio, context, sigma_v, cond=None):
            seen.append(cond)
            return mx.zeros_like(video), mx.zeros_like(audio)

    cfg = _tiny_config()  # helper already defined in this file (adapt the name if it differs)
    prepare_calls = []
    real_prepare = sampling_mod.prepare_condition

    def counting_prepare(payload, patch_size, noise_fn=cond_mod.default_noise):
        prepare_calls.append(payload)
        return real_prepare(payload, patch_size, noise_fn)

    monkeypatch.setattr(sampling_mod, "prepare_condition", counting_prepare)
    payload = cond_mod.ConditionPayload(seed=3)
    video = mx.zeros((1, cfg.latents_dim, 1, 4, 4))
    audio = mx.zeros((1, cfg.audio_latents_dim, 2, 2))
    sampling_mod.run_minimax_h3_sampling(FakeModel(cfg), video, audio, mx.zeros((3, cfg.text_dim)), 3, payload=payload)
    assert len(prepare_calls) == 1
    assert len(seen) == 3 and all(c is seen[0] and c is not None for c in seen)


def test_no_payload_keeps_the_old_model_call():
    class OldStyleModel:  # no `cond` parameter at all
        def __init__(self, cfg):
            self.config = cfg

        def __call__(self, video, audio, context, sigma_v):
            return mx.zeros_like(video), mx.zeros_like(audio)

    cfg = _tiny_config()
    video = mx.zeros((1, cfg.latents_dim, 1, 4, 4))
    audio = mx.zeros((1, cfg.audio_latents_dim, 2, 2))
    sampling_mod.run_minimax_h3_sampling(OldStyleModel(cfg), video, audio, mx.zeros((3, cfg.text_dim)), 2)
```

Ajouter à `tests/test_minimax_h3_loaders.py` :

```python
def test_payload_from_conditioning_reads_tags_keyframes_and_refs():
    import numpy as np
    import mlx.core as mx

    from apple_silicon_nodes.native.minimax_h3.condition import KeyframeCond, RefBlock

    assert nodes_module.payload_from_conditioning({"type": "minimax_h3", "hidden_states": mx.zeros((2, 3))}, 5) is None
    kf, ref = KeyframeCond(0, mx.zeros((1, 4, 1, 4, 4))), RefBlock(kind="image", latent_t=1, latent_h=4, latent_w=4)
    payload = nodes_module.payload_from_conditioning(
        {"type": "minimax_h3", "token_tags": mx.array([1, 0, 1]), "keyframes": [kf], "refs": [ref]}, 11
    )
    assert payload.seed == 11 and payload.keyframes == (kf,) and payload.refs == (ref,)
    assert payload.text_token_tags.tolist() == [1, 0, 1]
```

(Adapter l'import de `KeyframeCond`/`RefBlock` au mécanisme d'import du fichier de tests de nœuds : ce fichier charge le module par `load_node_module` ; si l'import direct ne fonctionne pas, utiliser `load_native_module("minimax_h3.condition")` comme ailleurs.)

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/native/minimax_h3/test_sampling.py tests/test_minimax_h3_loaders.py -q`
Expected: FAIL (`payload` inconnu, `payload_from_conditioning` absent).

- [ ] **Step 3: Implémenter le sampler**

Dans `sampling.py`, ajouter `from .condition import ConditionPayload, default_noise, prepare_condition` aux imports, changer la signature :

```python
def run_minimax_h3_sampling(
    model: MiniMaxH3Model,
    video_latent: mx.array,
    audio_latent: mx.array,
    context: mx.array,
    steps: int,
    payload: ConditionPayload | None = None,
    noise_fn=None,
) -> tuple[mx.array, mx.array]:
```

compléter la docstring (« `payload`: keyframe/reference conditions, prepared once for the whole run (the augmentation noise is identical at every step) »), ajouter juste avant la boucle :

```python
    cond = (
        prepare_condition(payload, cfg.patch_size, noise_fn or default_noise) if payload is not None else None
    )
```

et remplacer l'appel du modèle par :

```python
        if cond is None:
            video_v, audio_v = model(video, audio, context, sigma_v=sigma_v)
        else:
            video_v, audio_v = model(video, audio, context, sigma_v=sigma_v, cond=cond)
```

- [ ] **Step 4: Implémenter le nœud**

Dans `minimax_h3_nodes.py`, ajouter la fonction module (près de `encode_minimax_h3_prompt`) :

```python
def payload_from_conditioning(conditioning: dict, seed: int):
    """Build the DiT's `ConditionPayload` from a MiniMax H3 conditioning dict
    (`token_tags`, `keyframes`, `refs` as written by the encode/i2v/ref nodes).
    Returns None when the dict carries none of them (plain t2v)."""
    import numpy as np

    from .native.minimax_h3.condition import ConditionPayload

    tags = conditioning.get("token_tags")
    keyframes = tuple(conditioning.get("keyframes") or ())
    refs = tuple(conditioning.get("refs") or ())
    if tags is None and not keyframes and not refs:
        return None
    return ConditionPayload(
        text_token_tags=None if tags is None else np.array(tags).astype(np.int64),
        keyframes=keyframes, refs=refs, seed=int(seed),
    )
```

et, dans `ASDX_MiniMaxH3Sampler.execute`, remplacer l'appel `run_minimax_h3_sampling(...)` par :

```python
        video_out, audio_out = run_minimax_h3_sampling(
            model["transformer"], video_noise, audio_noise, conditioning["hidden_states"], steps,
            payload=payload_from_conditioning(conditioning, seed),
        )
```

- [ ] **Step 5: Vérifier le succès**

Run: `uv run pytest tests/native/minimax_h3/test_sampling.py tests/test_minimax_h3_loaders.py -q -rs` puis `uv run pytest tests -q`
Expected: PASS (seul échec permis : `tests/support/test_krea2_module_loader.py::test_plain_import_of_model_fails_outside_comfyui`, préexistant). Les tests de sampling existants passent sans modification d'attentes.

- [ ] **Step 6: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add apple_silicon_nodes/native/minimax_h3/sampling.py apple_silicon_nodes/minimax_h3_nodes.py tests/native/minimax_h3/test_sampling.py tests/test_minimax_h3_loaders.py
git commit -m "feat: run the MiniMax H3 sampler with keyframe and reference conditions"
```

---

### Task 5: Vérification sur les vrais fichiers (fl2va et ref2va)

**Files:**
- Test: `tests/native/minimax_h3/test_real_dit_cond.py` (créer)

**Interfaces:**
- Consumes: `load_minimax_h3_checkpoint(path, dtype)` (existant), `ConditionPayload`, `KeyframeCond`, `RefBlock`, `prepare_condition` (tâche 1), `MiniMaxH3Model.__call__(cond=)` (tâche 3).
- Produces: un test réel (`ASDX_FULL_GGUF_TEST=1`) : pour chacun des DiT `minimax_h3_fl2va_pruned_int8_convrot.safetensors` et `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (dossier `/Volumes/X10Pro/Images/models/diffusion_models/MiniMax H3/base model/`), un forward sur de petites formes réelles avec un keyframe puis avec une référence image : sortie finie, forme correcte, et la condition change la sortie.

- [ ] **Step 1: Écrire le test**

```python
"""Real fl2va / ref2va DiT weights with keyframe and reference conditions (small shapes)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

wm = load_native_module("minimax_h3.weight_map")
cond = load_native_module("minimax_h3.condition")

BASE = Path("/Volumes/X10Pro/Images/models/diffusion_models/MiniMax H3/base model")
FILES = ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"]


@pytest.mark.skipif(os.environ.get("ASDX_FULL_GGUF_TEST") != "1", reason="loads real 20GB DiTs; set ASDX_FULL_GGUF_TEST=1")
@pytest.mark.parametrize("fname", FILES)
def test_real_dit_runs_with_conditions(fname):
    path = BASE / fname
    if not path.exists():
        pytest.skip(f"{fname} not present")
    model = wm.load_minimax_h3_checkpoint(path, dtype="float16")
    c = model.config
    rng = np.random.default_rng(0)
    video = mx.array(rng.standard_normal((1, c.latents_dim, 2, 8, 8)).astype(np.float32))
    audio = mx.array(rng.standard_normal((1, c.audio_latents_dim, 2, 4)).astype(np.float32))
    context = mx.array(rng.standard_normal((6, c.text_dim)).astype(np.float32))

    def run(payload):
        prepared = cond.prepare_condition(payload, c.patch_size) if payload is not None else None
        kw = {} if prepared is None else {"cond": prepared}
        v, a = model(video, audio, context, sigma_v=0.5, **kw)
        mx.eval(v, a)
        return np.array(v.astype(mx.float32)), np.array(a.astype(mx.float32))

    plain = run(None)
    kf = cond.ConditionPayload(keyframes=(cond.KeyframeCond(0, mx.array(rng.standard_normal((1, c.latents_dim, 1, 8, 8)).astype(np.float32))),), seed=1)
    ref = cond.ConditionPayload(refs=(cond.RefBlock(kind="image", latent=mx.array(rng.standard_normal((1, c.latents_dim, 1, 8, 8)).astype(np.float32)), latent_t=1, latent_h=8, latent_w=8),), seed=1)
    with_kf, with_ref = run(kf), run(ref)
    for out in (plain, with_kf, with_ref):
        assert out[0].shape == tuple(video.shape) and np.isfinite(out[0]).all() and np.isfinite(out[1]).all()
    print(f"[real {fname}] delta keyframe {np.abs(plain[0] - with_kf[0]).max():.3f}, delta ref {np.abs(plain[0] - with_ref[0]).max():.3f}, peak {mx.get_peak_memory() / 1e9:.2f} GB")
    assert np.abs(plain[0] - with_kf[0]).max() > 1e-3 and np.abs(plain[0] - with_ref[0]).max() > 1e-3
```

- [ ] **Step 2: Exécuter sur les vrais fichiers (une fois)**

Run: `ASDX_FULL_GGUF_TEST=1 uv run pytest tests/native/minimax_h3/test_real_dit_cond.py -q -rs -s`
Expected: 2 passés. Relever verbatim les lignes `[real ...]` (écarts et pic mémoire). Si un forward échoue sur les vrais poids alors que la parité synthétique passe, ne pas contourner : rapporter l'erreur complète.

- [ ] **Step 3: Commit (sur ordre de l'utilisateur uniquement)**

```bash
git add tests/native/minimax_h3/test_real_dit_cond.py
git commit -m "test: run real fl2va and ref2va DiT weights with keyframe and reference conditions"
```

---

## Auto-revue du plan

- **Couverture de la spec, brique 3** : segments `cond`/`cond_audio`/`ref_img`/`ref_audio` et positions (T2), lignes de condition avec bruit d'augmentation et timestep 0,999 / 1,0 (T1, T3), timestep par segment `max(t, aug)` et tags de modulation (T3), assemblage par tranches et sortie sur les seules lignes cibles (T3), charge utile par le dict de conditioning (`keyframes`, `refs`, `token_tags`, `seed`, T4), acceptation « forward MLX contre `MiniMaxH3Model` de ComfyUI avec et sans keyframes/refs » (T3, y compris le t2v).
- **Placeholders** : aucun. Les deux incertitudes assumées sont nommées avec la conduite à tenir : l'exécution du `_forward` ComfyUI sur CPU (T3 étape 2) et un éventuel écart du t2v actuel (T3 étape 5).
- **Cohérence des types** : `KeyframeCond`, `RefBlock`, `ConditionPayload`, `PreparedCondition`, `prepare_condition(payload, patch_size, noise_fn)`, `PackedLayout(..., keyframes, refs)`, `build_mod_segments(..., vis_aug, aud_aug, text_tags)`, `model(..., cond=)`, `run_minimax_h3_sampling(..., payload, noise_fn)`, `payload_from_conditioning(conditioning, seed)` sont utilisés à l'identique partout.
- **Hors périmètre** : encodage VAE des keyframes et références (aller-retour fp16, posterior), nœuds i2v/ref, gate mémoire compté en lignes (brique 4).
