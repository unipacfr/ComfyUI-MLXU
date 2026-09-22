# Qwen Image 2.1 brique 3 : intégration (loader / nœuds text encoder / bridge)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un pipeline T2I Qwen Image 2.1 utilisable dans un vrai workflow ComfyUI : détection
de famille dans les loaders génériques (bf16 + GGUF), deux nœuds dédiés pour le text encoder
MLX natif, fonctions de pont conditioning/latent/noise, VAE générique (déjà compatible,
vérifié).

**Architecture:** Modifie 3 fichiers partagés (`loader.py`, `bridge.py`, `__init__.py`) et crée
`apple_silicon_nodes/qwen_image21_nodes.py` (pattern `minimax_h3_nodes.py`).

**Tech Stack:** MLX, PyTorch (pont), `comfy.text_encoders.qwen_image21` (tokenizer réel, sans
poids), pytest + `comfy_stub` pour les tests hors ComfyUI, `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-22-qwen-image-21-brick3-integration-design.md`.
Référence : `apple_silicon_nodes/loader.py` (`_detect_model_type`, `_load_transformer_for_type`,
`_MODEL_TYPE_CAPABILITY`), `apple_silicon_nodes/bridge.py` (`conditioning_flux2_to_mlx`,
`mlx_to_comfy_latent_sdxl`, `prepare_noise_from_latent_sdxl`), `apple_silicon_nodes/
minimax_h3_nodes.py` (`ASDX_MiniMaxH3TextEncoderLoader`, `encode_minimax_h3_prompt`,
`_register_gguf_extension`).

## Global Constraints

- Toujours `uv run pytest` / `uv run python` (jamais `python` nu).
- Style du dépôt : Python 3.10+, annotations complètes, docstrings en anglais, pas d'emoji ni
  de tiret cadratin.
- Fail-closed : clé/forme/dtype inattendu -> exception.
- **Le DiT Qwen Image 2.1 opère en NCHW directement** (`QwenImage21Transformer2DModel.__call__`,
  `x: [B,C,H,W]`), pas en NHWC comme SDXL — la conversion latent au pont n'a besoin d'aucune
  transposition de layout, seulement `mx.array`<->`torch.Tensor`.
- Le text encoder retourne `[S, hidden_size]` (pas de dimension batch, convention brique 1) —
  le pont doit ajouter la dimension batch (`[1, S, hidden_size]`) avant de la passer au DiT.
- **Aucun commit automatique** (règle utilisateur) : chaque étape « Commit » signifie « préparer
  le message et attendre l'ordre explicite de l'utilisateur ». Pas de ligne d'attribution.

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `apple_silicon_nodes/loader.py` (modifier) | `_QWEN_IMAGE21_HINTS`, repli structurel, dispatch bf16/GGUF dans `_load_transformer_for_type`, entrée `_MODEL_TYPE_CAPABILITY` |
| `apple_silicon_nodes/bridge.py` (modifier) | `QWEN_IMAGE21_LATENT_CHANNELS`/`QWEN_IMAGE21_VAE_DOWNSCALE`, `conditioning_qwen_image21_to_mlx`, `mlx_to_comfy_latent_qwen_image21`, `prepare_noise_from_latent_qwen_image21` |
| `apple_silicon_nodes/qwen_image21_nodes.py` (créer) | `ASDX_QwenImage21TextEncoderLoader`, `ASDX_QwenImage21TextEncode`, `NODE_LIST` |
| `apple_silicon_nodes/__init__.py` (modifier) | enregistrement du nouveau `NODE_LIST` |
| `tests/test_loader_qwen_image21.py` (créer) | détection de famille contre les vrais fichiers |
| `tests/test_bridge_qwen_image21.py` (créer) | fonctions de pont (formes, round-trip) |
| `tests/test_qwen_image21_nodes.py` (créer) | nœuds via `comfy_stub`, encodage réel via l'install ComfyUI locale |

---

### Task 1: Détection de famille + dispatch bf16/GGUF dans `loader.py`

**Files:**
- Modify: `apple_silicon_nodes/loader.py`
- Test: `tests/test_loader_qwen_image21.py`

**Interfaces:**
- Consumes: `load_qwen_image21_dit_checkpoint`, `load_qwen_image21_dit_from_gguf`
  (`native/qwen_image21/weight_map.py`, brique 2).
- Produces: `_detect_model_type(path)` reconnaît `"qwen_image21"` ; `_load_transformer_for_type`
  route bf16 et GGUF vers le bon loader. Consommé par `ASDX_DiffusionLoader`/
  `ASDX_CheckpointLoader` existants (aucune modification de ces classes).

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""Family detection for Qwen Image 2.1 against the real checkpoint files."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
loader_mod = load_node_module("loader")

_BF16_REAL = Path("/Volumes/X10Pro/Images/models/diffusion_models/Qwen 2/base model/qwen_image_2.1_bf16.safetensors")
_GGUF_REAL = Path("/Volumes/X10Pro/Images/models/diffusion_models/Qwen 2/gguf/qwen_image_2.1_Q8.gguf")


def test_qwen_image21_hints_present():
    assert any("qwen_image_2.1" in h or "qwen-image-2.1" in h or "qwen_image21" in h
                for h in loader_mod._QWEN_IMAGE21_HINTS)


@pytest.mark.skipif(not _BF16_REAL.exists(), reason="no local qwen_image_2.1_bf16.safetensors")
def test_detects_real_bf16_checkpoint():
    assert loader_mod._detect_model_type(_BF16_REAL) == "qwen_image21"


def test_structural_fallback_key_does_not_collide_with_other_families():
    # The distinctive key must not appear in any other family's own marker set.
    other_markers = ["input_blocks.", "noise_refiner.", "double_stream_modulation_img.", "txtfusion."]
    assert loader_mod._QWEN_IMAGE21_STRUCTURAL_KEY not in other_markers
    for marker in other_markers:
        assert marker not in loader_mod._QWEN_IMAGE21_STRUCTURAL_KEY
        assert loader_mod._QWEN_IMAGE21_STRUCTURAL_KEY not in marker


def test_capability_entry_present():
    assert "qwen_image21" in loader_mod._MODEL_TYPE_CAPABILITY
```

Run: `uv run pytest tests/test_loader_qwen_image21.py -q` -> FAIL (`AttributeError`).

- [ ] **Step 2: Modifier `loader.py`**

Ajouter après `_FLUX2_HINTS` (ligne ~164) :

```python
_QWEN_IMAGE21_HINTS = {
    "qwen_image_2.1": "qwen_image21",
    "qwen-image-2.1": "qwen_image21",
    "qwen_image21": "qwen_image21",
}

# Distinctive tensor key for the DiT (verified against the real checkpoint header in
# brick 2: `transformer_blocks.0.img_mlp.gate_up.weight`, unique to this family's fused
# SwiGLU MLP naming -- SDXL/Z-Image/Flux2/Krea2 use their own distinct markers, none of
# which contain or are contained in this string).
_QWEN_IMAGE21_STRUCTURAL_KEY = "transformer_blocks.0.img_mlp.gate_up."
```

Dans `_detect_model_type`, ajouter la vérification `_QWEN_IMAGE21_HINTS` à la chaîne de hints
(avant `_FLUX2_HINTS`, ordre arbitraire entre familles structurellement distinctes) :

```python
    if hint_type is None:
        for hint in _QWEN_IMAGE21_HINTS:
            if hint in name:
                hint_type = "qwen_image21"
                break
    if hint_type is None:
        for hint in _FLUX2_HINTS:
```

Dans `_detect_model_type_from_keys`, ajouter avant le `return "dev"` final :

```python
    if any(_QWEN_IMAGE21_STRUCTURAL_KEY in k for k in keys):
        return "qwen_image21"
```

**Note** : `_detect_model_type_from_keys` ouvre le fichier via `safe_open(path, framework="pt")`
— ne fonctionne QUE pour les fichiers safetensors, pas GGUF. Un fichier `.gguf` nommé sans hint
reconnu tombera dans l'exception `except Exception` (safe_open échoue sur un GGUF) et retournera
`"dev"` par défaut, silencieusement faux. **Corriger `_detect_model_type`** pour tester
`path.suffix.lower() == ".gguf"` AVANT le repli structurel, et sauter directement aux hints
`_QWEN_IMAGE21_HINTS` uniquement dans ce cas (aucune autre famille de ce dispatch générique n'a
de variante GGUF à ce jour, donc un `.gguf` sans hint reconnu doit lever une erreur explicite
plutôt que retourner `"dev"` par défaut) :

```python
    if path.suffix.lower() == ".gguf":
        for hint in _QWEN_IMAGE21_HINTS:
            if hint in name:
                return "qwen_image21"
        raise RuntimeError(
            f"ASDX: {path.name} is a .gguf file but its name doesn't match any known "
            "GGUF-supporting family (qwen_image21) -- no other family in this dispatch "
            "supports GGUF yet."
        )
```
(insérer cette branche tout en haut de `_detect_model_type`, avant la boucle `_QWEN_IMAGE21_HINTS`
de l'étape précédente -- fusionner les deux en une seule branche `.gguf` qui vérifie les hints
qwen_image21 et lève sinon.)

Dans `_load_transformer_for_type`, ajouter avant le `else:` final (bloc FLUX.1) :

```python
    elif model_type == "qwen_image21":
        from .native.qwen_image21 import (
            load_qwen_image21_dit_checkpoint,
            load_qwen_image21_dit_from_gguf,
        )
        if path.suffix.lower() == ".gguf":
            transformer = load_qwen_image21_dit_from_gguf(path, dtype=dtype)
        else:
            transformer = load_qwen_image21_dit_checkpoint(path, dtype=dtype)
        return transformer, transformer.config
```

Dans `_MODEL_TYPE_CAPABILITY`, ajouter :

```python
    "qwen_image21": "qwen_image21_base",
```

**Note pour l'implémenteur** : `"qwen_image21_base"` doit correspondre à une entrée réelle
dans `capability.py::_CAPABILITY_DISPATCH` (ou un équivalent) pour que le calibrage mémoire
fonctionne -- lire `apple_silicon_nodes/capability.py` avant d'ajouter cette entrée, et créer le
profil de capacité manquant si nécessaire (dimensionné sur le DiT 7B/14GB bf16 vérifié en
brique 2 -- pas un chiffre à deviner, à dériver des mesures déjà faites : `matched 265/265`,
poids bf16 ~14GB sur disque).

- [ ] **Step 3: Vérifier le succès**

Run: `uv run pytest tests/test_loader_qwen_image21.py -q` -> PASS.

- [ ] **Step 4: Commit**

```bash
git add apple_silicon_nodes/loader.py apple_silicon_nodes/capability.py tests/test_loader_qwen_image21.py
git commit -m "feat: Qwen Image 2.1 family detection + bf16/GGUF dispatch in loader.py"
```

---

### Task 2: Fonctions de pont (`bridge.py`)

**Files:**
- Modify: `apple_silicon_nodes/bridge.py`
- Test: `tests/test_bridge_qwen_image21.py`

**Interfaces:**
- Consumes: rien de nouveau (mx, torch, numpy déjà importés dans `bridge.py`).
- Produces: `conditioning_qwen_image21_to_mlx(conditioning: dict, precision: mx.Dtype) ->
  mx.array` (prend le dict `{"type": "qwen_image21", "hidden_states": mx.array[S,4096], ...}`
  produit par `ASDX_QwenImage21TextEncode`, retourne `[1, S, 4096]`).
  `mlx_to_comfy_latent_qwen_image21(latents: mx.array, template: dict) -> dict`.
  `prepare_noise_from_latent_qwen_image21(latent: dict, seed: int, precision: mx.Dtype) ->
  tuple[mx.array, int, int, tuple[int,int]]`. Consommé par la tâche 3 (nœuds) et le sampler
  générique existant.

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""Bridge functions for Qwen Image 2.1: conditioning dict -> MLX, MLX latent <-> Comfy LATENT."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
bridge_mod = load_node_module("bridge")


def test_conditioning_qwen_image21_to_mlx_adds_batch_dim():
    hidden_states = mx.random.normal((7, 4096))
    conditioning = {"type": "qwen_image21", "hidden_states": hidden_states, "text": "a cat"}
    cond = bridge_mod.conditioning_qwen_image21_to_mlx(conditioning, mx.float16)
    assert cond.shape == (1, 7, 4096)
    assert cond.dtype == mx.float16


def test_conditioning_qwen_image21_to_mlx_rejects_wrong_type():
    with pytest.raises(RuntimeError, match="qwen_image21"):
        bridge_mod.conditioning_qwen_image21_to_mlx({"type": "flux2"}, mx.float16)


def test_mlx_to_comfy_latent_qwen_image21_shape():
    latents = mx.random.normal((1, 64, 8, 8))
    template = {"samples": None}
    out = bridge_mod.mlx_to_comfy_latent_qwen_image21(latents, template)
    assert isinstance(out["samples"], torch.Tensor)
    assert tuple(out["samples"].shape) == (1, 64, 8, 8)


def test_prepare_noise_from_latent_qwen_image21_shape_and_dims():
    samples = torch.zeros((1, 64, 8, 8))
    latent = {"samples": samples}
    noise, height, width, out_shape = bridge_mod.prepare_noise_from_latent_qwen_image21(
        latent, seed=0, precision=mx.float32
    )
    assert noise.shape == (1, 64, 8, 8)
    assert height == 8 * 16 and width == 8 * 16
    assert out_shape == (8, 8)


def test_prepare_noise_from_latent_qwen_image21_rejects_wrong_channels():
    latent = {"samples": torch.zeros((1, 4, 8, 8))}
    with pytest.raises(RuntimeError, match="64-channel"):
        bridge_mod.prepare_noise_from_latent_qwen_image21(latent, seed=0, precision=mx.float32)
```

Run: `uv run pytest tests/test_bridge_qwen_image21.py -q` -> FAIL.

- [ ] **Step 2: Modifier `bridge.py`**

Ajouter près des autres constantes de famille (`FLUX2_LATENT_CHANNELS`, etc.) :

```python
QWEN_IMAGE21_LATENT_CHANNELS = 64
QWEN_IMAGE21_VAE_DOWNSCALE = 16
```

Ajouter les trois fonctions (near `conditioning_flux2_to_mlx`/`mlx_to_comfy_latent_sdxl`/
`prepare_noise_from_latent_sdxl` pour rester groupé par famille, convention du fichier) :

```python
def conditioning_qwen_image21_to_mlx(conditioning: Any, precision: mx.Dtype) -> mx.array:
    """Extract the Qwen3-VL-8B hidden states for Qwen Image 2.1's DiT.

    Unlike Flux2/SDXL (which route through ComfyUI's real CLIP pipeline and a
    standard conditioning list), Qwen Image 2.1's text encoder is a native MLX
    module wired through a dedicated pair of nodes
    (`ASDX_QwenImage21TextEncoderLoader`/`ASDX_QwenImage21TextEncode`, same pattern
    as MiniMax H3) -- `conditioning` here is that pair's own dict output, not a
    ComfyUI CONDITIONING list. `hidden_states` is `[S, hidden_size]` (no batch dim,
    brick 1's convention); this adds one."""
    if not isinstance(conditioning, dict) or conditioning.get("type") != "qwen_image21":
        raise RuntimeError(
            "ASDX: Qwen Image 2.1 sampler expected the output of ASDX_QwenImage21TextEncode "
            f"(dict with type='qwen_image21'), got {conditioning!r}."
        )
    hidden_states = conditioning["hidden_states"]
    cond = hidden_states[None].astype(precision)
    mx.eval(cond)
    return cond


def mlx_to_comfy_latent_qwen_image21(latents: mx.array, template: dict[str, Any]) -> dict[str, Any]:
    """Convert MLX Qwen Image 2.1 DiT output to a ComfyUI LATENT dict.

    The DiT operates on the latent grid directly in NCHW (matching ComfyUI's own
    LATENT convention) -- no patch packing, no NHWC transpose (unlike SDXL's
    native-MLX channel-last convention)."""
    latents = latents.astype(mx.float32)
    mx.eval(latents)
    samples = torch.from_numpy(np.array(latents, dtype=np.float32))
    out = dict(template)
    out["samples"] = samples
    return out


def prepare_noise_from_latent_qwen_image21(
    latent: dict[str, Any], seed: int, precision: mx.Dtype
) -> tuple[mx.array, int, int, tuple[int, int]]:
    """Prepare unit-gaussian noise from a Comfy latent for Qwen Image 2.1, in MLX.

    NCHW throughout, matching the DiT's own input convention -- no layout
    conversion needed, unlike SDXL's NCHW->NHWC bridge.

    Returns (noise, height, width, output_shape)."""
    if "samples" not in latent:
        raise RuntimeError("ASDX: latent must be a Comfy LATENT with 'samples'.")

    samples = latent["samples"]
    if tuple(samples.shape)[1] != QWEN_IMAGE21_LATENT_CHANNELS:
        raise RuntimeError(
            f"ASDX: needs {QWEN_IMAGE21_LATENT_CHANNELS}-channel Qwen Image 2.1 latent, "
            f"got {tuple(samples.shape)}"
        )

    import comfy.sample
    batch_inds = latent.get("batch_index") if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(samples, int(seed), batch_inds)
    noise_np = _to_numpy(noise)

    batch, channels, latent_h, latent_w = noise_np.shape
    height = latent_h * QWEN_IMAGE21_VAE_DOWNSCALE
    width = latent_w * QWEN_IMAGE21_VAE_DOWNSCALE

    noise_mlx = mx.array(noise_np).astype(precision)
    mx.eval(noise_mlx)
    return noise_mlx, height, width, (latent_h, latent_w)
```

**Note pour l'implémenteur** : vérifier le nom exact de l'utilitaire `_to_numpy` déjà présent
dans `bridge.py` (utilisé par `prepare_noise_from_latent_flux2`/`_sdxl` ci-dessus) et le
réutiliser tel quel -- ne pas le redéfinir.

- [ ] **Step 3: Vérifier le succès**

Run: `uv run pytest tests/test_bridge_qwen_image21.py -q` -> PASS.

- [ ] **Step 4: Commit**

```bash
git add apple_silicon_nodes/bridge.py tests/test_bridge_qwen_image21.py
git commit -m "feat: Qwen Image 2.1 bridge functions (conditioning/latent/noise)"
```

---

### Task 3: Nœuds text encoder dédiés

**Files:**
- Create: `apple_silicon_nodes/qwen_image21_nodes.py`
- Modify: `apple_silicon_nodes/__init__.py`
- Test: `tests/test_qwen_image21_nodes.py`

**Interfaces:**
- Consumes: `load_qwen_image21_text_encoder_checkpoint` (brique 1, `native/qwen_image21/
  text_encoder_weight_map.py`), `comfy.text_encoders.qwen_image21.QwenImage21Tokenizer` (réel,
  sans poids).
- Produces: `ASDX_QwenImage21TextEncoderLoader` (nœud, sortie `asdx_qwen_image21_text_encoder`),
  `ASDX_QwenImage21TextEncode` (nœud, sortie un dict `{"type": "qwen_image21", "hidden_states":
  mx.array[S,4096], "text": str}`). Consommé par `bridge.py::conditioning_qwen_image21_to_mlx`
  (tâche 2).

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""Tests for ASDX_QwenImage21TextEncoderLoader / ASDX_QwenImage21TextEncode.

Loader-only tests run via comfy_stub (no real ComfyUI needed). The real
tokenize+encode path needs the local ComfyUI install (comfy.text_encoders.
qwen_image21) and is gated accordingly, matching this project's established
convention for real-reference tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
nodes_module = load_node_module("qwen_image21_nodes")

ASDX_QwenImage21TextEncoderLoader = nodes_module.ASDX_QwenImage21TextEncoderLoader
ASDX_QwenImage21TextEncode = nodes_module.ASDX_QwenImage21TextEncode
encode_qwen_image21_prompt = nodes_module.encode_qwen_image21_prompt


def test_encode_rejects_wrong_loader_output():
    with pytest.raises(RuntimeError, match="ASDX_QwenImage21TextEncoderLoader"):
        encode_qwen_image21_prompt({"type": "wrong"}, "a cat")


_COMFYUI_ROOT = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")
_COMFYUI_VENV_SITE_PACKAGES = _COMFYUI_ROOT / ".venv" / "lib" / "python3.13" / "site-packages"
_TEXT_ENCODER_REAL = Path("/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_8b_bf16.safetensors")


@pytest.mark.skipif(
    not (_COMFYUI_ROOT.exists() and _COMFYUI_VENV_SITE_PACKAGES.exists() and _TEXT_ENCODER_REAL.exists()),
    reason="needs local ComfyUI install + real text encoder checkpoint",
)
def test_encode_real_prompt_end_to_end():
    import sys as _sys
    _sys.path.insert(0, str(_COMFYUI_ROOT))
    _sys.path.insert(0, str(_COMFYUI_VENV_SITE_PACKAGES))

    from apple_silicon_nodes.native.qwen_image21.text_encoder_weight_map import (
        load_qwen_image21_text_encoder_checkpoint,
    )

    encoder = load_qwen_image21_text_encoder_checkpoint(_TEXT_ENCODER_REAL, dtype="float16")
    text_encoder = {"type": "asdx_qwen_image21_text_encoder", "encoder": encoder}
    out = encode_qwen_image21_prompt(text_encoder, "a red apple on a table")
    assert out["type"] == "qwen_image21"
    assert out["hidden_states"].ndim == 2
    assert out["hidden_states"].shape[1] == 4096
```

Run: `uv run pytest tests/test_qwen_image21_nodes.py -q` -> FAIL.

- [ ] **Step 2: Créer `qwen_image21_nodes.py`**

```python
"""ComfyUI nodes for Qwen Image 2.1's text encoder (Qwen3-VL-8B, MLX native).

Dedicated loader + encode nodes, NOT a clip_type on ASDX_CLIPLoader -- that
node routes through comfy.sd.CLIPType/comfy.sd.load_clip (ComfyUI's real
PyTorch CLIP pipeline), which would load ComfyUI's own reference Qwen3-VL
implementation instead of this project's native MLX port. Same pattern as
MiniMax H3's ASDX_MiniMaxH3TextEncoderLoader/encode_minimax_h3_prompt
(minimax_h3_nodes.py) -- a dedicated node pair returning this project's own
conditioning dict, not the standard ComfyUI CONDITIONING type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_api.latest import io

import mlx.core as mx

_TEXT_ENCODER_CACHE: dict[str, Any] = {}


class ASDX_QwenImage21TextEncoderLoader(io.ComfyNode):
    """Load Qwen Image 2.1's Qwen3-VL-8B text encoder (bf16 safetensors, brick 1)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_QwenImage21TextEncoderLoader",
            display_name="🍏 ASDX Qwen Image 2.1 Text Encoder Loader",
            category="ASDX/Loaders",
            inputs=[
                io.Combo.Input("encoder_name", options=cls._get_encoders()),
                io.Combo.Input("precision", options=["float16", "bfloat16", "float32"], default="float16"),
            ],
            outputs=[
                io.Custom("asdx_qwen_image21_text_encoder").Output(display_name="text_encoder"),
            ],
        )

    @staticmethod
    def _get_encoders() -> list[str]:
        try:
            import folder_paths
            return [n for n in folder_paths.get_filename_list("text_encoders") if "qwen3vl_8b" in n.lower()]
        except Exception:
            return []

    @classmethod
    def execute(cls, encoder_name: str, precision: str = "float16") -> io.NodeOutput:
        import folder_paths

        found = folder_paths.get_full_path("text_encoders", encoder_name)
        if not found:
            raise RuntimeError(f"ASDX Qwen Image 2.1 Text Encoder Loader: could not find '{encoder_name}'.")
        path = Path(found)

        cache_key = f"{path}:{precision}"
        if cache_key in _TEXT_ENCODER_CACHE:
            print(f"[ASDX] Qwen Image 2.1 text encoder cache hit: {encoder_name}")
            return io.NodeOutput(_TEXT_ENCODER_CACHE[cache_key])

        from .native.qwen_image21.text_encoder_weight_map import load_qwen_image21_text_encoder_checkpoint

        _TEXT_ENCODER_CACHE.clear()
        encoder = load_qwen_image21_text_encoder_checkpoint(path, dtype=precision)
        result = {
            "type": "asdx_qwen_image21_text_encoder",
            "name": encoder_name,
            "path": str(path),
            "encoder": encoder,
            "precision": precision,
        }
        _TEXT_ENCODER_CACHE[cache_key] = result
        return io.NodeOutput(result)


def encode_qwen_image21_prompt(text_encoder: dict, prompt: str) -> dict:
    """Tokenize (ComfyUI's real, weight-free QwenImage21Tokenizer) and encode
    with the native MLX Qwen3VL8BTextEncoder. Returns the conditioning dict
    `conditioning_qwen_image21_to_mlx` (bridge.py) consumes."""
    if not isinstance(text_encoder, dict) or text_encoder.get("type") != "asdx_qwen_image21_text_encoder":
        raise RuntimeError(
            "ASDX Qwen Image 2.1 Text Encode: expected the output of ASDX_QwenImage21TextEncoderLoader."
        )

    import comfy.text_encoders.qwen_image21

    tokenizer = comfy.text_encoders.qwen_image21.QwenImage21Tokenizer()
    tokens = tokenizer.tokenize_with_weights(prompt)["qwen3vl_8b"][0]
    input_ids = mx.array([t[0] for t in tokens], dtype=mx.int32)

    hidden_states = text_encoder["encoder"](input_ids)
    print(f"[ASDX] Qwen Image 2.1 Text Encode: {len(prompt)} chars, {hidden_states.shape[0]} rows")
    return {"type": "qwen_image21", "hidden_states": hidden_states, "text": prompt}


class ASDX_QwenImage21TextEncode(io.ComfyNode):
    """Encode a T2I prompt with Qwen Image 2.1's Qwen3-VL-8B text encoder."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_QwenImage21TextEncode",
            display_name="🍏 ASDX Qwen Image 2.1 Text Encode",
            category="ASDX/Conditioning",
            inputs=[
                io.Custom("asdx_qwen_image21_text_encoder").Input("text_encoder"),
                io.String.Input("prompt", multiline=True),
            ],
            outputs=[
                io.Custom("asdx_qwen_image21_conditioning").Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder: dict, prompt: str) -> io.NodeOutput:
        return io.NodeOutput(encode_qwen_image21_prompt(text_encoder, prompt))


NODE_LIST = [
    ASDX_QwenImage21TextEncoderLoader,
    ASDX_QwenImage21TextEncode,
]
```

**Note pour l'implémenteur** : le nom exact de la méthode de tokenisation
(`tokenize_with_weights`) et la structure de retour (`tokens["qwen3vl_8b"][0]`, chaque élément
`(token_id, weight)`) sont repris du pattern MiniMax H3/brique 1 (déjà vérifiés dans ce projet
pour `Qwen3VLTokenizer`, dont `QwenImage21Tokenizer` hérite) mais PAS re-vérifiés ligne à ligne
pour `QwenImage21Tokenizer` spécifiquement dans ce plan -- vérifier contre le vrai
`comfy/text_encoders/qwen_image21.py` avant de committer, comme pour toute divergence dans ce
projet (CLAUDE.md règle 6).

- [ ] **Step 3: Enregistrer dans `__init__.py`**

Ajouter (près de `_minimax_h3_nodes`) :

```python
    from .qwen_image21_nodes import NODE_LIST as _qwen_image21_nodes
```

et dans la liste finale des nœuds agrégés :

```python
        *_qwen_image21_nodes,
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_qwen_image21_nodes.py -q` -> PASS (au moins le test hors
ComfyUI ; le test réel passe si l'install locale + le checkpoint sont présents).

- [ ] **Step 5: Commit**

```bash
git add apple_silicon_nodes/qwen_image21_nodes.py apple_silicon_nodes/__init__.py tests/test_qwen_image21_nodes.py
git commit -m "feat: Qwen Image 2.1 text encoder nodes (loader + encode)"
```

---

### Task 4: Vérification end-to-end sur ComfyUI réel

**Files:** aucun nouveau fichier.

- [ ] **Step 1: Lancer ComfyUI réel** (`comfy-mcp` si disponible, sinon manuellement) et
  construire un workflow minimal : `ASDX_QwenImage21TextEncoderLoader` →
  `ASDX_QwenImage21TextEncode` → `ASDX_DiffusionLoader` (sélectionner
  `qwen_image_2.1_bf16.safetensors`) → sampler générique → `ASDX_VAEDecode`
  (`ASDX_VAELoader` sur `qwen_image_2.1_vae_bf16.safetensors`) → `SaveImage`.

- [ ] **Step 2: Générer une image** avec un prompt simple, confirmer qu'elle n'est pas du bruit
  et correspond grossièrement au prompt.

- [ ] **Step 3: Répéter avec le GGUF** (`qwen_image_2.1_Q8.gguf`) — confirmer que
  `ASDX_DiffusionLoader` détecte et charge correctement.

- [ ] **Step 4: Documenter le résultat** (captures, seed, prompt, temps de génération) dans le
  ledger du plan, et mettre à jour `README.md` avec une section Qwen Image 2.1 (dette signalée
  en brique 2, à combler maintenant que le pipeline est utilisable).

- [ ] **Step 5: Corriger tout écart trouvé**, relancer les suites des tâches 1-3, commit si
  nécessaire.
