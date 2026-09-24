# Plan de conversion MPS → MLX natif

**Date de creation :** 2026-08-02
**Status :** Planification — en attente d'implementation
**Objectif :** Convertir les nodes fonctionnant sur PyTorch MPS en MLX natif pour exploiter la memoire unifiée d'Apple Silicon (zero-copy, moins de transfers CPU/GPU).

---

## 1. Contexte

Le projet ComfyUI-MLXU est actuellement un melange de :
- **Nodes 100% MLX natif** (sampler, LoRA, loader)
- **Nodes PyTorch MPS** (CLIP, Conditioning, Image Chain, Depth Map) qui utilisent l'infrastructure ComfyUI existante

L'objectif est de migrer progressivement les nodes MPS vers du MLX natif pour :
- **Réduire l'empreinte memoire** (zero-copy via Unified Memory)
- **Réduire les transfers** entre CPU/GPU
- **Gagner en performance** sur les operations lourdes (T5-XXL)

---

## 2. Inventaire des nodes MPS

| # | Node | Fichier actuel | Dependances | Type MPS |
|---|------|---------------|-------------|----------|
| 1 | ASDX CLIP Text Encode | `conditioning.py` | `comfy.sd.CLIP` | Encodeur texte |
| 2 | ASDX CLIP Loader | `conditioning.py` | `comfy.sd.load_clip` | Chargement poids |
| 3 | ASDX Dual CLIP Loader | `conditioning.py` | `comfy.sd.load_clip` | Chargement poids |
| 4 | ASDX Conditioning Merger | `conditioning.py` | — | Python dict (pas de MPS) |
| 5 | ASDX Mask From Image | `image_chain.py` | `torch` | Ops tensorielles |
| 6 | ASDX Mask Blur | `image_chain.py` | `torch.nn.functional` | Conv2D |
| 7 | ASDX Image Compositor | `image_chain.py` | `torch` | Ops tensorielles |
| 8 | ASDX Depth Map | `depth_map.py` | `transformers` + MPS | DepthPro |
| 9 | ASDX Live Preview | `live_preview.py` | `torch`, `comfy.utils` | Decode latent→JPEG |

---

## 3. Phases de conversion

### Phase 0 — Preparatoire (facultatif mais recommande)

**Objectif :** Mettre en place les fondations pour faciliter les conversions futures.

#### 3.1 Nouveau module : `apple_silicon_nodes/native/text_encoder.py`

Creer un module dedie aux encodeurs de texte MLX :

```python
"""
MLX Text Encoders
=================
Native MLX implementations of text encoders used by diffusion models.

Models:
  - CLIPLike : OpenAI CLIP (clip_l, clip_g) — SD/SDXL/Pony
  - T5XXL : Google T5-XXL — FLUX.1
  - UnifiedEncoder : wrapper qui dispatche selon le modele
"""

from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TextEncoderConfig:
    """Configuration pour un encodeur de texte."""
    name: str              # "clip_l", "clip_g", "t5xxl", "umt5xxl"
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    max_position_embeddings: int


class TokenizerWrapper:
    """Wrapper unifie pour les tokenizers.
    
    Peut encapsuler :
    - transformers.AutoTokenizer (pour T5-XXL, clip)
    - Un tokenizer custom MLX (si on reimplemente)
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def encode(self, texts: list[str], max_tokens: int = 256) -> dict:
        """Tokenize et retourne input_ids + attention_mask."""
        ...
```

**Fichier a creer :** `apple_silicon_nodes/native/text_encoder.py`
**Lignes estimees :** ~50
**Effort :** 2h

---

### Phase 1 — T5-XXL MLX (priorite haute, ROI max)

**Objectif :** Reimplementer l'encodeur T5-XXL en MLX natif.

#### 1.1 Architecture T5-XXL

T5-XXL a :
- **24 couches encoder**
- **Vocabulaire :** 32128 tokens (google/t5-xxl)
- **d_model :** 4096
- **d_ff :** 10240
- **8 têtes d'attention** (head_dim = 512)
- **Poids :** ~11GB en fp32, ~5.5GB en fp16

#### 1.2 Modules a implementer

```
apple_silicon_nodes/native/t5_encoder.py
├── T5Config              # Configuration dataclass
├── T5LayerFF             # Feed-forward layer (with GELU/swish)
├── T5LayerSelfAttention  # Self-attention + relative positional bias
├── T5LayerNorm           # RMS/LayerNorm
├── T5DenseActDense       # Dense-FF
├── T5DenseGatedActDense  # Gated-FF (T5-XXL utilise gated)
├── T5Stack               # Empilement des 24 couches
└── T5Encoder             # Wrapper de haut niveau
```

**Fichier a creer :** `apple_silicon_nodes/native/t5_encoder.py`
**Lignes estimees :** ~800-1200
**Effort :** 1-2 jours

#### 1.3 Chargement des poids

```python
def load_t5xxl(path: str | Path, dtype: str = "float16") -> T5Stack:
    """Charge les poids T5-XXL depuis un fichier safetensors ou un checkpoint HuggingFace."""
    # 1. Charger les poids depuis safetensors ou config.json + pytorch_model.bin
    # 2. Renommer les clés (HuggingFace → MLX naming)
    # 3. Assigner aux modules MLX
    # 4. Retourner le modele charge
```

**Source des poids :**
- `google/t5-xxl-lm-adapt` (HuggingFace) — encoder seul
- Fichiers T5-XXL dans les checkpoints FLUX (souvent dans `text_encoders/`)

#### 1.4 Integration dans CLIP Text Encode

Modifier `ASDX_CLIPTextEncode` dans `conditioning.py` :

```python
# Avant (actuel) — utilise comfy.sd.CLIP
conditioning = mlx_clip.encode_from_tokens_scheduled({"t5xxl": tokens_t5})

# Apres — utilise l'encodeur MLX direct
if t5xxl and use_mlx_t5:
    from .native.t5_encoder import load_t5xxl
    t5_model = load_t5xxl(t5_model_path, dtype=precision)
    embeddings = t5_model(input_ids=tokens_t5, attention_mask=mask)
    # embeddings: [B, seq_len, 4096]
```

**Fichier a modifier :** `apple_silicon_nodes/conditioning.py`
**Lignes a modifier :** ~30 lignes changees, ~20 ajoutees
**Effort :** 3-4h

#### 1.5 Acceptance criteria

- [ ] T5-XXL MLX charge un checkpoint HuggingFace sans erreur
- [ ] Les embeddings T5-XXL MLX sont numeriquement proches de ceux de PyTorch (cosine similarity > 0.99)
- [ ] `ASDX_CLIPTextEncode` utilise l'encodeur MLX quand `t5xxl` est fourni
- [ ] Memoire MLX active < 6GB pour un prompt de 100 tokens (vs ~8-10GB avec MPS)
- [ ] Temps d'encode < 2s pour un prompt de 100 tokens sur M2 Max

---

### Phase 2 — CLIP-L MLX (priorite moyenne)

**Objectif :** Reimplementer l'encodeur OpenAI CLIP (clip_l) en MLX.

#### 2.1 Architecture clip_l

- **12 couches transformer**
- **Vocabulaire :** 49408 tokens (GPT2 BPE)
- **d_model :** 768
- **12 têtes** (head_dim = 64)
- **Poids :** ~267MB en fp16

#### 2.2 Modules a implementer

```
apple_silicon_nodes/native/clip_encoder.py
├── CLIPLikeConfig
├── CLIPEncoderLayer
├── CLIPTextModel
└── CLIPTextModelWithProjection  # FLUX utilise la version avec projection
```

**Fichier a creer :** `apple_silicon_nodes/native/clip_encoder.py`
**Lignes estimees :** ~300-400
**Effort :** 4-6h

#### 2.3 Acceptance criteria

- [ ] clip_l MLX charge `clip_l.safetensors` sans erreur
- [ ] Embeddings compatibles avec les nodes existants
- [ ] Memoire < 500MB
- [ ] Temps d'encode < 0.5s

---

### Phase 3 — Depth Map MLX (priorite basse)

**Objectif :** Remplacer DepthPro (transformers + MPS) par un modele MLX.

#### 3.1 Options

| Option | Avantages | Inconvénients |
|--------|-----------|---------------|
| A. Monodepth2 en MLX | modele léger (~25MB) | Moins precis que DepthPro |
| B. DepthPro en MLX | Meilleure qualite | modele lourd (~300MB), complexe |
| C. Garder DepthPro sur MPS | Pas de travail | Pas de gain memoire |

**Recommandation :** Option A (Monodepth2) pour commencer. On peut toujours ajouter DepthPro plus tard.

**Fichier a creer :** `apple_silicon_nodes/native/depth_encoder.py`
**Effort :** 1-2 jours (si Option A), 3-5 jours (si Option B)

---

### Phase 4 — Image Chain MLX (priorite basse, ROI faible)

**Objectif :** Convertir MaskFromImage, MaskBlur, ImageCompositor.

**Decision :** NON recommande pour l'instant.

**Raisons :**
1. Operations simples (threshold, blur, composite) — pas de gain de performance significatif
2. ComfyUI passe les images en `torch.Tensor [B,H,W,C]` — conversion constante MLX↔PyTorch
3. Le cout de conversion peut annuler le gain
4. Sauf si on travaille sur de tres grandes images (>4K) ou batch important

**Exception :** Si on cree un pipeline image-chain complet, on pourrait convertir le batch entier en MLX une seule fois et faire toutes les operations en MLX, puis convertir le resultat final.

---

### Phase 5 — Live Preview MLX (priorite tres basse)

**Objectif :** Convertir le decode latent→JPEG en MLX.

**Decision :** NON recommande.

**Raisons :**
1. Le decode latent→image utilise deja `comfy.latent_formats.Flux().process_out()` + conversion PIL
2. Le bottleneck est PIL/JPEG encoding, pas le calcul
3. Gain memoire negligeable (< 100MB)

---

## 4. Fichiers a creer / modifier — Resume

### A creer

| Fichier | Contenu | Lignes estimees | Effort |
|---------|---------|-----------------|--------|
| `native/text_encoder.py` | TokenizerWrapper, TextEncoderConfig | ~50 | 2h |
| `native/t5_encoder.py` | T5-XXL complet | 800-1200 | 1-2j |
| `native/clip_encoder.py` | CLIP-L complet | 300-400 | 4-6h |
| `native/depth_encoder.py` | Monodepth2 (optionnel) | 200-300 | 1-2j |

### A modifier

| Fichier | Modification | Lignes changees | Effort |
|---------|-------------|-----------------|--------|
| `conditioning.py` | Utiliser T5-XXL MLX + clip_l MLX | ~30 changees, ~20 ajoutees | 3-4h |
| `__init__.py` | Registration des encodeurs (optionnel) | ~5 | 15min |

---

## 5. Ordre d'execution recommande

```
Phase 0 (prep) ──→ Phase 1 (T5-XXL) ──→ Phase 2 (clip_l) ──→ Phase 3 (depth)
     │                    │                      │
     │                    │                      └─→ Phase 5 (live preview) — optionnel
     │                    │
     └────────────────────┴─→ Phase 4 (image chain) — NON recommande
```

**Temps total estime :** 2-4 jours de developpement.

---

## 6. Risques et mitigations

| Risque | Probabilite | Impact | Mitigation |
|--------|-------------|--------|------------|
| Incompatibilite numerique (MLX vs PyTorch) | Moyenne | Eleve | Tests de validation avec cosine similarity |
| Memoire insufisante pour charger T5-XXL | Faible | Eleve | T5-XXL en fp16 = ~5.5GB, suffisant sur M2/M4 Max |
| Rupture de compatibilite avec workflows existants | Moyenne | Eleve | Garder le fallback MPS, activer MLX via flag |
| Temps de developpement > estime | Moyenne | Moyen | T5-XXL est le plus gros morceau — le faire en premier |

---

## 7. Tests de validation

### 7.1 Tests numeriques

```python
# tests/test_text_encoder.py

def test_t5xxl_numerical_alignment():
    """Verifier que les embeddings T5-XXL MLX correspondent a PyTorch."""
    # 1. Charger un prompt test
    # 2. Encoder avec PyTorch (comfy.sd.CLIP)
    # 3. Encoder avec MLX (native.t5_encoder)
    # 4. Comparer embeddings (cosine similarity > 0.99)
    ...

def test_clip_l_numerical_alignment():
    """Verifier que les embeddings clip_l MLX correspondent a PyTorch."""
    ...
```

### 7.2 Tests fonctionnels

```python
def test_full_flux_workflow():
    """Workflow complet : CLIP Text Encode (MLX) → Sampler (MLX) → VAE (MLX)."""
    # 1. Charger checkpoint FLUX
    # 2. Encoder prompt avec ASDX_CLIPTextEncode (mode MLX active)
    # 3. Sampler avec ASDX_MLXSampler
    # 4. Decoder avec ASDX_VAEDecode
    # 5. Verifier que l'image est generatee correctement
    ...

def test_memory_reduction():
    """Verifier la reduction memoire."""
    # 1. Mesurer memoire avec MPS
    # 2. Mesurer memoire avec MLX
    # 3. Verifier reduction > 30%
    ...
```

### 7.3 Tests de performance

```python
def test_encode_speed():
    """Verifier que le temps d'encode est acceptable."""
    prompt = "A beautiful landscape with mountains and a lake"
    # T5-XXL MLX < 2s
    # clip_l MLX < 0.5s
    ...
```

---

## 8. Migration progressive (fallback)

Pour ne pas briser les workflows existants, on peut ajouter un flag de commutation :

```python
# Dans conditioning.py

_USE_MLX_ENCODERS = os.environ.get("ASDX_MLX_ENCODERS", "0") == "1"

class ASDX_CLIPTextEncode:
    def encode(self, mlx_clip, text, t5xxl="", guidance=3.5):
        if _USE_MLX_ENCODERS and t5xxl:
            # Mode MLX natif
            return self._encode_mlx(text, t5xxl, guidance)
        else:
            # Mode MPS (fallback ComfyUI)
            return self._encode_comfy(mlx_clip, text, t5xxl, guidance)
```

**Activation :**
```bash
# Via variable d'environnement
export ASDX_MLX_ENCODERS=1
python main.py

# Ou via un parametre global dans __init__.py
```

---

## 9. Checklist de livraison

- [ ] `native/text_encoder.py` cree et teste
- [ ] `native/t5_encoder.py` cree, charge un checkpoint, genere des embeddings valides
- [ ] Tests numeriques passes (cosine similarity > 0.99 vs PyTorch)
- [ ] `native/clip_encoder.py` cree et teste
- [ ] `conditioning.py` modifie pour utiliser les encodeurs MLX
- [ ] Fallback MPS actif quand MLX non disponible
- [ ] Tests fonctionnels passes (workflow complet)
- [ ] Tests de performance passes (temps d'encode acceptable)
- [ ] Tests memoire passes (reduction > 30% pour T5-XXL)
- [ ] Documentation a jour (README, docstrings)

---

## 10. References

- T5-XXL HuggingFace : https://huggingface.co/google/t5-xxl
- CLIP-L HuggingFace : https://huggingface.co/openai/clip-vit-large-patch14
- ComfyUI comfy.sd : https://github.com/comfyanonymous/ComfyUI
- MLX documentation : https://ml-explore.github.io/mlx/
- mflux-AnyModel CLIP : https://github.com/marianoAbad/ComfyUI-mflux-AnyModel (reference pour le pattern)
