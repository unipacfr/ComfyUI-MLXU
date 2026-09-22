# Qwen Image 2.1 — brique 1 : génération texte-à-image (design)

Date: 2026-09-22
Statut: approuvé (brainstorming), en attente du plan d'implémentation

## Contexte

Le projet ASDX porte des familles de diffusion (Flux.2, Krea2, Z-Image, SDXL,
MiniMax H3) en MLX natif pour Apple Silicon. Qwen Image 2.1 est un nouveau
modèle Qwen (7B, 32 blocks single-stream DiT, mixed-granularity attention,
prefix KV cache reuse) sorti chez Civitai le 2026-09-21/21, avec T2I,
édition, transparence RGBA native et jusqu'à 10 images de référence.

Fichiers modèles disponibles localement :
- `diffusion_models/Qwen 2/base model/qwen_image_2.1_bf16.safetensors` (checkpoint complet bf16)
- `diffusion_models/Qwen 2/gguf/qwen_image_2.1_Q8.gguf` (GGUF Q8)
- `vae/qwen_image_2.1_vae_bf16.safetensors` (VAE — layout Wan 2.2, kernel temporel 1, RGBA)
- `text_encoders/qwen3vl_8b_bf16.safetensors` / `qwen3vl_8b_int8_convrot.safetensors`
- `text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` /
  `qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors`

Référence : ComfyUI natif a déjà le support (`comfy/ldm/qwen_image21/model.py`,
`comfy/text_encoders/qwen_image21.py`, `comfy/text_encoders/qwen35.py`,
`comfy/text_encoders/qwen3vl.py`, `comfy/supported_models.py::QwenImage21`).
C'est la référence de portage, conformément à CLAUDE.md.

## Périmètre — brique 1

**Dans le périmètre** :
- Génération texte-à-image uniquement (pas d'édition, pas de références
  multi-images, pas de masques locaux — ce sont des features phares du
  modèle mais hors brique 1)
- Les deux formats DiT : bf16 safetensors et GGUF Q8
- Qwen3-VL-8B comme text encoder — c'est le seul câblé par la référence
  ComfyUI native (`comfy/text_encoders/qwen_image21.py`, commit `6bfaacc6`,
  2026-09-19 : `QwenImage21TEModel` fixe `model_type="qwen3vl_8b"` en dur,
  aucune référence à `qwen35.py`)
- Le VAE Qwen Image 2.1 (layout Wan 2.2 / RGBA) via le chemin générique
  `ASDX_VAELoader`/`ASDX_VAEDecode`/`ASDX_VAEEncode` existant

**Hors périmètre (briques futures)** :
- Édition d'image, références multi-images, édition locale par masque/cercle
- Transparence RGBA native (le modèle la supporte, pas cette brique)
- INT8_convrot pour le text encoder (fichier disponible sur disque mais non
  traité ici — bf16 d'abord, quantization en suivi si besoin)
- **Qwen3.5-9B comme text encoder** (`qwen3.5_9b_qwen_image_2.1_pe_t2i/i2i`
  sur disque) : aucune implémentation de référence trouvée dans la ComfyUI
  locale (voir Risques). Abandonné pour cette spec sauf découverte d'une
  référence réelle ; le champ `clip_type` reste ouvert à une extension
  future si une référence apparaît.

## Architecture

Nouveau package `apple_silicon_nodes/native/qwen_image21/`, même schéma que
`native/flux2/` et `native/krea2/` :

- `config.py` — dimensions du DiT (32 blocks single-stream), détection de
  config depuis le header safetensors/GGUF, constantes latent (canaux VAE,
  downscale spatial), suit le pattern `detect_flux2_config`/`Flux2Config`
- `model.py` — port MLX de `QwenImage21Transformer2DModel`
  (`comfy/ldm/qwen_image21/model.py`, 353 lignes), incluant
  `block_causal_attention` et le cache KV préfixe (`prefix_cached_attention`,
  `prefix_cache_key`) — mécanisme absent des autres familles du projet,
  point le plus délicat du portage. Porter la math exacte depuis la
  référence ComfyUI avant toute vérif numérique (règle CLAUDE.md point 6).
- `weight_map.py` — mapping clés checkpoint → clés natives, couvre bf16 et
  GGUF Q8 (le GGUF passe par `native/gguf/reader.py` + `dequant.py` déjà en
  place, indépendamment de `native/weight_format.py::classify_quant_format`
  qui ne couvre que les marqueurs safetensors FP8_SCALED/FP4_PACKED/
  INT8_TENSORWISE — le bf16 n'a pas besoin de classification, c'est un
  dtype direct)
- `text_encoder.py` — port MLX Qwen3-VL-8B, adapte le pattern déjà validé
  par `native/minimax_h3/text_encoder.py` (port complet Qwen3-VL-32B, backbone
  `comfy/text_encoders/llama.py::Llama2_`/`Attention`/`MLP`/`TransformerBlock`
  générique, configuré par `Qwen3VL_8BConfig`) à la taille 8B. Différence clé
  avec MiniMax H3 : ici le modèle est utilisé jusqu'à sa dernière couche avec
  gabarit T2I dédié (`comfy/text_encoders/qwen_image21.py::T2I_TEMPLATE`), pas
  de tour vision à porter pour cette brique (pas de conditionnement image de
  référence en scope).

  **Décision explicite contre le canon par défaut** : le canon
  `Porting CLIP/T5/Qwen text encoders to MLX has weak memory ROI, except FP8
  sources` recommande par défaut le bridge CLIP réel ComfyUI (pattern
  `krea2_grounded_encode.py`) plutôt qu'un portage natif, et qualifie le port
  MiniMax H3 de "one-off, not a reversal of the general guidance". Choix
  utilisateur, en connaissance de cause, de faire un deuxième one-off plutôt
  que de suivre le bridge — pas une mesure de ROI nouvelle, une préférence
  explicite de garder Qwen Image 2.1 sur pipeline MLX natif.

## Intégration

- **`loader.py`** : ajout de `_QWEN_IMAGE21_HINTS` (indices de nom de
  fichier : "qwen_image_2.1", "qwen-image-2.1", "qwen_image21"...) plus un
  repli structurel sur clés tensor distinctives du DiT, suivant exactement
  le pattern `_FLUX2_HINTS`/`_KREA2_HINTS`/`_SDXL_HINTS`/`_ZIMAGE_HINTS` déjà
  en place. Câblé sur `ASDX_DiffusionLoader` et `ASDX_CheckpointLoader`
  existants — pas de nouveau nœud loader.
- **`ASDX_CLIPLoader`/`ASDX_DualCLIPLoader`** : ajout du `clip_type`
  `qwen_image21` (Qwen3-VL-8B, gabarit T2I). Canon existant : le `clip_type`
  générique par défaut charge silencieusement le mauvais encodeur — donc
  valeur explicite obligatoire, documentée dans le nœud, pas de détection
  auto ambiguë.
- **`sampler/bridge.py`** : nouvelles fonctions
  `conditioning_qwen_image21_to_mlx`, `mlx_to_comfy_latent_qwen_image21`,
  `prepare_noise_from_latent_qwen_image21`, plus les constantes de latent
  (canaux, downscale), suivant le pattern déjà présent pour flux2/sdxl/zimage.
- **VAE** : `ASDX_VAELoader` (générique, charge n'importe quel fichier VAE
  via `comfy.sd.VAE`) + `ASDX_VAEDecode`/`ASDX_VAEEncode` (déjà génériques,
  gèrent déjà le layout `latent_dim==3` façon Wan pour Krea2 — Qwen Image 2.1
  VAE utilise le même layout Wan 2.2). À confirmer sur checkpoint réel, mais
  **aucun nouveau code VAE attendu**.
- **Nœuds UI neufs** : aucun pour le T2I de base — réutilise
  `ASDX_DiffusionLoader`/`ASDX_CheckpointLoader`, `ASDX_CLIPLoader`, le
  sampler générique, `ASDX_VAEDecode`.

## Data flow (T2I)

```
prompt texte
  → ASDX_CLIPLoader(clip_type="qwen35" | "qwen3vl_8b")
  → text_encoder_qwen35.py | text_encoder_qwen3vl.py (encode MLX natif)
  → CONDITIONING (comfy)
  → conditioning_qwen_image21_to_mlx (bridge)
  → sampler générique + qwen_image21/model.py (DiT MLX)
  → mlx_to_comfy_latent_qwen_image21 (bridge)
  → ASDX_VAEDecode (VAE générique, comfy.sd.VAE réel)
  → IMAGE
```

Chargement du DiT : `ASDX_DiffusionLoader`/`ASDX_CheckpointLoader` détecte
`qwen_image21` via `_QWEN_IMAGE21_HINTS`, dispatch vers
`load_qwen_image21_transformer` (bf16 direct ou GGUF via `native/gguf/`),
`weight_map.py` fait le mapping vers les modules MLX natifs.

## Tests / vérification

Suit le protocole déjà établi dans ce projet :

1. **`verify-checkpoint` skill** — py_compile, forward pass à poids
   aléatoires, chargement N/M matché sur le checkpoint bf16 réel, sanity
   std-vs-random-init, pour le DiT ET les deux text encoders.
2. **Agent `weight-map-reviewer`** sur `weight_map.py` et la boucle de
   chargement (`load_qwen_image21_transformer`), pour la classe de bugs
   silencieux déjà rencontrée dans ce projet.
3. **Agent `comfy-reference-diff`** — diff du DiT porté contre
   `comfy/ldm/qwen_image21/model.py` (noms de couches, shapes, constantes,
   ordre des opérations, biais).
4. **Test manuel end-to-end sur ComfyUI réel** (pas un script standalone —
   canon : le dispatch famille doit être vérifié contre un run ComfyUI réel
   à cause du renommage de module par symlink) : bf16 puis GGUF Q8,
   Qwen3-VL-8B puis Qwen3.5-9B, les 4 combinaisons.

## Risques identifiés

- **Prefix KV cache / block-causal attention** : mécanisme neuf, aucune
  famille existante du projet ne l'a. Porter la math exacte avant
  vérification numérique (règle CLAUDE.md #6).
- **VAE Wan 2.2 layout** : supposé compatible avec le chemin générique
  existant (comme Krea2), mais non vérifié sur un checkpoint Qwen Image 2.1
  réel — premier test du protocole doit confirmer avant de considérer ce
  point acquis.
- **Qwen3.5-9B abandonné pour cette spec** : `qwen3.5_9b_qwen_image_2.1_pe_t2i/
  i2i.int8_convrot.safetensors` (Civitai, 2026-09-21) n'a aucune
  implémentation de référence dans la ComfyUI locale (commit `6bfaacc6`,
  2026-09-19, câble uniquement `qwen3vl_8b`). Ne pas porter à l'aveugle sans
  référence vérifiable (règle CLAUDE.md #6 : "port the exact math from a real
  reference implementation... verify numerically... do not trust a formula
  derived by inspection alone").
