# Qwen Image 2.1 — brique 3 : intégration (loader / CLIPLoader / bridge)

Date: 2026-09-22
Statut: approuvé (brainstorming), en attente du plan d'implémentation
Complète : brique 1 (text encoder, mergée) et brique 2 (DiT, mergée)

## Contexte

Les briques 1 et 2 ont livré le text encoder Qwen3-VL-8B et le DiT
`QwenImage21Transformer2DModel` en MLX, tous deux testés directement en pytest contre les
vrais checkpoints. Aucun des deux n'est câblé à un nœud ComfyUI réel — cette brique fait ce
câblage : dispatch de famille dans les loaders génériques, `clip_type` pour le text encoder,
fonctions de pont conditioning/latent/noise pour le sampler.

## Périmètre — brique 3

**Dans le périmètre** :
- Détection de famille `qwen_image21` dans `loader.py` (hints de nom de fichier + repli
  structurel), branchée sur `ASDX_DiffusionLoader`/`ASDX_CheckpointLoader`, avec dispatch
  bf16/GGUF
- **Deux nouveaux nœuds dédiés pour le text encoder** (correction post-brainstorming, voir
  ci-dessous) : `ASDX_QwenImage21TextEncoderLoader` + `ASDX_QwenImage21TextEncode`
- Fonctions de pont dans `apple_silicon_nodes/bridge.py` : `conditioning_qwen_image21_to_mlx`,
  `mlx_to_comfy_latent_qwen_image21`, `prepare_noise_from_latent_qwen_image21`, constantes
  `QWEN_IMAGE21_LATENT_CHANNELS=64`, `QWEN_IMAGE21_VAE_DOWNSCALE=16`
- Test end-to-end sur ComfyUI réel (T2I, bf16 puis GGUF Q8, prompt → image)

**Hors périmètre** :
- Nouveau nœud pour le DiT (reste sur le dispatch générique `ASDX_DiffusionLoader`)
- Édition, références multi-images, cache préfixe (déjà exclus brique 1/2)
- Streaming du chargeur GGUF DiT (dette signalée en brique 2, à traiter séparément)

**Correction post-brainstorming (important)** : la conception initiale supposait que
`ASDX_CLIPLoader`/`ASDX_DualCLIPLoader` pourraient charger le text encoder MLX natif de la
brique 1 via un nouveau `clip_type`. Vérifié faux en lisant `conditioning.py` :
`ASDX_CLIPLoader` route entièrement vers `comfy.sd.CLIPType`/`comfy.sd.load_clip` — le **vrai**
chargement PyTorch de ComfyUI. Lui ajouter `clip_type="qwen_image21"` chargerait
l'implémentation PyTorch de référence de ComfyUI, pas notre `Qwen3VL8BTextEncoder` MLX —
rendant le portage de la brique 1 inutilisé dans le pipeline réel. Le bon point d'intégration
pour un text encoder MLX natif est le pattern déjà établi par MiniMax H3 : un nœud de
chargement dédié + un nœud d'encodage dédié, retournant un dict de conditioning maison
(`{"type": "qwen_image21", "hidden_states": ..., ...}`) consommé directement par le pont du
sampler, pas le type `CONDITIONING` standard de ComfyUI.

## Architecture

- **`loader.py`** : `_QWEN_IMAGE21_HINTS` (ex. `"qwen_image_2.1"`, `"qwen-image-2.1"`,
  `"qwen_image21"`) suivant exactement le pattern `_FLUX2_HINTS`/`_KREA2_HINTS`/`_SDXL_HINTS`/
  `_ZIMAGE_HINTS` déjà en place, plus un repli structurel dans
  `_detect_model_type_from_keys` sur une clé distinctive du DiT (`"transformer_blocks.0.img_mlp.
  gate_up."` — confirmé unique à cette famille par inspection du header réel en brique 2, ne
  collisionne avec aucune des clés déjà utilisées par SDXL/Z-Image/Flux2/Krea2). Nouvelle
  branche dans `_load_transformer_for_type` : **premier cas de dispatch générique gérant le
  GGUF** — teste `path.suffix.lower() == ".gguf"` pour choisir entre
  `load_qwen_image21_dit_checkpoint` (bf16) et `load_qwen_image21_dit_from_gguf` (Q8_0).
  Entrée `"qwen_image21"` ajoutée à `_MODEL_TYPE_CAPABILITY` pour le calibrage mémoire.

  **Listing des fichiers `.gguf`** : `folder_paths` ne liste `.gguf` par défaut pour aucune
  clé de dossier standard. `minimax_h3_nodes.py` patche déjà `"diffusion_models"`/
  `"text_encoders"` pour inclure `.gguf` (`_register_gguf_extension`, exécuté au niveau module
  — donc actif dès que `apple_silicon_nodes` charge, puisque `minimax_h3_nodes` est importé
  inconditionnellement dans `__init__.py`). `ASDX_DiffusionLoader._get_models()` interroge la
  même clé `"diffusion_models"` — **aucune nouvelle registration nécessaire**, mais c'est une
  dépendance implicite à documenter dans le code (si `minimax_h3_nodes.py` était un jour rendu
  optionnel, ce filet de sécurité casserait silencieusement).

  **Différence avec MiniMax H3** : MiniMax H3 n'utilise PAS ce dispatch générique (nœuds de
  chargement dédiés, `ASDX_MiniMaxH3ModelLoader`/`ASDX_MiniMaxH3TextEncoderLoader`) car son
  architecture (vidéo, audio, conditioning lourd) s'écarte trop du pipeline image-diffusion
  standard. Qwen Image 2.1 reste un DiT image classique T2I — le dispatch générique convient,
  comme pour Flux2/Krea2/SDXL/Z-Image, et c'est la première famille de ce dispatch à avoir
  besoin du GGUF.
- **`ASDX_QwenImage21TextEncoderLoader`** (nouveau nœud, pattern `ASDX_MiniMaxH3TextEncoderLoader`,
  `minimax_h3_nodes.py`) : charge `Qwen3VL8BTextEncoder` via
  `load_qwen_image21_text_encoder_checkpoint` (brique 1), sortie `asdx_qwen_image21_text_encoder`
  (dict avec `encoder`, `precision`, etc., mis en cache comme les autres loaders).
- **`ASDX_QwenImage21TextEncode`** (nouveau nœud, pattern `ASDX_MiniMaxH3TextEncode`) : tokenise
  via le tokenizer réel, sans poids, de ComfyUI (`comfy.text_encoders.qwen_image21.
  QwenImage21Tokenizer`, gabarit T2I déjà construit), encode via l'encodeur natif, retourne un
  dict `{"type": "qwen_image21", "hidden_states": ..., "text": ...}` — pas le type
  `CONDITIONING` standard de ComfyUI, consommé directement par `conditioning_qwen_image21_to_mlx`
  dans le pont du sampler.
- **`bridge.py`** : nouvelles fonctions suivant le pattern `conditioning_flux2_to_mlx`/
  `mlx_to_comfy_latent_flux2`/`prepare_noise_from_latent_flux2` (`apple_silicon_nodes/
  bridge.py:396,454,674`), avec les constantes réelles `QWEN_IMAGE21_LATENT_CHANNELS=64`,
  `QWEN_IMAGE21_VAE_DOWNSCALE=16` (confirmées contre `comfy.latent_formats.QwenImage21` et le
  VAE réel chargé directement : `latent_dim=2`, `latent_channels=64`, `downscale_ratio=16`).
- **VAE** : **aucun changement**. Vérifié directement contre le vrai checkpoint
  (`qwen_image_2.1_vae_bf16.safetensors` chargé via `comfy.sd.VAE` réel) : `latent_dim=2` —
  un VAE 2D standard (comme SDXL/Z-Image), **pas** un VAE vidéo 3D style Wan comme le
  supposait la spec de la brique 1 par lecture de la description du modèle. `ASDX_VAELoader`/
  `ASDX_VAEDecode`/`ASDX_VAEEncode` génériques fonctionnent tels quels.
- **Sampler** : générique (`apple_silicon_nodes/sampler/`), pas de nœud dédié à écrire.

## Data flow (T2I, rappel de la brique 1, maintenant complet)

```
prompt texte
  → ASDX_QwenImage21TextEncoderLoader (nouveau, charge Qwen3VL8BTextEncoder)
  → ASDX_QwenImage21TextEncode (nouveau, tokenize + encode MLX natif)
  → dict conditioning maison {"type": "qwen_image21", "hidden_states": ...}
  → conditioning_qwen_image21_to_mlx (bridge, nouveau)
  → ASDX_DiffusionLoader/ASDX_CheckpointLoader (détecte "qwen_image21", charge le DiT MLX, bf16/GGUF)
  → sampler générique + QwenImage21Transformer2DModel
  → mlx_to_comfy_latent_qwen_image21 (bridge, nouveau)
  → ASDX_VAEDecode (générique, inchangé)
  → IMAGE
```

## Tests / vérification

1. Tests unitaires pour les fonctions de pont (formes, round-trip conditioning/latent)
2. Test de détection de famille dans `loader.py` (hints + repli structurel) contre les
   fichiers réels (`qwen_image_2.1_bf16.safetensors`, `qwen_image_2.1_Q8.gguf`)
3. **Test manuel end-to-end sur ComfyUI réel** (canon : le dispatch de famille doit être
   vérifié contre un run ComfyUI réel, pas un script standalone, à cause du renommage de
   module par symlink) : bf16 puis GGUF Q8, prompt → image, les deux text encoders déjà
   testés en isolation (brique 1) mais jamais dans le pipeline complet.

## Risques identifiés

- **Repli structurel** : la clé `"transformer_blocks.0.img_mlp.gate_up."` doit être vérifiée
  comme n'entrant pas en collision avec une autre famille déjà supportée (SDXL a
  `input_blocks.`, Z-Image `noise_refiner.`, Flux2 `double_stream_modulation_img.`, Krea2
  `txtfusion.` — aucune ne ressemble à `transformer_blocks.`, mais à confirmer explicitement
  dans le code de détection, pas supposé).
- **Premier run ComfyUI réel** : les briques 1 et 2 n'ont jamais été exercées dans un vrai
  process ComfyUI (seulement pytest direct) — c'est le premier test qui pourrait révéler un
  problème d'intégration invisible en isolation (symlink de module, cache, etc. — voir le
  canon `Why the symlink target matters`).
