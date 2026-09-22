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
  structurel), branchée sur `ASDX_DiffusionLoader`/`ASDX_CheckpointLoader`
- `clip_type="qwen_image21"` dans `ASDX_CLIPLoader`/`ASDX_DualCLIPLoader`
- Fonctions de pont dans `apple_silicon_nodes/bridge.py` : `conditioning_qwen_image21_to_mlx`,
  `mlx_to_comfy_latent_qwen_image21`, `prepare_noise_from_latent_qwen_image21`, constantes
  `QWEN_IMAGE21_LATENT_CHANNELS=64`, `QWEN_IMAGE21_VAE_DOWNSCALE=16`
- Test end-to-end sur ComfyUI réel (T2I, bf16 puis GGUF Q8, prompt → image)

**Hors périmètre** :
- Nouveaux nœuds UI dédiés (le T2I de base réutilise les loaders/sampler/VAE génériques)
- Édition, références multi-images, cache préfixe (déjà exclus brique 1/2)
- Streaming du chargeur GGUF (dette signalée en brique 2, à traiter séparément)

## Architecture

- **`loader.py`** : `_QWEN_IMAGE21_HINTS` (ex. `"qwen_image_2.1"`, `"qwen-image-2.1"`,
  `"qwen_image21"`) suivant exactement le pattern `_FLUX2_HINTS`/`_KREA2_HINTS`/`_SDXL_HINTS`/
  `_ZIMAGE_HINTS` déjà en place, plus un repli structurel dans
  `_detect_model_type_from_keys` sur une clé distinctive du DiT (`"transformer_blocks.0.img_mlp.
  gate_up."` — confirmé unique à cette famille par inspection du header réel en brique 2).
  **Différence avec MiniMax H3** : MiniMax H3 n'utilise PAS ce dispatch générique (nœuds de
  chargement dédiés, `ASDX_MiniMaxH3ModelLoader`/`ASDX_MiniMaxH3TextEncoderLoader`) car son
  architecture (vidéo, audio, conditioning lourd) s'écarte trop du pipeline image-diffusion
  standard. Qwen Image 2.1 reste un DiT image classique T2I — le dispatch générique convient,
  comme pour Flux2/Krea2/SDXL/Z-Image.
- **`ASDX_CLIPLoader`/`ASDX_DualCLIPLoader`** : nouveau `clip_type="qwen_image21"` chargeant
  `Qwen3VL8BTextEncoder` via `load_qwen_image21_text_encoder_checkpoint` (brique 1). Explicite
  obligatoire (canon existant : le générique par défaut charge silencieusement le mauvais
  encodeur).
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
  → ASDX_CLIPLoader(clip_type="qwen_image21")
  → text_encoder.py (Qwen3VL8BTextEncoder, MLX natif)
  → CONDITIONING (comfy)
  → conditioning_qwen_image21_to_mlx (bridge, nouveau)
  → ASDX_DiffusionLoader/ASDX_CheckpointLoader (détecte "qwen_image21", charge le DiT MLX)
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
