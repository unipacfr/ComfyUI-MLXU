# Reference des nodes ComfyUI-MLXU

**Date de generation :** 2026-08-02
**Version du projet :** 0.2.1

---

## Tableau complet des nodes

```
┌─────────────────────────────────────────────┬──────────┬──────────────────────────────┬────────────────────────────────────────────────┐
│ Node                                        │ Statut   │ Apple Silicon / MLX / Metal  │ Modeles compatibles                            │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  LOADERS                                     │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX Diffusion Loader                    │ ✅ OK    │ ✅ MLX natif (transformer)   │ FLUX.1-dev, FLUX.1-schnell, FLUX.2-Klein       │
│ 🍏 ASDX Checkpoint Loader                   │ ✅ OK    │ ✅ MLX natif (transformer)   │ FLUX.1-dev, FLUX.1-schnell, FLUX.2-Klein       │
│ 🍏 ASDX Dual CLIP Loader                    │ ✅ OK    │ ⚠️ PyTorch MPS (ComfyUI)    │ FLUX, SDXL, SD3, Hunyuan, Mochi, Wan, PixArt.. │
│ 🍏 ASDX CLIP Loader                         │ ✅ OK    │ ⚠️ PyTorch MPS (ComfyUI)    │ 33 types: SD1.5, SDXL, Pony, FLUX, FLUX2, Wan..│
│ 🍏 ASDX CLIP Text Encode                    │ ✅ OK    │ ⚠️ PyTorch MPS (ComfyUI)    │ FLUX (t5xxl+clip_l), SD/SDXL/Pony (clip seul)  │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  SAMPLER                                     │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX MLX Native Sampler                  │ ✅ OK    │ ✅ MLX natif (boucle Euler)  │ FLUX.1-dev, FLUX.1-schnell, FLUX.1-fill,       │
│                                             │          │                              │ FLUX.1-depth, FLUX.2-Klein                     │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  VAE                                         │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX VAE Decode (MLX)                    │ ✅ OK    │ ✅ MLX natif (via mlx_vae)   │ FLUX.1 (16 canaux latent)                      │
│ 🍏 ASDX VAE Encode (MLX)                    │ ✅ OK    │ ✅ MLX natif (via mlx_vae)   │ FLUX.1 (16 canaux latent)                      │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  LATENT                                      │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX Empty FLUX Latent                   │ ✅ OK    │ ✅ MPS (torch.zeros)         │ FLUX.1 uniquement (16 canaux)                  │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  CONDITIONING                                │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX Conditioning Merger                 │ ✅ OK    │ ⚠️ PyTorch (dict merge)     │ Tous (compatible FLUX + SD-style)              │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  LoRA                                        │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX LoRA Loader                         │ ✅ OK    │ ✅ MLX natif (delta weights) │ FLUX.1 (double_blocks + single_blocks)         │
│ 🍏 ASDX Multi LoRA Loader                   │ ✅ OK    │ ✅ MLX natif (delta weights) │ FLUX.1 (jusqu'a 5 LoRAs simultanes)            │
│ 🍏 ASDX LoRA Schedule                       │ ✅ OK    │ ✅ MLX natif (weight delta)  │ FLUX.1 (modulation per-step)                   │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  CONTROLNET                                  │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX ControlNet Union Loader             │ ✅ OK    │ ✅ MLX natif (ControlNet)    │ FLUX.1 (models ControlNet Union)               │
│ 🍏 ASDX Apply ControlNet                    │ ✅ OK    │ ✅ MLX natif (residuals)     │ FLUX.1-dev (8 types: pose, depth, soft_edge,   │
│                                             │          │                              │ line_canny, normal, segment, tile, repaint)     │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  IP-ADAPTER                                  │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX IP-Adapter Loader                   │ ✅ OK    │ ✅ MLX natif (proj weights)  │ FLUX.1 (models IP-Adapter plus)                │
│ 🍏 ASDX CLIP Vision Encode                  │ ✅ OK    │ ⚠️ MLX (encodage simple)    │ FLUX.1 (images reference pour style)           │
│ 🍏 ASDX Apply IP-Adapter                    │ ✅ OK    │ ✅ MLX natif (cross-attn)    │ FLUX.1 (injection cross-attention)             │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  IMAGE CHAIN                                 │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX Image → Latent                      │ ✅ OK    │ ✅ MLX natif (VAE encoder)   │ FLUX.1 (img2img workflow)                      │
│ 🍏 ASDX Mask From Image                     │ ✅ OK    │ ⚠️ PyTorch (tensor ops)     │ Tous (masque binaire depuis image)             │
│ 🍏 ASDX Mask Blur                           │ ✅ OK    │ ⚠️ PyTorch (conv2d)         │ Tous (flou gaussien sur masque)                │
│ 🍏 ASDX Image Compositor                    │ ✅ OK    │ ⚠️ PyTorch (tensor ops)     │ Tous (composite image + masque)                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  DEPTH                                       │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX Depth Map                           │ ✅ OK    │ ⚠️ MPS (DepthPro) + fallback│ Tous (generation carte de profondeur)          │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│  UTILITIES                                   │          │                              │                                                │
├─────────────────────────────────────────────┼──────────┼──────────────────────────────┼────────────────────────────────────────────────┤
│ 🍏 ASDX Memory Profiler                     │ ✅ OK    │ ✅ MLX + MPS (stats memoire)│ Tous (debug memoire Apple Silicon)             │
│ 🍏 ASDX Cache Manager                       │ ✅ OK    │ ✅ MLX + MPS (clear cache)   │ Tous (nettoyage caches)                        │
│ 🍏 ASDX Live Preview                        │ ✅ OK    │ ⚠️ MPS (decode latent→JPEG) │ FLUX.1 (apercu temps reel par step)            │
└─────────────────────────────────────────────┴──────────┴──────────────────────────────┴────────────────────────────────────────────────┘
```

---

## Resume par niveau d'integration Apple Silicon

| Niveau | Description | Nodes |
|--------|-------------|---------|
| **100% MLX natif** | Tout tourne dans MLX, zero copie Unified Memory | Sampler, VAE Encode/Decode, LoRA (3), ControlNet (2), IP-Adapter (3), Diffusion/Checkpoint Loader, Empty Latent, Memory Profiler, Cache Manager |
| **Hybride** | Some MLX, some PyTorch MPS | Image Chain (4), Depth Map, Live Preview |
| **PyTorch MPS (bridge)** | Utilise l'infrastructure ComfyUI standard | CLIP Loader, Dual CLIP Loader, CLIP Text Encode, Conditioning Merger |

---

## Compatibilite modeles par fonctionnalite

| Modele | Sampler | LoRA | ControlNet | IP-Adapter | img2img | inpaint | depth |
|--------|---------|------|------------|------------|---------|---------|-------|
| FLUX.1-dev | ✅ | ✅ | ✅ | ✅ | ✅ (mode routing) | ❌ | ✅ (via node externe) |
| FLUX.1-schnell | ✅ | ✅ | ❌ (hard-block) | ✅ | ✅ | ❌ | ❌ |
| FLUX.1-fill | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ (mode fill) | ❌ |
| FLUX.1-depth | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ (mode depth) |
| FLUX.2-Klein | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

---

## Liste complete des nodes par categorie

### Loaders (4)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_DiffusionLoader | 🍏 ASDX Diffusion Loader | `loader.py` | `(asdx_model,)` |
| ASDX_CheckpointLoader | 🍏 ASDX Checkpoint Loader | `loader.py` | `(asdx_model, mlx_clip, mlx_vae)` |
| ASDX_CLIPLoader | 🍏 ASDX CLIP Loader | `conditioning.py` | `(mlx_clip,)` |
| ASDX_DualCLIPLoader | 🍏 ASDX Dual CLIP Loader | `conditioning.py` | `(mlx_clip,)` |

### Conditioning (2)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_CLIPTextEncode | 🍏 ASDX CLIP Text Encode | `conditioning.py` | `(mlx_conditioning,)` |
| ASDX_ConditioningMerger | 🍏 ASDX Conditioning Merger | `conditioning.py` | `(mlx_conditioning,)` |

### Sampler (1)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_MLXSampler | 🍏 ASDX MLX Native Sampler | `sampler/__init__.py` | `(LATENT,)` |

### VAE (2)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_VAEDecode | 🍏 ASDX VAE Decode (MLX) | `vae.py` | `(IMAGE,)` |
| ASDX_VAEEncode | 🍏 ASDX VAE Encode (MLX) | `vae.py` | `(LATENT,)` |

### Latent (1)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_EmptyFLUXLatent | 🍏 ASDX Empty FLUX Latent | `latent.py` | `(LATENT,)` |

### LoRA (3)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_LoraLoader | 🍏 ASDX LoRA Loader | `lora.py` | `(asdx_model,)` |
| ASDX_MultiLoraLoader | 🍏 ASDX Multi LoRA Loader | `lora.py` | `(asdx_model,)` |
| ASDX_LoraSchedule | 🍏 ASDX LoRA Schedule | `lora.py` | `(asdx_model,)` |

### ControlNet (2)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_ControlNetUnionLoader | 🍏 ASDX ControlNet Union Loader | `controlnet/__init__.py` | `(controlnet,)` |
| ASDX_ApplyControlNet | 🍏 ASDX Apply ControlNet | `controlnet/__init__.py` | `(asdx_model,)` |

### IP-Adapter (3)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_IPAdapterLoader | 🍏 ASDX IP-Adapter Loader | `ip_adapter.py` | `(ip_adapter,)` |
| ASDX_IPAdapterCLIPVisionEncode | 🍏 ASDX CLIP Vision Encode | `ip_adapter.py` | `(mlx_conditioning,)` |
| ASDX_ApplyIPAdapter | 🍏 ASDX Apply IP-Adapter | `ip_adapter.py` | `(mlx_conditioning,)` |

### Image Chain (4)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_ImageToLatent | 🍏 ASDX Image → Latent | `image_chain.py` | `(mflux_image,)` |
| ASDX_MaskFromImage | 🍏 ASDX Mask From Image | `image_chain.py` | `(MASK, mflux_image)` |
| ASDX_MaskBlur | 🍏 ASDX Mask Blur | `image_chain.py` | `(MASK, mflux_image)` |
| ASDX_ImageCompositor | 🍏 ASDX Image Compositor | `image_chain.py` | `(IMAGE, mflux_image)` |

### Depth (1)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_DepthMap | 🍏 ASDX Depth Map | `depth_map.py` | `(mflux_image,)` |

### Utilities (3)

| Nom interne | Nom affiche | Fichier | Retourne |
|-------------|-------------|---------|----------|
| ASDX_MemoryProfiler | 🍏 ASDX Memory Profiler | `memory.py` | `(STRING, FLOAT, FLOAT, FLOAT)` |
| ASDX_CacheManager | 🍏 ASDX Cache Manager | `memory.py` | `(STRING,)` |
| ASDX_LivePreview | 🍏 ASDX Live Preview | `live_preview.py` | `(ASDX_PREVIEW_HANDLE,)` |

---

## Types de sortie utilises

| Type | Description | Utilise par |
|------|-------------|-------------|
| `asdx_model` | Descriptor modele FLUX (transformer + config + capability) | Loader, LoRA, ControlNet, IP-Adapter, Sampler |
| `mlx_clip` | Handle encodeur de texte (wrapper ComfyUI CLIP) | CLIP Loader, CLIP Text Encode |
| `mlx_vae` | Handle VAE MLX | Checkpoint Loader |
| `mlx_conditioning` | Conditioning encodee (sortie CLIP Text Encode) | CLIP Text Encode, IP-Adapter, Sampler |
| `controlnet` | Model ControlNet charge | ControlNet Loader, Apply ControlNet |
| `ip_adapter` | Model IP-Adapter charge | IP-Adapter Loader, Apply IP-Adapter |
| `mflux_image` | Dataclass image chain (image + latent + mask + depth) | ImageToLatent, MaskFromImage, MaskBlur, Compositor, DepthMap |
| `LATENT` | Latent ComfyUI standard (dict with samples tensor) | Sampler, VAE Encode/Decode, Empty FLUX Latent |
| `IMAGE` | Image ComfyUI standard [B,H,W,C] float32 [0,1] | VAE Decode, Image Chain nodes |
| `MASK` | Masque ComfyUI [B,H,W] float32 [0,1] | MaskFromImage, MaskBlur, ImageCompositor |
| `STRING` | Chaîne de caracteres | Memory Profiler, Cache Manager |

---

## Parametres du sampler (ASDX_MLXSampler)

### Requis

| Parametre | Type | Defaut | Description |
|-----------|------|--------|-------------|
| model | asdx_model | — | Modele diffusion charge |
| positive | mlx_conditioning | — | Conditioning positive |
| latent_image | LATENT | — | Image latente de depart |
| seed | INT | 0 | Graine (control_after_generate: true) |
| steps | INT | 20 | Nombre d'etapes (1-100) |
| guidance | FLOAT | 3.5 | Guidance scale (0-20) |
| teacache | BOOLEAN | false | Activer TeaCache |
| teacache_threshold | FLOAT | 0.08 | Seuil TeaCache (0.01-1.0) |
| kontext | BOOLEAN | false | Activer Kontext KV cache |
| kontext_reference_latent | LATENT | null | Image reference pour Kontext |
| kontext_reference_strength | FLOAT | 1.0 | Force Kontext (0-2) |
| seacache | BOOLEAN | false | Activer SeaCache |
| preview | BOOLEAN | false | Activer preview temps reel |

### Optionnels (mode routing)

| Parametre | Type | Defaut | Description |
|-----------|------|--------|-------------|
| mode | STRING | auto | text2img, img2img, inpaint, fill, depth |
| image | IMAGE | null | Image source pour img2img/inpaint |
| mask | MASK | null | Masque pour inpainting |
| image_strength | FLOAT | 0.8 | Force img2img (0-1) |
| mask_blur | INT | 0 | Flou masque (0-64) |
| mask_padding | INT | 48 | Padding masque (32-256) |
| depth_image | IMAGE | null | Image de profondeur |
| depth_strength | FLOAT | 1.0 | Force depth (0-2) |
| noise_aug | FLOAT | 0.0 | Augmentation bruit (0-1) |
| lora_schedule | ASDX_LORA_SCHEDULE | null | Schedule LoRA per-step |

---

## Capability Profiles

| Profile | Modele | guidance | img2img | inpaint | depth | controlnet |
|---------|--------|----------|---------|---------|-------|------------|
| flux1_dev | FLUX.1-dev | ✅ | ❌ | ❌ | ❌ | ✅ |
| flux1_schnell | FLUX.1-schnell | ❌ (hard-block) | ❌ | ❌ | ❌ | ❌ |
| flux1_fill | FLUX.1-fill | ✅ | ✅ | ✅ | ❌ | ❌ |
| flux1_depth | FLUX.1-depth | ✅ | ✅ | ❌ | ✅ | ❌ |
| flux2_klein | FLUX.2-Klein | ✅ | ❌ | ❌ | ❌ | ✅ |

**Dispatch par nom de fichier :**
- `fill` → flux1_fill
- `depth` → flux1_depth
- `klein` → flux2_klein
- `schnell` → flux1_schnell
- `dev`, `kontext` → flux1_dev (defaut)

---

## Modes de sampling supports

| Mode | Description | Inputs requis |
|------|-------------|---------------|
| text2img | Generation depuis le bruit | latent_image seul |
| img2img | Variation d'une image | image + image_strength |
| inpaint | Modification region masquee | image + mask |
| fill | Inpainting FLUX (noise augmentation) | image + mask + noise_aug |
| depth | Controle par profondeur | depth_image + depth_strength |

**Detection automatique :** Le mode `auto` deduit le mode depuis les inputs connects.

---

## ControlNet Union — types supports

| Type | Index | Description |
|------|-------|-------------|
| pose | 0 | Pose skeleton |
| depth | 1 | Carte de profondeur |
| soft_edge | 2 | Detection contours doux |
| line_canny | 3 | Detection contours Canny |
| normal | 4 | Carte de normales |
| segment | 5 | Segmentation semantique |
| tile | 6 | Tile/repaint haute resolution |
| repaint | 7 | Repaint general |

---

## Chemines modeles attendus

| Type | Dossier ComfyUI | Exemple |
|------|-----------------|---------|
| Checkpoint | `models/checkpoints/` | `flux1-dev-fp16.safetensors` |
| Diffusion | `models/diffusion_models/` ou `models/unet/` | `flux1-dev-fp16.safetensors` |
| CLIP | `models/text_encoders/` | `clip_l.safetensors`, `t5xxl.safetensors` |
| LoRA | `models/loras/` | `example_lora.safetensors` |
| ControlNet | `models/controlnet/` | `controlnet_union.safetensors` |
| IP-Adapter | `models/ipadapter/` ou `models/unet/` | `ip-adapter-plus.safetensors` |
