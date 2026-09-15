# Plan : Multi-modeles Apple Silicon

**Date de creation :** 2026-08-02
**Dernier elagage :** 2026-09-14 (verifie contre le code, pas contre les cases a cocher)
**Status :** Partiellement realise — ce document ne conserve que le travail restant.
**Objectif :** Transformer le projet en plateforme multi-modeles optimisee Apple Silicon, au-dela de FLUX.1.

---

## 0. Etat des lieux (2026-09-14)

Verifie en lisant le code, pas les checklists (qui n'ont jamais ete tenues a jour).

| Phase | Etat | Preuve |
|-------|------|--------|
| **0 — Fondations** | ✅ Fait, mais **pas sous la forme prevue** (voir §4) | `loader.py::_detect_model_type` / `_load_transformer_weights`, `capability.py::CapabilityProfile`, `sampler/scheduling.py` + `sampler/solvers.py` |
| **1 — SDXL** | ✅ Fait (txt2img) — commit `7046a5e` | `native/sdxl/` (config/model/weight_map), `sampler/core.py::_run_sdxl` |
| **2 — FLUX.2 Klein** | ✅ Fait — commit `7046a5e` | `native/flux2/`, `sampler/core.py::_run_flux2` |
| **3 — Wan 2.1** | ❌ Rien | pas de `native/wan/`, aucun `wan_video`/`umt5` dans le code (les occurrences `Wan21` concernent le *format de latent* du VAE Krea2, pas le modele video) |
| **4 — Hunyuan** | ❌ Rien | pas de `native/hunyuan/`, aucun `hunyuan_dit`/`bert_encoder` (les mentions dans `conditioning.py` sont la liste des `CLIPType` de ComfyUI, pas une implementation) |
| **5 — Krea2** | ✅ Fait — commit `7046a5e` | `native/krea2/` (model, rope, text_encoder, nag), `sampler/core.py::_run_krea2` |
| **5 — PixArt / Ideogram / SVD** | ❌ Rien | aucun fichier correspondant |
| **6 — MiniMax H3** | ❌ Rien | pas de `native/minimax_h3/` ; ComfyUI a un port natif complet a partir duquel travailler : `comfy/ldm/minimax/model.py` (784 lignes), `comfy/text_encoders/minimax.py` + `qwen3vl.py`, `comfy_extras/nodes_minimax_h3.py` (626 lignes) — voir Phase 6 |
| **Bonus hors plan** | ✅ **Z-Image** (base + turbo), absent du plan d'origine | `native/zimage/`, `sampler/core.py::_run_zimage`, profils `zimage_base`/`zimage_turbo` dans `capability.py` |

### Limites connues des familles deja livrees

- **SDXL : txt2img uniquement.** `sampler/core.py::_run_sdxl` le documente explicitement :
  img2img / inpainting / depth / ControlNet ne sont pas cables (le dispatch de `run()`
  route vers `_run_sdxl` *avant* `_detect_mode()`, qui est specifique FLUX).
- **ControlNet : indisponible hors FLUX.1.** Le package ControlNet est deplace dans
  `disabled_nodes/controlnet/`, et `capability.py` declare `supports_controlnet=False`
  pour `sdxl_base`, `zimage_base`, `zimage_turbo`.
- **LoRA SDXL : fonctionnel.** Detection de famille par cles (`lora.py:100`,
  signature `input_blocks.`) + `native/sdxl/weight_map.py::native_key_to_checkpoint_stem`
  (`lora.py:2299`). SDXL passe par la boucle de merge, pas par le chemin residuel FLUX.
- **Text encoders et VAE restent ceux de ComfyUI.** Aucun encodeur MLX natif generique
  n'a ete ecrit : SDXL consomme le `comfy.sd.CLIP` dual (`bridge.py::conditioning_sdxl_to_mlx`)
  et `vae.py` instancie `comfy.sd.VAE`. Seul Krea2 a un encodeur natif
  (`native/krea2/text_encoder.py`).

---

## 1. Vision

Passer de **FLUX.1-only** a **tous les modeles de diffusion** sur infrastructure MLX native.

### Principes directeurs

1. **Interface commune** — chaque famille de modeles implemente les memes interfaces (transformer, VAE, text encoder)
2. **Dispatch automatique** — le loader detecte le type de modele depuis le checkpoint
3. **Zero copie superflue** — tenseurs en memoire unifiee, bridge MLX↔PyTorch minimal
4. **Retrocompatibilite** — workflows existants continuent de fonctionner
5. **Extensibilite** — ajouter un modele = ajouter un module, pas modifier le noyau

> Note 2026-09-14 : les principes 2 a 5 sont tenus. Le principe 1 (interfaces
> abstraites communes) n'a **pas** ete retenu a l'implementation — voir §4.

---

## 2. Modeles cibles

| Famille | Modeles | VAE | Latent | Encoder | Scheduler | Etat |
|---------|---------|-----|--------|---------|-----------|------|
| **FLUX.1** | dev, schnell, fill, depth | 16 can. | 16 | T5-XXL + CLIP-L | Euler | ✅ Supporte (anterieur au plan) |
| **FLUX.2** | Klein | 16 can. | 16 | T5-XXL + CLIP-L | Euler | ✅ Fait (`native/flux2/`) |
| **Krea2** | Base, Turbo | Wan21 | 16 | Krea2 CLIP | Euler | ✅ Fait (`native/krea2/`) |
| **SDXL** | base (+ Illustrious/Pony/NoobAI) | 4 can. | 4 | CLIP-L + OpenCLIP | Euler, DDIM, DPM++ | ✅ Fait, txt2img seul (`native/sdxl/`) |
| **Z-Image** | base, turbo | 16 can. | 16 | — | Euler | ✅ Fait, hors plan initial (`native/zimage/`) |
| **Wan 2.1** | 480p, 720p, 14B | 4 can. | 4 | UMT5-XXL | Euler | A faire — Phase 3 |
| **Hunyuan** | DiT XL/2 | 4 can. | 4 | CLIP + BERT | DDIM | A faire — Phase 4 |
| **PixArt** | Sigma 2B | 4 can. | 4 | T5-XXL + CLIP-L | Euler | A faire — Phase 5 |
| **SVD** | SVD, SV3D | 4 can. | 4 | CLIP-L | Euler | A faire — Phase 5 |
| **MiniMax H3** | H3 (FL2VA / REF2VA) | video 24 can. (ratio 16, ratio_t 4) + audio 32 can. (BigVGAN, 32kHz) | video 24 (patch 1x2x2) + audio 32, paquetes en un seul flux `[text\|cond\|audio\|video]` | Qwen3-VL-32B (couche 50) | flow-matching sigma-shift double (video 12.0 / audio 3.0) | A faire — Phase 6, architecture connue (voir §5) |

---

## 3. Architecture — reelle vs cible

L'arborescence cible d'origine (`native/transformer/`, `native/vae/`, `native/text_encoder/`,
`native/rope/`) **n'a pas ete construite**. Le decoupage retenu est **un sous-package par
famille de modele**, chacun portant son propre `config.py` / `model.py` / `weight_map.py` :

```
apple_silicon_nodes/native/
├── config.py            # config FLUX.1 + helpers de latent (process_wan21_latent_*)
├── weight_format.py     # classification des conventions de quantification
├── weight_map.py        # mapping de cles FLUX.1
├── flux2/               # FLUX.2-Klein
├── krea2/               # Krea2 (+ rope.py, text_encoder.py, nag.py)
├── sdxl/                # SDXL UNet
└── zimage/              # Z-Image (NextDiT)
```

Les VAE et les text encoders restent ceux de ComfyUI (`comfy.sd.VAE`, `comfy.sd.CLIP`),
relayes via `bridge.py`. **Toute nouvelle famille (Wan, Hunyuan, PixArt, SVD) doit suivre
ce patron `native/<famille>/`**, pas l'arborescence par couche du plan d'origine.

Points d'extension a modifier pour chaque nouvelle famille :

| Fichier | Ce qu'il faut y ajouter |
|---------|-------------------------|
| `native/<famille>/` | `config.py`, `model.py`, `weight_map.py` |
| `native/__init__.py::_load_safetensors` | chargement + dequantification (point d'entree unique) |
| `loader.py::_detect_model_type` + `_load_transformer_weights` | detection par nom puis par cles, puis construction |
| `capability.py` | un `CapabilityProfile` + les alias de nom |
| `sampler/core.py` | une boucle `_run_<famille>()` + le dispatch dans `run()` |
| `sampler/scheduling.py` / `solvers.py` | sigma schedule et preconditionnement si le modele n'est pas flow-matching |
| `latent.py::_LATENT_FORMATS` | `(channels, downscale)` de la famille |
| `lora.py` | signature de cles de la famille + `native_key_to_checkpoint_stem` |

---

## 4. Interfaces generiques — abandonnees

**✅ Tranche : non retenu.** Les ABC prevues a l'origine (`DiffusionTransformer`,
`DiffusionVAE`, `TextEncoder`, `DiffusionScheduler`) n'existent pas — un
`grep abstractmethod apple_silicon_nodes/` ne retourne rien, et `TransformerConfig`
non plus (`native/config.py` n'a jamais ete renomme).

Ce qui a ete construit a la place, et qui fait office de contrat :

- **Dispatch par chaine `model_type`** : `loader.py::_detect_model_type` (nom de fichier,
  puis detection par cles du header safetensors) renvoie `"dev"` / `"flux2"` / `"krea2"` /
  `"sdxl"` / `"zimage"`..., consommee par `sampler/core.py::run()` qui route vers une des
  cinq boucles de denoising dediees.
- **`capability.py::CapabilityProfile`** : declaratif par famille (`generate_params`,
  `requires`, `hard_block`, `latent_channels`, `supports_controlnet`) — c'est le seul
  "contrat" formel qu'une famille doit remplir.
- **Schedulers** : `sampler/scheduling.py` (`generate_sigmas`, `generate_sigmas_sdxl`,
  `SDXLSampling`, schedulers karras/simple/sgm_uniform/beta) + `sampler/solvers.py`
  (`euler`, `euler_a`, `ddim`, `dpmpp_2m`, `dpmpp_2m_sde`, `dpmpp_2s_ancestral`, `deis`),
  selectionnes par chaine, sans hierarchie de classes.

**Consequence pour les phases 3-5** : ne pas commencer par ecrire des interfaces.
Ajouter une famille = ajouter un sous-package + une boucle + un profil, selon le tableau
de §3.

---

## 5. Phases restantes

### Phase 0 — Fondations

✅ **Fait**, sous une forme differente de celle planifiee — voir §4 et §0.
Rien a reprendre ici.

### Phase 1 — SDXL

✅ **Fait pour SDXL** (commit `7046a5e` "feat(native): add SDXL, Flux2/Klein, Krea2,
Z-Image native MLX architectures", puis `a093e82` et `94559cd` pour le LoRA).
Livre : `native/sdxl/`, `_run_sdxl`, LoRA SDXL, `latent.py` multi-canaux
(`"sdxl": (4, 8)`), sigmas discrets EPS/DDPM.

**Reste ouvert sur cette phase :**

| Item | Etat | Detail |
|------|------|--------|
| img2img / inpainting SDXL | ❌ | `_run_sdxl` court-circuite `_detect_mode()`. Effort : 1-2j |
| ControlNet SD | ❌ | `disabled_nodes/controlnet/` a reactiver puis porter. Effort : 2-3j |

### Phase 2 — FLUX.2 Klein

✅ **Fait** (commit `7046a5e`, puis `335f586`) — `native/flux2/`, `_run_flux2`,
signature LoRA `double_stream_modulation_img/_txt` (`lora.py`).

---

### Phase 3 — Wan 2.1 (priorite moyenne)

**Objectif :** Supporter Wan 2.1 — modele video. **Rien n'existe a ce jour.**

| Fichier | Contenu | Lignes | Effort |
|---------|---------|--------|--------|
| `native/wan/model.py` | Wan 2.1 video MLX | 1000 | 3-5j |
| `native/wan/config.py` + `weight_map.py` | Config + mapping de cles | 200 | 1j |
| VAE Wan | Via `comfy.sd.VAE` d'abord ; port MLX seulement si le bridge est le goulot | 250 | 1-2j |
| Encodeur UMT5-XXL | Via le `CLIPType "wan"` de ComfyUI d'abord (deja expose dans `conditioning.py`) ; port MLX optionnel | 300 | 1-2j |
| `sampler/core.py::_run_wan` | Boucle de denoising + rope temporel 3D | 150 | 1-2j |

**Total phase 3 : 6-10 jours**

Attention : le format de latent `Wan21` existe deja dans `native/config.py`
(`process_wan21_latent_in/out`) — il est utilise par le **VAE de Krea2**, pas par le
modele video. Le reutiliser, ne pas le redefinir.

---

### Phase 4 — Hunyuan (priorite moyenne)

**Objectif :** Supporter Hunyuan DiT. **Rien n'existe a ce jour.**

| Fichier | Contenu | Lignes | Effort |
|---------|---------|--------|--------|
| `native/hunyuan/model.py` | Hunyuan DiT MLX | 600 | 2-3j |
| `native/hunyuan/config.py` + `weight_map.py` | Config + mapping de cles | 200 | 1j |
| VAE Hunyuan | Via `comfy.sd.VAE` d'abord | 250 | 1-2j |
| Encodeur BERT-Large | Via le `CLIPType "hunyuan_dit"` de ComfyUI d'abord | 250 | 1-2j |
| `sampler/core.py::_run_hunyuan` | Boucle DDIM + CFG deux passes | 150 | 1j |

**Total phase 4 : 5-8 jours**

---

### Phase 5 — Modeles avances (priorite basse)

Krea2 est **✅ fait** (`native/krea2/`, commit `7046a5e`) et sort de cette phase.
Z-Image, absent du plan d'origine, est **✅ fait** egalement (`native/zimage/`).

| Fichier | Modele | Lignes | Effort |
|---------|--------|--------|--------|
| `native/pixart/` | PixArt Sigma | 400 | 2j |
| `native/ideogram/` | Ideogram | 400 | 2j |
| `native/svd/` | SVD | 500 | 2-3j |

**Total phase 5 : 6-7 jours**

---

### Phase 6 — MiniMax H3 (investigation faite, effort re-estime a la hausse)

**Objectif :** Supporter MiniMax H3. **Rien n'existe a ce jour cote MLX**, mais
contrairement a l'hypothese initiale, **ComfyUI a deja un port PyTorch complet et
maintenu** qui sert de reference directe (meme statut que Wan/Hunyuan/FLUX.2/Krea2,
pas un cas "architecture a deviner") :

- `comfy/ldm/minimax/model.py` (784 lignes) — le DiT
- `comfy/ldm/minimax/controlnet.py` (85 lignes)
- `comfy/text_encoders/minimax.py` (217 lignes) + `comfy/text_encoders/qwen3vl.py` (215 lignes)
- `comfy_extras/nodes_minimax_h3.py` (626 lignes) — nodes de sampling/latent
- `comfy/supported_models.py::MiniMaxH3`, `comfy/model_base.py::MiniMaxH3`,
  `comfy/latent_formats.py::MiniMaxH3AV`

**Investigation menee sur les checkpoints reels de l'utilisateur** (headers
safetensors, `/Volumes/X10Pro/Images/models/`) :

- `diffusion_models/.../minimax_h3_fl2va_pruned_int8_convrot.safetensors` (21 Go,
  variante `h3ErosMax_beta5_fp8.safetensors` en FP8, 14 Go) : DiT **single-stream**
  de 50 blocs, hidden 5376, 56 tetes (head_dim 128), MLP SwiGLU (fc1 5376→28672
  split en 2×14336, fc2 14336→5376). Quantifie en **INT8 tensorwise** (marqueur
  `.comfy_quant` par tenseur + `.weight_scale`) — format deja reconnu par
  `native/weight_format.py::classify_quant_format` (meme convention que les
  checkpoints Flux.2/Krea2 deja portes), pas un nouveau format a gerer.
  `token_refiner` (2 blocs, non quantifies) affine les tokens texte avant injection.
  `adaln_t_table` [1025, 8] + `adaln_proj` par bloc (18 groupes de modulation/bloc)
  pour un timestep discret. Tetes de sortie separees `video_out` (5376→96) et
  `audio_out` (5376→32).
- `vae/minimax_h3_video_vae_fp16.safetensors` : encodeur causal 3D-conv (`ch_mult`
  jusqu'a 8, `use_3d_conv`), **decodeur ViT** (36 blocs, rope propre `rope_theta=100`,
  `rope_dim_ratio=0.75`) — pas un decodeur convolutif classique. Latent 24 canaux,
  ratio spatial 16, ratio temporel 4.
- `vae/minimax_h3_audio_vae_fp32.safetensors` : encodeur/decodeur **BigVGAN**,
  32kHz, latent 32 canaux.
- `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (27 Go) : Qwen3
  standard (50 couches, GQA q/k/v 8192/1024/1024, MLP gate/up/down 5120↔25600) +
  tour visuelle Qwen3-VL (naViT, deepstack merger). Meme quantification INT8
  tensorwise. `condition_proj` du DiT projette 5120 (hidden Qwen3VL) → 5376.

**Ce que ca change par rapport a l'estimation initiale :** l'architecture n'est
pas un DiT image/video classique a la Wan — c'est un **flux unique text+audio+video
packe** (`[text | cond | audio | video]`), avec deux schedules de sigma distincts
(shift video 12.0 / shift audio 3.0, reconvertis en forme close dans `model.py`),
un `ModelType.FLOW_AV` et un latent "nested" (`NestedTensor`, deux tenseurs de
forme differente portes ensemble) cote ComfyUI. Rien de comparable n'existe dans
`sampler/core.py` ou `sampler/scheduling.py` aujourd'hui — c'est un nouveau
paradigme de sampling a ajouter, pas juste un nouveau `_run_xxx`.

**Nodes ComfyUI reels, verifies sur le workflow de l'utilisateur**
(`user/default/workflows/video_minimax_h3_t2v.json`, sous-graphe "Image to Video
(MiniMax H3)", 21 nodes). Ca confirme precisement la surface a couvrir cote UI,
en plus du moteur :

- `comfy_extras/nodes_minimax_h3.py` expose 6 nodes `io.ComfyNode` (meme base
  class que ce projet utilise deja) : `EmptyMiniMaxH3LatentAV` (23 lignes, latent
  packe vide, calage sur la grille 17k+5 images a 24fps), `MiniMaxH3ImageToVideo`
  (51 lignes — le node utilise dans le workflow reel : prompt + first/last frame
  optionnels → conditioning + latent, logique fine mais courte, reutilise
  `clip.tokenize`/`encode_from_tokens_scheduled` et `vae.encode` standard),
  `MiniMaxH3SigmaShift` (45 lignes, patch de modele pour le double schedule).
  **`MiniMaxH3AddGuide`, `MiniMaxH3ReferenceToVideo` (REF2VA) et les 3 nodes
  `FunControlPatch`/`BlockPatch`/`FunControlNetApply` (ControlNet-style) sont
  hors scope du premier port** — pas utilises dans le workflow t2v/i2v de base,
  a traiter en increment separe (ControlNet est deja limite a FLUX.1 dans ce
  projet, §0).
- Le reste du workflow (`VAELoader`, `VAEDecode`, `VAEDecodeAudio`, `UNETLoader`,
  `CLIPLoader`, `KSamplerSelect` (`res_multistep`), `BasicScheduler` (`simple`,
  4 steps), `SamplerCustomAdvanced`, `RandomNoise`, `BasicGuider`, `CreateVideo`,
  `SaveVideo`) sont des nodes ComfyUI core deja couverts par le pattern de bridge
  existant, a l'exception de **`VAEDecodeAudio` et de la sortie audio en general
  (`AUDIO`)** : premiere fois que ce projet doit emettre ce type. `CreateVideo`
  (assemblage frames + audio en fichier video) est egalement nouveau.
- Le `LoraLoaderModelOnly` du workflow pointe vers
  `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (LoRA turbo deja
  present dans `models/loras/`) — mapping de cles LoRA specifique a prevoir,
  pas couvert par `lora.py` actuel (aucune famille MiniMax H3 dedans).
- **Divergence relevee, non bloquante :** le `CLIPLoader` du workflow sauvegarde
  reference `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (variante NVFP4+AWQ) —
  **ce fichier n'existe pas sur le disque de l'utilisateur** (seuls l'INT8 25GB
  et le GGUF Q4_K_M 17GB sont presents). A ignorer pour le dimensionnement tant
  qu'il n'est pas obtenu ; NVFP4 n'est pas un format gere par
  `native/weight_format.py` aujourd'hui si jamais il devient pertinent.

| Fichier | Contenu | Lignes (ref. Comfy) | Effort |
|---------|---------|---------------------|--------|
| `native/minimax_h3/model.py` | DiT single-stream MLX, port fidele de `comfy/ldm/minimax/model.py` | 784 | 5-8j |
| `native/minimax_h3/config.py` + `weight_map.py` | Config + mapping INT8 tensorwise (reutilise `weight_format.py`) | 250 | 1-2j |
| `native/minimax_h3/text_encoder.py` | Qwen3 (+ tour visuelle Qwen3-VL) MLX ; aucun encodeur Qwen n'existe encore dans le projet (seul Krea2 a un encodeur natif) | 400-600 | 3-5j |
| VAE video MiniMax H3 | Decodeur ViT + rope propre — `comfy.sd.VAE` d'abord, port MLX seulement si goulot | 300 | 1-2j |
| VAE audio MiniMax H3 (BigVGAN) + sortie `AUDIO`/`CreateVideo` | Via `comfy.sd.VAE` d'abord ; premiere fois que ce projet emet de l'audio et assemble un fichier video | 200 | 1-2j |
| `sampler/scheduling.py` | Double schedule sigma (video/audio) + conversion closed-form (port `MiniMaxH3SigmaShift`) | 150 | 1-2j |
| `sampler/core.py::_run_minimax_h3` | Boucle de denoising sur flux packe text+audio+video (pas de precedent dans le projet) | 300-400 | 3-5j |
| Nodes ASDX dedies (equivalents `EmptyMiniMaxH3LatentAV`, `MiniMaxH3ImageToVideo`) | Logique courte, reutilise le bridge CLIP/VAE existant | ~120 | 1-2j |
| LoRA MiniMax H3 (mapping de cles, cf. `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`) | Nouvelle famille dans `lora.py` | — | 1-2j |
| Verification numerique vs `comfy/ldm/minimax/model.py` (skill `verify-checkpoint` + `comfy-reference-diff`) | — | — | 1-2j |

**Total phase 6 (t2va/fl2va + LoRA turbo, sans REF2VA/AddGuide/ControlNet) :
18-30 jours** — plus lourd que Wan (6-10j) et Hunyuan (5-8j) reunis, a cause du
flux audio+video joint et du nouvel encodeur Qwen3-VL. REF2VA, AddGuide et le
ControlNet-style MiniMax H3 : effort supplementaire non chiffre, a evaluer une
fois le t2va/fl2va de base valide.

**Contrainte materielle cible : MacBook Pro M5 Max, 64GB de memoire unifiee.**
Les checkpoints DiT de l'utilisateur pesent tous environ **20GB SUR DISQUE**
(variante retenue : `..._pruned_int8_convrot`, INT8 tensorwise — `h3ErosMax_beta5_fp8`
existe aussi en 13GB mais n'est pas la reference a dimensionner). Texte
encoder Qwen3-VL-32B : 25GB sur disque (INT8). VAE : 4.9GB video + 0.6GB audio.

**Correction (implementation Phase 6, apres coup) : ces tailles sont celles du
fichier QUANTIFIE sur disque, pas la taille reelle en memoire une fois
dequantifie en dense** — la convention historique de ce projet pour toutes les
autres familles (`_load_safetensors` : tout dequantifier en dense, aucun kernel
GEMM quantifie). Mesure sur le vrai nombre de parametres (pas la taille fichier) :
DiT 20,1 milliards de parametres → **~40GB dense en fp16, ~80GB en fp32** ;
text encoder 25,8 milliards → **~51,5GB dense en fp16, ~103GB en fp32**. Cette
convention **ne tient pas dans 64GB**, meme un seul etage a la fois pour le
text encoder. Solution retenue et implementee : garder les gros poids (les
Linear par bloc : `qkv_proj`/`out_proj`/`fc1`/`fc2`) au format quantifie natif
MLX (`mx.quantize`/`nn.QuantizedLinear`, meme mecanisme que `mlx-lm` pour les
gros modeles) au lieu de les dequantifier en dense — chaque tenseur source est
dequantifie de façon transitoire (un seul a la fois) puis immediatement
requantifie, jamais materialise en dense pour tout le modele. Voir
`native/minimax_h3/quantized_linear.py` + `weight_map.py`, verifie sur le vrai
checkpoint DiT (13.9GB, 532 tenseurs, chargement complet en ~3 min).

**Chargement sequentiel etage par etage reste necessaire** (encoder → libere →
DiT → libere → VAE → decode) : meme avec le format quantifie, avoir les trois
etages residents simultanement n'est pas souhaitable — mais la marge est
desormais dictee par la taille quantifiee-en-memoire de chaque etage (proche
de la taille fichier, pas 2-4x plus grande), pas par la taille dense.

**GGUF — requis a la fois pour le text encoder et le DiT (diffusion_models),
et verifiable de bout en bout des maintenant.** Contrairement a ce qui avait ete
dit precedemment dans cette session, **un GGUF MiniMax H3 pour le DiT existe
bien en local** : `models/unet/MiniMax H3/minimax_h3_fl2va_pruned-Q5_0.gguf`
(13Go). Header verifie directement avec le lecteur reel (voir ci-dessous) :
532 tenseurs, **exactement la meme architecture que le safetensors** — 50 blocs
principaux + 2 blocs `token_refiner`, memes noms/formes de tenseurs
(`blocks.N.attn.qkv_proj.weight` etc.), les poids lineaires en **Q5_0** (`210`
tenseurs) et les normes/adaLN en F16/BF16/F32 non quantifies. Q5_0 est un quant
"legacy" simple (bloc de 32 valeurs, 5 bits + une seule echelle F16) — **pas un
K-quant a superblocs**, plus simple a dequantifier que le Q4_K_M du text
encoder. Avec le Qwen3-VL-32B Q4_K_M (17GB) deja present aussi, **les deux
chemins (text encoder ET DiT) sont verifiables contre un vrai checkpoint des
la premiere implementation** — plus de verification differee.

**Reference d'implementation confirmee en local :**
`/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/custom_nodes/gguf` (paquet `calcuis/gguf`,
PyPI `gguf-node`, ComfyUI Registry — pas le plus connu `city96/ComfyUI-GGUF`,
mais un vrai node communautaire maintenu, 12800 lignes). Le lecteur de cette
reference (`gguf_connector.reader.GGUFReader`) a ete utilise directement en
session pour parser le fichier DiT reel et confirmer les 532 tenseurs ci-dessus
— pas une lecture de code a l'aveugle, un test reel. Points utiles pour le
portage MLX :

- `gguf_connector/reader.py` + `gguf_connector/const.py` : lecteur de header GGUF
  standard (magique, version, `GGMLQuantizationType`) — reprend le format du
  paquet `gguf` officiel llama.cpp.
- `gguf_connector/quant.py` (832 lignes) : dequantiseurs complets — legacy
  `Q4_0`/`Q5_0`/`Q8_0` (blocs simples, **couvrent le DiT MiniMax H3 reel**) et
  K-quant `Q4_K`/`Q5_K`/`Q6_K` (blocs a superblocs, **couvrent le text encoder
  Q4_K_M reel**). Les deux familles de dequant sont necessaires — le DiT et le
  text encoder n'utilisent pas le meme schema de quantification.
- `pig.py::load_gguf_sd(path, handle_prefix='model.diffusion_model.')` : loader
  dedie aux **diffusion_models** GGUF (le node `LoaderGGUF`), distinct du chemin
  `ClipLoaderGGUF` utilise pour le text encoder. Confirme sur le fichier reel :
  les noms de tenseurs GGUF du DiT correspondent tels quels a ceux du safetensors
  (`blocks.N....`), donc `handle_prefix` s'applique proprement.
  La reference charge les tenseurs quantifies en `GGMLTensor` (lazy, dequantifie
  a la volee dans le forward via `lazy.py`) plutot que de tout dequantifier au
  chargement — a evaluer si ce choix vaut la peine en MLX (memoire unifiee =
  dequantifier une fois au chargement dans un `mx.array` est probablement plus
  simple et suffisant vu la marge RAM calculee plus haut) ou si le lazy dequant
  reste preferable pour limiter le pic memoire pendant le chargement lui-meme.

**Etat du classifieur existant :** `native/weight_format.py::classify_quant_format`
ne couvre que le safetensors (FP8_SCALED/FP4_PACKED/INT8_TENSORWISE) — GGUF est
un conteneur binaire completement different (header llama.cpp, blocs a quant
legacy ou K-quant) et demande un lecteur + un dequantizer dedies, branches a
cote du classifieur existant plutot que dedans.

| Fichier | Contenu | Effort |
|---------|---------|--------|
| Lecteur GGUF (header + tensor info) | Port MLX de `gguf_connector/reader.py` + `const.py` | 1-2j |
| Dequantizer legacy (Q4_0/Q5_0/Q8_0) — cible DiT | Port MLX de `gguf_connector/quant.py`, verifie contre `minimax_h3_fl2va_pruned-Q5_0.gguf` reel | 1-2j |
| Dequantizer K-quant (Q4_K/Q5_K/Q6_K) — cible text encoder | Port MLX, verifie contre `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` reel | 2-3j |
| `native/minimax_h3/text_encoder.py` — chemin de chargement GGUF | Equivalent MLX de `ClipLoaderGGUF` ; verifie sur fichier reel | 1j |
| `native/minimax_h3/model.py` — chemin de chargement GGUF (diffusion_models) | Equivalent MLX de `LoaderGGUF`/`load_gguf_sd(handle_prefix='model.diffusion_model.')` ; verifie sur fichier reel | 1-2j |
| Branchement `loader.py` (routage GGUF vs safetensors par extension de fichier) | — | 0.5-1j |

**Total support GGUF (text encoder + DiT, verifie de bout en bout) : 6.5-11.5
jours supplementaires**, a ajouter au total Phase 6.

---

## 6. Resume des efforts restants

| Phase | Etat | Effort restant |
|-------|------|----------------|
| **0 — Fondations** | ✅ Fait (forme differente) | — |
| **1 — SDXL** | ✅ Fait (txt2img) | — |
| **1bis — img2img/inpaint SDXL + ControlNet SD** | Ouvert | **3-5j** |
| **2 — FLUX.2** | ✅ Fait | — |
| **3 — Wan 2.1** | Ouvert | **6-10j** |
| **4 — Hunyuan** | Ouvert | **5-8j** |
| **5 — PixArt / Ideogram / SVD** | Ouvert (Krea2 fait) | **6-7j** |
| **6 — MiniMax H3** | Ouvert, architecture investiguee | **25-42j** (18-30j t2va/fl2va+LoRA + 6.5-11.5j GGUF text encoder+DiT ; REF2VA/AddGuide/ControlNet en sus, non chiffre) |
| **TOTAL RESTANT** | | **~45-72j** |

---

## 7. Memoire unifiee — strategie

### Regles d'or

1. **Un seul buffer par tenseur** — jamais de copie inutile entre MLX et PyTorch
2. **Eval strategique** — `mx.eval()` uniquement aux points de bridge
3. **Cache MLX limite** — `mx.set_cache_limit()` adapte a la RAM disponible
4. **Nettoyage inter-phases** — `mx.clear_cache()` entre loader, encode, sampling, decode

### Conversion MLX ↔ PyTorch (bridge)

```
PyTorch → MLX (input):
  tensor.detach().cpu().numpy() → mx.array() → mx.eval()

MLX → PyTorch (output):
  mx.eval() → np.array() → torch.from_numpy().to("mps")

Regles:
  - Minimiser les conversions, batcher quand possible
  - Toujours appeler mx.eval() avant conversion MLX→NumPy
  - Utiliser des buffers numpy reutilisables quand possible
```

### Memoire par modele (FP16) — estimations pour les familles restantes

| Modele | RAM min. recommandee | Poids + Etat | Avec TeaCache |
|--------|---------------------|--------------|---------------|
| Wan 2.1 | 36GB | ~18GB | ~10GB |
| Hunyuan | 16GB | ~6GB | ~4GB |
| MiniMax H3 | **64GB (M5 Max de l'utilisateur), a condition de charger sequentiellement ET de garder les poids quantifies (pas de dequant dense)** | Poids **sur disque** : text encoder Qwen3-VL-32B INT8 25GB (ou GGUF Q4_K_M 17GB), DiT INT8 ~20GB (ou GGUF Q5_0 ~14GB). **En memoire dense (evite) : DiT ~40GB fp16/~80GB fp32 (20,1 Md param.), text encoder ~51,5GB fp16/~103GB fp32 (25,8 Md param.)** — mesure reelle, corrige une estimation anterieure de cette session basee a tort sur la taille fichier. Solution implementee : `mx.quantize`/`nn.QuantizedLinear` (poids quantifies natifs MLX, memoire proche de la taille fichier) pour le DiT, verifie sur le vrai checkpoint (13.9GB, chargement ~3 min, 948/948 params). Meme approche requise pour le text encoder. | non evalue |

---

## 8. Tests de validation (par nouveau modele)

Pour SDXL / FLUX.2 / Krea2 / Z-Image, la recette de verification etablie du projet
(skill `verify-checkpoint`) fait foi et a deja tourne. Les listes ci-dessous ne
concernent plus que les familles a livrer (Wan, Hunyuan, PixArt, Ideogram, SVD, MiniMax H3).

### Tests numeriques

- [ ] Predictions stables (pas de NaN/Inf)
- [ ] Similarite cosinus > 0.99 vs PyTorch reference
- [ ] Chargement d'un vrai checkpoint : N/M cles appariees, std vs init aleatoire

### Tests fonctionnels

- [ ] Workflow complet : charge → encode → sample → decode → image
- [ ] img2img fonctionne
- [ ] inpainting fonctionne (si supporte)
- [ ] LoRA fonctionne
- [ ] ControlNet fonctionne (si supporte — actuellement FLUX.1 uniquement)
- [ ] IP-Adapter fonctionne (si supporte — actuellement dans `disabled_nodes/`)

### Tests de performance

- [ ] Temps de generation mesure et documente
- [ ] Memoire maximale mesuree
- [ ] Comparaison avec implementation PyTorch reference

### Tests de compatibilite

- [ ] Resolutions supportees documentees
- [ ] Batch size 1 garanti
- [ ] Float16 et bfloat16 testes
- [ ] Fallback CPU teste

---

## 9. Ordre d'execution restant

```
Phase 1bis (img2img/inpaint SDXL + ControlNet SD) ──────── 3-5j
    │
    ├──→ Phase 3 (Wan 2.1) ──────────────────────────────── 6-10j
    │       └──→ Tests Wan
    │
    ├──→ Phase 4 (Hunyuan) ──────────────────────────────── 5-8j
    │       └──→ Tests Hunyuan
    │
    ├──→ Phase 5 (PixArt, Ideogram, SVD) ────────────────── 6-7j
    │
    └──→ Phase 6 (MiniMax H3) ───────────────────────────── a chiffrer
            └──→ Investigation architecture d'abord
```

Les quatre branches sont independantes : les fondations (§4) sont en place, chacune
ajoute un sous-package `native/<famille>/` sans toucher au noyau.

---

## 10. Risques et mitigations

| Risque | Probabilite | Impact | Mitigation |
|--------|-------------|--------|------------|
| Architecture modele inconnue (pas de specs) | Moyenne | Eleve | Analyser les poids pour deduire l'architecture (inspecter le header safetensors AVANT d'ecrire le weight_map) |
| Memoire insuffisante (Wan 14B) | Moyenne | Eleve | Quantification MLX (int8, fp8), streaming |
| Incompatibilite numerique MLX vs PyTorch | Haute | Moyen | Recette `verify-checkpoint` avant toute annonce de support |
| ControlNet non disponible hors FLUX.1 | Confirme | Moyen | `disabled_nodes/controlnet/` a reactiver et porter par famille |
| Modele video (Wan/SVD) : rope temporel + VAE 3D | Haute | Eleve | Aucun precedent dans le repo — prevoir une marge sur la phase 3 |

---

## 11. Migration progressive

✅ **Resolu autrement.** Ni le dispatcher a flag (`ASDX_MULTI_MODEL`, absent du code)
ni `load_diffusion_model()` generique n'ont ete necessaires : `loader.py::_detect_model_type`
detecte la famille depuis le nom puis depuis les cles du checkpoint, et retombe sur
`"dev"` (FLUX.1) si la detection echoue — la retrocompatibilite est donc le comportement
par defaut, sans variable d'environnement.

Pour les phases 3-5, ajouter une branche dans `_detect_model_type_from_keys` avec un
marqueur de cle propre a la famille (voir `weight_format.py::classify_quant_format` pour
la meme discipline cote quantification : jamais de devinette plausible, une erreur plutot
qu'un faux positif).

---

## 12. Checklist de livraison (restante)

### Phase 1bis (completion SDXL)
- [ ] img2img / inpainting cables pour SDXL
- [ ] ControlNet SD fonctionne (reactivation de `disabled_nodes/controlnet/`)
- [ ] Memoire et performances mesurees

### Phases 3-5
- [ ] Chaque modele passe la recette `verify-checkpoint` sur un vrai checkpoint
- [ ] Chaque modele passe les tests de §8
- [ ] README.md mis a jour (tableau des familles supportees)
- [ ] Documentation a jour

### Phase 6 (MiniMax H3)
- [x] Architecture investiguee (headers safetensors des checkpoints reels + reference `comfy/ldm/minimax/`, 2026-09-14)
- [x] Effort re-estime (25-42j : 18-30j t2va/fl2va+LoRA + 6.5-11.5j GGUF text encoder+DiT ; REF2VA/AddGuide/ControlNet en sus)
- [x] `native/minimax_h3/config.py` : detection d'architecture (`detect_minimax_h3_config`) verifiee sur les headers reels des deux formats (safetensors + GGUF), config identique des deux cotes
- [x] `native/minimax_h3/model.py` : DiT complet pour le chemin minimal t2va (rope.py, layout.py, patchify.py, model.py — RMSNorm/Attention/MLP/AdalnProj/TokenRefiner/DiTBlock/FinalLayer/MiniMaxH3Model), chaque brique verifiee numeriquement contre la vraie reference (comfy_kitchen eager backend pour le rope fusionne, `comfy.ldm.minimax.model` importe directement pour layout/patchify/curve embedding). Hors scope : REF2VA, AddGuide, gate_compress/VSA, PDD head bank, pad/crop non-aligne — voir le docstring de module de `model.py`.
- [x] `native/minimax_h3/weight_map.py` (GGUF) : chargement complet du DiT verifie sur le vrai fichier (948/948 params, ~3 min). **Decouverte critique en cours de route : dequantifier en dense (convention existante du projet pour les autres familles) ne tient PAS dans 64 Go pour ce modele** — DiT 20,1 Md param. = ~40 Go dense fp16/~80 Go fp32 ; text encoder 25,8 Md = ~51,5 Go dense fp16/~103 Go fp32 (mesure sur le vrai nombre d'elements, pas la taille fichier quantifiee). Solution : poids gardes en format quantifie natif MLX (`mx.quantize`/`nn.QuantizedLinear`, memes mecanismes que mlx-lm pour les gros modeles) — voir `native/minimax_h3/quantized_linear.py`. Chargement safetensors (INT8) pas encore implemente (GGUF est la cible principale de l'utilisateur).
- [ ] Profil `capability.py` + `native/minimax_h3/text_encoder.py` (Qwen3-VL, meme approche de chargement quantifie requise vu les 25,8 Md de parametres)
- [x] Encodeur Qwen3-VL natif (`native/minimax_h3/text_encoder.py`) : **scope reduit — texte seul, tour vision hors perimetre**. Decouverte en implementant : le M-RoPE multimodal de Qwen3-VL degenere en RoPE standard des lors qu'aucune image n'est presente (`position_ids.shape[0]==1`), donc la tour vision (27 blocs ViT + DeepStack) n'est jamais invoquee pour le chemin t2va sans keyframes — non implementee, pas juste differee. Backbone Qwen3 causal tronque a 50 couches (sur 64), GQA (64 heads/8 kv heads), verifie contre `comfy.text_encoders.llama` reel. Chargement GGUF quantifie (`text_encoder_weight_map.py`, meme strategie que le DiT) verifie sur le vrai fichier (18Go, 902 tenseurs, 1253/1253 params, ~2min45).
- [ ] Sortie audio (`AUDIO`) cablee — premiere fois dans ce projet
- [ ] Chargement sequentiel encodeur → DiT → VAE verifie (pic RAM mesure < 64GB sur M5 Max, aucun etage resident en meme temps qu'un autre)
- [x] Lecteur GGUF (header-only, `native/gguf/reader.py` + tests dans `tests/native/gguf/test_reader.py`) — verifie contre les deux fichiers reels (2026-09-15) : DiT `minimax_h3_fl2va_pruned-Q5_0.gguf` (532 tenseurs, 50+2 blocs, torch_shape identique au safetensors, 210 tenseurs Q5_0) et text encoder `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (902 tenseurs, meme structure que le safetensors, mix Q4_K/Q6_K — Q6_K sur `down_proj`/`v_proj`, coherent avec le schema "Q4_K_M" standard llama.cpp)
- [x] Dequantizer legacy (Q5_0) — cible DiT, verifie bit-exact contre `calcuis/gguf` + contre le vrai fichier (`native/gguf/dequant.py`). **Finding : un poids lineaire quantifie ne matche PAS entre le GGUF Q5_0 et le safetensors INT8-convrot** meme apres dequant des deux cotes — le safetensors applique une rotation Hadamard "ConvRot" (`_dequantize_comfy_quant_int8`) que le GGUF n'a pas ; les deux ne convergent qu'apres la chaine de reconstruction complete propre a chaque format. Un poids non quantifie (BF16, ex. `norm1.weight`) matche bit-exact entre les deux fichiers, confirmant que la lecture/l'offset/le reshape sont corrects.
- [x] Dequantizer K-quant (Q4_K + Q6_K) — cible text encoder, verifie bit-exact contre `calcuis/gguf` et contre le vrai fichier (`native/gguf/dequant.py`). Confirme le schema Q4_K_M standard llama.cpp (Q6_K sur `down_proj`/`v_proj`, Q4_K ailleurs)
- [x] `ASDX_MiniMaxH3EmptyLatentAV` + `ASDX_MiniMaxH3SigmaShift` portes (deux LATENT separes video/audio au lieu du NestedTensor packe ComfyUI ; SigmaShift ecrit directement `MiniMaxH3Config.sigma_shift_*` au lieu de patcher `ModelPatcher`)
- [x] VAE : aucun port MLX necessaire — `ASDX_VAEDecode` gere deja le latent video 5D generiquement ; ajout de `ASDX_VAEDecodeAudio` (premiere sortie AUDIO du projet, miroir de `comfy_extras.nodes_audio.vae_decode_audio`)
- [ ] `MiniMaxH3ImageToVideo` (conditioning texte->video) : bloque sur le tokenizer Qwen3-VL reel (a cabler via un objet `comfy.sd.CLIP`, meme motif que `krea2_grounded_encode.py`) — pas devine, differe
- [ ] `sampler/core.py::_run_minimax_h3` (boucle de denoising) — pas commence
- [ ] LoRA MiniMax H3 (mapping de cles) fonctionne avec `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`
- [ ] Passe la recette `verify-checkpoint` + `comfy-reference-diff` contre `comfy/ldm/minimax/model.py`

### General
- [ ] Tous les tests passes
- [ ] Benchmarks compares avec PyTorch
- [ ] Graphify run pour verifier couverture

---

## 13. References

- Wan 2.1: https://github.com/Wan-Video/Wan2.1
- Hunyuan DiT: https://github.com/Tencent/HunyuanDiT
- PixArt Sigma: https://github.com/PixArt-alpha/PixArt-sigma
- MiniMax H3 : reference d'implementation = ComfyUI lui-meme, deja a jour dans
  l'installation locale (`comfy/ldm/minimax/model.py`, `comfy/text_encoders/minimax.py`
  + `qwen3vl.py`, `comfy_extras/nodes_minimax_h3.py`) — meme statut que Wan/Hunyuan/
  FLUX.2/Krea2 pour la regle CLAUDE.md "ComfyUI et le stack SceneWorks sont les
  references". Verifie en session le 2026-09-14 (§5, Phase 6).
- MiniMax H3 GGUF : reference d'implementation = `calcuis/gguf` (PyPI `gguf-node`,
  https://github.com/calcuis/gguf), installe localement dans
  `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/custom_nodes/gguf`. Lecteur + dequant
  K-quant dans `gguf_connector/reader.py`, `const.py`, `quant.py` ; loader
  diffusion_models dans `pig.py::load_gguf_sd`. Verifie en session le 2026-09-14.
- MiniMax H3 nodes/workflow : `user/default/workflows/video_minimax_h3_t2v.json`
  (installation ComfyUI locale) — workflow reel utilise par l'utilisateur, sous-graphe
  "Image to Video (MiniMax H3)". A servi a identifier precisement les nodes
  MiniMax H3 en scope (`EmptyMiniMaxH3LatentAV`, `MiniMaxH3ImageToVideo`,
  `MiniMaxH3SigmaShift`) vs hors scope (`AddGuide`, `ReferenceToVideo`,
  `FunControlPatch`/`BlockPatch`/`FunControlNetApply`).
- MLX: https://ml-explore.github.io/mlx/
- ComfyUI: https://github.com/comfyanonymous/ComfyUI (implementation de reference — voir le
  record de canon `ComfyUI and the SceneWorks stack are the reference implementations`)
