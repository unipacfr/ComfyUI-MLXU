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
| **6 — MiniMax H3** | ❌ Rien | pas de `native/minimax_h3/` ; l'installation locale a des packs tiers (`ComfyUI-MiniMax-H3-Turbo`, `ComfyUI-Spectrum-MiniMax-H3`, `ComfyUI-MiniMaxH3-Cache`, `ComfyUI-MiniMaxH3-FirstBlockCache`) qui tournent en PyTorch/comfy standard, aucun port MLX |
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
| **MiniMax H3** | H3 (+ Turbo) | ? | ? | ? | ? | A faire — Phase 6 (architecture a determiner, voir §5) |

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

### Phase 6 — MiniMax H3 (priorite a determiner)

**Objectif :** Supporter MiniMax H3. **Rien n'existe a ce jour — architecture non
investiguee.** Des packs tiers PyTorch/comfy existent deja dans l'installation locale
(`ComfyUI-MiniMax-H3-Turbo`, `ComfyUI-Spectrum-MiniMax-H3`, `ComfyUI-MiniMaxH3-Cache`,
`ComfyUI-MiniMaxH3-FirstBlockCache`, cf. "TJ NODE STUDIO ONE" dans les logs ComfyUI) —
utiles comme reference d'implementation PyTorch, mais aucun n'est un port MLX.

Avant d'estimer un effort : suivre la regle du §10 ("Architecture modele inconnue —
analyser les poids AVANT d'ecrire le weight_map") — inspecter le header safetensors
d'un checkpoint H3 reel pour determiner nombre de canaux VAE, dimension latente,
text encoder(s), et si c'est un modele image ou video (comme Wan, ce dernier cas
impliquerait un rope temporel et gonflerait fortement l'effort).

| Fichier | Contenu | Lignes | Effort |
|---------|---------|--------|--------|
| Investigation architecture (header safetensors, poids reels) | Prealable obligatoire | — | 0.5-1j |
| `native/minimax_h3/model.py` | MiniMax H3 MLX | ? | ? |
| `native/minimax_h3/config.py` + `weight_map.py` | Config + mapping de cles | ? | ? |
| VAE MiniMax H3 | Via `comfy.sd.VAE` d'abord | ? | ? |
| Encodeur texte | Via le `CLIPType` ComfyUI correspondant d'abord, si expose | ? | ? |
| `sampler/core.py::_run_minimax_h3` | Boucle de denoising | ? | ? |

**Total phase 6 : a chiffrer apres investigation**

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
| **6 — MiniMax H3** | Ouvert, a chiffrer | **a determiner** |
| **TOTAL RESTANT** | | **~20-30j + Phase 6** |

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
| MiniMax H3 | ? (a mesurer apres investigation architecture) | ? | ? |

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
- [ ] Architecture investiguee (header safetensors d'un vrai checkpoint)
- [ ] Effort re-estime une fois l'architecture connue
- [ ] Profil `capability.py` + `native/minimax_h3/` crees
- [ ] Passe la recette `verify-checkpoint`

### General
- [ ] Tous les tests passes
- [ ] Benchmarks compares avec PyTorch
- [ ] Graphify run pour verifier couverture

---

## 13. References

- Wan 2.1: https://github.com/Wan-Video/Wan2.1
- Hunyuan DiT: https://github.com/Tencent/HunyuanDiT
- PixArt Sigma: https://github.com/PixArt-alpha/PixArt-sigma
- MiniMax H3: pas de source officielle verifiee dans cette session — reference
  d'implementation la plus proche disponible localement : les packs tiers PyTorch/comfy
  de "TJ NODE STUDIO ONE" (`ComfyUI-MiniMax-H3-Turbo`, `ComfyUI-Spectrum-MiniMax-H3`,
  `ComfyUI-MiniMaxH3-Cache`, `ComfyUI-MiniMaxH3-FirstBlockCache`), a localiser sur la
  machine avant de commencer l'investigation architecture (§5, Phase 6)
- MLX: https://ml-explore.github.io/mlx/
- ComfyUI: https://github.com/comfyanonymous/ComfyUI (implementation de reference — voir le
  record de canon `ComfyUI and the SceneWorks stack are the reference implementations`)
