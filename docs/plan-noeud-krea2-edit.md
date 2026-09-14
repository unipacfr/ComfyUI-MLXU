# Plan : nœud dédié `ASDX_Krea2Edit` (Krea2 Identity Edit)

**Date de création :** 2026-08-14
**Statut :** ✅ Réalisé (2026-09-14) — voir `apple_silicon_nodes/krea2_edit.py` et
`docs/superpowers/plans/2026-09-13-node-extraction-and-depth-control.md` Phase A.
Confirmé sans artefact visage par l'utilisateur. Ce document reste comme spec de
référence pour le nœud tel qu'implémenté.
**Objectif :** sortir l'Identity Edit du sampler et en faire un nœud composable, qui
porte aussi le chargement du LoRA Krea2 Identity Edit et débloque le fit en espace
pixel.

---

## 1. Contexte : pourquoi ce nœud

Session du 2026-08-14. Un test d'Identity Edit sous ASDX donnait un résultat où
l'instruction d'édition ne s'appliquait que partiellement, alors que le même prompt
sur deux implémentations indépendantes — le workflow de référence
`comfyui-krea2edit` et SceneWorks — donnait des résultats identiques entre elles.

Cause identifiée : **le latent source n'était jamais ajusté à la grille cible.**

- Référence `comfyui-krea2edit` : `_fit_src` (`__init__.py:59`), appelée par
  `krea2_edit_forward:192-194` sur toute source dont la grille diffère de la cible.
  Le chemin recommandé fait même le crop+resize **en pixels** avant l'encodage VAE
  (`_fit_encode_image:74`), parce que redimensionner un latent l'adoucit.
- SceneWorks : `fit_edit_references` (`crates/sceneworks-worker/src/image_jobs/flux2.rs:112`),
  appelée depuis `krea_edit.rs:211-214`, qui ajuste chaque référence à la résolution
  de sortie en pixels via `fit_rgb` mode `crop` (`image_jobs/base.rs:22`) —
  scale-to-cover puis center-crop. Le mode `stretch` y est étiqueté « legacy ».

Mesure sur le run réel : source `[1, 16, 166, 250]` soit une grille de 83×125 = 10 375
tokens, contre une cible 96×168 soit 48×84 = 4 032 tokens. La source occupait 72 % des
tokens image, tirant le résultat vers la reproduction de la référence et au détriment
de l'instruction, avec une géométrie RoPE hors distribution d'entraînement et un coût
d'attention quadratique (~31 s/step).

**Déjà corrigé** (non commité au moment d'écrire) : `_SamplerCore._fit_source_latent`
dans `sampler/core.py`, portage de `_fit_src` vérifié bit-identique à la référence sur
six cas de formes. C'est un fit en espace **latent**, le chemin de repli de la
référence. Le présent plan vise le chemin **pixel**, structurellement impossible depuis
le sampler qui ne reçoit qu'un latent déjà encodé.

Le nœud suit aussi la convention déjà en place : le sampler lit `controlnet`,
`capability`, `memory_shape` et `lora_schedule` depuis le dict `asdx_model`, pas
depuis ses propres entrées (voir le commentaire à `sampler/__init__.py:154-157`).

---

## 2. Le nœud

**Nouveau fichier** `apple_silicon_nodes/krea2_edit.py`, exportant `NODE_LIST`,
importé dans `__init__.py` (autour de la ligne 72) et ajouté à `_ALL_NODES` ainsi
qu'au dict `_DISPLAY`.

- `node_id` : `ASDX_Krea2Edit`
- `display_name` : `🍏 ASDX Krea2 Identity Edit`
- `category` : `ASDX/Conditioning`

### Entrées

| Entrée | Type | Rôle |
|---|---|---|
| `model` | `io.Custom("asdx_model")` | requis — reçoit le LoRA, porte la config vers le sampler |
| `image` | `io.Image` | requis — l'image source |
| `vae` | `io.Vae` | requis — encodage de la source |
| `lora_name` | `io.Combo` | le LoRA Identity Edit |
| `lora_strength` | `io.Float` 1.0 | |
| `ref_boost` | `io.Float` (défaut à trancher, voir §5) | dial de fidélité vers la source |
| `target_latent` | `io.Latent` optionnel | donne la grille cible, débloque le fit pixel |

**Sortie :** `io.Custom("asdx_model")`.

### Le champ `lora_name`

Liste tous les fichiers LoRA (même source que `ASDX_LoraLoader._get_loras`), avec
ceux dont le nom correspond à `identity[_ -]?edit` (insensible à la casse) triés en
tête, le plus récent en valeur par défaut. Toute mise à jour future du LoRA est ainsi
prise automatiquement, sans modification du code.

**Pas de détection par contenu, et c'est un constat vérifié sur les artefacts**, pas
un renoncement. Inspection des dix fichiers de `/Volumes/MBP2021/Images/models/loras/Krea 2/tool/` :

- `krea2_style_reference.safetensors` a exactement la même signature que les
  identity-edit — 512 clés, mêmes cibles `blocks.N.{attn,mlp}` + `txtfusion.{layerwise,refiner}_blocks` —
  donc la structure de clés ne discrimine pas.
- Six des huit fichiers identity-edit n'ont **aucune** metadata ; seul
  `krea2_identity_edit_v1.safetensors` en porte (`ss_output_name: krea2_edit_identity_1024`).

Le nom de fichier est le seul discriminant disponible. Le tri le met en avant sans
jamais masquer un autre fichier, ce qui laisse au passage `krea2_style_reference`
utilisable par le même nœud puisqu'il repose sur le même mécanisme de tokens source.

### `execute()`

1. Refuser si `model["capability"].family` n'est pas Krea2 — message explicite, jamais
   un silence (même principe que le `raise` de `_prepare_krea2_identity_edit`).
2. Charger et appliquer le LoRA en réutilisant l'existant, sans dupliquer la logique :
   `ASDX_LoraLoader._resolve_lora_path`, `_check_lora_compatibility` (`lora.py:129`),
   `ASDX_LoraLoader._load_lora_file`, `ASDX_LoraLoader._apply_lora_to_transformer`.
3. Préparer la source :
   - `target_latent` branché → `W, H = samples.shape[-1] * 8, samples.shape[-2] * 8`,
     puis scale-to-cover + center-crop de l'IMAGE **en pixels**, puis encodage via
     `ASDX_VAEEncode._fallback_encode` (`vae.py:291`), qui gère déjà le retour 5D des
     VAE Wan21 et le repli tuilé MPS.
   - sinon → encodage direct ; `_fit_source_latent` du sampler reste le filet.

   **Le resize pixel se fait en MLX**, canon *Porting from a reference means
   converting it to Apple Silicon, not transcribing it*. Contrairement au fit
   latent du sampler (~1,6 M éléments, exception mesurée et documentée sur place),
   celui-ci travaille sur une image pleine résolution — 2000×1328×3 ≈ 8 M éléments,
   deux ordres de grandeur au-dessus — donc le passage par l'hôte n'est plus
   négligeable. `mlx.nn.Upsample(mode="linear")` couvre le besoin : layout NHWC
   (transposition depuis le `[B,H,W,C]` de ComfyUI, qui est déjà le bon ordre),
   `scale_factor` fractionnaire qui tombe sur la taille exacte, écart mesuré de
   2,2e-05 relatif contre le bilinéaire torch. Le crop est une simple sélection,
   donc gratuit dans les deux cas. L'encodage VAE lui-même reste sur PyTorch-MPS,
   exception déjà au canon (*Porting VAE encode/decode to MLX has negative ROI*).
4. Retourner `{**model, "transformer": new_transformer, "identity_edit": {...}}` —
   copie superficielle, jamais de mutation du dict d'entrée, exactement comme
   `ASDX_LoraLoader` à `lora.py:1940`. Muter polluerait le modèle mis en cache.

### Ce que le nœud ne fait pas

**Pas de widget `fit_mode`.** Les deux références utilisent `crop` par défaut,
`stretch` est legacy chez SceneWorks et `pad` n'existe pas côté Krea2. Une option de
plus serait de la configurabilité non demandée (CLAUDE.md §2).

**Pas de grounded encode.** Il vit déjà dans `ASDX_CLIPTextEncode` (`conditioning.py:345-349`)
et a été vérifié ligne pour ligne identique à `Krea2EditGroundedEncode` de la
référence : même `_prep`, même `grounding_px=768`, même template, même appel
`tokenize(prompt, images=…, llama_template=…)`. Ne pas y toucher. L'image source
reste donc branchée à deux endroits, comme dans le workflow de référence.

---

## 3. Modifications des nœuds existants

**`sampler/__init__.py`** — `execute()` lit `model.get("identity_edit")` en priorité
sur les entrées `source_latent` / `ref_boost`, sur le modèle exact de `lora_schedule`
à la ligne 157. Les deux entrées **restent en place et optionnelles** : les retirer
casserait les workflows sauvegardés, ce que le canon *Node package migrated fully to
ComfyUI's V3 API* pose comme contrainte du projet. Logger si les deux chemins sont
alimentés simultanément, en indiquant lequel gagne (le dict).

**`sampler/core.py`** — inchangé. `_fit_source_latent` devient un no-op quand la
source arrive déjà fittée depuis le nœud, et garde son rôle sur les graphes câblés à
l'ancienne.

**`__init__.py`** — enregistrement (import, `_ALL_NODES`, `_DISPLAY`).

**`README.md`** — documenter le nœud dans le tableau Conditioning et ajuster le
paragraphe Identity Edit ajouté sous la section Sampling.

---

## 4. Vérifications

1. `py_compile` sur les fichiers touchés.
2. Le nœud en isolation : image 2000×1328 + `target_latent` 1344×768 → source en
   `[1, 16, 96, 168]`, comparée bit-à-bit à un crop+resize de référence calculé hors
   du nœud.
3. Chemin sans `target_latent` : la source arrive non fittée, le sampler doit logger
   `source latent fitted … -> 96x168`. Les deux chemins doivent converger vers la même
   grille finale.
4. Génération réelle, avec dans le log : `256/256 adapters`, la ligne de fit,
   `packed [1, 4032, 64] grid=48x84`, et un temps par step nettement inférieur aux
   ~31 s mesurés avant le fit.
5. Non-régression : un workflow câblé à l'ancienne (`source_latent` sur le sampler)
   doit produire le même résultat qu'avant.

Rappel du canon *Coverage and magnitude are separate checks* : vérifier aussi qu'un
forward réel reste fini et d'amplitude saine, pas seulement que les formes concordent.

---

## 5. Points à trancher avant implémentation

**`ref_boost` par défaut.** Le workflow de référence ship 4.0 (`Krea2EditModelPatch | [4, 1, 'fit']`),
décrit comme « much stronger face + body likeness, more reliable edits », alors que le
sampler ASDX est à 1.0. Mettre 4.0 sur le nouveau nœud change le rendu par défaut de
façon visible. À confirmer par l'utilisateur.

**Le double LoRA.** Le nœud appliquant lui-même l'Identity Edit, il faut retirer du
graphe le LoRA Loader qui le porte aujourd'hui, sinon il s'applique deux fois.
Option : détecter des adaptateurs identity-edit déjà présents sur le transformer et
avertir, plutôt que d'empiler silencieusement.

---

## 6. Hors périmètre

Sortir également `mode` / `image` / `mask` / `depth_image` / `kontext` du sampler.
Le diagnostic tient — 25 entrées font un god-node — mais c'est un autre chantier, il
casse tous les workflows existants et ne résout aucun bug connu.

---

## 7. Réglages de référence, pour comparer

Relevés dans `custom_nodes/comfyui-krea2edit/workflows/krea2_identity_edit.json` :

| Paramètre | Référence |
|---|---|
| `ref_boost` / `fit_mode` | 4.0 / `fit` |
| Sampler | 10 steps, `euler`, `simple`, cfg 1 |
| Résolution | 1024×1024 (1 MP ; « inputs around 1MP work best ») |
| VAE | `qwen_image_vae.safetensors` |
| Modèle | `krea2_turbo_fp8_scaled` |
| Enhancer Krea2T | **absent du workflow** |

À noter : `krea2_enhancer_strength` vaut 1.0 par défaut dans `ASDX_MLXSampler` alors
qu'aucune des deux références ne l'applique. Le mettre à 0 pour toute comparaison
qui se veut valide.
