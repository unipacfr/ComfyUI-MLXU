# Qwen Image 2.1 — brique 3b : boucle d'échantillonnage (`_run_qwen_image21`)

Date: 2026-09-22
Statut: approuvé (brainstorming), en attente du plan d'implémentation
Complète : brique 3 (tâches 1-3, mergées) — découvert manquant par le test end-to-end réel
(brique 3, tâche 4)

## Contexte

Le test end-to-end sur ComfyUI réel (brique 3, tâche 4) a révélé que le sampler générique
(`ASDX_MLXSampler`) n'a PAS de dispatch générique par `model_type` pour le conditioning/latent
— chaque famille (SDXL, Flux2, Z-Image, Krea2) a sa propre méthode `_run_<famille>` dans
`sampler/core.py::_SamplerCore` (~140-260 lignes chacune : ordonnancement du bruit, boucle de
débruitage, appels au DiT). Les fonctions de pont écrites en brique 3 tâche 2
(`conditioning_qwen_image21_to_mlx`, `mlx_to_comfy_latent_qwen_image21`,
`prepare_noise_from_latent_qwen_image21`) sont nécessaires mais jamais appelées nulle part sans
cette brique.

De plus, le test a révélé un bug de type de socket : `ASDX_QwenImage21TextEncode` (brique 3,
tâche 3) sort `"asdx_qwen_image21_conditioning"`, mais `ASDX_MLXSampler` attend
`"mlx_conditioning"` sur son entrée `positive` — incompatibilité de connexion.

## Découverte critique (évite un bug silencieux type Krea2)

`comfy/model_base.py::QwenImage21.__init__(model_type=ModelType.FLUX, ...)` — Qwen Image 2.1
s'enregistre `ModelType.FLUX` dans ComfyUI, **exactement comme Krea2**, PAS comme Flux2/Z-Image.
Cela signifie que son échantillonnage utilise `ModelSamplingFlux`/`flux_time_shift` (pas
`ModelSamplingDiscreteFlow`/`time_snr_shift`) — cette confusion a déjà causé un bug silencieux
réel pour Krea2 par le passé (canon existant, `sampler/scheduling.py` module docstring :
"Using time_snr_shift here was a real, previously-uncaught bug").

`comfy/supported_models.py::QwenImage21.sampling_settings = {"multiplier": 1.0, "shift": 0.69}`
— un shift FIXE (pas la formule mu résolution-dépendante de FLUX.1-dev), même famille de
formule que Krea2 (shift=1.15) et Flux2 (shift=2.02, mais eux avec `time_snr_shift`). Le
commentaire "scheduler mu at 1024x1024 (base 0.5 @ 256 tokens, max 0.9 @ 8192)" est de la
documentation sur l'origine du chiffre 0.69, pas une formule à réimplémenter.

`apple_silicon_nodes/sampler/scheduling.py::_flux_fixed_shift_sigmas(shift, steps)` existe
déjà (écrite pour Krea2) et est générique — réutilisable directement avec `shift=0.69`.

## Périmètre

**Dans le périmètre** :
- `sampler/scheduling.py` : branch `model_type == "qwen_image21"` dans `generate_sigmas` et
  `_flow_shift_fn`, réutilise `_flux_fixed_shift_sigmas(0.69, steps)`
- `sampler/core.py` : nouvelle méthode `_SamplerCore._run_qwen_image21(steps, seed)`
- `sampler/__init__.py` : branch `model_type == "qwen_image21"` dans le dispatch de
  préparation du bruit (`bridge.prepare_noise_from_latent_qwen_image21`) + appel à
  `_run_qwen_image21` dans `run()`
- Correction du type de socket de sortie de `ASDX_QwenImage21TextEncode` :
  `"asdx_qwen_image21_conditioning"` → `"mlx_conditioning"`
- Test end-to-end réel (reprise de la brique 3 tâche 4) : bf16 puis GGUF, T2I complet

**Hors périmètre** :
- CFG / negative prompt (le modèle n'a pas d'embedding de guidance, pas de branche two-pass)
- img2img / inpainting / ControlNet pour cette famille
- TeaCache/SeaCache (peuvent être ajoutés plus tard si mesuré utile, comme pour Flux2)

## Architecture

`_run_qwen_image21` suit le pattern de `_run_flux2` (flow-matching, pas de CFG) avec les
différences suivantes, dictées par l'architecture réelle du DiT (brique 2) :
- **Pas de `.predict()`/`rope` externe** : `QwenImage21Transformer2DModel.__call__(x, timestep,
  context)` calcule le RoPE en interne (`_build_sequence`) — appel direct
  `self.transformer(x, timestep, context)`, pas de `get_rope()` préalable.
- **`x` reste `[B,C,H,W]` (NCHW) tout du long** — pas de flattening en tokens côté appelant
  (contrairement à Flux2 qui passe `img` comme tokens `[B,N,C]`) : le DiT gère lui-même la
  conversion grille↔séquence en interne.
- **Pas de guidance** : un seul passage conditionnel par pas, pas de paramètre `guidance` à
  transmettre au DiT (sa signature n'en a pas).
- **Conversion latent** : `bridge.prepare_noise_from_latent_qwen_image21`/
  `mlx_to_comfy_latent_qwen_image21` (déjà écrites, brique 3 tâche 2) — cas le plus simple du
  projet, aucune transposition de layout (NCHW des deux côtés).
- **Update flow-matching** : `denoised = x - noise_pred * sigma` (même convention que Flux2/
  Z-Image), via `solvers.step(...)` générique déjà utilisé par toutes les familles.

## Tests / vérification

1. Tests unitaires pour la nouvelle branche de `scheduling.py` (comparaison numérique contre
   `_flux_fixed_shift_sigmas` appelée directement avec shift=0.69, et contre Krea2's
   `flux_time_shift` à shift différent pour confirmer que c'est la même fonction, pas une
   coïncidence)
2. Test manuel end-to-end sur ComfyUI réel (canon : le dispatch famille doit être vérifié
   contre un run ComfyUI réel) — REPREND la brique 3 tâche 4, bloquée par ce manque

## Risques identifiés

- **`_run_qwen_image21` est un nouveau code non testé par petits tests unitaires isolés**
  (comme les autres `_run_<famille>`, qui ne le sont pas non plus dans ce projet) — la seule
  vérification réelle est le test end-to-end sur ComfyUI. Accepté, cohérent avec le reste du
  projet.
- **Shift=0.69 est un chiffre du commentaire ComfyUI, pas re-dérivé** — si une future version
  de Qwen Image 2.1 change son `sampling_settings`, ce chiffre devient périmé silencieusement
  (mais c'est aussi le cas pour Flux2/Z-Image/Krea2's shifts fixes existants, pas un risque
  propre à cette brique).
