# Qwen Image 2.1 brique 3b : boucle d'échantillonnage + reprise du test end-to-end

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Câbler `QwenImage21Transformer2DModel` (brique 2) dans le sampler générique
(`ASDX_MLXSampler`), corriger le type de socket du text encoder (brique 3), et compléter la
vérification end-to-end sur ComfyUI réel restée bloquée en brique 3 tâche 4.

**Architecture:** Ajoute `_SamplerCore._run_qwen_image21` (`sampler/core.py`), une branche
`generate_sigmas`/`_flow_shift_fn` (`sampler/scheduling.py`, réutilise
`_flux_fixed_shift_sigmas` déjà écrite pour Krea2), une branche de dispatch dans
`sampler/__init__.py`, et corrige le type de sortie de `ASDX_QwenImage21TextEncode`.

**Tech Stack:** MLX, PyTorch (pont), `uv run` pour les tests unitaires, API HTTP directe de
ComfyUI (curl) pour le test end-to-end réel (`comfy-mcp` est cassé dans cette session, tous ses
appels échouent).

**Spec:** `docs/superpowers/specs/2026-09-22-qwen-image-21-brick3b-sampler-design.md`.
Référence directe : `apple_silicon_nodes/sampler/core.py::_run_flux2` (pattern le plus proche :
flow-matching, pas de CFG), `apple_silicon_nodes/sampler/scheduling.py::_flux_fixed_shift_sigmas`
(déjà générique, réutilisée telle quelle), `comfy/model_base.py::QwenImage21` (confirme
`ModelType.FLUX`), `comfy/supported_models.py::QwenImage21.sampling_settings` (`shift=0.69`).

## Global Constraints

- Toujours `uv run pytest` / `uv run python` (jamais `python` nu).
- Style du dépôt : Python 3.10+, annotations complètes, docstrings en anglais, pas d'emoji ni
  de tiret cadratin.
- **`QwenImage21Transformer2DModel.__call__(x, timestep, context)`** : `x` reste `[B,C,H,W]`
  (NCHW) du début à la fin, RoPE calculé en interne — PAS de `.predict()`/`rope` externe comme
  Flux2.
- **Aucune CFG** : le modèle n'a pas d'embedding de guidance, un seul passage conditionnel par
  pas.
- **`ModelType.FLUX`, pas `ModelType.FLOW`** : le shift utilise `flux_time_shift`
  (`ModelSamplingFlux`), pas `time_snr_shift` (`ModelSamplingDiscreteFlow`) — même famille de
  formule que Krea2, différente de Flux2/Z-Image malgré leur shift fixe similaire.
- **Aucun commit automatique** (règle utilisateur) : chaque étape « Commit » signifie « préparer
  le message et attendre l'ordre explicite de l'utilisateur ». Pas de ligne d'attribution.

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `apple_silicon_nodes/sampler/scheduling.py` (modifier) | branche `"qwen_image21"` dans `generate_sigmas`/`_flow_shift_fn` |
| `apple_silicon_nodes/sampler/core.py` (modifier) | nouvelle méthode `_run_qwen_image21`, routage dans `run()` |
| `apple_silicon_nodes/sampler/__init__.py` (modifier) | branche de préparation du bruit |
| `apple_silicon_nodes/qwen_image21_nodes.py` (modifier) | type de sortie `mlx_conditioning` |
| `tests/test_scheduling_qwen_image21.py` (créer) | vérifie la nouvelle branche de sigma |

---

### Task 1: Câblage complet du sampler

**Files:**
- Modify: `apple_silicon_nodes/sampler/scheduling.py`
- Modify: `apple_silicon_nodes/sampler/core.py`
- Modify: `apple_silicon_nodes/sampler/__init__.py`
- Modify: `apple_silicon_nodes/qwen_image21_nodes.py`
- Test: `tests/test_scheduling_qwen_image21.py`

**Interfaces:**
- Consumes: `bridge.conditioning_qwen_image21_to_mlx`, `bridge.mlx_to_comfy_latent_qwen_image21`,
  `bridge.prepare_noise_from_latent_qwen_image21` (brique 3 tâche 2, déjà commitées),
  `QwenImage21Transformer2DModel.__call__` (brique 2), `_flux_fixed_shift_sigmas` (déjà en
  place, `scheduling.py`).
- Produces: `_SamplerCore._run_qwen_image21(steps, seed) -> dict`, routé depuis `run()` pour
  `model_type == "qwen_image21"`. `ASDX_QwenImage21TextEncode` sort maintenant
  `"mlx_conditioning"` (connectable à `ASDX_MLXSampler`).

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""qwen_image21's sigma schedule: same flux_time_shift/ModelSamplingFlux family as Krea2,
fixed shift=0.69 (comfy/supported_models.py::QwenImage21.sampling_settings)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
scheduling_mod = load_node_module("sampler.scheduling")


def test_generate_sigmas_matches_flux_fixed_shift_directly():
    steps = 10
    expected = scheduling_mod._flux_fixed_shift_sigmas(0.69, steps)
    got = scheduling_mod.generate_sigmas(steps, "qwen_image21")
    assert got == expected


def test_generate_sigmas_differs_from_krea2_shift():
    # Confirms it's really using shift=0.69, not accidentally reusing Krea2's 1.15
    # or falling through to a different branch entirely.
    steps = 10
    qwen = scheduling_mod.generate_sigmas(steps, "qwen_image21")
    krea2 = scheduling_mod.generate_sigmas(steps, "krea2")
    assert qwen != krea2


def test_flow_shift_fn_matches_flux_time_shift_at_shift_0_69():
    fn = scheduling_mod._flow_shift_fn("qwen_image21")
    expected = scheduling_mod.flux_time_shift(0.69, 1.0, 0.5)
    assert abs(fn(0.5) - expected) < 1e-9


def test_ends_at_zero():
    sigmas = scheduling_mod.generate_sigmas(10, "qwen_image21")
    assert sigmas[-1] == 0.0
    assert len(sigmas) == 11
```

Run: `uv run pytest tests/test_scheduling_qwen_image21.py -q` -> FAIL.

- [ ] **Step 2: Modifier `scheduling.py`**

Dans `generate_sigmas`, ajouter avant le `if model_type == "flux2":` (ou juste après le bloc
`("krea2", "krea2_turbo")`, peu importe l'ordre — ce sont des `if` indépendants qui `return`) :

```python
    if model_type == "qwen_image21":
        # Registers as ModelType.FLUX in ComfyUI (comfy/model_base.py::QwenImage21),
        # same family as Krea2 above -- flux_time_shift/ModelSamplingFlux, NOT
        # time_snr_shift/ModelSamplingDiscreteFlow like Flux2/Z-Image below, despite
        # sharing their "fixed shift, not resolution-dependent" simplicity. Fixed
        # shift=0.69 (comfy/supported_models.py::QwenImage21.sampling_settings).
        shift = 0.69
        sigmas = _flux_fixed_shift_sigmas(shift, steps)
        return sigmas
```

Dans `_flow_shift_fn`, ajouter le branch correspondant (même position relative, avant
`if model_type == "flux2": return lambda t: time_snr_shift(2.02, t)`) :

```python
    if model_type == "qwen_image21":
        return lambda t: flux_time_shift(0.69, 1.0, t)
```

- [ ] **Step 3: Ajouter `_run_qwen_image21` à `core.py`**

Dans `run()`, ajouter le routage (avant `if model_type == "sdxl":`, aux côtés des routages
zimage/flux2 existants) :

```python
        if model_type == "qwen_image21":
            return self._run_qwen_image21(steps, seed)
```

Ajouter la méthode (près de `_run_flux2`) :

```python
    def _run_qwen_image21(self, steps: int, seed: int) -> dict:
        """Run the Qwen Image 2.1 DiT sampling loop.

        Flow-matching, single conditional pass per step (no CFG -- the model has
        no guidance embedding, unlike FLUX-dev/Flux2). Registers as ModelType.FLUX
        in ComfyUI (comfy/model_base.py::QwenImage21), same family as Krea2 -- NOT
        ModelType.FLOW like Flux2/Z-Image -- so its sigma schedule uses
        flux_time_shift (ModelSamplingFlux), not time_snr_shift
        (ModelSamplingDiscreteFlow). Fixed shift=0.69
        (comfy/supported_models.py::QwenImage21.sampling_settings), handled by
        scheduling.py's "qwen_image21" branch via _flux_fixed_shift_sigmas.

        Unlike Flux2, the DiT has no `.predict()`/external `rope` --
        `QwenImage21Transformer2DModel.__call__(x, timestep, context)` computes
        RoPE internally (`_build_sequence`) and consumes/returns `x` as `[B,C,H,W]`
        (NCHW) directly -- no token-grid flattening at this call site, unlike
        Flux2's packed-token `img` parameter.
        """
        precision = self.config.mlx_dtype
        model_type = self.model_type

        context = bridge.conditioning_qwen_image21_to_mlx(self.positive, precision)

        sigmas = calculate_sigmas(model_type, self.scheduler_name, steps, self.width, self.height)
        solver_state: dict[str, Any] = {}

        mx.reset_peak_memory()
        step_times: list[float] = []
        t_sampling_start = time.perf_counter()

        for t in range(steps):
            step_start = time.perf_counter()
            sigma_t = sigmas[t]
            sigma_next = sigmas[t + 1] if t + 1 < len(sigmas) else 0.0

            if self.lora_schedule is not None:
                self.lora_schedule["step"] = t
                self.transformer = self._update_lora_schedule(
                    self.transformer, self.config, self.lora_schedule, t, steps
                )

            timestep = mx.array([sigma_t], dtype=mx.float32)
            noise_pred = self.transformer(self.noise, timestep, context)
            mx.eval(noise_pred)

            denoised = self.noise - noise_pred * sigma_t

            def _model_call(x_at, sigma_at):
                timestep_at = mx.array([sigma_at], dtype=mx.float32)
                out = self.transformer(x_at, timestep_at, context)
                mx.eval(out)
                return x_at - out * sigma_at

            self.noise, solver_state = solvers.step(
                self.sampler_name,
                x=self.noise,
                sigma=sigma_t,
                sigma_next=sigma_next,
                denoised=denoised,
                state=solver_state,
                seed=seed,
                step_index=t,
                is_flow_matching=self._is_flow_matching,
                model_call=_model_call,
            )
            mx.eval(self.noise)

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            if (t + 1) % 5 == 0 or t == 0:
                print(f"[ASDX] Qwen Image 2.1 Step {t + 1}/{steps} - {step_time:.3f}s")

        total_time = time.perf_counter() - t_sampling_start

        if self.low_memory_mode:
            from ..loader import clear_model_cache
            clear_model_cache()
            self.transformer = None
            print("[ASDX] Low memory: model cache evicted, next load will read from disk")

        out_latent = bridge.mlx_to_comfy_latent_qwen_image21(self.noise, {"samples": self.noise})

        mem = bridge.collect_mlx_memory()
        avg_step = sum(step_times) / len(step_times) if step_times else 0
        print(
            f"[ASDX] Qwen Image 2.1 Sampling complete: {total_time:.1f}s total, "
            f"{avg_step:.3f}s/step, {mem['peak_gb']:.1f}GB peak"
        )

        if self.memory_shape is not None:
            record_observation(self.memory_shape, mx.get_peak_memory())

        bridge.clear_mlx_cache()

        return out_latent
```

**Note pour l'implémenteur** : vérifier que `Any`, `time`, `mx`, `solvers`, `calculate_sigmas`,
`record_observation` sont déjà importés en tête de `core.py` (ils le sont, utilisés par
`_run_flux2`) — ne rien ré-importer localement dans la méthode.

- [ ] **Step 4: Ajouter le dispatch de bruit à `sampler/__init__.py`**

Ajouter une branche dans la chaîne `if model_type == "sdxl": ... elif ...` existante :

```python
        elif model_type == "qwen_image21":
            noise, height, width, output_shape = bridge.prepare_noise_from_latent_qwen_image21(
                latent_image, int(seed), config.mlx_dtype
            )
```

- [ ] **Step 5: Corriger le type de socket dans `qwen_image21_nodes.py`**

Dans `ASDX_QwenImage21TextEncode.define_schema`, changer :
```python
            outputs=[
                io.Custom("asdx_qwen_image21_conditioning").Output(display_name="conditioning"),
            ],
```
en :
```python
            outputs=[
                io.Custom("mlx_conditioning").Output(display_name="conditioning"),
            ],
```
(la valeur réelle retournée par `execute()` — le dict `{"type": "qwen_image21", ...}` — ne
change pas, seul le tag de type de socket ComfyUI change, pour se connecter à
`ASDX_MLXSampler.positive`.)

- [ ] **Step 6: Vérifier le succès**

Run: `uv run pytest tests/test_scheduling_qwen_image21.py -q` -> PASS (4 tests).
Run: `uv run pytest tests/ -q` -> pas de nouvelle régression (baseline : 1 échec pré-existant
sans rapport, `test_krea2_module_loader.py`).

- [ ] **Step 7: Commit**

```bash
git add apple_silicon_nodes/sampler/scheduling.py apple_silicon_nodes/sampler/core.py apple_silicon_nodes/sampler/__init__.py apple_silicon_nodes/qwen_image21_nodes.py tests/test_scheduling_qwen_image21.py
git commit -m "feat: wire Qwen Image 2.1 DiT into the generic sampler (_run_qwen_image21)"
```

---

### Task 2: Reprise du test end-to-end sur ComfyUI réel

**Files:** aucun nouveau fichier.

**Contexte** : `comfy-mcp` est cassé dans cette session (tous les appels échouent, y compris en
lecture seule) — utiliser l'API HTTP de ComfyUI directement (`curl`/Python `requests` vers
`http://127.0.0.1:8188`). Le symlink `custom_nodes/ComfyUI-MLXU` pointe vers le dépôt principal
(`master`), pas un worktree — merger la tâche 1 sur `master` AVANT de demander à l'utilisateur
de redémarrer ComfyUI Desktop (comfy-cli ne peut pas redémarrer un process qu'il n'a pas lancé
lui-même — c'est l'app Desktop qui l'a lancé).

- [ ] **Step 1: Merger la tâche 1 sur `master`**, demander à l'utilisateur de redémarrer
  ComfyUI Desktop, confirmer via `curl http://127.0.0.1:8188/object_info/ASDX_MLXSampler` que
  les inputs sont à jour (optionnel, ou juste confirmer que le workflow tourne au step suivant).

- [ ] **Step 2: Construire le workflow JSON minimal** (format API, pas format UI-export) :
  `ASDX_QwenImage21TextEncoderLoader` → `ASDX_QwenImage21TextEncode` →
  `ASDX_DiffusionLoader(model_name="qwen_image_2.1_bf16.safetensors")` → `ASDX_EmptyLatent` →
  `ASDX_MLXSampler` → `ASDX_VAELoader(vae_name="qwen_image_2.1_vae_bf16.safetensors")` →
  `ASDX_VAEDecode` → `SaveImage` (nœud core ComfyUI).

- [ ] **Step 3: Soumettre via `POST /prompt`**, poller `GET /history/<prompt_id>` jusqu'à
  complétion, télécharger l'image via `/view`, confirmer visuellement/statistiquement qu'elle
  n'est pas du bruit (ex. écart-type/moyenne des pixels dans une plage plausible, pas uniforme).

- [ ] **Step 4: Répéter avec `qwen_image_2.1_Q8.gguf`** comme `model_name`.

- [ ] **Step 5: Documenter le résultat** (prompt, seed, temps de génération, image obtenue) et
  mettre à jour `README.md` avec une section Qwen Image 2.1.

- [ ] **Step 6: Corriger tout écart trouvé**, relancer les suites pytest concernées, commit si
  nécessaire.
