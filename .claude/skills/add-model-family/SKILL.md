---
name: add-model-family
description: Scaffold a new native MLX model family (native/<x>/{config,model,weight_map,__init__}.py) following the pattern established by native/krea2, native/sdxl and native/zimage. Use when adding architecture support for a new diffusion model family (e.g. FLUX.2/Klein per Phase B of the multi-model plan).
disable-model-invocation: true
---

# add-model-family

Scaffolds a new `native/<x>/` subpackage for a diffusion model family, replicating the
pattern used by every family already ported in this project (FLUX.1 → `native/__init__.py`,
Krea2 → `native/krea2/`, SDXL → `native/sdxl/`, Z-Image → `native/zimage/`). This is a
project-specific skill: it does not exist outside this repo, and it does not generate
architecture code for you — it generates the skeleton, the checklist, and the places you
must NOT forget to touch, then lets you fill in the real math from the real comfy reference.

## Before scaffolding: read the reference first

Every family in this project was ported only after reading the real ComfyUI source for it
end to end (`comfy/ldm/...`, `comfy/model_base.py`, `comfy/supported_models.py`,
`comfy/model_detection.py`, `comfy/latent_formats.py`). Do not scaffold from memory or guess
shapes. If the family is already scoped in `~/.claude/plans/dynamic-splashing-boot.md` (check
there first — Phase B/FLUX.2-Klein and Phase E/LoRA-ControlNet are already researched), reuse
that research instead of re-deriving it.

If a real checkpoint for the family exists on this machine, inspect its keys/shapes with
`safetensors.safe_open(path, framework="pt")` (header only, no weight data) BEFORE writing
`config.py` — this project has twice caught scope gaps this way (SDXL's
`transformer_depth_output`, Z-Image's `pad_tokens_multiple`) that the plan/reference code
alone did not surface.

## What to ask the user (if not already given)

1. Family name / package name (lowercase, matches loader hints), e.g. `flux2`.
2. Path to the comfy reference source file(s) for this architecture.
3. Path to a real checkpoint on this machine, if one exists (for later verification — see the
   `verify-checkpoint` skill).
4. Latent space: channels, patch_size, and whether it reuses an existing scale/shift constant
   (e.g. FLUX's 0.3611/0.1159) or needs new ones.

## Steps

### 1. Create the subpackage skeleton

```
apple_silicon_nodes/native/<x>/
├── __init__.py
├── config.py
├── model.py
└── weight_map.py
```

Use `native/krea2/` as the direct template for style (frozen dataclass config with a
`validate()` + `__post_init__`, `mlx_dtype`/derived-shape `@property`s, latent
scale/shift module-level constants + `process_<x>_latent_in/out()` helpers).
Reuse `native/common.py` (`to_mlx_dtype`, `timestep_embedding`, `rms_norm`,
`_check_weight_match`) instead of copying them into the new family.

`config.py` skeleton:

```python
"""<Family> model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class <X>Config:
    # TODO: fill in from the real comfy reference / real checkpoint shapes.
    # Do not guess a value that can be read from comfy/model_detection.py or
    # from the checkpoint's own tensor shapes.
    dtype: str = "float16"

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return to_mlx_dtype(self.dtype, "<X>")  # from ..common import to_mlx_dtype

    def validate(self) -> None:
        # TODO: cross-field assertions (e.g. hidden_dim % num_heads == 0,
        # rope_axes_dim summing to head_dim — see native/krea2/config.py for examples).
        ...

    def __post_init__(self) -> None:
        self.validate()


# ── <X> latent space constants ──────────────────────────────────────────
# TODO: reuse an existing FLUX_LATENT_SCALE/SHIFT-style constant if the family
# shares its VAE with an already-ported family (check comfy/latent_formats.py
# inheritance chain first — Z-Image reused FLUX's because ZImage(Lumina2) and
# Lumina2.latent_format = latent_formats.Flux).
```

`model.py` skeleton — bottom of file must define `load_<x>_transformer(path, dtype)`
following the **exact 6-step recipe** used by every family in this project (do not deviate):

```python
def load_<x>_transformer(path: str, dtype: str = "float16") -> <X>Transformer:
    from apple_silicon_nodes.native import _load_safetensors
    from .weight_map import normalize_<x>_keys, map_<x>_to_native
    from .config import <X>Config

    state_dict = _load_safetensors(path)
    state_dict = normalize_<x>_keys(state_dict)
    state_dict = map_<x>_to_native(state_dict)

    config = <X>Config(dtype=dtype)
    model = <X>Transformer(config)

    from mlx.utils import tree_flatten, tree_unflatten
    flat_params = dict(tree_flatten(model.parameters()))

    matched, missing, extra = {}, [], []
    for key in flat_params:
        if key in state_dict:
            matched[key] = state_dict[key].astype(config.mlx_dtype)
        else:
            missing.append(key)
    extra = [k for k in state_dict if k not in flat_params]

    print(f"ASDX: <X> matched {len(matched)}/{len(flat_params)} params "
          f"({len(missing)} missing, {len(extra)} unused in checkpoint)")

    model.update(tree_unflatten(list(matched.items())))
    mx.eval(model.parameters())
    return model
```

**Gotcha to never reproduce** (Session 11 bug, cost real debugging time): the checkpoint-key
match in the loop above MUST compare **strings directly** — `key in state_dict` — never
convert either side to a tuple first. A tuple-vs-string comparison silently matches zero keys
without raising, and the model runs forward on 100% randomly-initialized weights with no error.

`weight_map.py` skeleton:

```python
"""Weight-key normalization and mapping for <Family> checkpoints."""

from __future__ import annotations


def normalize_<x>_keys(state_dict: dict) -> dict:
    """Strip any checkpoint-distribution prefix (e.g. 'model.diffusion_model.')."""
    # TODO: confirm the real prefix against the actual checkpoint — do not assume
    # it matches another family's prefix.
    ...


def map_<x>_to_native(state_dict: dict) -> dict:
    """Rename checkpoint keys to match this project's MLX attribute names.

    Most renames needed here are `.layers.` insertions for any MLX nn.Sequential
    (MLX preserves PyTorch child indices even for parameter-free layers like SiLU —
    verified empirically for SDXL's time_embed/label_emb and Z-Image's adaLN_modulation).
    Prefer naming native model attributes to match the checkpoint 1:1 in the first
    place (Z-Image's model.py did this deliberately) — it needs zero rename rules.
    """
    ...
```

`__init__.py`: re-export `<X>Config`, the transformer class(es), `load_<x>_transformer`,
`normalize_<x>_keys`, `map_<x>_to_native` — mirror `native/zimage/__init__.py`.

### 2. Integration checklist (4 points — do not skip any)

1. **`loader.py`** — add `_<X>_HINTS` (filename-based detection dict) + a branch in
   `_load_transformer_for_type()`. Also extend `_detect_model_type_from_keys()` with an
   architecture-distinctive key marker for the content-based fallback (filenames lie —
   SDXL/Illustrious/Pony finetunes and Z-Image finetunes have both shipped with zero
   filename hints in this project already).
2. **`capability.py`** — add a `CapabilityProfile` entry to `CAPABILITY_PROFILES` (family,
   generate_params, `hard_block`/`requires`, `latent_channels`) + an entry in
   `_CAPABILITY_DISPATCH`. If a shared capability-resolution helper exists
   (`_capability_for_model_type()` — added in Session 14 to deduplicate the SDXL/Z-Image
   logic), extend that table instead of re-duplicating the branch a third time.
3. **`sampler/core.py`** — add an early dispatch in `run()`
   (`if model_type in (...): return self._run_<x>(steps)`, placed BEFORE any FLUX-specific
   call) + a new `_run_<x>()` method. Check whether the family is flow-matching (reuse the
   Euler-in-`t∈[0,1]` loop pattern) or discrete/EPS like SDXL (needs its own sigma-space loop
   — do not force a flow-matching update onto a DDPM-scheduled family).
4. **`sampler/scheduling.py`** — if flow-matching with a fixed or resolution-dependent shift,
   add a branch to the existing `generate_sigmas()` dispatcher. If the schedule type differs
   entirely (DDPM/EPS), write a dedicated `generate_sigmas_<x>()` instead of forcing it into
   the shared dispatcher (this is what SDXL did — its two-pass cond/uncond semantics don't fit
   the shared function's signature).

Also check `bridge.py` for a `conditioning_<x>_to_mlx()` function, and whether the family can
reuse an existing pack/unpack latent function outright (Z-Image reused FLUX's 16ch/patch=2
packing verbatim because of a shared latent_format inheritance chain in comfy — check for this
before writing new pack/unpack code).

### 3. Verify

Run the `verify-checkpoint` skill against the new family once `config.py`/`model.py`/
`weight_map.py` are filled in. Do not report the family as done without a `matched N/M` run
against a real checkpoint, if one exists on this machine.
