# Anima LoRA Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ASDX_LoraLoader`, `ASDX_MultiLoraLoader` and `ASDX_LoraSchedule` apply Anima LoRAs to the native MLX Anima model with the same effect as ComfyUI's LoRA loader on the same model, for every dialect present in the user's library, and fail loudly on any LoRA key they cannot route.

**Architecture:** Replace the current "LoRA is not supported for Anima" guard in `lora.py::ASDX_LoraLoader._apply_lora_to_transformer` with a new `_apply_lora_residual_to_anima`, following the existing Z-Image/Krea2 forward-time-residual pattern (low-rank `(A, B)` pairs attached to `AdaptableLinear` leaves, full-size deltas merged once, untouched modules shared by reference, copy-on-write clones). Targets are derived from the live model (every `nn.Linear` leaf, DiT and `llm_adapter`), mirroring ComfyUI's `model_lora_keys_unet`, which maps every `.weight` of the diffusion model under both `lora_unet_<key_with_underscores>` and `diffusion_model.<key>`. Any LoRA key that routes nowhere raises (mlx-gen-anima's strict no-silent-drop policy).

**Tech Stack:** Python 3.13, MLX, ComfyUI V3 nodes, pytest via `uv run pytest`.

**Spec:** user request 2026-09-26 "implementer le support des Lora pour Anima". Research findings in "Reference facts" below.

## Global Constraints

- Ground truth: ComfyUI `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/comfy/lora.py::model_lora_keys_unet` (key map) and `comfy/weight_adapter/lora.py` (scale = strength * alpha/rank, 1.0 when no alpha). Second reference: `/Volumes/X10Pro/Images/Projet/inference/crates/media/mlx-gen/mlx-gen-anima/src/adapters.rs` (routing `llm_adapter.*` into the conditioner; strict: unrouted target = hard error).
- Canon "Full-size LoRA deltas are merged once into `.weight`, never held as a residual" and "`AdaptableLinear` adapters must upsert by array identity, not append": reuse `_upsert_lora_factor` / `merge_delta`; never append.
- Canon "Coverage and magnitude are separate checks": verify both the attached count AND the numeric effect against ComfyUI.
- Canon "A parity test must be proven by mutation" and "MLX GPU fp32 matmul is not exact: parity tests run on the CPU stream".
- Non-destructive: the base transformer (which may be `loader.py::_MODEL_CACHE`'s cached object) must be bit-identical after applying a LoRA.
- Behavior for every other family (Krea2, FLUX, Flux2, Z-Image, SDXL, MiniMax H3) must not change. The full suite must stay green (baseline 499 passed, 26 skipped, 0 failed).
- Python via `uv run`. Commit messages `type: description`, no scope, no attribution trailer (repo hook). Stage explicit paths only; never `git add docs` (`docs/architecture/` stays untracked). Do not push.

## Reference facts (inspected 2026-09-26)

- User library: 36 files in `/Volumes/X10Pro/Images/models/loras/Anima/`, three dialects:
  - **A (33 files)** kohya DiT-only: `lora_unet_blocks_{i}_{self_attn|cross_attn}_{q_proj|k_proj|v_proj|output_proj}` and `lora_unet_blocks_{i}_mlp_{layer1|layer2}`, each `.lora_down.weight` / `.lora_up.weight` / `.alpha` (840 keys = 28 x 10 x 3). BF16 or F16. Example: `ANID1sn3yAlic3.safetensors` (alpha 4, rank 8), `Mavis_AnimaBaseV10_byKonan.safetensors` (F16, alpha 1, rank 16).
  - **B (2 files)** kohya with adaLN: A plus `lora_unet_blocks_{i}_adaln_modulation_{self_attn|cross_attn|mlp}_{1|2}` (1344 keys). Examples: `tifa-anima.safetensors` (alpha 16, rank 16), `Hercule V1.2.safetensors`. Shapes: `adaln_modulation_*_1` down `[r,2048]` up `[256,r]`; `*_2` down `[r,256]` up `[6144,r]`. Metadata `ss_network_module = networks.lora_anima`.
  - **C (1 file)** PEFT: `diffusion_model.blocks.{i}.{...}.lora_A.weight` / `.lora_B.weight`, including `adaln_modulation_*.{1,2}`, no alpha (scale 1.0), rank 64: `Disney_Animation_Style.safetensors` (896 keys).
- No local file targets `llm_adapter.*`, but published ones do: mlx-gen documents `anima-turbo-lora-v0.2` = 448 DiT + 60 `llm_adapter.*` targets. ComfyUI maps them because they are `.weight`s of the diffusion model (`diffusion_model.llm_adapter.blocks.{i}.self_attn.q_proj`, kohya `lora_unet_llm_adapter_blocks_{i}_self_attn_q_proj`).
- Alpha and rank are uniform inside every one of the 36 files (checked), so `LoRAAdapter`'s single per-file `scale` is correct for them.
- Native module tree (`apple_silicon_nodes/native/anima/model.py`, `adapter.py`): parameter names equal the checkpoint keys minus prefix. `adaln_modulation_{self_attn,cross_attn,mlp}`, `final_layer.adaln_modulation`, `t_embedder`, `x_embedder.proj` and the adapter's `blocks.N.mlp` are **plain python lists** (index 0 is a parameter-free `nn.SiLU`/`nn.Identity`/`nn.GELU`), NOT `nn.Sequential`. `lora.py::_adapt_leaf`'s all-digit branch looks for `module.layers` (the MLX `nn.Sequential` list) and returns None for a plain list: using it unchanged would silently drop every adaLN and adapter-MLP target.
- LoRA plumbing in `apple_silicon_nodes/lora.py`: `_load_lora_file` strips `diffusion_model.`/`model.`/`transformer.` prefixes and keeps kohya flat keys as-is; `_lookup_native_or_kohya(lora, native_key)` tries the dotted native key then the forward-built kohya key `lora_unet_<stem_with_underscores>.<suffix>`; `_adapt_leaf`/`_clone_block_path`/`_ensure_adaptable` do the copy-on-write wrap; `_upsert_lora_factor(leaf, a, b, scale)` attaches a pair; `leaf.merge_delta(delta, scale)` merges a full delta; `_rescale_attached_lora` is the `ASDX_LoraSchedule` fast path; `_apply_lora_residual_to_zimage` (lora.py ~1577) is the closest model to follow. `detect_lora_family`/`_check_lora_compatibility`/`_LORA_COMPATIBLE_BASE` refuse a confidently-mismatched family.
- The Anima capability family string is `"anima"` (`capability.py::anima_base`).

## Review Focus

- A kohya adaLN key (`..._adaln_modulation_self_attn_1`) must reach the list element `adaln_modulation_self_attn[1]`; a silent drop here is the single most likely bug. Pinned in Task 1 by a per-target-class test and in Task 2 by exact attached counts on real files.
- Stacking two LoRAs (`ASDX_MultiLoraLoader`) and scheduling one (`ASDX_LoraSchedule`) must neither double-apply nor mutate the cached base model. Pinned in Task 1.
- A Krea2/FLUX/SDXL LoRA plugged into an Anima model must be refused with the existing mismatch error, and an Anima LoRA plugged into another family must be refused too. Pinned in Task 1.
- A LoRA containing keys that route to nothing (e.g. a TE-only key the node does not send to CLIP, or a typo'd module) must raise listing examples, not report a partial "attached N/M". Pinned in Task 1.
- F16 LoRA factors on a bf16/fp16 model must not overflow or silently change dtype of the base weights. Pinned in Task 2 (Mavis/Pocahontas are F16).

---

### Task 1: Anima LoRA routing and residual attach

**Files:**
- Modify: `apple_silicon_nodes/lora.py` (detection signature, compatible-base entry, new `_apply_lora_residual_to_anima`, dispatch in `_apply_lora_to_transformer` replacing the guard, list-aware leaf adaptation)
- Delete: `tests/test_lora_anima_guard.py` (its guard is removed; its import pattern may be reused)
- Test: `tests/test_lora_anima.py`

**Interfaces:**
- Consumes: `AnimaTransformer` (tiny config for tests: `AnimaConfig(dtype="float32", model_channels=256, num_blocks=2, num_heads=2)`), existing lora.py helpers named above.
- Produces: `_apply_lora_residual_to_anima(transformer, lora) -> AnimaTransformer` (new object; base untouched), called from `_apply_lora_to_transformer` for `isinstance(transformer, AnimaTransformer)`; `detect_lora_family` returns `"anima"` for dialects A, B, C (and for `llm_adapter`-bearing files).

Requirements:
1. **Target enumeration from the model.** Walk the live `AnimaTransformer` and collect every `nn.Linear` leaf with its dotted native name (e.g. `blocks.3.self_attn.q_proj`, `blocks.3.adaln_modulation_mlp.2`, `final_layer.adaln_modulation.1`, `t_embedder.1.linear_2`, `x_embedder.proj.1`, `llm_adapter.blocks.5.cross_attn.o_proj`, `llm_adapter.blocks.5.mlp.2`, `llm_adapter.out_proj`). Use `transformer.named_modules()` or an equivalent walk; do not hand-write a table. For each, resolve the LoRA entry with `_lookup_native_or_kohya(lora, f"{name}.weight")` (covers dotted/PEFT after prefix strip, and kohya built forward from the native name). Also resolve `.bias` targets the same way if the LoRA carries `diff_b`-style bias deltas, using the existing `_apply_bias_deltas` helper if it fits; otherwise leave biases out and let rule 3 catch them.
2. **Copy-on-write attach.** Attach pairs with `_upsert_lora_factor(leaf, a, b, lora.scale)` and full deltas with `leaf.merge_delta(delta, lora.scale)` on an `AdaptableLinear` obtained via `_ensure_adaptable`, cloning every module on the path from the root to the leaf so the input transformer is untouched and untouched sub-trees are shared by reference. Handle the plain-list containers (`blocks`, `llm_adapter.blocks`, `adaln_modulation_*`, `final_layer.adaln_modulation`, `t_embedder`, `x_embedder.proj`, adapter `mlp`): either extend `_adapt_leaf`/`_clone_block_path` with a plain-list branch (only reached when `.layers` is absent, so existing families keep their exact behavior) or write a small Anima-local path cloner. LoKr targets: use the existing `_lookup_lokr`/`_upsert_lokr_factor` path the other residual families use, if they apply generically.
3. **Strict routing.** After the walk, every key of `lora.factors`, `lora.deltas`, `lora.lokr_factors` and `lora.loha_factors` must have been consumed. If any remains, raise `RuntimeError` naming the LoRA, the count, and up to 5 unrouted keys. Exception: keys that belong to the text encoder (kohya `lora_te*` or ComfyUI `text_encoders.`/`qwen3_06b` prefixes) are not DiT keys; check how `_load_lora_file` and `ASDX_LoraLoader.execute` already separate TE keys for `_apply_lora_to_clip`, and exclude exactly those.
4. **Logging:** `[ASDX] LoRA (residual, Anima): attached N/N adapters (dit=X, llm_adapter=Y)`.
5. **Detection:** in `detect_lora_family`, add an `"anima"` signature matching `adaln_modulation_self_attn`/`self_attn_output_proj`/`.self_attn.output_proj.`/`_self_attn_output_proj` style keys under `blocks` (dotted or kohya-flat) or `llm_adapter` keys; verify it does not match any Krea2/FLUX/Flux2/Z-Image/SDXL file in `/Volumes/X10Pro/Images/models/loras` (run the detector over the whole library and report the family histogram before and after). Add `"anima": "anima"` to `_LORA_COMPATIBLE_BASE`.
6. **Schedule:** `_rescale_attached_lora` must work unchanged on an Anima transformer after the first attach (it walks `_iter_adaptable_leaves`); verify, do not special-case.

- [ ] **Step 1: Write failing tests** in `tests/test_lora_anima.py` (import pattern: copy `tests/test_lora_anima_guard.py`'s header, which already builds a tiny `AnimaTransformer` under the comfy stub; run MLX math under `mx.stream(mx.cpu)`):
  - `test_each_target_class_matches_merged_reference`: for each of `blocks.0.self_attn.q_proj`, `blocks.1.cross_attn.v_proj` (in 1024 -> out 256 in the tiny model), `blocks.0.mlp.layer1`, `blocks.0.adaln_modulation_self_attn.1`, `blocks.1.adaln_modulation_mlp.2`, `final_layer.adaln_modulation.2`, `llm_adapter.blocks.0.self_attn.q_proj`, `llm_adapter.blocks.1.mlp.0`: build a synthetic random `(A, B)` LoRA for that single target in kohya form (`lora_unet_<underscored>.lora_down/.lora_up/.alpha`) AND in PEFT form (`diffusion_model.<dotted>.lora_A/.lora_B`), load it through the real `_load_lora_file` (write a temp safetensors file), apply, and assert the transformer output equals the output of a reference copy whose `.weight` was manually replaced by `W + scale * B @ A` (float32, atol 1e-5). Assert attached == 1.
  - `test_mutation_breaks_reference`: with one target, perturb the reference weight and assert the comparison fails (canon mutation rule).
  - `test_base_model_untouched`: base transformer parameters bit-identical after apply (compare flattened arrays).
  - `test_two_loras_stack_without_double_apply`: apply LoRA 1 then LoRA 2 to the result; output equals reference with both deltas merged; LoRA 1 alone still gives its own result from the base.
  - `test_schedule_rescale`: apply once at scale s1, then call `_apply_lora_to_transformer` again with the same LoRA object at incremental scale (as `ASDX_LoraSchedule` does); output equals reference at the summed scale.
  - `test_unrouted_key_raises`: a LoRA with one valid key and one bogus `lora_unet_blocks_0_nonexistent` key raises `RuntimeError` naming the bogus key.
  - `test_detect_lora_family_anima`: synthetic headers for dialects A, B, C and an `llm_adapter`-only file return `"anima"`; a synthetic Krea2 (`blocks.0.attn.wq`) header does not.
  - `test_foreign_lora_on_anima_refused`: `_check_lora_compatibility` raises for a Krea2-signature LoRA against a model dict whose capability family is `"anima"`.
- [ ] **Step 2: Run to verify they fail** (`uv run pytest tests/test_lora_anima.py -q`; expected: the current guard raises "not supported").
- [ ] **Step 3: Implement** per requirements 1-6.
- [ ] **Step 4: Run** `uv run pytest tests/test_lora_anima.py -q`, then the other families' LoRA tests (`uv run pytest tests -q -k lora`), then the full suite once.
- [ ] **Step 5: Commit** `feat: anima LoRA residual attach with strict routing`.

---

### Task 2: Real-library coverage, ComfyUI parity, docs

**Files:**
- Create: `tests/test_lora_anima_real.py`
- Modify: `scripts/anima_parity.py` (add a `--lora PATH --strength S` leg)
- Modify: `README.md` (Anima section: LoRA now supported, dialects, strict routing), `apple_silicon_nodes/native/anima/__init__.py` docstring
- Test: `tests/test_lora_anima_real.py`

**Interfaces:**
- Consumes: Task 1's `_apply_lora_residual_to_anima` via `ASDX_LoraLoader._apply_lora_to_transformer`, `ASDX_LoraLoader._load_lora_file`, `native.anima.load_anima_checkpoint`.

- [ ] **Step 1: Coverage test on every real file.** `tests/test_lora_anima_real.py`, skipped when the files are absent: load `WaiHassakuAnima.safetensors` once (module-scoped fixture, bfloat16), then for every `*.safetensors` under `/Volumes/X10Pro/Images/models/loras/Anima/` apply it and assert: no exception, attached count == number of LoRA targets in the file (A: 280, B: 448, C: 448 — derive the expected count from the file's own keys, don't hard-code), `detect_lora_family == "anima"`, output of one tiny forward (latent `[1,16,16,16]`) is finite and differs from the base output. Release each applied model before the next.
- [ ] **Step 2: ComfyUI parity leg.** Extend `scripts/anima_parity.py`: ComfyUI side loads the model as today, then `comfy.sd.load_lora_for_models(model, None, comfy.utils.load_torch_file(lora_path), strength, 0)` and runs the same forward with the patched model (make sure the patches are actually applied — e.g. `model_patcher.patch_model()` or by calling through `comfy.model_management.load_models_gpu` — and prove it: the ComfyUI output with LoRA must differ from without); MLX side applies the LoRA through `ASDX_LoraLoader._load_lora_file` + `_apply_lora_to_transformer` with the same strength. Report, in fp32 CPU: cosine of full outputs, and — the real check — cosine and norm ratio of the LoRA-induced difference `(v_lora - v_base)` between ComfyUI and MLX. Pass: difference cosine > 0.999 and norm ratio within 1%. Harness sanity first (each side vs itself).
- [ ] **Step 3: Run the parity leg** on three dialect representatives at strength 1.0: `ANID1sn3yAlic3.safetensors` (A, bf16), `tifa-anima.safetensors` (B, adaLN), `Disney_Animation_Style.safetensors` (C, PEFT); and once on `Mavis_AnimaBaseV10_byKonan.safetensors` (F16) at strength 0.7. Record all numbers in the report.
- [ ] **Step 4: Docs.** README Anima section: remove "LoRA not supported"; state supported dialects (kohya incl. adaLN, PEFT `diffusion_model.`), `llm_adapter` targets routed, unrouted keys raise, works with `ASDX_LoraLoader`/`ASDX_MultiLoraLoader`/`ASDX_LoraSchedule`. Update the `native/anima/__init__.py` docstring.
- [ ] **Step 5:** full suite once; commit `test: anima LoRA real-library coverage and ComfyUI parity` and `docs: anima LoRA support`.

## Out of scope

- Per-module alpha (all 36 files are uniform; `LoRAAdapter` keeps one scale per file like every other family).
- Anima text-encoder LoRA keys beyond what `ASDX_LoraLoader` already forwards to ComfyUI's CLIP.
