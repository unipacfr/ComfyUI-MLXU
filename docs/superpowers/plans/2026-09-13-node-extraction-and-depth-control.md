# Node Extraction (Sampler + CLIP Text Encode) + Real Depth Control Implementation Plan

> **Status (2026-09-14):** Phases A, B, D, E are done and committed. Phase C is done except
> Task C4 (real-checkpoint verification — no `flux1-depth-dev` checkpoint has been available
> in this environment). This doc has been trimmed to keep only what remains; see git log for
> the full implementation history of the completed phases.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `ASDX_MLXSampler` and `ASDX_CLIPTextEncode` — both god-nodes bundling multiple unrelated features — into focused, composable nodes, and implement real FLUX depth-conditioning (currently a declared-but-inert no-op) so there is something real to extract for that feature.

**Architecture:** Each bolt-on feature currently living as extra optional inputs on the sampler or text-encode node moves to its own upstream node that writes its result into the `asdx_model` dict (the project's existing pattern for `controlnet`/`capability`/`lora_schedule`/`memory_shape`) or produces a ready-made `mlx_conditioning`/`LATENT`. The sampler/text-encode read from the model dict **in priority over** their existing legacy inputs, which stay in place as a documented fallback — this is the same pattern already used for `lora_schedule` (`sampler/__init__.py:170`) and is what keeps every saved workflow working unchanged while new workflows get small, focused nodes.

**Tech Stack:** MLX (`mlx.core`/`mlx.nn`), ComfyUI V3 node API (`comfy_api.latest.io`), PyTorch-MPS bridge points for VAE encode (per canon, never ported to MLX).

**Spec:** This plan absorbs and supersedes the scope question left open in `docs/plan-noeud-krea2-edit.md` §6 ("hors périmètre") — that doc's §§1-5 are the spec for Phase A below and are not repeated here; Phase A executes it. `ROADMAP.md`'s "Feature à évaluer" sections on samplers/schedulers are NOT in scope here.

## Global Constraints

- Never break a saved workflow: every new node's output is consumed by the old node **in addition to**, not instead of, its existing optional inputs (model-dict priority, legacy input as fallback with a one-line log noting which one won if both are wired).
- No MLX port of VAE encode/decode (canon: `Porting VAE encode/decode to MLX has negative ROI`). All new nodes that need VAE encoding call the real `comfy.sd.VAE`/`ASDX_VAEEncode._fallback_encode`, never `MLXVAE()`.
- Any new architecture code (depth channel concat) is verified against the real ComfyUI reference (`comfy/model_base.py::Flux.concat_cond`, already read for this plan) before being written, per canon `ComfyUI and the SceneWorks stack are the reference implementations`.
- `py_compile` every touched file before commit. Real-checkpoint verification (matched N/M params, finite forward pass) before claiming any task "done" — per canon `Coverage and magnitude are separate checks`.
- Commit message "type" must be one of `feat, fix, refactor, docs, style, test, build` (local hook rejects `chore` and any AI-attribution trailer).

---

## Phase A — Extract Krea2 Identity Edit into `ASDX_Krea2Edit`

✅ **Done** (`apple_silicon_nodes/krea2_edit.py`, commits `54eac6d`/`b32c492`/`d8e7ca7`).
`ref_boost` default kept at `1.0`; double-LoRA guard implemented. Sampler prefers
`model["identity_edit"]` over its legacy `source_latent`/`source_image` inputs (fallback
kept for old workflows). Confirmed working end-to-end without face artifacts (2026-09-14,
after two follow-up bugfixes — see git log for `krea2_edit.py`/`sampler/core.py`).

---

## Phase B — Extract Krea2 grounded-encode into `ASDX_Krea2GroundedEncode`

✅ **Done** (`apple_silicon_nodes/krea2_grounded_encode.py`, commits `3d7986d`/`1ea040b`/`105c6fc`).
`ASDX_CLIPTextEncode` no longer has `image`/`grounding_px`/`system_prompt` inputs (breaking
change, as flagged — any old workflow using those must rewire to the new node).

---

## Phase C — Implement real FLUX depth conditioning

Currently `depth_image`/`depth_strength` are accepted by the sampler and declared in `capability.py`'s `flux1_depth` profile, but never actually used: `_prepare_depth_noise()` (`sampler/core.py:873-883`) is a no-op that returns the noise unchanged, and `FluxTransformer.img_in` (`native/__init__.py:455`) is hardcoded `nn.Linear(64, HIDDEN_DIM)`, which cannot even load a real flux1-depth-dev checkpoint (whose `img_in.weight` is `[3072, 128]`, not `[3072, 64]`).

Reference mechanism, verified by reading `comfy/model_base.py::Flux.concat_cond` (lines 987-1022) on this machine:
1. `num_channels` = the checkpoint's *actual* `img_in.weight.shape[1] // patch_size²`, read from the live weight, not a static config — a depth-dev checkpoint has `num_channels=32` (16 noise + 16 depth-latent channels, vs. the plain FLUX.1-dev's 16).
2. The control image (the depth map, rendered as RGB) is **VAE-encoded** to a 16-channel latent, scaled through `process_latent_in` (the same scale/shift as the noisy latent), resized to the noise's spatial shape, then concatenated along the channel axis to the noise **before** `img_in` — once per generation, not per step (the transformer's own weights are what make it "depth-aware"; there is no separate per-step depth injection).
3. Fill/inpaint models concatenate a further 1-channel mask on top (`num_channels<=out_channels*3`) — out of scope here, ASDX's Fill mode already works differently (`_prepare_inpainting_noise`) and is not part of this phase.

### Tasks C1-C3

✅ **Done.** Checkpoint channel detection (`detect_flux_in_channels`, commit `cd82565`),
VAE-encode + channel-concat of the depth latent (`_prepare_depth_noise`, commit `7fe0d7b`),
capability-profile doc clarification (commit `9ee61ef`). Regression-verified against the
real `flux1-dev.safetensors` (still 780/780 params, `in_channels` resolves to 64 — no
behavior change for non-depth checkpoints). Structural test passes
(`tests/test_depth_conditioning.py`).

### Task C4: Real-checkpoint verification — REMAINING

- [ ] **Step 1:** If a real `flux1-depth-dev` (or `flux1-canny-dev`, same mechanism) checkpoint is reachable from this environment: load it end-to-end, generate one image with a real depth map, confirm non-NaN output and a visually plausible depth-following result.
- [ ] **Step 2:** If no such checkpoint is reachable in this environment (the pattern seen repeatedly in this project's own `ROADMAP.md`, e.g. the Krea2 Identity Edit grounding fix): state explicitly in the commit/PR description which parts were verified structurally (header shapes, synthetic forward pass) vs. which remain "not yet re-tested on a real checkpoint" — do not claim the feature "works" without this distinction, per this project's own stated convention (`CLAUDE.md` §"Before shipping a new dequantization path... verify numerically against it on a real checkpoint").

---

## Phase D — Extract depth into `ASDX_DepthConditioning`

✅ **Done** (`apple_silicon_nodes/depth_conditioning.py`, commit `954a67f`). Validates
`model["capability"].family == "flux1_depth"`, refuses otherwise. Sampler prefers
`model["depth_cond"]` over legacy `depth_image`/`depth_strength` inputs.

---

## Phase E — Extract img2img/inpaint/Kontext mode-routing

✅ **Done.** `ASDX_KontextReference` (`kontext_reference.py`, commit `74387e5`) and
`ASDX_LatentNoisePrep` (`latent_noise_prep.py`, commit `92e338e`) both follow the
model-dict-priority pattern; sampler's own inputs kept as fallback (no schema removal,
unlike Phase B2). Docs updated (commit `2aad268`).

---

## Self-Review

**Spec coverage:** Phase A covers the pre-existing Krea2Edit spec in full. Phase B covers the CLIP text-encode god-node complaint (grounded-encode extracted; FLUX-dual/SD-single dispatch kept, since that dispatch IS the node's stated single responsibility, not a bolt-on). Phases C+D cover depth control per the user's explicit choice to implement it for real before extracting it. Phase E covers the remaining sampler inputs (img2img/inpaint/fill/Kontext) named in the "sûres + mode routing" option. ControlNet and `krea2_enhancer_strength` are deliberately left on the sampler: ControlNet already lives in the `model` dict (set by `ASDX_ApplyControlNet`, not a sampler input at all — nothing to extract), and `krea2_enhancer_strength` is a single float dial with no prep logic behind it (nothing to extract either).

**Placeholder scan:** Task E2's "Produces" section documents a real design correction found during self-review (seed-dependent noise can't move upstream) rather than leaving a wrong interface in place — this is the one spot where the original extraction idea doesn't fully work, and it's called out explicitly rather than glossed over.

**Type consistency:** `model["identity_edit"]`, `model["depth_cond"]`, `model["kontext"]`, `model["latent_prep"]` all follow the same shallow-copy-of-`model`-dict convention already established by `ASDX_LoraLoader`/`ASDX_Krea2Edit`. `_SamplerCore` constructor parameter names in Phase A Task A2 / Phase D Task D1 / Phase E Task E1 match the existing parameter names already present in `sampler/__init__.py`'s current call to `_SamplerCore(...)` (`kontext_reference_latent`, `kontext_reference_strength`, `source_latent`, `source_image`, `vae`, `depth_image`, `depth_strength`) — no renames introduced.

---

**Plan complete and saved to `docs/superpowers/plans/2026-09-13-node-extraction-and-depth-control.md`.**
