# Node Extraction (Sampler + CLIP Text Encode) + Real Depth Control Implementation Plan

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

This phase executes the already-complete spec in `docs/plan-noeud-krea2-edit.md`. Two points that doc left open are decided here so the task is unblocked:

- **`ref_boost` default stays `1.0`** (not the reference's `4.0`). Rationale: `ref_boost=1.0` already means "off" everywhere else in ASDX (sampler tooltip, capability profile); changing the default on the new node alone would make the same dial mean two different "normal" values depending on which node set it. The node's tooltip documents "reference ships 4.0 for stronger likeness" so users can opt in explicitly.
- **Double-LoRA guard**: `execute()` scans `transformer`'s `AdaptableLinear` leaves for an already-attached adapter whose source file stem matches `identity[_-]?edit` (reuse the same regex idea as `lora_name` sorting) before attaching its own; if found, raise `RuntimeError` naming both files rather than silently stacking two copies of the same LoRA.

### Task A1: Create `ASDX_Krea2Edit` node

**Files:**
- Create: `apple_silicon_nodes/krea2_edit.py`
- Modify: `apple_silicon_nodes/__init__.py` (import + `_ALL_NODES`/`_DISPLAY`, mirroring how `krea2_edit.py`'s neighbors are registered around the existing `controlnet`/`image_chain` imports)
- Test: `apple_silicon_nodes/tests/test_krea2_edit.py` (new)

**Interfaces:**
- Consumes: `ASDX_LoraLoader._resolve_lora_path`, `ASDX_LoraLoader._check_lora_compatibility` (`lora.py:129`), `ASDX_LoraLoader._load_lora_file`, `ASDX_LoraLoader._apply_lora_to_transformer` (all existing, `lora.py`) — call them as-is, do not duplicate.
- Consumes: `ASDX_VAEEncode._fallback_encode(latent_dict_shape, vae, image)` (`vae.py:291`) for the pixel-fit path.
- Produces: `model["identity_edit"] = {"source_latent": mx.array, "source_grid": (h, w), "ref_boost": float}` — the exact shape `_SamplerCore._prepare_krea2_identity_edit` already expects internally for its pixel path (verify field names against `sampler/core.py::_prepare_krea2_identity_edit_pixels` before wiring Task A2).

- [ ] **Step 1:** Read `docs/plan-noeud-krea2-edit.md` §2 in full and write `apple_silicon_nodes/krea2_edit.py` per its node/input/`execute()` spec verbatim, substituting the two decisions above for `ref_boost` default and double-LoRA handling.
- [ ] **Step 2:** `uv run python -c "import py_compile; py_compile.compile('apple_silicon_nodes/krea2_edit.py', doraise=True)"` — must pass.
- [ ] **Step 3:** Write `test_krea2_edit.py`: construct a 2-block reduced `Krea2Config` model (mirrors the pattern already used in Session 10's manual tests, see `.claude/debugging-log.md`), call `ASDX_Krea2Edit.execute()` with a synthetic 64x64 image + a real `krea2_identity_edit_v1_2.safetensors` if locally available, else skip with a clear `pytest.skip("no local Krea2 Identity Edit LoRA")`. Assert: `model["identity_edit"]["source_latent"]` is finite, shape matches `(1, N, 64)` for some grid.
- [ ] **Step 4:** Run it: `uv run pytest apple_silicon_nodes/tests/test_krea2_edit.py -v`. Expected: PASS or SKIP (not FAIL).
- [ ] **Step 5:** Register the node in `apple_silicon_nodes/__init__.py`, add display name `🍏 ASDX Krea2 Identity Edit`, category `ASDX/Conditioning`.
- [ ] **Step 6:** Commit: `git add apple_silicon_nodes/krea2_edit.py apple_silicon_nodes/__init__.py apple_silicon_nodes/tests/test_krea2_edit.py && git commit -m "feat: add ASDX_Krea2Edit node for Identity Edit pixel-space fit"`

### Task A2: Wire sampler to prefer `model["identity_edit"]`

**Files:**
- Modify: `apple_silicon_nodes/sampler/__init__.py:161-270` (the `execute()` → `_SamplerCore(...)` construction)
- Modify: `apple_silicon_nodes/sampler/core.py` (`_prepare_krea2_identity_edit`, wherever it currently reads `self.source_latent`/`self.source_image`/`self.vae`/`self.ref_boost`)

**Interfaces:**
- Consumes: `model.get("identity_edit")` from Task A1.
- Produces: no change to `_SamplerCore`'s public surface — only its *source of truth* for Identity Edit params changes.

- [ ] **Step 1:** In `sampler/__init__.py::execute()`, after `lora_schedule = model.get("lora_schedule") or lora_schedule` (line 170), add:
  ```python
  identity_edit = model.get("identity_edit")
  if identity_edit is not None and (source_latent is not None or source_image is not None):
      print("[ASDX] Identity Edit: both ASDX_Krea2Edit and legacy source_latent/"
            "source_image inputs are wired -- ASDX_Krea2Edit wins.")
  ```
- [ ] **Step 2:** In `_SamplerCore.__init__` / `_prepare_krea2_identity_edit`, check `identity_edit` dict first; fall back to the existing `source_latent`/`source_image`+`vae` path only when it is `None`. Pass `identity_edit` through the constructor call in `sampler/__init__.py` the same way `lora_schedule` already is.
- [ ] **Step 3:** `uv run python -c "import py_compile; py_compile.compile('apple_silicon_nodes/sampler/core.py', doraise=True); py_compile.compile('apple_silicon_nodes/sampler/__init__.py', doraise=True)"`.
- [ ] **Step 4:** Regression: re-run any existing Identity Edit test/manual script with the OLD wiring (`source_latent` input only, no `ASDX_Krea2Edit` node) and confirm identical output to before this task (non-regression requirement from the existing plan's §4.5).
- [ ] **Step 5:** Commit: `git add apple_silicon_nodes/sampler/__init__.py apple_silicon_nodes/sampler/core.py && git commit -m "refactor: prefer ASDX_Krea2Edit's model dict over legacy sampler inputs"`

### Task A3: Docs

**Files:**
- Modify: `README.md` (node table, Identity Edit paragraph under Sampling)
- Modify: `ROADMAP.md` (mark "Feature planifiée : nœud dédié ASDX_Krea2Edit" as done, link to this plan)

- [ ] **Step 1:** Add `ASDX_Krea2Edit` to the README's Conditioning node table with its inputs/outputs.
- [ ] **Step 2:** Update the Identity Edit paragraph to show the new recommended graph (`ASDX_Krea2Edit` → `model` input of `ASDX_MLXSampler`) with the legacy wiring noted as still-supported.
- [ ] **Step 3:** Commit: `git add README.md ROADMAP.md && git commit -m "docs: document ASDX_Krea2Edit node and its legacy-input priority"`

---

## Phase B — Extract Krea2 grounded-encode into `ASDX_Krea2GroundedEncode`

`ASDX_CLIPTextEncode` currently does three unrelated jobs: (1) generic FLUX dual-encode (`t5xxl`+`guidance`), (2) generic SD-style single encode, (3) Krea2-specific image-grounded encode (`image`/`grounding_px`/`system_prompt`, `conditioning.py:345-380`). (1) and (2) are the node's stated purpose (its own docstring: "auto-detects FLUX vs SD-style"); (3) is the bolt-on to extract.

### Task B1: Create `ASDX_Krea2GroundedEncode` node

**Files:**
- Create: `apple_silicon_nodes/krea2_grounded_encode.py`
- Modify: `apple_silicon_nodes/__init__.py`
- Test: `apple_silicon_nodes/tests/test_krea2_grounded_encode.py` (new)

**Interfaces:**
- Consumes: `comfy.text_encoders.krea2.Krea2Tokenizer` (isinstance check), `_krea2_grounding_template`, `_KREA2_DEFAULT_GROUNDING_SYSTEM` — move these two module-level helpers from `conditioning.py` into the new file (they exist only for this feature).
- Produces: `io.Custom("mlx_conditioning").Output()` — same dict shape already returned by the SD-style branch today: `{"type": "clip", "conditioning": conditioning, "text": text}`.

- [ ] **Step 1:** Move `_KREA2_DEFAULT_GROUNDING_SYSTEM`, `_krea2_grounding_template`, `ASDX_CLIPTextEncode._prep_grounding_image` (as a module function, it has no other `cls`/`self` use) into `krea2_grounded_encode.py`.
- [ ] **Step 2:** Write the new node class with inputs `mlx_clip`, `text` (multiline), `image` (required — this node's whole purpose is grounding), `grounding_px` (default 768), `system_prompt` (optional). `execute()` body = the exact `if image is not None:` branch body currently at `conditioning.py:346-354` plus the encode/NaN-guard/result-building code at `conditioning.py:355-379`, minus the "ignored" warning branch (grounding is this node's only job now, so a non-Krea2 `mlx_clip` should `raise RuntimeError`, not warn-and-ignore).
- [ ] **Step 3:** `py_compile` the new file.
- [ ] **Step 4:** Write `test_krea2_grounded_encode.py`: assert `RuntimeError` is raised when `mlx_clip.tokenizer` is not a `Krea2Tokenizer` (use a `unittest.mock.Mock()` with a plain tokenizer attribute — no real checkpoint needed for this one assertion).
- [ ] **Step 5:** Run: `uv run pytest apple_silicon_nodes/tests/test_krea2_grounded_encode.py -v`. Expected: PASS.
- [ ] **Step 6:** Register in `__init__.py`, display name `🍏 ASDX Krea2 Grounded Encode`, category `ASDX/Conditioning`.
- [ ] **Step 7:** Commit: `git add apple_silicon_nodes/krea2_grounded_encode.py apple_silicon_nodes/__init__.py apple_silicon_nodes/tests/test_krea2_grounded_encode.py && git commit -m "feat: add ASDX_Krea2GroundedEncode node"`

### Task B2: Simplify `ASDX_CLIPTextEncode`

**Files:**
- Modify: `apple_silicon_nodes/conditioning.py:254-381`

- [ ] **Step 1:** Remove the `image`, `grounding_px`, `system_prompt` inputs from `define_schema()` (lines 265-283) and their parameters from `execute()`.
- [ ] **Step 2:** Remove the `if image is not None:` branch (lines 346-354); the SD-style branch becomes a plain `tokens = mlx_clip.tokenize(text)`.
- [ ] **Step 3:** Keep the NaN/Inf finite-check — it is generic (applies to any CLIP, not just grounded Krea2) and is still cheap insurance against a 7-minute wasted sampling run.
- [ ] **Step 4:** Remove `_prep_grounding_image` (moved in Task B1) and the now-unused `_KREA2_DEFAULT_GROUNDING_SYSTEM`/`_krea2_grounding_template` from `conditioning.py` if Task B1 already copied them (don't leave two copies).
- [ ] **Step 5:** `py_compile apple_silicon_nodes/conditioning.py`.
- [ ] **Step 6:** Run the existing conditioning test suite (`uv run pytest apple_silicon_nodes/tests/ -k conditioning -v` or equivalent) — confirm no regression on the FLUX-mode and plain SD-mode paths.
- [ ] **Step 7:** Commit: `git add apple_silicon_nodes/conditioning.py && git commit -m "refactor: drop Krea2 grounded-encode branch from ASDX_CLIPTextEncode"`

### Task B3: Docs

- [ ] **Step 1:** Update `README.md`'s Identity Edit workflow diagram: the image now wires to `ASDX_Krea2GroundedEncode` instead of `ASDX_CLIPTextEncode`'s optional `image` input (which no longer exists).
- [ ] **Step 2:** Commit: `git add README.md && git commit -m "docs: update Identity Edit workflow for ASDX_Krea2GroundedEncode"`

**Breaking-change note for B2** (flag to user before executing): unlike Phase A/D/E's model-dict-priority pattern, B2 removes inputs outright — any saved workflow using `ASDX_CLIPTextEncode`'s `image` input for grounding must be rewired to the new node. This is unavoidable here because the grounding branches on `image is not None` inside a single `execute()`, there is no "model dict" equivalent to fall back through for a CLIP-encode node. Confirm before running Task B2.

---

## Phase C — Implement real FLUX depth conditioning

Currently `depth_image`/`depth_strength` are accepted by the sampler and declared in `capability.py`'s `flux1_depth` profile, but never actually used: `_prepare_depth_noise()` (`sampler/core.py:873-883`) is a no-op that returns the noise unchanged, and `FluxTransformer.img_in` (`native/__init__.py:455`) is hardcoded `nn.Linear(64, HIDDEN_DIM)`, which cannot even load a real flux1-depth-dev checkpoint (whose `img_in.weight` is `[3072, 128]`, not `[3072, 64]`).

Reference mechanism, verified by reading `comfy/model_base.py::Flux.concat_cond` (lines 987-1022) on this machine:
1. `num_channels` = the checkpoint's *actual* `img_in.weight.shape[1] // patch_size²`, read from the live weight, not a static config — a depth-dev checkpoint has `num_channels=32` (16 noise + 16 depth-latent channels, vs. the plain FLUX.1-dev's 16).
2. The control image (the depth map, rendered as RGB) is **VAE-encoded** to a 16-channel latent, scaled through `process_latent_in` (the same scale/shift as the noisy latent), resized to the noise's spatial shape, then concatenated along the channel axis to the noise **before** `img_in` — once per generation, not per step (the transformer's own weights are what make it "depth-aware"; there is no separate per-step depth injection).
3. Fill/inpaint models concatenate a further 1-channel mask on top (`num_channels<=out_channels*3`) — out of scope here, ASDX's Fill mode already works differently (`_prepare_inpainting_noise`) and is not part of this phase.

### Task C1: Detect checkpoint channel count before construction

**Files:**
- Modify: `apple_silicon_nodes/native/weight_map.py` (add `detect_flux_in_channels`)
- Modify: `apple_silicon_nodes/native/__init__.py` (`FluxConfig`, `FluxTransformer.__init__`, `load_transformer`)

**Interfaces:**
- Produces: `detect_flux_in_channels(normalized: dict[str, mx.array]) -> int` — reads `normalized["img_in.weight"].shape[1]`, returns it directly (already raw channel count, not patch-multiplied, since `img_in.weight` shape is `[hidden, in_channels]` with `in_channels` already `= packed_channels` in the checkpoint's own convention — verify this against the real 64-channel flux1-dev checkpoint already used for the 780/780 validation in `.claude/debugging-log.md`/session-state before trusting it for depth).
- Produces: `FluxConfig.in_channels: int = 64` (new field, default preserves current behavior for every non-depth checkpoint).

- [ ] **Step 1:** Add `detect_flux_in_channels` to `weight_map.py`:
  ```python
  def detect_flux_in_channels(normalized: dict) -> int:
      """Read img_in's real input width from the checkpoint, mirroring
      comfy/model_base.py::Flux.concat_cond's own try/except on the live
      weight shape. 64 = plain FLUX.1 (16ch * 2x2 patch); 128 = depth/canny
      dev variants (32ch: 16 noise + 16 depth-or-canny latent).
      """
      weight = normalized.get("img_in.weight")
      if weight is None:
          return 64
      return int(weight.shape[1])
  ```
- [ ] **Step 2:** In `native/__init__.py`, add `in_channels: int = 64` to `FluxConfig`. Change `FluxTransformer.__init__` line 455 from `self.img_in = nn.Linear(64, HIDDEN_DIM)` to `self.img_in = nn.Linear(config.in_channels, HIDDEN_DIM)`.
- [ ] **Step 3:** In `load_transformer`, call `detect_flux_in_channels(normalized)` right after `map_flux_to_native(normalized)` (line 941), pass the result into `FluxConfig(dtype=dtype, in_channels=detected)` before constructing `FluxTransformer(config)`.
- [ ] **Step 4:** `uv run python -c "import py_compile; py_compile.compile('apple_silicon_nodes/native/weight_map.py', doraise=True); py_compile.compile('apple_silicon_nodes/native/__init__.py', doraise=True)"`.
- [ ] **Step 5:** Regression: reload the real `flux1-dev.safetensors` checkpoint already validated at 780/780 params (per `.claude/debugging-log.md`/session-state) and confirm it still reports 780/780 and `in_channels` resolves to 64 (no behavior change for non-depth checkpoints).
- [ ] **Step 6:** If a real flux1-depth-dev checkpoint is available locally, load it and confirm the new matched-param count (should now include `img_in.weight`/`.bias` where it previously would have failed or mismatched); if none is available, inspect the checkpoint's safetensors header directly (`read_safetensors_header`, same tool used throughout this project for header-only checks) and confirm `img_in.weight` is `[3072, 128]`, without loading the full file.
- [ ] **Step 7:** Commit: `git add apple_silicon_nodes/native/weight_map.py apple_silicon_nodes/native/__init__.py && git commit -m "feat: detect FLUX img_in channel count from checkpoint for depth/canny variants"`

### Task C2: VAE-encode and concat the depth latent in the sampler

**Files:**
- Modify: `apple_silicon_nodes/sampler/core.py` (`_prepare_depth_noise`, and the packing block around line 233-242 in `run()`)

**Interfaces:**
- Consumes: `self.depth_image` (already a node input), `self.vae` — **new requirement**: depth mode needs a VAE the same way the Identity Edit pixel path and ControlNet already do; add a `depth_vae` parameter to `_SamplerCore.__init__` (reuse `self.vae` if Phase A's identity-edit `vae` param is already threaded through — check for a naming collision and use one shared `vae` param across both features if so, rather than two differently-named VAE inputs).
- Produces: `self._depth_concat: mx.array | None` (the packed depth-latent channels), consumed by `run()`'s patchify block.

- [ ] **Step 1:** Rewrite `_prepare_depth_noise` to VAE-encode `self.depth_image`, following the exact pattern of `_prepare_controlnet_latent` (`sampler/core.py:885-922`) for the encode+pack mechanics, but WITHOUT the `(ctrl_np - FLUX_LATENT_SHIFT) * FLUX_LATENT_SCALE` line replaced — use the same `process_in`-equivalent scale/shift (this project's `FLUX_LATENT_SHIFT`/`FLUX_LATENT_SCALE` constants already match comfy's `process_latent_in` for FLUX, confirmed in `.claude/canon.md`/session-state history). Store the packed result on `self._depth_concat`; return `self.noise` unchanged (noise itself is NOT modified — only a channel-concat partner is produced, matching the reference's `concat_cond`, which concatenates rather than blends).
- [ ] **Step 2:** In `run()`, right after the `img_h`/`img_w`/token-grid check (line 238-242) and before `get_rope()` is called, add:
  ```python
  img_tokens = self.noise
  if getattr(self, "_depth_concat", None) is not None:
      img_tokens = mx.concatenate([self.noise, self._depth_concat], axis=-1)
  ```
  Use `img_tokens` (not `self.noise`) as the value passed into `self.transformer.predict(...)`'s image-token argument for every step of the denoising loop — the concat channels are a per-generation constant, not recomputed per step, but they must ride along with `x` on every call since `img_in` now expects the wider input on every forward pass.
- [ ] **Step 3:** `py_compile apple_silicon_nodes/sampler/core.py`.
- [ ] **Step 4:** Structural test (no live checkpoint required): construct a reduced `FluxConfig(in_channels=128)` / `FluxTransformer`, feed a synthetic `(1, 64, 128)` token tensor through `predict()`, confirm no shape error and a finite, non-NaN output — mirrors the existing "random-weight forward pass" step of this project's `verify-checkpoint` skill.
- [ ] **Step 5:** Run: `uv run pytest apple_silicon_nodes/tests/ -k depth -v` (new test from Step 4, saved as `test_depth_conditioning.py`).
- [ ] **Step 6:** Commit: `git add apple_silicon_nodes/sampler/core.py apple_silicon_nodes/tests/test_depth_conditioning.py && git commit -m "feat: concat VAE-encoded depth latent to FLUX image tokens (concat_cond port)"`

### Task C3: Capability profile

**Files:**
- Modify: `apple_silicon_nodes/capability.py:106-121`

- [ ] **Step 1:** `flux1_depth`'s `generate_params` already lists `depth_image`/`depth_strength` (lines 114-115) — no change needed there. Add a comment noting `latent_channels=16` refers to the *output* latent (what `ASDX_VAEDecode` expects back), not the transformer's `in_channels` (32, detected dynamically per Task C1) — these are two different numbers and a future reader should not "fix" one to match the other.
- [ ] **Step 2:** `py_compile apple_silicon_nodes/capability.py`. No functional change in this task — documentation only, to prevent a future regression from someone "correcting" the channel count here instead of in `native/__init__.py`.
- [ ] **Step 3:** Commit: `git add apple_silicon_nodes/capability.py && git commit -m "docs: clarify flux1_depth's latent_channels is output-latent, not transformer in_channels"`

### Task C4: Real-checkpoint verification

- [ ] **Step 1:** If a real `flux1-depth-dev` (or `flux1-canny-dev`, same mechanism) checkpoint is reachable from this environment: load it end-to-end, generate one image with a real depth map, confirm non-NaN output and a visually plausible depth-following result.
- [ ] **Step 2:** If no such checkpoint is reachable in this environment (the pattern seen repeatedly in this project's own `ROADMAP.md`, e.g. the Krea2 Identity Edit grounding fix): state explicitly in the commit/PR description which parts were verified structurally (header shapes, synthetic forward pass) vs. which remain "not yet re-tested on a real checkpoint" — do not claim the feature "works" without this distinction, per this project's own stated convention (`CLAUDE.md` §"Before shipping a new dequantization path... verify numerically against it on a real checkpoint").

---

## Phase D — Extract depth into `ASDX_DepthConditioning`

Only start this phase after Phase C lands and is at least structurally verified — there is nothing real to extract before that.

### Task D1: Create the node

**Files:**
- Create: `apple_silicon_nodes/depth_conditioning.py`
- Modify: `apple_silicon_nodes/__init__.py`
- Modify: `apple_silicon_nodes/sampler/__init__.py`, `apple_silicon_nodes/sampler/core.py`

**Interfaces:**
- Consumes: `vae.encode` (real ComfyUI VAE, same as Task C2).
- Produces: `model["depth_cond"] = {"depth_image": torch.Tensor, "strength": float}` (raw inputs, not pre-encoded — the actual VAE-encode stays in `_SamplerCore` since it needs `self.noise`'s shape to resize against, matching how `_prepare_controlnet_latent` already reads `self.controlnet["image"]`/`self.controlnet["vae"]` lazily rather than pre-encoding at the ControlNet loader node).

- [ ] **Step 1:** Write `ASDX_DepthConditioning`: inputs `model` (`asdx_model`, passthrough), `depth_image` (required), `strength` (default 1.0), `vae` (required). `execute()` validates `model["capability"].family == "flux1_depth"` (raise `RuntimeError` otherwise, same explicit-refusal convention as `ASDX_Krea2Edit` Task A1 Step 1), then returns `{**model, "depth_cond": {"depth_image": depth_image, "vae": vae, "strength": strength}}` (shallow copy, never mutate the input dict — same rule as `ASDX_LoraLoader`).
- [ ] **Step 2:** `py_compile`.
- [ ] **Step 3:** In `sampler/__init__.py::execute()`, same priority pattern as Task A2: `depth_cond = model.get("depth_cond")`; if present, it wins over the legacy `depth_image`/`depth_strength` inputs (log when both are wired).
- [ ] **Step 4:** In `sampler/core.py::_prepare_depth_noise` (from Task C2), read from `self.depth_cond` first, fall back to `self.depth_image`/`self.depth_strength`/`self.vae`.
- [ ] **Step 5:** Register node, display name `🍏 ASDX Depth Conditioning`, category `ASDX/Conditioning`.
- [ ] **Step 6:** Regression: a workflow wired the old way (depth_image/depth_strength directly on the sampler) must still produce the same result as Phase C's verification.
- [ ] **Step 7:** Commit: `git add apple_silicon_nodes/depth_conditioning.py apple_silicon_nodes/__init__.py apple_silicon_nodes/sampler/ && git commit -m "feat: add ASDX_DepthConditioning node, sampler prefers it over legacy inputs"`

---

## Phase E — Extract img2img/inpaint/Kontext mode-routing

Largest, most invasive phase — touches the sampler's primary execution path. Do this last, after A-D have proven out the model-dict-priority pattern on two smaller features.

### Task E1: `ASDX_KontextReference` node

**Files:**
- Create: `apple_silicon_nodes/kontext_reference.py`
- Modify: `apple_silicon_nodes/__init__.py`, `apple_silicon_nodes/sampler/__init__.py`, `apple_silicon_nodes/sampler/core.py`

**Interfaces:**
- Produces: `model["kontext"] = {"reference_latent": dict, "strength": float}`.

- [ ] **Step 1:** Write the node: inputs `model`, `reference_latent` (LATENT, required), `strength` (default 1.0). `execute()` returns `{**model, "kontext": {"reference_latent": reference_latent, "strength": strength}}`.
- [ ] **Step 2:** `py_compile`.
- [ ] **Step 3:** `sampler/__init__.py`: `kontext_cfg = model.get("kontext")`; if present, `kontext=True` and `kontext_reference_latent`/`kontext_reference_strength` are taken from it, overriding the node's own `kontext`/`kontext_reference_latent`/`kontext_reference_strength` inputs when both are wired (log which wins).
- [ ] **Step 4:** Register, display name `🍏 ASDX Kontext Reference`, category `ASDX/Conditioning`.
- [ ] **Step 5:** Regression: existing Kontext workflow (reference wired directly on sampler) unchanged in output.
- [ ] **Step 6:** Commit: `git add apple_silicon_nodes/kontext_reference.py apple_silicon_nodes/__init__.py apple_silicon_nodes/sampler/ && git commit -m "feat: add ASDX_KontextReference node"`

### Task E2: `ASDX_LatentNoisePrep` node (img2img / inpaint / fill)

**Files:**
- Create: `apple_silicon_nodes/latent_noise_prep.py`
- Modify: `apple_silicon_nodes/sampler/__init__.py`, `apple_silicon_nodes/sampler/core.py`

**Interfaces:**
- Consumes: `_SamplerCore._prepare_img2img_noise`, `_prepare_inpainting_noise`, `_prepare_mask_latent` — these move from `_SamplerCore` methods to plain functions in the new file (they only need `latent_image`/`image`/`mask`/`image_strength`/`mask_blur`/`mask_padding`/`noise`/`config.mlx_dtype` as arguments, no other `self` state — verify this by re-reading each method's body before moving, since `_prepare_inpainting_noise` was just rewritten in this session and may still reference `self.latent_image`/`self.image` directly).
- Produces: `model["prepared_noise"] = {"mode": "img2img"|"inpaint"|"fill", "blended": mx.array}` — wait: this CANNOT be computed at this node's `execute()` time, because it needs the seeded `noise` tensor that `ASDX_MLXSampler` generates from `latent_image`+`seed` (`bridge.prepare_noise_from_latent`, `sampler/__init__.py:220`), and the seed is a sampler input, not available upstream. **Resolution**: this node cannot fully replace the sampler's mode-prep the way Phases A/C/D's nodes could, because seed-dependent noise generation is inherently sampler-side. Produce `model["latent_prep"] = {"mode": ..., "latent_image": ..., "image": ..., "mask": ..., "image_strength": ..., "mask_blur": ..., "mask_padding": ...}` (the raw ingredients, validated and mode-tagged) instead, and leave the actual blend-with-noise step in `_SamplerCore` where the noise tensor already lives. This still removes `mode`/`image`/`mask`/`image_strength`/`mask_blur`/`mask_padding` from the sampler's primary input list down to a single optional `latent_prep` passthrough.

- [ ] **Step 1:** Re-read `_prepare_img2img_noise`, `_prepare_inpainting_noise`, `_prepare_mask_latent` in their current, just-fixed state (this session's commit `2182e81`) to confirm the exact `self.*` fields each one touches.
- [ ] **Step 2:** Write `ASDX_LatentNoisePrep`: inputs `model`, `mode` (combo, same 5 options as today), `image` (optional), `mask` (optional), `image_strength`/`mask_blur`/`mask_padding` (optional, same defaults as today). `execute()` does input validation only (e.g. `mode="inpaint"` requires `mask`) — it does NOT call VAE or build latents (no VAE available here without adding one more required input; `latent_image` for content comes from the sampler's own required input, unchanged). Returns `{**model, "latent_prep": {...}}`.
- [ ] **Step 3:** `py_compile`.
- [ ] **Step 4:** `sampler/__init__.py`: `latent_prep = model.get("latent_prep")`; when present, it supplies `mode`/`image`/`mask`/`image_strength`/`mask_blur`/`mask_padding` in place of the sampler's own same-named inputs (log when both wired). The sampler's own inputs stay, `optional=True`, unchanged — no schema removal in this task (unlike Phase B2, there's no `execute()`-internal branching problem here, so full backward compatibility is achievable, unlike grounded-encode).
- [ ] **Step 5:** Register, display name `🍏 ASDX Latent Noise Prep`, category `ASDX/Conditioning`.
- [ ] **Step 6:** Regression: run the existing img2img and inpaint manual/smoke tests with the OLD wiring (inputs directly on the sampler) and confirm unchanged output; then with the NEW node wired instead and confirm identical output between the two wirings.
- [ ] **Step 7:** Commit: `git add apple_silicon_nodes/latent_noise_prep.py apple_silicon_nodes/sampler/ && git commit -m "feat: add ASDX_LatentNoisePrep node for img2img/inpaint/fill mode routing"`

---

## Self-Review

**Spec coverage:** Phase A covers the pre-existing Krea2Edit spec in full. Phase B covers the CLIP text-encode god-node complaint (grounded-encode extracted; FLUX-dual/SD-single dispatch kept, since that dispatch IS the node's stated single responsibility, not a bolt-on). Phases C+D cover depth control per the user's explicit choice to implement it for real before extracting it. Phase E covers the remaining sampler inputs (img2img/inpaint/fill/Kontext) named in the "sûres + mode routing" option. ControlNet and `krea2_enhancer_strength` are deliberately left on the sampler: ControlNet already lives in the `model` dict (set by `ASDX_ApplyControlNet`, not a sampler input at all — nothing to extract), and `krea2_enhancer_strength` is a single float dial with no prep logic behind it (nothing to extract either).

**Placeholder scan:** Task E2's "Produces" section documents a real design correction found during self-review (seed-dependent noise can't move upstream) rather than leaving a wrong interface in place — this is the one spot where the original extraction idea doesn't fully work, and it's called out explicitly rather than glossed over.

**Type consistency:** `model["identity_edit"]`, `model["depth_cond"]`, `model["kontext"]`, `model["latent_prep"]` all follow the same shallow-copy-of-`model`-dict convention already established by `ASDX_LoraLoader`/`ASDX_Krea2Edit`. `_SamplerCore` constructor parameter names in Phase A Task A2 / Phase D Task D1 / Phase E Task E1 match the existing parameter names already present in `sampler/__init__.py`'s current call to `_SamplerCore(...)` (`kontext_reference_latent`, `kontext_reference_strength`, `source_latent`, `source_image`, `vae`, `depth_image`, `depth_strength`) — no renames introduced.

---

**Plan complete and saved to `docs/superpowers/plans/2026-09-13-node-extraction-and-depth-control.md`.**
