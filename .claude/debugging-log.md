# Debugging Log

Detailed investigation narratives, bug reports, and real-file audit findings.
This is NOT canon — it's the "how we got here" that's recoverable from git
but worth keeping in one place for context.

---

## Auditing the residual LoRA port against real files found real coverage gaps synthetic tests couldn't

After Phases 1-3 were implemented and passed synthetic tests, running `_load_lora_file` against the actual LoRA library surfaced THREE real, previously-undetected coverage gaps:

1. **kohya-ss flat naming** — most real community-trained FLUX.1 LoRAs use flat keys (`lora_unet_double_blocks_0_img_attn_proj.lora_up.weight`) not dotted native keys. Fixed by `_lookup_native_or_kohya`.

2. **Krea2's `txtfusion` sub-transformer** — majority of Krea2 LoRAs target ONLY `txtfusion.*` (`layerwise_blocks`/`refiner_blocks`/`projector`) and never `blocks.*`. Phase 1's attach loop only walked `blocks.*`. Fixed by adding Krea2's `txtfusion.*` and other top-level Linears.

3. **FLUX.1/Flux.2's top-level embedders/output layer** — `img_in`, `txt_in`, `time_in`/`vector_in`/`guidance_in`, `final_layer.linear`, and Flux.2's THREE global `Modulation` instances were unreachable. Fixed by extending the target list.

Two gaps predated the porting work and affected the old merge path identically; both FIXED 2026-08-14:
- (a) Krea2's `tmlp`/`tproj`/`txtmlp` are `nn.Sequential`-wrapped; real files use flat `tmlp.0.weight`. Fixed in `_adapt_leaf`, which now treats an all-digit leaf as an index into the Sequential's `.layers` list — `getattr(list, "0")` is None, so the attribute walk could never reach these.
- (b) ~9 real Krea2 LoRA files and most Z-Image files use genuine HF diffusers keys. Fixed by `_resolve_krea2_diffusers_lora` / `_resolve_zimage_diffusers_lora`; Z-Image additionally needs its split `to_q`/`to_k`/`to_v` assembled into the native FUSED `qkv`.

Key lesson: synthetic tests prove the ATTACH MECHANISM is correct, but cannot prove TARGET COVERAGE is complete. Only loading real, independently-authored LoRA files and counting resolved keys closes this gap.

---

## The SDXL schedule fix was incomplete -- Krea2/Z-Image/Flux2 had the identical wiring gap

The original fix wired `_update_lora_schedule` into `_run_sdxl` and incorrectly stated that the shared FLUX/Krea2/Z-Image denoise loop already called it. Only `run()` (FLUX.1's inline loop) had the call; `_run_krea2`, `_run_zimage`, and `_run_flux2` were three MORE independent loops with the exact same missing wiring. Fixed by adding the identical block to all three loops.

**Two caveats, both since FIXED (2026-08-14) -- kept for the reasoning:**

1. **Krea2's precomputed text context ignores schedule changes.** Krea2 precomputes `context = self.transformer.encode_text(...)` ONCE before the step loop (to avoid ~12s/step re-encode cost). Most real Krea2 LoRAs target `txtfusion` only — changing `self.transformer` every step has NO visible effect after step 0 since `context` is never recomputed. Fixed by re-encoding only on steps that actually moved a text-side weight (`_text_lora_state`): measuring showed the real cost is 86ms at seq=256, not the ~12s quoted (that figure is seq=4444 WITH the enhancer).

2. **Schedule is much more expensive on the residual families than on SDXL's merge path.** A real FLUX.1 run with a densely-targeted diffusers LoRA (494 factors) showed step time jump from ~3s to ~13-14s (4-5x) and peak memory rise from ~48GB to 68.7GB once `delta_scale != 0`. Each call re-runs the full non-destructive clone-and-wrap traversal. SDXL's equivalent merge-based re-apply only cost ~0.70s->~1.05s (1.5x). Fixed by `_rescale_attached_lora`, which mutates the already-attached scales in place instead of re-cloning: 81.6ms -> 0.5ms per step, bit-identical output.

---

## LoKr: three wrong diagnoses before the real one

A real Krea2 generation peaked at 118.6GB on a 64GB machine, then OOM'd in VAE
decode. Three hypotheses, tested in order:

1. **Eager materialization at load** — real but minor. The loader built all 256
   kron deltas up front (23GB of deltas plus ~47GB of float32 intermediates).
   Made lazy: load dropped 23GB → 3GB. Peak unchanged.
2. **Lazy-graph accumulation** — refuted. Added `mx.eval`/`mx.clear_cache` after
   each merge on the theory that MLX was holding every intermediate until a
   final eval. Effect: 97.24GB → 97.30GB, i.e. none.
3. **The targets are the model** — correct. Weighing them showed the 256 LoKr
   targets cover 45.3GB of Krea2's 47.8GB. Merging each one creates a new
   full-size weight while the source transformer still holds the original, so
   the "merge" is a near-complete second copy of the model.

Only the third was found by measuring rather than reasoning. The fix (structured
kron residual) is in the canon.

## Black images: two independent bugs, both invisible to the existing tests

User testing produced black (NaN) images on both ASDX_LoraSchedule and
ASDX_MultiLoraLoader while the same workflow without a LoRA was correct.

1. **Sentinel alpha as the application scale.** Simulating the real 8-step
   schedule showed `scale=9.99e9` and activations at 9.5e9 — instant float16
   overflow. The delta was correct; the multiplier was not.
2. **Shared adapter list.** `_ensure_adaptable` copied `_lora_factors` but not
   the newly added `_lokr_factors`. `copy.copy` on an MLX Module (a dict
   subclass) is shallow, so two leaves shared one list and a second LoRA
   appended into the first one's adapters.

Bug 1 explains the schedule test, bug 2 the multi-loader test. Both passed every
coverage and bit-exactness test that existed, because neither of those looks at
the magnitude of an activation after application — see the canon record.

## Test-harness errors that produced false findings

Four times this session a measurement script, not the code, was wrong. Recording
them because each looked like a real defect first:

- A coverage script that did not reproduce the loader's prefix normalization
  reported "0 deltas applied" across the whole Krea2 library, contradicting a
  known-good observation.
- Comparing two separately-constructed `NextDiT`/`SingleStreamDiT` instances:
  random init differs, so outputs never match. Compare weight DELTAS, not
  absolutes.
- Reading adapter state off the pre-clone tree while the full attach path
  returns a re-cloned one — the fast path mutates in place, the slow path does
  not.
- Importing `lora.py` under a second package name, so `isinstance` dispatch
  failed and Krea2 silently fell through to the SDXL merge (`0/264`). This is
  the module-identity trap already documented for the ComfyUI symlink.

## Krea2 Identity Edit: an empty prompt field first, then a missing source fit

2026-08-14. A Krea2 Identity Edit run produced an image that did not follow the
prompt. Two independent causes, found in order.

**Cause 1 — the prompt was never sent.** The log line
`[ASDX] Text encoded (SD-style): 0 chars, type=clip, grounded on source image`
prints `len(text)` (`conditioning.py:376`). The instruction had been typed into
`system_prompt`, which only fills the grounding template's system block; the `{}`
that receives the user text (`conditioning.py:53-54`) got an empty string. The
model was asked to describe an image and given no instruction. `system_prompt`
should stay empty in almost every case -- it steers what the vision tower
attends to, it is not the prompt.

Ruled out along the way, both benign: `matched 430/686 params` is the expected
count for a bias-free-trained checkpoint (the 256 missing are all `.bias`,
zero-filled at `native/krea2/model.py:1030-1039`, not left random), and the
grounded-encode path is line-for-line identical to the reference's
`Krea2EditGroundedEncode` -- same `_prep`, `grounding_px=768`, template, and
`tokenize(prompt, images=…, llama_template=…)` call.

**Cause 2 — the source latent was never fitted to the target grid.** With the
prompt filled in, the edit still applied only partially, while the same prompt on
two independent implementations agreed with each other. Both fit every source to
the output resolution before it reaches the transformer, and ASDX did not:
`comfyui-krea2edit` calls `_fit_src` from `krea2_edit_forward:192-194` on any
source whose grid differs, and SceneWorks pre-fits in PIXELS via
`fit_edit_references` -> `fit_rgb` "crop" (`image_jobs/base.rs:22`).

Measured on the real run: source `[1, 16, 166, 250]` = an 83x125 grid = 10 375
tokens against a 96x168 target = 48x84 = 4 032 tokens. The source held 72% of the
image tokens, pulling the result toward reproducing the reference instead of
following the instruction, on RoPE coordinates that run past anything training
saw, at ~31 s/step from the quadratic attention cost.

Fixed by `_SamplerCore._fit_source_latent` (`sampler/core.py`), a port of
`_fit_src` verified bit-identical (max|diff| = 0.0) against the reference on six
shape cases, with a null check confirming the comparison can actually see a
difference. Tokens drop 10 375 -> 4 032.

**Transferable lesson.** Two independent references agreeing against ASDX
localized the fault in one step; that is now canon (`ComfyUI and the SceneWorks
stack are the reference implementations`). And read the encode log's character
count before theorizing about conditioning -- a config divergence hunt was
started on a run where the prompt field was simply empty.

## Krea2 Identity Edit face artifacts: the source was fitted in latent space

2026-09-07. A Krea2 Identity Edit run produced wavy, smeared faces (all three
subjects present but distorted). The 2026-08-14 record had fixed the missing
source fit, but a later session had "fixed" residual artifacts by switching
`_fit_source_latent` to a zero-letterbox in LATENT space (bilinear resize of the
VAE latent + zero padding to the target grid, full ref grid, no RoPE offset).

Root cause: both references fit in PIXEL space, not latent space.
`comfyui-krea2edit`'s `_fit_encode_image` (fit mode) resamples the IMAGE
(contain + /16 floor + bicubic), VAE-encodes it, and centers the content-only
ref grid with a RoPE offset; its docstring is explicit -- "latent-space resizing
softens VAE latents, this path never resizes latents at all." SceneWorks does a
direct Lanczos pixel resize. The v1_2 LoRA was trained with the fit geometry.
The zero padding, after Wan21 whitening, becomes arbitrary values; with
ref_boost=4 the target is pulled toward that garbage, smearing the faces.

Confirmed identical on both sides (ruled out): ref_boost=4, LoRA
krea2_identity_edit_v1_2 scale 1.0, guidance, whitening, template. The only
divergence was the SPACE of the fit.

Fixed by `_prepare_krea2_identity_edit_pixels` (sampler/core.py): fit the image
in pixel space (contain + /16 floor + bicubic, or a minimal center-crop when the
AR nearly matches), VAE-encode via the real ComfyUI VAE, whiten, pack; the ref
grid is content-only and centered with a RoPE offset (`get_rope_grid` now takes
a per-block offset, matching the reference `_imgids_offset`). Verified: fit
geometry byte-identical to the reference across 5 aspect cases (incl. the
portrait case that reproduces the 2026-09-06 log `48x72 centered in 48x86`),
RoPE offset matches `_imgids_offset`, and a same-seed run yields clean faces
(3/3). Commit 08fcd27. See canon `Krea2 Identity Edit source is fitted in pixel
space, never in latent space`.

**Transferable lesson.** When a reference offers two code paths (a preferred one
and a fallback), port the PREFERRED one and read its docstring for why the
fallback exists -- the fallback's limitations are the bug you will hit. And a
"fix" that changes the SPACE of an operation (pixels -> latents) without
checking which space the reference and the training used is a regression, not a
fix.

## Krea2 NAG's negative pass silently dropped the ref_boost mask -- caught only by a whole-branch review, not any of 10 per-task reviews

2026-09-10. Porting NAG (Normalized Attention Guidance) for Krea2/Krea2 Identity
Edit, `_nag_edit_block` (`native/krea2/nag.py`) passed `ref_boost` to the
positive pass's raw attention but `None` to the negative pass. The design spec
asserted this "mirrors the reference: source attention bias must not leak into
the negative/text-only pass" -- that claim was wrong. The real reference
(`krea2_nag.py:303-326`) builds a SECOND mask, `negative_mask`, from the same
`_ref_attn_bias` call sized to the negative text length, and passes it into the
negative pass too (`krea2_nag.py:146-152`). Both passes get the source-fidelity
bias; only the text length differs between them.

Every one of the 10 task-level reviews approved the code that shipped this bug,
because each review saw only its own task's diff: the task that wrote
`_nag_edit_block` (Task 5) tested it with `ref_boost=None`; the task that wired
`predict_nag` into the sampler (Task 7) never touched `ref_boost` math at all;
no single task's brief asked "does Identity Edit + NAG + ref_boost != 1.0 ever
get exercised end to end." Only the FINAL whole-branch review -- reading the
whole diff against the real reference file, not just each task's own brief --
caught it, and confirmed it numerically: with `ref_boost=4.0` and a null-case
negative (`neg_context == context`, which must reproduce `predict()` bit-for-bit
like every other null case in this feature), the bug produced a 0.0174-0.0253
max abs diff instead of 0.0 (three independent reproductions, unseeded random
inits, all agreeing it should be exactly 0.0 and wasn't).

Fixed by threading a `neg_ref_boost` mask (built the same way as the positive
one, via the same `_krea2_ref_attn_bias` helper called again with the negative
text's length) through `sampler/core.py` -> `SingleStreamDiT.predict_nag` ->
`_nag_edit_block`'s negative `_raw_attention` call. A regression test was added
that fails on the old code and passes on the fix.

**Transferable lesson.** A design spec's own stated rationale for a divergence
from the reference ("this mirrors the reference") is a claim, not a fact --
verify it by reading the actual reference file line by line, the way this
project's own canon record (`ComfyUI and the SceneWorks stack are the reference
implementations`) already says to. And a chain of correct per-task reviews does
not add up to a correct whole: a bug that requires seeing two tasks' code
together (what Task 5 built, and what no task's tests ever combined it with)
needs a review pass that reads the whole diff, not just each task's own slice
of it.
