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

Two unfixed gaps predate the porting work and affect the old merge path identically:
- (a) Krea2's `tmlp`/`tproj`/`txtmlp` are `nn.Sequential`-wrapped; real Krea2 LoRA files use flat checkpoint-style keys (`tmlp.0.weight`) matching neither dotted-native nor kohya-flat form.
- (b) ~9 real Krea2 LoRA files and most Z-Image LoRA files use genuine HF diffusers-style keys; neither family has ever had a diffusers-key resolver.

Key lesson: synthetic tests prove the ATTACH MECHANISM is correct, but cannot prove TARGET COVERAGE is complete. Only loading real, independently-authored LoRA files and counting resolved keys closes this gap.

---

## The SDXL schedule fix was incomplete -- Krea2/Z-Image/Flux2 had the identical wiring gap

The original fix wired `_update_lora_schedule` into `_run_sdxl` and incorrectly stated that the shared FLUX/Krea2/Z-Image denoise loop already called it. Only `run()` (FLUX.1's inline loop) had the call; `_run_krea2`, `_run_zimage`, and `_run_flux2` were three MORE independent loops with the exact same missing wiring. Fixed by adding the identical block to all three loops.

**Two unfixed caveats:**

1. **Krea2's precomputed text context ignores schedule changes.** Krea2 precomputes `context = self.transformer.encode_text(...)` ONCE before the step loop (to avoid ~12s/step re-encode cost). Most real Krea2 LoRAs target `txtfusion` only — changing `self.transformer` every step has NO visible effect after step 0 since `context` is never recomputed. Fixing this properly means re-encoding text per step, reintroducing the cost the precompute exists to avoid — a real trade-off.

2. **Schedule is much more expensive on the residual families than on SDXL's merge path.** A real FLUX.1 run with a densely-targeted diffusers LoRA (494 factors) showed step time jump from ~3s to ~13-14s (4-5x) and peak memory rise from ~48GB to 68.7GB once `delta_scale != 0`. Each call re-runs the full non-destructive clone-and-wrap traversal. SDXL's equivalent merge-based re-apply only cost ~0.70s->~1.05s (1.5x). Worth optimizing (mutate existing `_lora_factors` scale in place instead of re-cloning) if Schedule + large residual-family LoRA becomes a common workflow.
