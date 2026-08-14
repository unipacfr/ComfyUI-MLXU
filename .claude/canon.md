# Canon

Settled engineering decisions for this repository. The protocol lives in
`.claude/rules/canon.md`. Append new records at the end; supersede, never
rewrite. Managed with Claudoscope, safe to edit by hand.

Detailed investigation narratives and bug reports live in `.claude/debugging-log.md`.

## Kontext ref_latents implements only the single-reference case
kind: choice | date: 2026-08-04 | status: canon
For a single Kontext reference latent, all three `ref_latents_method` variants reduce to the same placement (RoPE axis-0 index=1, h_offset=w_offset=0). The multi-reference stacking/offset logic was intentionally not ported because this project's Kontext node only ever supplies one reference latent. `FluxTransformer.get_rope` already accepts a `ref_grids` list for future extension.

## ASDX_VAEDecode's _fallback_decode double-indexed the latent
kind: gotcha | date: 2026-08-04 | status: canon
PyTorch's advanced indexing treats a bare string as a character sequence — calling `vae.decode(latent["samples"])` on a 4D tensor with the string "samples" (7 chars) produces exactly `IndexError: too many indices for tensor of dimension 4`. When an error looks intermittent or environment-specific, re-derive the repro from the ACTUAL calling code path before reaching for a GPU/timing explanation.

## Node package migrated fully to ComfyUI's V3 API
kind: choice | date: 2026-08-09 | status: canon
All ASDX nodes use `io.ComfyNode` with `define_schema()`/`execute()` via a single `ComfyExtension`/`comfy_entrypoint()`, not V1 `NODE_CLASS_MAPPINGS`. Every `node_id` was kept identical to its V1 key so saved workflows continue to resolve.

## Krea2T enhancer's ~x75 amplification needs a finite-value guard and bfloat16
kind: gotcha | date: 2026-08-09 | status: canon
`krea2t_enhance_conditioning` amplifies Qwen3-VL taps by ~75x, which can overflow float16 on real hidden-state outliers, producing 100% NaN embeddings and black images. Fixed by a finite-check fallback and recommending bfloat16 precision.

## CLIP `clip_type` must be set manually outside ASDX_CheckpointLoader
kind: gotcha | date: 2026-08-09 | status: canon
Only `ASDX_CheckpointLoader` auto-detects the text-encoder architecture from checkpoint weights. Standalone `ASDX_CLIPLoader`/`ASDX_DualCLIPLoader` require manual `clip_type` — leaving it at a generic default silently loads a plain single-layer Qwen3-VL encoder instead of `Krea2TEModel`, producing a valid non-NaN image that doesn't match the prompt (not a crash).

## `resource.ru_maxrss` measures a loading-time peak, not steady-state memory
kind: gotcha | date: 2026-08-11 | status: canon
`resource.ru_maxrss` is a monotonic high-water mark for the whole process; it captures transient buffers during `load_state_dict()` and never drops, producing false-positive memory-waste readings. Always cross-check suspicious peaks against live RSS post-GC before concluding a framework/dtype is wasting memory.

## Porting CLIP/T5/Qwen text encoders to MLX has weak memory ROI, except FP8 sources
kind: choice | date: 2026-08-11 | status: canon
PyTorch-CPU text encoder live RAM is already within 2-8% of on-disk size on Apple Silicon Unified Memory, so porting CLIP/T5/Qwen to MLX yields no memory benefit. Exception: Flux.2/Klein's FP8 text encoder doubles live RAM (CPU dequant to fp16), which is a quantization-handling gap, not a framework gap.

## Porting VAE encode/decode to MLX has negative ROI -- PyTorch-MPS kernels win
kind: choice | date: 2026-08-11 | status: canon
VAE already runs on MPS (not CPU), so there is no CPU-RAM gap to close. PyTorch-MPS conv2d+GroupNorm+SiLU kernels are consistently faster than MLX at the high-spatial-resolution regime that dominates decode time. Do not port VAE to MLX.

## Flux.2/Klein's FP8 text encoder and a native-BF16 alternative cost the same live RAM
kind: choice | date: 2026-08-11 | status: non-canon, superseded by: An FP8 text encoder can be kept in FP8 by faking llama_detect's two probe keys
FP8 and BF16 variants of Flux.2's Qwen3-8B text encoder have identical live RAM (~15.79GB). FP8's on-disk compactness is fully cancelled by CPU-side dequant-to-fp16 cost. The FP8 file is strictly better for disk at equal RAM.

## LoRA merge OOM has two independent duplication sources, not one
kind: gotcha | date: 2026-08-12 | status: canon
`_load_lora_file` eagerly computes `delta = B @ A` for every target at load time; `_apply_lora_to_transformer` then builds a SECOND full parameter set via `type(transformer)(config)`. Both structures are alive simultaneously during the merge loop on top of the original base weights — explains the observed ~2x/98GB peaks. Prior fix attempts (chunked `mx.eval()`, `mx.clear_cache()`) had zero effect because neither addressed either structure's *lifetime*, only transient per-chunk scratch. Source #1 fixed in Phase 0; Source #2 addressed by forward-time-residual architecture (Phases 1-3).

## comfy's automatic tiled-VAE OOM fallback never fires on Apple Silicon MPS
kind: gotcha | date: 2026-08-12 | status: non-canon, superseded by: MPS OOM is detected by message, and the tiled retry runs outside the except block
`comfy.sd.VAE.decode()`'s OOM detection only matches `torch.cuda.OutOfMemoryError` or `torch.AcceleratorError`. A real MPS OOM raises a plain `RuntimeError` ("MPS backend out of memory..."), which is NOT an instance of either type — so `is_oom()` returns `False`, the node crashes instead of falling back to tiled decode. Not yet implemented: wrap `vae.decode`/`vae.encode` with a `RuntimeError` + `"out of memory"` message check.

## `AdaptableLinear` adapters must upsert by array identity, not append, or `ASDX_LoraSchedule` grows unbounded
kind: gotcha | date: 2026-08-12 | status: canon
`sampler/core.py::_update_lora_schedule` re-applies the LoRA every sampling step. A naive `.append()` on every attach call would add one MORE redundant adapter entry per step, since each call sees the same `(a, b)` but has no way to recognize "this is the same adapter" — the forward pass would re-run one more low-rank matmul per step, unbounded over the whole sampling run. Fixed by `_upsert_lora_factor`/`_upsert_lora_delta`: find an existing entry with the same `a`/`b` array IDENTITY and ADD the incoming scale (not replace, not append).

## Full-size LoRA deltas are merged once into `.weight`, never held as a residual
kind: gotcha | date: 2026-08-13 | status: canon
Full-size LoRA deltas (from diffusers/PEFT multi-component assembly) must be merged once into `AdaptableLinear.weight`, not held as forward-time residuals. Holding them permanently resident causes memory doubling and worsening per-step latency across cached model reuse. A full-size delta gains nothing from residual treatment — it isn't cheap like a low-rank pair. Fixed: `merge_delta(delta, scale)` does `self.weight = self.weight + scale * delta` ONCE; `_lora_deltas` and its forward-pass loop removed.

## MPS OOM is detected by message, and the tiled retry runs outside the except block
kind: gotcha | date: 2026-08-14 | status: canon
`vae.py::_is_mps_oom` matches a plain `RuntimeError` whose message contains "out of memory", because `torch.AcceleratorError` is a `RuntimeError` SUBCLASS (not the reverse) so a real MPS OOM is an instance of neither type comfy's `is_oom()` checks. The tiled retry is flagged inside `except` and executed AFTER it: a live exception keeps every tensor allocated at raise-time referenced, so retrying inside the block runs against memory that has not been released. Confirmed firing in a real generation.

## An FP8 text encoder can be kept in FP8 by faking llama_detect's two probe keys
kind: gotcha | date: 2026-08-14 | status: canon
comfy takes `dtype_llama` from the dtype of two LayerNorm probe keys (`hunyuan_video.py::llama_detect`), then `flux2_te` applies it unconditionally -- so neither `model_options["dtype"]` nor `--fp8_e4m3fn-text-enc` can keep an FP8 encoder in FP8 (both measured to have zero effect). Casting ONLY those two probes to FP8 and restoring the real BF16 norms afterwards gives 7.72GB live instead of 15.35GB with bit-identical output. The restore is not optional: FP8 norms alone (0.3M params) shifted every embedding by 6.4% rel. L2, against 0.998 cosine between two *different* prompts.

## LyCORIS scale is alpha/rank, except both-full LoKr which forces 1.0
kind: constraint | date: 2026-08-14 | status: canon
A LoKr storing BOTH Kronecker factors in full has no low-rank dimension, so its `alpha` is not a scale and the module applies at 1.0 (LyCORIS `LokrModule.__init__`, confirmed against mlx-gen's `adapters/loader.rs::scale`). A real file here carries a sentinel alpha of 9999220736.0; using it as the multiplier drove activations to 9.5e9 and produced black images in float16. `rank <= 0` or a non-finite alpha must RAISE -- `0/0 = NaN` baked into a delta poisons every weight it touches while the load still reports success.

## An adapter whose targets are most of the model must stay a residual, not merge
kind: choice | date: 2026-08-14 | status: canon
The merge-once rule for full-size deltas holds only when the targets are a small slice of the model. A real LoKr's 256 targets covered 45.3GB of Krea2's 47.8GB, so merging built a near-complete second copy: +46.6GB permanently, a 118.6GB peak on a 64GB machine. A Kronecker delta has a structured form (`_kron_matmul` contracts against each factor in turn) that applies it without ever materializing `[out, in]` -- 1.46GB of factors instead, a 32x difference. Before choosing merge, weigh what fraction of the model the targets actually represent.

## Coverage and magnitude are separate checks; a correct delta can still be mis-scaled
kind: convention | date: 2026-08-14 | status: canon
Two LoRA bugs reached real generations while every existing test passed: both counted resolved keys and verified delta values bit-exact against a reference, and neither looked at the magnitude of an activation after the delta was applied. The deltas were right; the scale multiplying them was ~1e10. Any adapter work must assert a real forward pass is finite AND within a sane magnitude, not just that keys resolve and math matches.

## MLX float32 matmul carries ~1e-3 relative error on this Metal backend
kind: gotcha | date: 2026-08-14 | status: canon
A plain `x @ K.T` in MLX float32 deviates ~7.5e-03 from the same product in numpy float32, while numpy f32-vs-f64 is 6.5e-07. Do not treat a ~1e-3 relative gap between two MLX formulations as a formula error -- it is backend accumulation noise. Compare against a float64 reference computed OUTSIDE MLX before concluding a rewrite is numerically wrong.
