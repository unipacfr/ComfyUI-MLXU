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

## ComfyUI and the SceneWorks stack are the reference implementations
kind: convention | date: 2026-08-14 | status: canon
When ASDX diverges from the expected output for a model family, ground truth is ComfyUI's own PyTorch source (plus the relevant custom node) and the SceneWorks Rust stack -- `SceneWorks`, `inference`, `mlx-gen`, `mlx-rs`. Read and port from them, verifying numerically, rather than reasoning from ASDX's own code alone. Because: every silent-divergence bug found so far has been ASDX's own, and two independent references agreeing against ASDX localizes the fault in one step -- that is exactly how Krea2 Identity Edit's missing source-to-target fit was found on 2026-08-14. The single exception is a bug discovered IN a reference: it stops being ground truth on that specific point, and only that point.

## Porting from a reference means converting it to Apple Silicon, not transcribing it
kind: convention | date: 2026-08-14 | status: canon
The reference implementations are ground truth for BEHAVIOR (geometry, layout, constants, operation order), never for the execution substrate: ComfyUI is PyTorch/CUDA-shaped and the SceneWorks stack is Rust. Every port must be rebuilt on MLX/Metal and unified memory -- `mlx.core`/`mlx.nn` on the hot path, no needless host round-trips, `mx.eval()` placed deliberately -- because a transcribed reference inherits an execution model this hardware does not have. An exception must be earned by a MEASUREMENT on real shapes, not by convenience: the canon already records two (VAE encode/decode and the text encoders, where PyTorch-MPS wins), and any new one belongs in the code comment that takes it. Keep the reference's numerical result as the acceptance test for the converted version.

## Krea2 Identity Edit source is fitted in pixel space, never in latent space
kind: constraint | date: 2026-09-07 | status: canon
The Identity Edit source must be fitted in PIXEL space (contain + /16 floor + bicubic, or a minimal center-crop when the AR nearly matches), then VAE-encoded, yielding a content-only ref grid centered with a RoPE offset -- the geometry the v1_2 LoRA was trained with and what `comfyui-krea2edit`'s `fit` mode does. Fitting in latent space (bilinear resize + zero letterbox) is wrong twice over: bilinear interpolation of VAE latents softens them, and the zero padding becomes arbitrary values after whitening that `ref_boost` then pulls the target toward, producing face artifacts. Because: the reference's own docstring states "latent-space resizing softens VAE latents -- this path never resizes latents at all," and the 2026-09-06 face-artifact bug was exactly this latent-space fit. The /16 (not /8) pixel snap keeps the centered offset byte-identical to training; a /8 snap shifts the grid and re-creates a seam.

## MiniMax H3's text encoder is ported to MLX despite the text-encoder weak-ROI record
kind: choice | date: 2026-09-15 | status: canon
`native/minimax_h3/text_encoder.py` (Qwen3-VL-32B, truncated to 50 layers) is a full MLX/`nn.QuantizedLinear` port, chosen deliberately over routing through ComfyUI's real `comfy.sd.CLIP` path (the pattern `krea2_grounded_encode.py` uses) even though "Porting CLIP/T5/Qwen text encoders to MLX has weak memory ROI, except FP8 sources" still holds in general -- this is NOT a measured exception to that record (no measurement was taken showing MLX wins here), it is a deliberate pipeline-purity choice: keep MiniMax H3 end-to-end on MLX/Metal with no PyTorch CLIP model resident at inference, since it is this project's largest text encoder (25.8B params) and the DiT it feeds is itself MLX-native. Tokenization still reuses ComfyUI's real, weight-free `comfy.text_encoders.minimax.MiniMaxH3Tokenizer` (BPE only, no model weights, safe to instantiate directly) -- only the encoder forward pass is native MLX. Future model families should default to the CLIP-bridge pattern per the existing record; this is a one-off, not a reversal of the general guidance.

## MLX requantize in a streaming loader must be evaluated per tensor
kind: gotcha | date: 2026-09-19 | status: canon
In the MiniMax H3 loaders, `mx.quantize` results are lazy, so the graph keeps every dequantized fp32 source tensor alive until the final `mx.eval`. Measured on the real 20B DiT: 79GB peak for 12.8GB resident and a 155s load; `mx.eval(w_q, scales, biases)` inside the per-tensor loop gave a 14GB peak in 26s. Because: `mx.get_active_memory()` reads near zero right after a lazy load and hides this, so any new streaming/requantizing loader must be checked with `mx.get_peak_memory()`, not active memory.

## MiniMax H3 memory gate is calibrated on measured peaks, not file size
kind: choice | date: 2026-09-19 | status: canon
memory_calibration._FAMILY_HEURISTIC_MULTIPLIER sets `minimax_h3_dit` and `minimax_h3_text_encoder` to 1.0 instead of the 2.3 default. Because: every big linear is requantized to MLX 4-bit at load, so resident size follows the parameter count and not the source format (measured peaks: DiT 14.1GB from a 20GB INT8 safetensors, 13.4GB from Q5_0 GGUF; text encoder 16.8GB from 26GB safetensors, 16.2GB from Q4_K_M GGUF). Activation memory at full video resolution is NOT covered by these numbers.
