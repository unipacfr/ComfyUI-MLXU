# LoRA Factor Dtype (mlx-gen convention) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the per-step cost of forward-time LoRA residuals (measured +30% on Anima with a 448-target LoRA at 1024², CFG) by keeping LoRA factors in their file dtype and computing the residual in the natural promoted dtype, as mlx-gen does, instead of upcasting factors and activations to float32.

**Architecture:** Two changes in `apple_silicon_nodes/lora.py`, both cross-family: (1) `_load_lora_file` keeps each low-rank factor in its file dtype (bf16 stays bf16; F16 stays F16; F32 stays F32) instead of the current bf16->float32 upcast; (2) `AdaptableLinear.__call__` computes `(x @ A.T) @ B.T` without casting `x` to the factor dtype (MLX's natural promotion: bf16 x bf16 -> bf16, bf16 x f16 -> f32, bf16 x f32 -> f32) and narrows the scaled residual to the host output dtype before the add. Every full-size delta materialization (`_delta_from_factors`, LoHa, LoKr materialization used by merge paths) keeps computing in float32 and returns float32, so merge-based families (SDXL, MiniMax H3) are numerically no worse than today.

**Tech Stack:** Python 3.13, MLX, pytest via `uv run pytest`.

**Spec:** user decision 2026-09-26 ("option 2") after measurement and a survey of the reference stack.

## Global Constraints

- Reference: `/Volumes/X10Pro/Images/Projet/inference/crates/media/mlx-gen/src/adapters.rs:384-423` — "LoRA -- `scale · (x·A)·B` with `A`/`B` kept at their loaded (file) dtype. The fork never casts the factors ... The result is NOT cast back here ... `AdaptableLinear`'s adapter accumulation narrows the residual to the host's output dtype before the add so installing an adapter cannot widen the host Linear". mlx-gen removed its former forced-f32 upcast (sc-2718) after validating against its goldens (Z-Image / Qwen LoRA+LoKr).
- Canon "`AdaptableLinear` adapters must upsert by array identity, not append": unchanged behavior.
- Canon "Full-size LoRA deltas are merged once into `.weight`": merge paths keep float32 materialization.
- Canon "Coverage and magnitude are separate checks" and "A parity test must be proven by mutation".
- LoKr structured residual (`_kron_matmul`) is out of scope unless the load change alters its factor dtype; if it does, keep its current math (it already casts `x` to the factor dtype) and report.
- The full suite must stay green (baseline 563 passed, 26 skipped, 0 failed). Python via `uv run`. Commit messages `type: description`, no scope, no attribution trailer. Stage explicit paths only; never `git add docs` or `.superpowers`. Do not push.

## Reference facts (measured 2026-09-26, WaiHassakuAnima bf16, tifa-anima 448 targets, 1024², CFG, interleaved 6 rounds)

- base 0.878 s/step; f32 factors (current) 1.141 (+30%); bf16 factors 0.980 (+12%); merged 0.943.
- LoRA-induced difference, bf16 factors vs f32 factors: cosine 0.997975, norm ratio 1.0003 (bf16 model).
- Library factor dtypes: Anima 33 BF16 + 3 F16; other families: inspect per family in Task 2.
- Timing on this machine drifts thermally by up to 20% within a run: every timing comparison must interleave variants over several rounds and report medians.

## Review Focus

- F16 factor files on a bf16 model must stay finite (natural promotion bf16 x f16 -> f32 should make them safer than today's fp16 cast; verify on the 3 real F16 Anima files).
- Merge-based families (SDXL merge loop, MiniMax H3 dequant/requant, LoHa, `.diff`) must produce the same merged weights as before within bf16 rounding.
- `ASDX_LoraSchedule` rescale and `ASDX_MultiLoraLoader` stacking must be unaffected (factor identity/upsert unchanged).
- A float32 model (e.g. FLUX.2 f32 image stream, parity tests at float32) must see no change: bf16 factors promote to f32 exactly.

---

### Task 1: Keep factor file dtype; natural-promotion residual

**Files:**
- Modify: `apple_silicon_nodes/lora.py` (`_load_lora_file` dtype handling, `AdaptableLinear.__call__`, and any materialization helper that must now upcast explicitly)
- Test: `tests/test_lora_factor_dtype.py`

Requirements:
1. `_load_lora_file`: today safetensors are read with `safetensors.torch`, and bf16 tensors are converted to float32 because numpy has no bf16. Keep that conversion path only as a transport, then restore each factor's file dtype in MLX (`mx.array(np_f32).astype(mx.bfloat16)`), or read the file with `mx.load` directly if that is simpler and handles every format `_load_lora_file` supports today (.safetensors, .pt, .bin). Scalars (`.alpha`) stay Python floats as today.
2. `AdaptableLinear.__call__`: residual `= (x @ a.T) @ b.T` with no cast of `x`; then `y = y + (scale * residual).astype(y.dtype)`. Keep the LoKr loop's math unchanged.
3. `_delta_from_factors` and every other full-size materialization (LoHa, LoKr materialized form) compute in float32 and return float32 (today they return `b.dtype`, which was float32 for bf16 files). Check each call site that merges (`merge_delta`, SDXL merge loop `value + delta.astype(value.dtype) * scale`, MiniMax H3) still casts correctly.
4. Tests in `tests/test_lora_factor_dtype.py` (tiny synthetic LoRA written to a temp safetensors file with bf16 factors, and one with F16 factors):
   - factors loaded keep their file dtype (bf16 -> mx.bfloat16, F16 -> mx.float16);
   - `AdaptableLinear` output with bf16 x and bf16 factors equals `x@W.T + scale*((x@A.T)@B.T)` computed in bf16, and its dtype is the host output dtype (bf16);
   - with float32 x the result equals the float32 reference exactly as before (bf16->f32 promotion is exact);
   - `_delta_from_factors` returns float32 and equals `(B.f32 @ A.f32)`;
   - mutation: forcing the old `x.astype(a.dtype)` upcast path changes the dtype/timing-relevant behavior the test pins (document what the test detects).
5. Run all existing LoRA tests (`uv run pytest tests -q -k lora`) and the full suite.

- [ ] Step 1: failing tests; Step 2: run (fail); Step 3: implement; Step 4: run focused + LoRA + full suite; Step 5: commit `perf: keep LoRA factors in file dtype and skip the float32 residual upcast`.

---

### Task 2: Real-weight verification per family + timing + docs

**Files:**
- Create: `scripts/lora_dtype_check.py` (reusable: for a given family checkpoint + LoRA file, compare the LoRA-induced output difference old-path vs new-path, and time both interleaved)
- Modify: `README.md` only if it documents LoRA precision; canon record proposal goes in the report, not the canon file.

Requirements:
1. Old-path reference without reverting code: in the script, emulate the old behavior by upcasting every attached factor to float32 and wrapping the residual with the old `x.astype(a.dtype)` cast (monkeypatch `AdaptableLinear.__call__` in-process), so old vs new run in the same process on the same inputs.
2. For each family with a native MLX residual path AND a checkpoint + LoRA available locally (Anima; and Krea2, Z-Image, FLUX.1, Flux.2/Klein if files exist — locate them under `/Volumes/X10Pro/Images/models/` and `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/models/`; skip a family if its checkpoint is too large to load twice safely on 64 GB and say so), report: LoRA-difference cosine new vs old, norm ratio, finite, and interleaved median s/step old vs new at the family's usual resolution (one forward pass pair per step if CFG applies, else one). Pass: difference cosine > 0.995 and norm ratio within 1% for every family; any family below that is reported with numbers as DONE_WITH_CONCERNS (do not tune code blindly).
3. Anima: run all 3 real F16 files (Mavis, Pocahontas, vanessa-ngai-anima) finite at [1,16,128,128] on the bf16 model; and re-run `scripts/anima_parity.py --lora` for `tifa-anima` to confirm fp32 parity vs ComfyUI is unchanged (expected identical, since bf16->f32 is exact).
4. Commit `test: per-family LoRA dtype check script and results`.

## Out of scope

- Fusing static LoRAs into the weights (option 1), deferred by the user.
- LoKr residual dtype policy beyond what the load change implies.
