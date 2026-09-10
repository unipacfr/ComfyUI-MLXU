# Krea2 Normalized Attention Guidance (NAG) — design

Date: 2026-09-10
Status: approved (user), pre-implementation

## Problem

Krea2/Krea2Edit in ASDX have no negative-prompt mechanism. The FLUX/Krea2
denoise loop only ever does a single conditional forward pass per step
(`sampler/core.py:1475-1480`); guidance is baked into `guidance_in`, not true
classifier-free guidance (unlike SDXL's two-pass CFG at `core.py:1546`).

NAG ("Normalized Attention Guidance", https://arxiv.org/abs/2505.21179) gives
Krea2/Krea2Edit a negative prompt without true CFG: both passes share the
same image query, only the text K/V context differs, and the two attention
outputs are combined *inside* attention (paper eqs. 7-10), not at the
sampler's eps level. Reference implementation:
`/Volumes/X10Pro/Images/Projet/krea2-nag` (ComfyUI native PyTorch,
`SingleStreamDiT`).

## Scope

Both Krea2 text-to-image and Krea2 Identity Edit (Krea2Edit). For Identity
Edit, NAG must apply **only to target-image tokens** — source-reference
attention (and `ref_boost`) must stay exactly as today on the positive path,
per the reference's `guide_attention_tail`/`_nag_edit_block`.

## Why not a ComfyUI-style "model patch" node

The reference exposes NAG as `ModelPatcher` wrappers
(`comfy.patcher_extension`) so it composes with any sampler. ASDX has no
such wrapper system — `ASDX_MLXSampler` runs the whole Krea2 loop directly in
`_run_krea2` (`sampler/core.py:1185`). NAG parameters are therefore new
**optional inputs on the existing sampler node**, not a separate patch node:
`nag_negative` (CONDITIONING), `nag_phi`, `nag_tau`, `nag_alpha`,
`nag_sigma_start`, `nag_sigma_end`. Absent `nag_negative` → today's behavior,
byte-identical (this is the regression-safety net).

## Why the block loop can't be reused as-is

`SingleStreamDiT.__call__` (`native/krea2/model.py:780-876`) builds one
sequence `combined = [context | src | img]` (`:842-850`) and runs it through
`self.blocks` in a single loop (`:863-869`), where each `SingleStreamBlock`
(`:456-514`) calls `self.attn(...)` which bakes `gate` and `wo` *inside*
`Attention.__call__` (`native/krea2/model.py:129-207`, gate at `:204`, `wo`
at `:207`). NAG operates on the attention output **before** gating/`wo`
(paper eqs. 7-10 need `z_pos`/`z_neg` in raw-attention space) — see the
reference's `_raw_attention` splitting `gate` off before `wo`
(`krea2_nag.py:45-66`). So a NAG pass needs a raw-attention primitive that
stops before `gate*out`/`wo`, matching `_raw_attention`, not the existing
`Attention.__call__`.

Both passes reuse the *same* image tokens (`img`/`src`), so the image-token
Q/K/V are byte-identical between the positive and negative pass — only the
text K/V differ. This is what lets NAG stay cheap: no doubled sampler eps
pass, only a second block-loop pass with a shorter (text-only) sequence
change per block.

## Components

### `native/krea2/nag.py` (new file)

1. **`normalized_attention_guidance(pos, neg, phi, tau, alpha) -> mx.array`**
   Direct port of `krea2-nag/nag_math.py::normalized_attention_guidance`
   (paper eqs. 7-10). Norms/ratio computed in `mx.float32` regardless of
   input dtype (mirrors the PyTorch port's `.float()`), clamped with
   `mx.clip` (`torch.clamp` → `mx.clip` is the only substitution). Raises on
   shape mismatch, matching the reference's explicit `ValueError`.

2. **`guide_attention_tail(pos, neg, positive_start, negative_start, phi, tau, alpha) -> mx.array`**
   Port of the same-named reference function: keeps the positive prefix
   (text + source-reference tokens) untouched, applies NAG only to the
   tail (target-image tokens).

3. **`_raw_attention(attn: Attention, x, freqs, ref_boost=None) -> tuple[mx.array, mx.array]`**
   Duplicates `Attention.__call__` (`model.py:129-207`) up to (not
   including) `out = out * gate; return self.wo(out)` — returns
   `(raw_out, gate)` both as `[B, L, D]`, so the caller can slice image rows,
   apply NAG, concat back, then apply `gate`/`wo` itself. This is the one
   piece of real duplication against `Attention.__call__`; a comment must
   point at both call sites so a future attention-math change is caught in
   both places (this project has hit exactly this class of drift before —
   see canon `Krea2T enhancer's ~x75 amplification` and the debugging-log's
   "recurred multiple times across separate call sites" pattern for VAE).

4. **`_nag_block(block, pos_text, neg_text, image, tvec, pos_freqs, neg_freqs, phi, tau, alpha, ref_boost=None) -> tuple[mx.array, mx.array, mx.array]`**
   Port of `krea2_nag.py::_nag_block`: builds `positive = concat([pos_text, image])`,
   `negative = concat([neg_text, image])`, runs `_raw_attention` on each
   (image rows are identical `x` on both, so Q/K/V for those rows match),
   slices the image-token tail from each raw output, calls
   `normalized_attention_guidance` on that tail only, reassembles, applies
   `block`'s own `gate`/`wo`/`mlp` (reading `block.attn.gate_proj`/`block.attn.wo`
   directly — no new modulation math, `DoubleSharedModulation` output is
   reused as-is). Returns `(pos_text_out, neg_text_out, image_out)`.

5. **`_nag_edit_block(block, pos_text, neg_text, image, target_offset, tvec, pos_freqs, neg_freqs, ref_boost, negative_ref_boost, phi, tau, alpha) -> tuple[mx.array, mx.array, mx.array]`**
   Port of `krea2_nag.py::_nag_edit_block`: same as above but calls
   `guide_attention_tail` instead of `normalized_attention_guidance`, so
   only tokens from `target_offset` onward (the target image, after
   `source_len` — reuses the existing `self._identity_edit_src_offset`
   concept from the pixel-space fix, see canon `Krea2 Identity Edit source
   is fitted in pixel space`) are guided. `ref_boost`/`negative_ref_boost`
   are applied to **both** passes' raw attention (mirrors the reference,
   `krea2_nag.py:303-326`: it builds a `positive_mask` and a `negative_mask`
   from the same bias math, each sized to its own pass's text length —
   the source-fidelity bias is not specific to the positive pass, only its
   text length differs between the two calls).

### `native/krea2/model.py` changes

`SingleStreamDiT` gains one new method, **not** a change to `__call__`:

```python
def predict_nag(
    self, img, context, neg_context, timestep, img_h, img_w,
    freqs, neg_freqs, ref_boost=None, src=None, src_h=None, src_w=None,
    src_offset=(0, 0), phi=4.0, tau=2.5, alpha=0.25,
) -> mx.array:
```

It duplicates `__call__`'s img-projection / `combined` assembly (`:823-850`)
for **two** sequences (`positive = [context | src | img]`,
`negative = [neg_context | src | img]`), then loops `self.blocks` calling
`_nag_block`/`_nag_edit_block` (edit variant iff `src is not None`) instead
of `block(...)` directly, then applies `self.last` to the positive branch's
image tail exactly like `__call__` does today (`:871-876`). This keeps
`__call__`/`predict` (the non-NAG path) completely untouched — zero risk to
existing behavior when `nag_negative` is absent.

`get_rope_grid` is reused unchanged for `neg_freqs`, called with the
negative text's own token length (positive and negative prompts are not
required to tokenize to the same length — only the image-token portion of
the freqs must match, which it does since both calls receive the same
`img_h`/`img_w`/`src_grids`/`src_offsets`).

### `sampler/core.py` changes (`_run_krea2`)

- New optional attrs on `_SamplerCore` (set from new `ASDX_MLXSampler`
  inputs): `nag_negative`, `nag_phi`, `nag_tau`, `nag_alpha`,
  `nag_sigma_start`, `nag_sigma_end`.
- If `nag_negative` is given: encode it the same way `txt_fused` is encoded
  (`conditioning_krea2_to_mlx` → `encode_text`), once before the step loop,
  same hoisting rationale as the existing `context` precompute
  (`core.py:1292-1293`).
- **CFG must be 1.0.** Krea2/Krea2Edit's guidance is baked into
  `guidance_in`, already separate from a CFG scale — but this sampler has no
  CFG-scale concept for Krea2 (SDXL is the only two-pass path). So "CFG 1.0"
  here means: raise if the caller also tries to combine NAG with anything
  that would imply a second conditioning branch. In practice, for this
  sampler that's automatically satisfied (Krea2 never runs a second
  eps-level branch) — the check is a defensive assertion, not new plumbing,
  to keep behavior identical to the reference's stated constraint.
- Per-step, all three `self.transformer.predict(...)` call sites
  (`core.py:1326, 1355, 1383`, including the higher-order-solver
  `_model_call` closure) must be replaced with
  `self.transformer.predict_nag(...)` when NAG is active. A single small
  wrapper closure (`_call_transformer(x_at, sigma_at)`) picks `predict` vs
  `predict_nag` once per step to avoid repeating the branch three times.
- Sigma gating (`nag_sigma_start`/`nag_sigma_end`) reuses `_nag_is_active`'s
  logic from the reference (`krea2_nag.py:36-42`): NAG only applies while
  `sigma_end <= sigma_t <= sigma_start`; outside that window, call `predict`
  instead of `predict_nag` for that step (matches the reference's "NAG only
  in a sigma band" behavior, e.g. paper's guidance-scheduling recommendation).

## Data flow summary

```
nag_negative (CONDITIONING) ──encode_text──► neg_context [B, txt_len_neg, 6144]
                                                    │
target img/src (existing) ──────────────────────────┤
                                                    ▼
                                     predict_nag() per step, per solver eval
                                     (replaces predict() only when active)
                                                    │
                                     _nag_block/_nag_edit_block per transformer block
                                     (raw attention twice, NAG-combine image tail, gate+wo once)
                                                    ▼
                                          noise_pred (positive branch only)
```

## Error handling

- `nag_negative` conditioning present but empty → raise (mirrors reference's
  `_make_state`'s empty check).
- `nag_sigma_start < nag_sigma_end` → raise.
- `nag_phi == 0.0` or `nag_alpha == 0.0` → NAG is a no-op for that
  configuration; reference treats this as "inactive", not an error
  (`_nag_is_active`) — same here, log once and fall back to `predict`.
- Shape mismatch inside `normalized_attention_guidance` → raise with actual
  shapes (already the reference's behavior, keep it).

## Testing

1. **Unit, `normalized_attention_guidance`**: synthetic `[B,H,L,D]` tensors,
   compare MLX output against the PyTorch reference formula computed with
   numpy/torch on CPU, tolerance ~1e-3 (canon: `MLX float32 matmul carries
   ~1e-3 relative error on this Metal backend` — do not treat that gap as a
   bug). Include a null case: `pos == neg` must return `pos` unchanged
   (guided = pos when phi>0 but z_pos-z_neg=0) — sanity-checks the harness
   per `feedback_inspect-artifacts-first`/4a discipline.
2. **Geometry parity (Identity Edit)**: for 2-3 aspect-ratio cases already
   used to verify the pixel-space fit, confirm `_nag_edit_block`'s
   `target_offset` lines up with `self._identity_edit_src_offset` and that
   `ref_boost` masks are byte-identical to the non-NAG path's mask.
3. **Real-run smoke test**: one Krea2 T2I run and one Identity Edit run with
   a real negative prompt (e.g. reference README's `big wings` /
   `brown feathers` cases) — confirm the negative concept visibly
   suppressed, source identity preserved on the edit case, and that
   `nag_negative` absent reproduces today's output bit-for-bit (regression
   guard for `__call__`/`predict` being untouched).

## Out of scope (this design)

- Multi-reference Identity Edit NAG (`source_latent_b` in the reference) —
  ASDX Identity Edit is currently single-reference only (see canon `Kontext
  ref_latents implements only the single-reference case` for the same
  precedent on a different feature).
- A generic model-patch/wrapper mechanism for ASDX — out of scope beyond
  what NAG itself needs; if a second guidance mechanism appears later, that
  refactor should be considered on its own merits then, not spec'd
  speculatively here.
