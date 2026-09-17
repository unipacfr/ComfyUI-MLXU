"""
LoRA Runtime Loading
====================
Load and apply LoRA adapters to FLUX transformers at runtime.

Supports:
  - Standard LoRA (A/B matrices)
  - ComfyUI diff format (.diff, .diff_b)
  - Per-LoRA alpha scaling (alpha / rank)
  - Multiple LoRA stacking with individual weights
  - LoRA baking (fuse into base weights for speed)
"""

from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import torch

from comfy_api.latest import io

from . import bridge
from .native import FluxTransformer
from .native.flux2 import Flux2Transformer
from .native.krea2 import SingleStreamDiT
from .native.minimax_h3.model import MiniMaxH3Model
from .native.zimage import NextDiT
from .native.safetensors_header import read_safetensors_header
from .native.sdxl import UNetModel as SDXLUNetModel


# ── LoRA↔model family compatibility ─────────────────────────────────

@dataclass(frozen=True)
class LoRAFamilySignature:
    family: str  # a CapabilityProfile.family value, or "unknown"
    evidence: tuple[str, ...]


# Base architecture a CapabilityProfile.family value is built on, for
# compatibility comparison -- flux1_fill/flux1_depth are FLUX.1 variants that
# share the exact double_blocks/single_blocks key namespace a plain "flux1"
# LoRA targets, so they must not be flagged as a mismatch.
_LORA_COMPATIBLE_BASE: dict[str, str] = {
    "flux1": "flux1",
    "flux1_fill": "flux1",
    "flux1_depth": "flux1",
    "flux2": "flux2",
    "krea2": "krea2",
    "sdxl": "sdxl",
    "zimage": "zimage",
}

# Conservative, substring-based signatures over RAW (unstripped) safetensors
# header keys -- a real LoRA file's keys still carry whatever prefix
# (diffusion_model./model./transformer.) `_load_lora_file` strips later, but
# substring matching doesn't care about a leading prefix.
#
# Verified against real LoRA files on this machine for flux1 (BFL-native),
# flux2, krea2, sdxl (kohya-ss, the dominant convention for
# Illustrious/Pony/NoobAI in this project's library) and zimage. A
# diffusers/PEFT-style FLUX/Flux2 LoRA (`transformer_blocks.{i}.attn.to_q...`,
# `single_transformer_blocks.{i}...`) returns "unknown" here rather than a
# false match -- `_apply_lora_to_transformer` resolves that naming separately
# via `_resolve_flux_diffusers_lora`/`_resolve_flux2_diffusers_lora`, so this
# check only needs to not misclassify it, not parse it.
#
# `.double_blocks.` alone (not `.single_blocks.`) is the flux1 signal here,
# but it is NOT conclusive by itself: Flux.2/Klein reuses FLUX.1's exact
# `double_blocks`/`single_blocks`/`img_attn`/`linear1`/`linear2` naming (both
# share the same `comfy.ldm.flux.model.Flux` class), so a genuine Flux.2 LoRA
# also carries `.double_blocks.` keys (confirmed on a real Civitai Flux.2
# Klein 9B LoRA). Flux.2's own `double_stream_modulation_img./_txt.` marker
# is used as a positive signal when present, but real-world LoRA files
# rarely train the shared top-level Modulation layers, so it's usually
# absent -- `_check_lora_compatibility` cross-checks actual block COUNTS
# against the loaded model's config before refusing a flux1/flux2 mismatch
# found here, to catch this ambiguity.
def detect_lora_family(path: Path) -> LoRAFamilySignature:
    """Header-only (no tensor data) detection of which base family a LoRA
    file targets, from distinctive key fragments. Zero or multiple matching
    signatures both return family="unknown" -- never a forced guess; callers
    should only refuse on a confident, single, DISAGREEING match, not on
    "unknown" (see `_check_lora_compatibility`).
    """
    header = read_safetensors_header(path)
    keys = header.tensors.keys()

    matches: dict[str, str] = {}  # family -> the key fragment that matched
    if any(".double_blocks." in k for k in keys):
        matches["flux1"] = "double_blocks."
    if any(".input_blocks." in k for k in keys) or any("_input_blocks_" in k for k in keys):
        matches["sdxl"] = "input_blocks."
    if any(".noise_refiner." in k for k in keys) or any(
        ".layers." in k and ".attention.to_" in k for k in keys
    ):
        matches["zimage"] = "noise_refiner./layers.*.attention.to_*"
    if any(".double_stream_modulation_img." in k or ".double_stream_modulation_txt." in k for k in keys):
        matches["flux2"] = "double_stream_modulation_img./_txt."
    # Krea2's SingleStreamDiT names attention projections attn.wq/wk/wv/wo
    # directly under blocks.{i} -- distinct from FLUX's fused img_attn.qkv/
    # txt_attn.qkv, so this never collides with the flux1 signature above.
    if any(".attn.wq." in k or ".attn.wk." in k or ".attn.wv." in k for k in keys):
        matches["krea2"] = "attn.wq/wk/wv"

    if len(matches) != 1:
        return LoRAFamilySignature(family="unknown", evidence=tuple(sorted(matches)))
    family, evidence = next(iter(matches.items()))
    return LoRAFamilySignature(family=family, evidence=(evidence,))


def _lora_block_counts(path: Path) -> tuple[int, int]:
    """Count of distinct double_blocks./single_blocks. indices present in a
    LoRA file's header (BFL-native naming). Header-only, no tensor data read.
    """
    header = read_safetensors_header(path)
    double_idx = {int(m.group(1)) for k in header.tensors for m in [re.search(r"double_blocks\.(\d+)\.", k)] if m}
    single_idx = {int(m.group(1)) for k in header.tensors for m in [re.search(r"single_blocks\.(\d+)\.", k)] if m}
    return len(double_idx), len(single_idx)


def _check_lora_compatibility(lora_path: Path, model: dict) -> None:
    """Refuse to apply a LoRA whose detected family disagrees with the
    loaded model's family -- silently applying deltas keyed by coincidental
    name overlap (or applying 0 deltas with only a quiet log line) is the
    failure mode this guards against. An "unknown" detection never blocks:
    it means "no confident signature found", not "confirmed mismatch".
    """
    signature = detect_lora_family(lora_path)
    if signature.family == "unknown":
        return
    model_family = model["capability"].family
    model_base = _LORA_COMPATIBLE_BASE.get(model_family, model_family)
    lora_base = _LORA_COMPATIBLE_BASE.get(signature.family, signature.family)
    if lora_base != model_base:
        # flux1 and flux2/Klein share the exact `comfy.ldm.flux.model.Flux`
        # class and its double_blocks/single_blocks/img_attn/linear1/linear2
        # naming -- only block COUNT and hidden_size differ (Klein 9B: 8
        # double + 24 single vs FLUX.1-dev: 19 double + 38 single), so a
        # "flux1 evidence: double_blocks." detection can be a false positive
        # on a genuine Flux2 LoRA. Cross-check the LoRA's actual block counts
        # against the loaded model's real config before trusting the naming
        # alone (confirmed false positive on a real Civitai-labeled Flux.2
        # Klein 9B LoRA: 8 double + 24 single blocks in the file, exactly
        # matching the loaded Klein checkpoint).
        if {lora_base, model_base} == {"flux1", "flux2"}:
            lora_double, lora_single = _lora_block_counts(lora_path)
            config = model["config"]
            if (
                lora_double == getattr(config, "num_double_blocks", -1)
                and lora_single == getattr(config, "num_single_blocks", -1)
            ):
                return
        raise ValueError(
            f"ASDX: '{lora_path.name}' looks like a {signature.family} LoRA "
            f"(evidence: {', '.join(signature.evidence)}), but the loaded "
            f"model is {model_family}. Applying it would silently apply "
            f"deltas to unrelated layers. Refusing."
        )


# ── LoRA Target Definition ────────────────────────────────────────────

@dataclass(frozen=True)
class LoRATarget:
    """Defines which transformer layer a LoRA weight key maps to."""
    # Path through the transformer module tree
    path_parts: tuple[str, ...]
    # Weight key suffixes in the LoRA file
    key_suffixes: tuple[str, ...]  # e.g. ("lora_A.weight", "lora_B.weight")


# FLUX.1 LoRA target patterns
_FLUX_LORA_TARGETS: tuple[LoRATarget, ...] = (
    # Double block attention
    LoRATarget(("double_blocks", "{i}", "img_attn"), ("qkv.weight",)),
    LoRATarget(("double_blocks", "{i}", "img_attn"), ("proj.weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_attn"), ("qkv.weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_attn"), ("proj.weight",)),
    # Double block MLP
    LoRATarget(("double_blocks", "{i}", "img_mlp_0"), ("weight",)),
    LoRATarget(("double_blocks", "{i}", "img_mlp_2"), ("weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_mlp_0"), ("weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_mlp_2"), ("weight",)),
    # Single block attention
    LoRATarget(("single_blocks", "{i}", "attn"), ("qkv.weight",)),
    LoRATarget(("single_blocks", "{i}", "attn"), ("proj.weight",)),
    # Single block MLP
    LoRATarget(("single_blocks", "{i}", "mlp_0"), ("weight",)),
    LoRATarget(("single_blocks", "{i}", "mlp_2"), ("weight",)),
)

# Krea2 (SingleStreamDiT) LoRA target patterns
# Keys in checkpoint: diffusion_model.blocks.{i}.attn.wq.lora_A.weight
# After stripping prefix: blocks.{i}.attn.wq
_KREA2_LORA_TARGETS: tuple[LoRATarget, ...] = (
    # Attention projections
    LoRATarget(("blocks", "{i}", "attn", "wq"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "wk"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "wv"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "wo"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "gate_proj"), ("weight",)),
    # MLP projections
    LoRATarget(("blocks", "{i}", "mlp", "up"), ("weight",)),
    LoRATarget(("blocks", "{i}", "mlp", "gate"), ("weight",)),
    LoRATarget(("blocks", "{i}", "mlp", "down"), ("weight",)),
)


# ── LoRA Adapter ─────────────────────────────────────────────────────

@dataclass
class LoRAAdapter:
    """A single LoRA adapter with its weights and scale."""
    name: str
    # Map from (block_index, layer_type, param) -> delta weight. Only holds
    # already-final deltas (ComfyUI .diff/.diff_b format, or the rare
    # single-sided A-only/B-only case) -- everything else lives in `factors`
    # below and is never materialized here.
    deltas: dict[tuple[int, str, str], mx.array] = field(default_factory=dict)
    # Raw (A, B) low-rank factor pairs, keyed the same way as `deltas`. Kept
    # UNMATERIALIZED (rank*in + out*rank elements, not out*in) until a
    # specific target is actually consumed by _materialize_delta -- see that
    # function's docstring for why (duplication source #1 in the canon).
    factors: dict[tuple[int, str, str], tuple[mx.array, mx.array]] = field(default_factory=dict)
    # LoKr (LyCORIS Kronecker) factor pairs, keyed the same way. Kept
    # UNMATERIALIZED for the same reason as `factors`, and even more so: a
    # kron delta is `w1.size * w2.size` elements, e.g. [4,4] x [1536,1536] ->
    # [6144,6144], a 16x blow-up. Materializing all of them at load cost 23GB
    # of deltas plus ~47GB of float32 intermediates on a real file, which
    # pushed a Krea2 generation to a 118.6GB peak on a 64GB machine.
    lokr_factors: dict[str, tuple[mx.array, mx.array]] = field(default_factory=dict)
    # LoHa (LyCORIS Hadamard) factors: (w1_a, w1_b, w2_a, w2_b, scale). Unlike
    # LoKr there is no structured deferred form -- the elementwise product of
    # two low-rank products is inherently full-size -- so these materialize on
    # demand like any other delta, one target at a time.
    loha_factors: dict[str, tuple[mx.array, mx.array, mx.array, mx.array, float]] = field(default_factory=dict)
    # None means the file has no ".alpha" key (see _load_lora_file) -- not
    # the same as alpha=1.0, the two fall back to different scales below.
    alpha: float | None = None
    rank: int = 0
    scale: float = 1.0

    def __post_init__(self):
        if self.rank == 0:
            # Infer rank from first delta
            if self.deltas:
                self.rank = next(iter(self.deltas.values())).shape[-1]
        if self.scale == 0:
            self.scale = base_lora_scale(self.alpha, self.rank)


def _normalize_native_lora_key(prefix: str) -> str:
    """Rename a stripped native BFL LoRA key prefix to match this project's
    live module tree, undoing the same checkpoint-convention renames
    `native/weight_map.py` applies at checkpoint-load time (see its own
    docstring) -- a LoRA trained against the real checkpoint's dotted
    `img_mlp.0`/`img_mlp.2` Sequential naming otherwise never matches the
    native module's flat `img_mlp_0`/`img_mlp_2` attributes, silently
    dropping every double-block MLP delta (confirmed: a real Flux.2 Klein
    LoRA lost exactly its 4 MLP keys/block, 32/112 deltas, this way).
    """
    if ".attn.gate." in prefix:
        prefix = prefix.replace(".attn.gate.", ".attn.gate_proj.")
    if ".img_mlp." in prefix:
        prefix = prefix.replace(".img_mlp.", ".img_mlp_")
    elif ".txt_mlp." in prefix:
        prefix = prefix.replace(".txt_mlp.", ".txt_mlp_")

    return prefix


def base_lora_scale(alpha: float | None, rank: int) -> float:
    """`alpha/rank` if the file declared its own alpha, else a flat 1.0 --
    matches comfy/weight_adapter/lora.py's fallback exactly (`alpha =
    v[2]/mat2.shape[0] if v[2] is not None else 1.0`). Do NOT default missing
    alpha to `1.0/rank` -- that under-applies the LoRA for any file that
    simply doesn't ship an alpha key.
    """
    return alpha / max(rank, 1) if alpha is not None else 1.0


def _delta_from_factors(a: mx.array, b: mx.array) -> mx.array:
    """Compute the full-size `[out, in]` delta from a raw low-rank `(A, B)`
    factor pair -- `lora_A`/`lora_B` (or kohya `lora_down`/`lora_up`)
    convention: A is `[rank, in_features]`, B is `[out_features, rank]`.

    Split out of `_load_lora_file`'s old eager conversion loop so the same
    logic can run lazily, once per consumed target, from `_materialize_delta`
    below (see its docstring for why this must not run for every target up
    front).
    """
    if a.ndim == 4:
        # Conv-style (LyCORIS/LoCon) LoRA: A keeps the full conv kernel
        # [rank, in, kh, kw], B is the 1x1 channel-mixing projection
        # [out, rank, 1, 1] -- a plain 2D matmul can't contract these
        # directly (their trailing dims don't align), so flatten A's
        # spatial extent into the matmul's contraction dim and reshape back
        # to the conv weight shape afterwards.
        rank, in_ch, kh, kw = a.shape
        out_ch = b.shape[0]
        a2d = a.reshape(rank, in_ch * kh * kw).astype(mx.float32)
        b2d = b.reshape(out_ch, rank).astype(mx.float32)
        delta = (b2d @ a2d).reshape(out_ch, in_ch, kh, kw)
        # The checkpoint (and this raw LoRA tensor) is PyTorch
        # [out, in, kh, kw], but every native MLX Conv2d parameter this
        # delta gets added to is [out, kh, kw, in] (see native/sdxl/model.py's
        # checkpoint-load transpose) -- without this, the delta silently has
        # the wrong shape for the target parameter it's summed into.
        return delta.transpose(0, 2, 3, 1).astype(b.dtype)
    # Standard LoRA: delta = B @ A
    return (b.astype(mx.float32) @ a.astype(mx.float32)).astype(b.dtype)


def _materialize_delta(key: str, lora: "LoRAAdapter") -> mx.array | None:
    """Resolve a single LoRA target key to its full-size `[out, in]` delta,
    computed on demand from `lora.factors`' tiny low-rank pair rather than
    pre-materialized for every target in the file at load time.

    `_load_lora_file` used to compute `delta = B @ A` for EVERY target up
    front, so the full set of deltas (near-full-size relative to every
    touched base weight, for a densely-targeted LoRA) stayed alive for the
    entire merge in `_apply_lora_to_transformer` on top of the whole-model
    rebuild that merge also builds -- one of two independent duplication
    sources behind the OOM crashes this was written to fix (see the canon).
    Calling this per-target, right where `_apply_lora_to_transformer`'s
    existing chunked eval/`clear_cache` loop already consumes the result,
    means at most one full-size delta is transiently alive per call instead
    of the whole file's worth concurrently.
    """
    delta = lora.deltas.get(key)
    if delta is not None:
        return delta
    pair = lora.factors.get(key)
    if pair is not None:
        return _delta_from_factors(*pair)
    lokr = lora.lokr_factors.get(key)
    if lokr is not None:
        return _delta_from_lokr(*lokr)
    loha = lora.loha_factors.get(key)
    if loha is not None:
        return _delta_from_loha(*loha)
    return None


def _kron_matmul(x: mx.array, w1: mx.array, w2: mx.array) -> mx.array:
    """`x @ kron(w1, w2).T` WITHOUT ever building `kron(w1, w2)`.

    With `K[i1*r2+i2, j1*c2+j2] = w1[i1,j1] * w2[i2,j2]`, the product
    factorises: reshape `x`'s input axis to `[c1, c2]`, contract `c2` with
    `w2`, then `c1` with `w1`. Verified against an explicit
    `x @ kron(w1,w2).T` at the real Krea2 dims -- agreement is 1.4e-03
    relative, which is float32 accumulation-order noise over 6144 summed
    terms, not a formula difference (the same comparison in float64 closes
    to 6e-07).

    This is what makes LoKr affordable as a residual: the explicit matrix for
    ONE Krea2 target is 0.141GB, and this file's 256 targets cover 45.3GB of
    the model's 47.8GB of weights -- merging them all cost a permanent
    +46.6GB (measured), versus 1.46GB to keep the factors.
    """
    r1, c1 = w1.shape
    r2, c2 = w2.shape
    lead = x.shape[:-1]
    xr = x.reshape(*lead, c1, c2)
    t = (xr @ w2.T)                        # contract c2 -> [..., c1, r2]
    t = mx.swapaxes(t, -1, -2)             # [..., r2, c1]
    t = t @ w1.T                           # contract c1 -> [..., r2, r1]
    t = mx.swapaxes(t, -1, -2)             # [..., r1, r2]
    return t.reshape(*lead, r1 * r2)


def _as_mx(value: Any) -> mx.array:
    """Raw safetensors entry -> `mx.array`, without copying an existing one."""
    return value if isinstance(value, mx.array) else mx.array(value)


def _strip_and_normalize_key(key: str) -> str:
    """Drop a ComfyUI wrapper prefix, then apply the native module renames.

    Same two steps every other branch of `_load_lora_file` performs inline;
    factored out here so the LyCORIS branches cannot drift from them.
    """
    for pfx in ("diffusion_model.", "model.", "transformer."):
        if key.startswith(pfx):
            key = key[len(pfx):]
            break
    return _normalize_native_lora_key(key)


def _alpha_of(stem: str, raw: dict) -> float | None:
    """This module's own `<stem>.alpha`, or None. LyCORIS writes alpha PER
    MODULE, so a shared file-level value is the wrong granularity here."""
    entry = raw.get(f"{stem}.alpha")
    if entry is None:
        return None
    try:
        return float(entry)
    except (TypeError, ValueError):
        return None


def _lokr_factor_pair(
    stem: str, raw: dict
) -> tuple[mx.array, mx.array, int | None] | None:
    """Resolve a LoKr module's two Kronecker factors, rebuilding either from
    its low-rank `_a @ _b` pair when it is not stored full.

    Returns `(w1, w2, rank)` where `rank` is None only when BOTH factors are
    full -- the LyCORIS both-full case whose scale is forced to 1.0 (see
    `_lycoris_scale`). Returns None for a variant this cannot represent
    (missing factor, tucker/CP `lokr_t2`, or a non-2-D factor), so the caller
    reports it instead of guessing.

    Rebuilding a low-rank factor is bounded by the FACTOR dims, not by
    `out x in`, so the result stays small and the delta stays deferrable --
    the reasoning mlx-gen's `build_lokr_factors` documents.
    """
    if f"{stem}.lokr_t2" in raw:
        return None                      # tucker/CP: conv-only, no 2-D form
    rank: int | None = None
    resolved: list[mx.array] = []
    for idx in ("1", "2"):
        full = raw.get(f"{stem}.lokr_w{idx}")
        if full is not None:
            arr = _as_mx(full)
        else:
            a = raw.get(f"{stem}.lokr_w{idx}_a")
            b = raw.get(f"{stem}.lokr_w{idx}_b")
            if a is None or b is None:
                return None
            a_arr, b_arr = _as_mx(a), _as_mx(b)
            if a_arr.ndim != 2 or b_arr.ndim != 2:
                return None
            arr = a_arr @ b_arr
            rank = a_arr.shape[1]
        if arr.ndim != 2:
            return None
        resolved.append(arr)
    return resolved[0], resolved[1], rank


def _lycoris_scale(alpha: float | None, rank: int | None) -> float:
    """LyCORIS per-module scale: `alpha / rank`, or 1.0 when there is no rank.

    Ported from mlx-gen (`adapters/loader.rs`, `fn scale`), which in turn
    mirrors LyCORIS `LokrModule.__init__` (`if use_w1 and use_w2: alpha =
    lora_dim`): a LoKr whose BOTH Kronecker factors are stored full has no
    low-rank dimension, so its `alpha` is not a scale at all and the module
    applies at 1.0. A real file here confirms it -- its alpha is a sentinel
    (9999220736.0) that, used as a multiplier, overflowed every generation to
    a black image.

    `rank <= 0` or a non-finite alpha raises rather than silently producing
    `0/0 = NaN` baked into the delta -- mlx-gen documents that exact failure
    ("NaN-poisoning every subsequent render while the load reports success"),
    which is the same silent-corruption class as the sentinel above.
    """
    if rank is None:
        return 1.0
    if rank <= 0:
        raise RuntimeError(
            f"LyCORIS adapter: invalid rank {rank} -- must be > 0. A rank of 0 "
            "would make the scale 0/0 = NaN and poison every weight it touches "
            "while the load still reported success."
        )
    if alpha is None:
        return 1.0
    if not math.isfinite(alpha):
        raise RuntimeError(
            f"LyCORIS adapter: non-finite alpha {alpha} -- refusing to scale by it."
        )
    return alpha / rank


def _delta_from_loha(
    w1a: mx.array, w1b: mx.array, w2a: mx.array, w2b: mx.array, scale: float
) -> mx.array:
    """LoHa (Hadamard-product) delta: `scale * ((w1_a @ w1_b) * (w2_a @ w2_b))`.

    Ported from mlx-gen's `reconstruct_loha_delta`, which mirrors LyCORIS
    `LohaModule.get_weight`. Unlike LoKr this is an ELEMENTWISE product of two
    low-rank products, so the result is inherently full-size `[out, in]` --
    there is no structured trick to defer it the way `_kron_matmul` defers a
    Kronecker product. It is therefore returned as a delta and merged once.

    The tucker/CP variant (`hada_t1`/`hada_t2`, conv-only in LyCORIS) is NOT
    handled here; callers detect and report it rather than guessing.
    """
    return (((w1a @ w1b) * (w2a @ w2b)) * scale)


def _delta_from_lokr(w1: mx.array, w2: mx.array) -> mx.array:
    """Build a LoKr target's full `[out, in]` delta as `kron(w1, w2)`.

    Math ported from comfy's reference (`comfy/weight_adapter/lokr.py::
    LokrDiff.__call__`) and verified bit-exact against it on real weights.
    Called lazily, per target, from `_materialize_delta`: the result is 16x
    the size of the factors it is built from ([4,4] x [1536,1536] ->
    [6144,6144]), so materializing every target up front is exactly the
    duplication the canon's "LoRA merge OOM has two independent duplication
    sources" record warns about -- it produced a 118.6GB peak on a real 64GB
    machine before this was made lazy.

    Kept in the factors' own dtype rather than promoted to float32: the
    product is a single elementwise multiply with no accumulation, so there
    is nothing for the wider type to protect, and float32 here doubles the
    transient cost of the largest array in the whole apply path.
    """
    r1, c1 = w1.shape
    r2, c2 = w2.shape
    return (w1.reshape(r1, 1, c1, 1) * w2.reshape(1, r2, 1, c2)).reshape(r1 * r2, c1 * c2)


# ── Forward-time-residual LoRA (Phase 1: Krea2 only) ────────────────────
#
# See canon "LoRA target architecture is mlx-gen's forward-time residual
# (AdaptableLinear), ported in phases; Phase 0 done" -- this is Phase 1.
# Unlike `_apply_lora_to_transformer`'s merge below (which computes a
# full-size `delta = B @ A` per target via `_materialize_delta` and rebuilds
# the ENTIRE parameter tree via `type(transformer)(config)` +
# `tree_unflatten`), this attaches each target's raw `(A, B)` factor pair
# (or, rarely, a full diff-format delta) directly to the one Linear it
# targets and evaluates the residual `x @ A.T @ B.T` at forward time --
# never materializing a full-size `[out, in]` array and never rebuilding
# anything but the object-graph path from the transformer root down to each
# touched Linear (`copy.copy` on an MLX `Module`, a `dict` subclass, is an
# O(1) shallow copy per level -- see mlx.nn.layers.base.Module). Krea2 is
# the Phase 1 target because its `SingleStreamDiT` has no fused qkv/linear1
# weights (unlike FLUX.1/Flux.2's `double_blocks.*.img_attn.qkv` etc.), so
# every raw file key maps to exactly one Linear with no multi-adapter slice
# assembly needed -- see Phase 2 in the canon for the fused case.

class AdaptableLinear(nn.Linear):
    """`nn.Linear` with zero or more LoRA contributions applied as a
    forward-time residual instead of merged into `.weight`/`.bias`.

    `_lora_factors` is a plain python list with a leading underscore --
    MLX's `Module.valid_parameter_filter` excludes any dict/list attribute
    whose key starts with `_` from `.parameters()`, so it never shows up in
    `tree_flatten(transformer.parameters())`, `mx.eval(transformer.
    parameters())`, or a checkpoint save -- it is per-application scratch,
    not a model weight.

    Only genuine low-rank `(A, B)` pairs live here as a forward-time
    residual (`base(x) + scale * (x @ A.T @ B.T)`) -- cheap in both memory
    (rank*dim, not out*in) and per-forward compute. A target that's already
    a full `[out, in]` delta (ComfyUI `.diff`/`.diff_b`, or a diffusers/
    PEFT LoRA's multi-component-assembled delta) is merged directly into
    `.weight` ONCE via `merge_delta` instead -- see that method's docstring
    for why (confirmed via a real generation log: holding a full-size delta
    as a forever-recomputed residual instead roughly doubled both peak
    memory and per-step latency, worsening across repeated cache-hit reuses
    of the same model).
    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__(input_dims, output_dims, bias=bias)
        self._lora_factors: list[tuple[mx.array, mx.array, float]] = []
        self._lokr_factors: list[tuple[mx.array, mx.array, float]] = []

    def __call__(self, x: mx.array) -> mx.array:
        y = super().__call__(x)
        for a, b, scale in self._lora_factors:
            residual = (x.astype(a.dtype) @ a.T) @ b.T
            y = y + (scale * residual).astype(y.dtype)
        for w1, w2, scale in self._lokr_factors:
            y = y + (scale * _kron_matmul(x.astype(w1.dtype), w1, w2)).astype(y.dtype)
        return y

    def merge_delta(self, delta: mx.array, scale: float) -> None:
        """Merge a full-size `[out, in]` delta directly into `.weight`, a
        ONE-TIME cost -- unlike `_lora_factors`' residual, this is never
        held resident beyond the merge itself and never recomputed on
        subsequent forward calls, reproducing the old merge path's exact
        cost profile for a target that can't be represented as a cheap
        low-rank residual. `self` must already be a private, freshly-cloned
        leaf (see `_ensure_adaptable`) -- `weight = weight + x` never
        mutates the underlying buffer in place (MLX arrays are immutable
        value objects), it rebinds `self`'s own `.weight` entry to a new
        array, so this is safe even though the clone initially shares the
        same underlying array reference as the leaf it was cloned from.
        """
        self.weight = self.weight + (scale * delta).astype(self.weight.dtype)
        # Force this merge to completion before returning, then drop the
        # buffers it allocated. MLX is lazy: without this, N merges build one
        # giant unevaluated graph that keeps every intermediate AND every
        # superseded weight alive until the final eval -- measured as active
        # memory climbing 50.7GB -> 97.2GB across a 256-target LoKr attach,
        # which is what a real generation hit as a 118.6GB peak. The residual
        # path never needed this because a low-rank pair is tiny; a full-size
        # delta is not, so the merge path has to bound its own lifetime the
        # same way the old merge loop's chunked eval/clear_cache did.
        mx.eval(self.weight)
        mx.clear_cache()

    def merge_bias_delta(self, delta: mx.array, scale: float) -> bool:
        """Merge a `[out]` bias delta into `.bias`, same one-time-merge
        contract as `merge_delta` (a bias delta is inherently full-size --
        there is no low-rank form to keep as a residual).

        These come from ComfyUI's `.diff_b` entries, which `_load_lora_file`
        already stored under a `<module>.bias` key -- but every attach loop
        only ever probed `<module>.weight`, so they were loaded and then
        silently never applied.

        Returns False when the target Linear has no bias at all, so the
        caller can count it as unapplied instead of inventing one: adding a
        bias where the architecture has none would change the module's shape
        contract, not just its values.
        """
        if "bias" not in self:
            return False
        self.bias = self.bias + (scale * delta).astype(self.bias.dtype)
        return True

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "AdaptableLinear":
        """Wrap an existing plain `nn.Linear`'s `.weight`/`.bias` BY
        REFERENCE (no copy, no re-init) -- only the wrapper object is new,
        the underlying base-model weight data is never duplicated.
        """
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new._lora_factors = []
        new._lokr_factors = []
        new.weight = linear.weight
        if "bias" in linear:
            new.bias = linear.bias
        return new


def _lookup_lokr(lora: "LoRAAdapter", native_key: str):
    """LoKr factors for `native_key`, trying the kohya-flat spelling too.

    A real FLUX.1 LoKr on this machine (`lora.TA_trained.safetensors`) ships
    kohya-flat keys (`lora_unet_double_blocks_0_img_attn_qkv.lokr_w1`), so a
    dotted-only lookup resolved none of its 304 targets -- the same gap the
    plain-LoRA path already closes via `_lookup_native_or_kohya`.
    """
    hit = lora.lokr_factors.get(native_key)
    if hit is not None:
        return hit
    stem, _, suffix = native_key.rpartition(".")
    return lora.lokr_factors.get(f"lora_unet_{stem.replace('.', '_')}.{suffix}")


def _upsert_lokr_factor(leaf: AdaptableLinear, w1: mx.array, w2: mx.array, scale: float) -> None:
    """LoKr counterpart of `_upsert_lora_factor` -- same identity-based
    accumulation, so `ASDX_LoraSchedule`'s per-step re-application converges
    on the new scale instead of appending a duplicate residual every step.
    """
    for pos, (a, b, existing) in enumerate(leaf._lokr_factors):
        if a is w1 and b is w2:
            leaf._lokr_factors[pos] = (a, b, existing + scale)
            return
    leaf._lokr_factors.append((w1, w2, scale))


def _apply_bias_deltas(transformer: Any, lora: "LoRAAdapter", probe: Any) -> int:
    """Apply every `<module>.bias` delta in `lora.deltas` to its target.

    ComfyUI `.diff_b` entries are stored by `_load_lora_file` under a
    `<module>.bias` key, but every attach loop probes only `<module>.weight`,
    so these were loaded and then silently never applied.

    `probe(native_weight_key)` is the caller's own resolver -- it returns the
    already-wrapped `AdaptableLinear` for a native `.weight` key, or None.
    Reusing it means the bias pass inherits each family's exact target
    coverage (including its diffusers/kohya fallbacks) instead of duplicating
    that key logic here. Returns the number of bias deltas actually applied.
    """
    applied = 0
    for key, delta in lora.deltas.items():
        if not key.endswith(".bias"):
            continue
        leaf = probe(key[: -len(".bias")] + ".weight")
        if leaf is None:
            continue
        if leaf.merge_bias_delta(delta, lora.scale):
            applied += 1
    return applied


def _rescale_attached_lora(transformer: Any, lora: "LoRAAdapter") -> Any | None:
    """Fast path for `ASDX_LoraSchedule`: add `lora.scale` onto the adapters
    this exact LoRA already attached, mutating scales in place.

    `_update_lora_schedule` (`sampler/core.py`) re-invokes the whole attach
    path EVERY sampling step with `lora.scale` set to the step's incremental
    `delta_scale`. After the first call, that work is entirely redundant: the
    object graph is already cloned and private, every target already resolves
    to the same `AdaptableLinear`, and `_upsert_lora_factor` finds the same
    `(a, b)` arrays -- the only thing that actually changes is a float. The
    full path still cost ~81ms/step on a real Z-Image LoRA (measured), which
    is pure overhead once the LoRA is attached.

    Returns the same transformer with scales updated, or None if the fast
    path does not apply -- in which case the caller must run the full attach.
    It deliberately does NOT apply when:
      - nothing is attached yet (first call), or
      - the LoRA has any full-size `deltas`, since those were folded into
        `.weight` by `merge_delta` and leave no `(a, b)` entry to rescale;
        rescaling only the low-rank half would silently desynchronise the two.
    Both cases fall back to the existing, known-correct path.
    """
    if lora.deltas or not (lora.factors or lora.lokr_factors):
        return None

    factor_ids = {(id(a), id(b)) for a, b in lora.factors.values()}
    factor_ids |= {(id(a), id(b)) for a, b in lora.lokr_factors.values()}
    matched = 0
    pending: list[tuple[list, int, float]] = []
    for module in _iter_adaptable_leaves(transformer):
        for entries in (module._lora_factors, module._lokr_factors):
            for pos, (a, b, scale) in enumerate(entries):
                if (id(a), id(b)) in factor_ids:
                    pending.append((entries, pos, scale + lora.scale))
                    matched += 1
    if not matched:
        return None
    for entries, pos, new_scale in pending:
        a, b, _ = entries[pos]
        entries[pos] = (a, b, new_scale)
    return transformer


def _iter_adaptable_leaves(module: Any, _seen: set[int] | None = None):
    """Yield every `AdaptableLinear` reachable from `module`.

    Walks the module tree generically (MLX `Module` is a dict subclass, and
    `nn.Sequential` keeps its children in a plain `.layers` list), so this
    stays correct for all five families without per-family target tables --
    and, unlike those tables, cannot silently miss a leaf a future family adds.
    """
    if _seen is None:
        _seen = set()
    if id(module) in _seen:
        return
    _seen.add(id(module))
    if isinstance(module, AdaptableLinear):
        yield module
        return
    children: Any
    if isinstance(module, dict):
        children = module.values()
    elif isinstance(module, (list, tuple)):
        children = module
    else:
        return
    for child in children:
        if isinstance(child, (nn.Module, dict, list, tuple)):
            yield from _iter_adaptable_leaves(child, _seen)


def _upsert_lora_factor(leaf: AdaptableLinear, a: mx.array, b: mx.array, scale: float) -> None:
    """Attach `(a, b, scale)` to `leaf`, ADDING `scale` onto an existing
    entry that already uses the exact same `a`/`b` arrays (by identity)
    instead of appending a duplicate.

    This is what makes `ASDX_LoraSchedule`'s per-step re-application of the
    SAME LoRA both correct and cheap: `sampler/core.py::_update_lora_schedule`
    calls back into the attach path every step with `lora.scale` temporarily
    set to `delta_scale = new_scale - scale_prev` (not the absolute new
    scale -- see its own docstring), relying on the MERGE path's `value =
    value + delta*delta_scale` to make repeated calls converge on
    `orig_value + delta*new_scale` without compounding. `a`/`b` are the
    exact same array objects across every one of those calls (they live in
    `lora.factors`, never recreated), so adding `delta_scale` onto the
    existing entry's scale reproduces that identical accumulation for the
    residual path -- `scale_prev + (new_scale - scale_prev) == new_scale` --
    with ZERO changes needed in sampler/core.py. Without this, every
    sampling step would instead APPEND a new redundant `(a, b, delta_scale)`
    entry, making the forward pass iterate one more low-rank matmul per step
    per touched leaf forever (unbounded, O(steps) growth per leaf).

    A genuinely different adapter targeting the same leaf (different `a`/`b`
    identity -- e.g. `ASDX_MultiLoraLoader` chaining a second, distinct LoRA
    file) is unaffected and still appends its own separate entry.
    """
    for i, (existing_a, existing_b, existing_scale) in enumerate(leaf._lora_factors):
        if existing_a is a and existing_b is b:
            leaf._lora_factors[i] = (a, b, existing_scale + scale)
            return
    leaf._lora_factors.append((a, b, scale))


def _lookup_native_or_kohya(
    lora: "LoRAAdapter", native_key: str
) -> tuple[tuple[mx.array, mx.array] | None, mx.array | None]:
    """Look up `native_key` in `lora.factors`/`lora.deltas` directly, then
    fall back to the kohya-ss (sd-scripts) flat naming convention (dots
    replaced by underscores, `lora_unet_` prefix) that a large fraction of
    real community-trained FLUX/Krea2/Z-Image LoRAs (not just SD/SDXL)
    actually ship as -- confirmed against real Civitai FLUX.1 LoRAs whose
    raw keys are e.g. `lora_unet_double_blocks_0_img_attn_proj.lora_up.
    weight`, not the dotted native form. The candidate is built FROM the
    known real key, forward (native -> flat), exactly like the original
    merge path's identical fallback -- never reverse-engineered from the
    flat string, which is ambiguous whenever a leaf name itself contains an
    underscore (e.g. `gate_proj`, `img_mlp_0`). Returns `(factor_pair,
    None)`, `(None, delta)`, or `(None, None)` if neither form matches.
    """
    pair = lora.factors.get(native_key)
    if pair is not None:
        return pair, None
    delta = lora.deltas.get(native_key)
    if delta is not None:
        return None, delta
    # LoHa: an elementwise product of two low-rank products -- inherently
    # full-size, so it can only be consumed as a delta (built here, for this
    # one target, and merged once).
    if native_key in lora.loha_factors:
        return None, _delta_from_loha(*lora.loha_factors[native_key])
    stem, _, suffix = native_key.rpartition(".")
    kohya_key = f"lora_unet_{stem.replace('.', '_')}.{suffix}"
    pair = lora.factors.get(kohya_key)
    if pair is not None:
        return pair, None
    delta = lora.deltas.get(kohya_key)
    if delta is not None:
        return None, delta
    if kohya_key in lora.loha_factors:
        return None, _delta_from_loha(*lora.loha_factors[kohya_key])
    return None, None


_KREA2_RESIDUAL_TARGETS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("attn",), "wq"), (("attn",), "wk"), (("attn",), "wv"),
    (("attn",), "wo"), (("attn",), "gate_proj"),
    (("mlp",), "gate"), (("mlp",), "up"), (("mlp",), "down"),
)

# Leaf-name renames between a Krea2 HF-diffusers LoRA and this project's
# native module tree. Unlike FLUX.1's `to_q`/`to_k`/`to_v`, Krea2's attention
# keeps q/k/v as three SEPARATE native Linears (`wq`/`wk`/`wv`, see
# `native/krea2/model.py::Attention`), so every entry here is a pure 1:1
# rename -- no fused-weight assembly, and therefore no full-size delta
# materialization: `_resolve_krea2_diffusers_lora` can hand back the raw
# low-rank `(A, B)` pair and stay a true forward-time residual.
_KREA2_DIFFUSERS_LEAF_RENAME: dict[str, str] = {
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.gate_proj.weight": "attn.to_gate.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "mlp.down.weight": "ff.down.weight",
}

# Top-level (non-block-list) renames. `time_embed`/`txt_in`/`time_mod_proj`
# are deliberately absent: they map to `tmlp`/`txtmlp`/`tproj`, which are
# `nn.Sequential`-wrapped natively (`tmlp.layers.0.weight`), the same
# pre-existing Sequential key-mapping gap flagged in the canon -- resolving
# those needs a Sequential-aware mapper, not a rename entry.
_KREA2_DIFFUSERS_TOPLEVEL_RENAME: dict[str, str] = {
    "first.weight": "img_in.weight",
    "last.linear.weight": "final_layer.linear.weight",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
    # The three `nn.Sequential`s, addressed by native position. diffusers
    # names each Linear individually instead of by index, so `tmlp` 0/2 ->
    # time_embed.linear_1/2 and `txtmlp` 1/3 -> txt_in.linear_1/2 (txtmlp's
    # position 0 is an RMSNorm, hence the 1/3 offset), while `tproj`'s single
    # Linear at position 1 is a standalone `time_mod_proj`.
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "tproj.1.weight": "time_mod_proj.weight",
}

_KREA2_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
_KREA2_TXTFUSION_RE = re.compile(
    r"^txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$"
)


def _resolve_krea2_diffusers_lora(
    native_key: str, lora: "LoRAAdapter"
) -> tuple[tuple[mx.array, mx.array] | None, mx.array | None]:
    """Resolve a native Krea2 key against an HF-diffusers/PEFT-style LoRA.

    Roughly a third of this machine's real Krea2 library (9/35 files, e.g.
    `krea2_softwatercolor.safetensors` and its style siblings) ships genuine
    diffusers module names -- `transformer_blocks.N.attn.to_q`, `ff.down`,
    `text_fusion.*` -- which match neither the dotted-native nor the
    kohya-flat form `_lookup_native_or_kohya` already covers. Every one of
    those files resolved 0/264 targets and applied ZERO deltas, silently:
    the LoRA loaded, logged no error, and simply had no effect on the image.

    Returns the same `(factor_pair, delta)` shape as
    `_lookup_native_or_kohya` so callers can treat both uniformly. The pair
    is returned raw and unmaterialized -- see `_KREA2_DIFFUSERS_LEAF_RENAME`
    for why Krea2 needs no fused assembly.
    """
    m = _KREA2_BLOCK_RE.match(native_key)
    if m:
        leaf = _KREA2_DIFFUSERS_LEAF_RENAME.get(m.group(2))
        if leaf is None:
            return None, None
        candidate = f"transformer_blocks.{m.group(1)}.{leaf}"
    else:
        m = _KREA2_TXTFUSION_RE.match(native_key)
        if m:
            leaf = _KREA2_DIFFUSERS_LEAF_RENAME.get(m.group(3))
            if leaf is None:
                return None, None
            candidate = f"text_fusion.{m.group(1)}.{m.group(2)}.{leaf}"
        else:
            candidate = _KREA2_DIFFUSERS_TOPLEVEL_RENAME.get(native_key)
            if candidate is None:
                return None, None

    pair = lora.factors.get(candidate)
    if pair is not None:
        return pair, None
    return None, lora.deltas.get(candidate)


def _clone_block_path(
    container: list, touched: dict[int, Any], idx: int, path: tuple[str, ...]
) -> Any | None:
    """Shallow-clone `container[idx]` (cached in `touched` so repeated
    targets in the same block/call reuse the same clone) and each attribute
    named in `path` in turn -- e.g. `("attn",)` for Krea2, `("img_attn",)`
    for a FLUX double-block's image-side attention, or `()` for a leaf that
    sits directly on the block (FLUX's `single_blocks.*.linear1`). Returns
    the final module the leaf attribute should be looked up on, or None if
    any name in `path` doesn't exist.

    Each clone in the chain is made FROM the already-updated parent, never
    from the pristine original, so repeated calls for the same block/path
    within one attach call keep carrying forward every previously wrapped
    sibling leaf by reference instead of reverting it -- no second cache is
    needed for that.
    """
    block = touched.get(idx)
    if block is None:
        block = copy.copy(container[idx])
        container[idx] = block
        touched[idx] = block
    module = block
    for name in path:
        sub = getattr(module, name, None)
        if sub is None:
            return None
        sub = copy.copy(sub)
        setattr(module, name, sub)
        module = sub
    return module


def _ensure_adaptable(leaf: nn.Linear) -> AdaptableLinear:
    """Return an `AdaptableLinear` for `leaf`: wrap it fresh if it's still a
    plain `nn.Linear`, or clone it (and its adapter lists) if it's already
    an `AdaptableLinear` from an EARLIER, separate LoRA attach call (e.g.
    `ASDX_MultiLoraLoader` chaining several LoRAs) -- the transformer that
    earlier leaf lives on may still be referenced elsewhere, so clone before
    appending rather than mutating a possibly-shared object in place.
    """
    if isinstance(leaf, AdaptableLinear):
        clone = copy.copy(leaf)
        # BOTH adapter lists must be copied. `copy.copy` on an MLX Module (a
        # dict subclass) is shallow, so any list left uncopied stays SHARED
        # with the original -- a second LoRA attaching to the same leaf then
        # appends into the first one's list too, double-applying every
        # adapter. Missing `_lokr_factors` here produced black (NaN) images
        # for both ASDX_LoraSchedule and ASDX_MultiLoraLoader.
        clone._lora_factors = list(leaf._lora_factors)
        clone._lokr_factors = list(getattr(leaf, "_lokr_factors", []))
        return clone
    return AdaptableLinear.from_linear(leaf)


def _adapt_leaf(
    container: list, touched: dict[int, Any], idx: int, path: tuple[str, ...], leaf_name: str
) -> AdaptableLinear | None:
    """Resolve `container[idx].<path...>.<leaf_name>` to an
    `AdaptableLinear`, shallow-cloning the path down to it (see
    `_clone_block_path`) and wrapping it on first touch (see
    `_ensure_adaptable`). Returns None if the path/leaf don't actually exist
    (a regex match that isn't a real attribute).
    """
    module = _clone_block_path(container, touched, idx, path)
    if module is None:
        return None
    # An all-digit leaf addresses an `nn.Sequential` element: MLX keeps those
    # in a plain `.layers` LIST, so `getattr(module, "0")` is None and the leaf
    # has to be reached (and written back) by index instead.
    if leaf_name.isdigit():
        elements = getattr(module, "layers", None)
        if not isinstance(elements, list):
            return None
        pos = int(leaf_name)
        if pos >= len(elements) or not isinstance(elements[pos], nn.Linear):
            return None
        elements = list(elements)
        leaf = _ensure_adaptable(elements[pos])
        elements[pos] = leaf
        module.layers = elements
        return leaf
    leaf = getattr(module, leaf_name, None)
    if leaf is None:
        return None
    leaf = _ensure_adaptable(leaf)
    setattr(module, leaf_name, leaf)
    return leaf


def _apply_lora_residual_to_krea2(transformer: Any, lora: "LoRAAdapter") -> Any:
    """Krea2-only Phase 1 forward-time-residual LoRA attach -- see the
    module comment above. Returns a NEW transformer object (never mutates
    `transformer` in place, matching `_apply_lora_to_transformer`'s own
    contract) whose untouched blocks/layers are the exact same objects as
    the input's.
    """
    new_blocks = list(transformer.blocks)
    touched_blocks: dict[int, Any] = {}
    applied = 0
    total = (len(lora.factors) + len(lora.deltas) + len(lora.lokr_factors)
             + len(lora.loha_factors))

    def _lookup(native_key: str):
        """Native-dotted / kohya-flat first, then the HF-diffusers rename.
        Diffusers last because it is the narrowest convention -- a file that
        matches either of the first two is not a diffusers file at all.
        """
        pair, delta = _lookup_native_or_kohya(lora, native_key)
        if pair is not None or delta is not None:
            return pair, delta
        return _resolve_krea2_diffusers_lora(native_key, lora)

    for idx in range(len(new_blocks)):
        for path, leaf_name in _KREA2_RESIDUAL_TARGETS:
            native_key = f"blocks.{idx}." + "".join(f"{p}." for p in path) + f"{leaf_name}.weight"
            lokr = _lookup_lokr(lora, native_key)
            if lokr is not None:
                leaf = _adapt_leaf(new_blocks, touched_blocks, idx, path, leaf_name)
                if leaf is None:
                    continue
                _upsert_lokr_factor(leaf, lokr[0], lokr[1], lora.scale)
                applied += 1
                continue
            pair, delta = _lookup(native_key)
            if pair is None and delta is None:
                continue
            leaf = _adapt_leaf(new_blocks, touched_blocks, idx, path, leaf_name)
            if leaf is None:
                continue
            if pair is not None:
                a, b = pair
                _upsert_lora_factor(leaf, a, b, lora.scale)
            else:
                leaf.merge_delta(delta, lora.scale)
            applied += 1

    # `txtfusion` (TextFusionTransformer) is a nested submodule with its
    # OWN two block lists (`layerwise_blocks`/`refiner_blocks`, both
    # `TextFusionBlock` -- identical `.attn`/`.mlp` shape to the main
    # `SingleStreamBlock`, so `_KREA2_RESIDUAL_TARGETS` applies unchanged)
    # plus a standalone `projector` Linear. Confirmed against real LoRA
    # files that the MAJORITY of this project's character/clothing Krea2
    # LoRAs target ONLY `txtfusion.*` and never touch the main `blocks.*`
    # at all -- omitting this would silently apply zero deltas for most of
    # a real Krea2 LoRA library, not just an edge case.
    txtfusion_layerwise = list(transformer.txtfusion.layerwise_blocks)
    txtfusion_refiner = list(transformer.txtfusion.refiner_blocks)
    touched_layerwise: dict[int, Any] = {}
    touched_refiner: dict[int, Any] = {}
    for sub_list_name, container, touched_map in (
        ("layerwise_blocks", txtfusion_layerwise, touched_layerwise),
        ("refiner_blocks", txtfusion_refiner, touched_refiner),
    ):
        for idx in range(len(container)):
            for path, leaf_name in _KREA2_RESIDUAL_TARGETS:
                native_key = (f"txtfusion.{sub_list_name}.{idx}."
                              + "".join(f"{p}." for p in path) + f"{leaf_name}.weight")
                lokr = _lookup_lokr(lora, native_key)
                if lokr is not None:
                    leaf = _adapt_leaf(container, touched_map, idx, path, leaf_name)
                    if leaf is None:
                        continue
                    _upsert_lokr_factor(leaf, lokr[0], lokr[1], lora.scale)
                    applied += 1
                    continue
                pair, delta = _lookup(native_key)
                if pair is None and delta is None:
                    continue
                leaf = _adapt_leaf(container, touched_map, idx, path, leaf_name)
                if leaf is None:
                    continue
                if pair is not None:
                    a, b = pair
                    _upsert_lora_factor(leaf, a, b, lora.scale)
                else:
                    leaf.merge_delta(delta, lora.scale)
                applied += 1

    new_projector = None
    proj_pair, proj_delta = _lookup("txtfusion.projector.weight")
    if proj_pair is not None or proj_delta is not None:
        new_projector = _ensure_adaptable(transformer.txtfusion.projector)
        if proj_pair is not None:
            a, b = proj_pair
            _upsert_lora_factor(new_projector, a, b, lora.scale)
        else:
            new_projector.merge_delta(proj_delta, lora.scale)
        applied += 1

    # `first` (image-token input projection) and `last.linear` (output
    # projection) are standalone top-level Linears, same treatment.
    new_first = None
    first_pair, first_delta = _lookup("first.weight")
    if first_pair is not None or first_delta is not None:
        new_first = _ensure_adaptable(transformer.first)
        if first_pair is not None:
            a, b = first_pair
            _upsert_lora_factor(new_first, a, b, lora.scale)
        else:
            new_first.merge_delta(first_delta, lora.scale)
        applied += 1

    new_last = None
    last_pair, last_delta = _lookup("last.linear.weight")
    if last_pair is not None or last_delta is not None:
        new_last_linear = _ensure_adaptable(transformer.last.linear)
        if last_pair is not None:
            a, b = last_pair
            _upsert_lora_factor(new_last_linear, a, b, lora.scale)
        else:
            new_last_linear.merge_delta(last_delta, lora.scale)
        new_last = copy.copy(transformer.last)
        new_last.linear = new_last_linear
        applied += 1

    # `tmlp`/`txtmlp`/`tproj` are top-level `nn.Sequential`s. MLX keeps their
    # submodules in a plain `.layers` LIST, so the Linear at position i lives
    # at `<module>.layers.{i}` -- while the real checkpoint AND every real
    # LoRA file use the flat `<module>.{i}` form. `native/krea2/weight_map.py`
    # already bridges that at CHECKPOINT load; it was simply never applied to
    # LoRAs, so these targets silently matched nothing (see canon). Verified
    # against the real library that the file's indices already equal the
    # native Sequential positions (tmlp 0/2, txtmlp 1/3, tproj 1), so the
    # positions are read straight off the live module rather than hard-coded.
    new_sequentials: dict[str, Any] = {}
    for seq_name in ("tmlp", "txtmlp", "tproj"):
        seq = getattr(transformer, seq_name, None)
        elements = getattr(seq, "layers", None)
        if not isinstance(elements, list):
            continue
        updated = None
        for pos, element in enumerate(elements):
            if not isinstance(element, nn.Linear):
                continue
            pair, delta = _lookup(f"{seq_name}.{pos}.weight")
            if pair is None and delta is None:
                continue
            leaf = _ensure_adaptable(element)
            if pair is not None:
                a, b = pair
                _upsert_lora_factor(leaf, a, b, lora.scale)
            else:
                leaf.merge_delta(delta, lora.scale)
            if updated is None:
                updated = list(elements)
            updated[pos] = leaf
            applied += 1
        if updated is not None:
            new_seq = copy.copy(seq)
            new_seq.layers = updated
            new_sequentials[seq_name] = new_seq

    # Bias deltas (`.diff_b`) target the SAME leaves already wrapped above, so
    # resolve them against those rather than re-walking the key tables.
    wrapped: dict[str, AdaptableLinear] = {}
    for idx, block in touched_blocks.items():
        for path, leaf_name in _KREA2_RESIDUAL_TARGETS:
            mod = block
            for p in path:
                mod = getattr(mod, p, None)
                if mod is None:
                    break
            leaf = getattr(mod, leaf_name, None) if mod is not None else None
            if isinstance(leaf, AdaptableLinear):
                wrapped[f"blocks.{idx}." + "".join(f"{p}." for p in path) + f"{leaf_name}.weight"] = leaf
    for sub_name, touched_map in (("layerwise_blocks", touched_layerwise),
                                  ("refiner_blocks", touched_refiner)):
        for idx, block in touched_map.items():
            for path, leaf_name in _KREA2_RESIDUAL_TARGETS:
                mod = block
                for p in path:
                    mod = getattr(mod, p, None)
                    if mod is None:
                        break
                leaf = getattr(mod, leaf_name, None) if mod is not None else None
                if isinstance(leaf, AdaptableLinear):
                    wrapped[f"txtfusion.{sub_name}.{idx}." + "".join(f"{p}." for p in path)
                            + f"{leaf_name}.weight"] = leaf
    if new_first is not None:
        wrapped["first.weight"] = new_first
    if new_last is not None and isinstance(new_last.linear, AdaptableLinear):
        wrapped["last.linear.weight"] = new_last.linear
    if new_projector is not None:
        wrapped["txtfusion.projector.weight"] = new_projector
    for seq_name, new_seq in new_sequentials.items():
        for pos, element in enumerate(new_seq.layers):
            if isinstance(element, AdaptableLinear):
                wrapped[f"{seq_name}.{pos}.weight"] = element
    applied += _apply_bias_deltas(transformer, lora, wrapped.get)

    touched_txtfusion = bool(touched_layerwise) or bool(touched_refiner) or new_projector is not None
    if (not touched_blocks and not touched_txtfusion and new_first is None
            and new_last is None and not new_sequentials):
        print("[ASDX] LoRA: no matching weights found")
        return transformer

    new_transformer = copy.copy(transformer)
    # Record which LoRA file(s) this transformer carries, so a downstream node
    # (ASDX_Krea2Edit) can detect an already-attached Identity Edit LoRA and
    # refuse to stack a second copy instead of silently doubling it. The base
    # transformer in the loader cache is never LoRA-applied (non-destructive
    # apply), so this attribute only ever appears on per-generation copies and
    # is always fresh; `copy.copy` shares the input's list by reference, so
    # rebind to a new list to leave the input (and the cached base) untouched.
    prev_names = list(getattr(transformer, "_applied_lora_names", []))
    if lora.name not in prev_names:
        prev_names.append(lora.name)
    new_transformer._applied_lora_names = prev_names
    if touched_blocks:
        new_transformer.blocks = new_blocks
    if touched_txtfusion:
        new_txtfusion = copy.copy(transformer.txtfusion)
        if touched_layerwise:
            new_txtfusion.layerwise_blocks = txtfusion_layerwise
        if touched_refiner:
            new_txtfusion.refiner_blocks = txtfusion_refiner
        if new_projector is not None:
            new_txtfusion.projector = new_projector
        new_transformer.txtfusion = new_txtfusion
    if new_first is not None:
        new_transformer.first = new_first
    if new_last is not None:
        new_transformer.last = new_last
    for seq_name, new_seq in new_sequentials.items():
        setattr(new_transformer, seq_name, new_seq)
    print(f"[ASDX] LoRA (residual): attached {applied}/{total} adapters")
    return new_transformer


# ── Forward-time-residual LoRA (Phase 2: FLUX.1 / Flux.2) ───────────────
#
# See canon Phase 2. A BFL-native LoRA file targets a fused qkv/linear1
# weight as ONE tensor (same shape as the whole Linear it patches), so it's
# structurally identical to Krea2's Phase 1 case -- a pure low-rank residual,
# zero full-size materialization. A diffusers/PEFT-trained LoRA instead
# splits that same fused weight into separate per-component factor pairs
# (to_q/to_k/to_v[/proj_mlp]), each with its own independent rank -- these
# can't be represented as a single low-rank residual, so they're assembled
# into one full-width delta with the EXACT SAME math the merge path already
# uses (`_assemble_fused_delta`, reached here via the existing
# `_resolve_flux_diffusers_lora`/`_resolve_flux2_diffusers_lora`, unchanged
# since Phase 0 already made them pull from `lora` lazily) and attached as a
# single full-width residual delta instead. That still avoids the
# whole-model rebuild and never touches any weight but this one fused
# Linear -- it just can't skip materializing this one target the way a
# native-format LoRA's targets can.
#
# Unlike the merge path, this never walks the model's full flat parameter
# list -- only the fixed, small set of known fused/leaf target names per
# block is probed (10 per double block, 3 per single block), so cost is
# O(num_blocks), not O(whole model).
#
# `img_mod.lin`/`txt_mod.lin`/`modulation.lin` are included even though
# FLUX.1's OWN training code rarely targets them, because
# `_FLUX_DOUBLE_DIFFUSERS_RENAME`/`_FLUX_SINGLE_DIFFUSERS_RENAME` already
# prove they're real, confirmed diffusers-trained LoRA targets (mapped from
# `norm1.linear`/`norm1_context.linear`/`norm.linear`) -- omitting them here
# would silently drop those deltas exactly like the img_mlp bug the
# `_normalize_native_lora_key` docstring describes. Flux.2's blocks have no
# per-block modulation Linear at all (global_modulation=True, see
# `flux2/model.py`'s `DoubleBlock` docstring), so these three entries are
# harmless no-ops there (`_adapt_leaf` returns None for a path/leaf that
# doesn't exist on the block).

_FLUX_DOUBLE_RESIDUAL_TARGETS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("img_attn",), "qkv"), (("img_attn",), "proj"),
    (("txt_attn",), "qkv"), (("txt_attn",), "proj"),
    (("img_mod",), "lin"), (("txt_mod",), "lin"),
    ((), "img_mlp_0"), ((), "img_mlp_2"),
    ((), "txt_mlp_0"), ((), "txt_mlp_2"),
)
_FLUX_SINGLE_RESIDUAL_TARGETS: tuple[tuple[tuple[str, ...], str], ...] = (
    ((), "linear1"), ((), "linear2"), (("modulation",), "lin"),
)


def _apply_lora_residual_to_flux(transformer: Any, lora: "LoRAAdapter", config: Any, is_flux2: bool) -> Any:
    """FLUX.1/Flux.2-only Phase 2 forward-time-residual LoRA attach -- see
    the module comment above. Returns a NEW transformer object whose
    untouched blocks/layers are the exact same objects as the input's.
    """
    double_container = list(transformer.double_blocks)
    single_container = list(transformer.single_blocks)
    touched_double: dict[int, Any] = {}
    touched_single: dict[int, Any] = {}
    applied = 0
    total = (len(lora.factors) + len(lora.deltas) + len(lora.lokr_factors)
             + len(lora.loha_factors))

    hidden_dim = config.hidden_size if is_flux2 else config.hidden_dim
    mlp_dim = None if is_flux2 else config.mlp_dim

    def _attach(container: list, touched: dict[int, Any], idx: int, native_key: str, path: tuple[str, ...], leaf_name: str) -> int:
        # Raw low-rank pair for THIS exact fused target, native-dotted or
        # kohya-flat -- the common case (both are whole, un-split targets),
        # zero full-size materialization either way.
        pair, delta = _lookup_native_or_kohya(lora, native_key)
        if pair is not None:
            leaf = _adapt_leaf(container, touched, idx, path, leaf_name)
            if leaf is None:
                return 0
            a, b = pair
            _upsert_lora_factor(leaf, a, b, lora.scale)
            return 1
        # Rare diff-format/single-sided delta already stored whole, or a
        # diffusers/PEFT-trained LoRA's assembled multi-component delta.
        consumed = 1 if delta is not None else 0
        if delta is None:
            if is_flux2:
                delta, consumed = _resolve_flux2_diffusers_lora(native_key, lora, hidden_dim)
            else:
                delta, consumed = _resolve_flux_diffusers_lora(native_key, lora, hidden_dim, mlp_dim)
        if delta is None:
            return 0
        leaf = _adapt_leaf(container, touched, idx, path, leaf_name)
        if leaf is None:
            return 0
        leaf.merge_delta(delta, lora.scale)
        return consumed

    for idx in range(len(double_container)):
        for path, leaf_name in _FLUX_DOUBLE_RESIDUAL_TARGETS:
            native_key = f"double_blocks.{idx}." + "".join(f"{p}." for p in path) + f"{leaf_name}.weight"
            applied += _attach(double_container, touched_double, idx, native_key, path, leaf_name)

    for idx in range(len(single_container)):
        for path, leaf_name in _FLUX_SINGLE_RESIDUAL_TARGETS:
            native_key = f"single_blocks.{idx}." + "".join(f"{p}." for p in path) + f"{leaf_name}.weight"
            applied += _attach(single_container, touched_single, idx, native_key, path, leaf_name)

    # Top-level, non-block-list targets -- `img_in`/`txt_in`/`final_layer.
    # linear`/the MLP embedders/Flux.2's global modulation. Confirmed
    # against a real Flux.2 "Turbo" LoRA that targets exactly
    # `double_stream_modulation_img.lin`/`double_stream_modulation_txt.lin`/
    # `final_layer.linear` and nothing else in double_blocks/single_blocks
    # -- omitting these would silently apply zero deltas for that file.
    # `LastLayer.adaLN_modulation` and any Sequential-wrapped embedder
    # internals are NOT covered (same pre-existing `.layers.` Sequential
    # key-mapping gap as Krea2's `tmlp`/`tproj`/`txtmlp`, out of scope
    # here -- see canon).
    def _attach_standalone(native_key: str, current_leaf: Any) -> AdaptableLinear | None:
        if current_leaf is None:
            return None
        pair, delta = _lookup_native_or_kohya(lora, native_key)
        if pair is None and delta is None:
            return None
        leaf = _ensure_adaptable(current_leaf)
        if pair is not None:
            a, b = pair
            _upsert_lora_factor(leaf, a, b, lora.scale)
        else:
            leaf.merge_delta(delta, lora.scale)
        return leaf

    def _attach_embedder(name: str) -> Any | None:
        embedder = getattr(transformer, name, None)
        if embedder is None:
            return None
        new_in = _attach_standalone(f"{name}.in_layer.weight", getattr(embedder, "in_layer", None))
        new_out = _attach_standalone(f"{name}.out_layer.weight", getattr(embedder, "out_layer", None))
        if new_in is None and new_out is None:
            return None
        new_embedder = copy.copy(embedder)
        if new_in is not None:
            new_embedder.in_layer = new_in
        if new_out is not None:
            new_embedder.out_layer = new_out
        return new_embedder

    new_img_in = _attach_standalone("img_in.weight", getattr(transformer, "img_in", None))
    new_txt_in = _attach_standalone("txt_in.weight", getattr(transformer, "txt_in", None))
    new_time_in = _attach_embedder("time_in")
    new_vector_in = _attach_embedder("vector_in")      # FLUX.1 only
    new_guidance_in = _attach_embedder("guidance_in")  # None on non-distilled models

    new_final_layer = None
    final_layer = getattr(transformer, "final_layer", None)
    if final_layer is not None:
        new_final_linear = _attach_standalone("final_layer.linear.weight", getattr(final_layer, "linear", None))
        if new_final_linear is not None:
            new_final_layer = copy.copy(final_layer)
            new_final_layer.linear = new_final_linear

    new_mod_img = new_mod_txt = new_mod_single = None
    if is_flux2:
        mod_img = getattr(transformer, "double_stream_modulation_img", None)
        mod_txt = getattr(transformer, "double_stream_modulation_txt", None)
        mod_single = getattr(transformer, "single_stream_modulation", None)
        new_mod_img_lin = _attach_standalone(
            "double_stream_modulation_img.lin.weight", getattr(mod_img, "lin", None))
        if new_mod_img_lin is not None:
            new_mod_img = copy.copy(mod_img)
            new_mod_img.lin = new_mod_img_lin
        new_mod_txt_lin = _attach_standalone(
            "double_stream_modulation_txt.lin.weight", getattr(mod_txt, "lin", None))
        if new_mod_txt_lin is not None:
            new_mod_txt = copy.copy(mod_txt)
            new_mod_txt.lin = new_mod_txt_lin
        new_mod_single_lin = _attach_standalone(
            "single_stream_modulation.lin.weight", getattr(mod_single, "lin", None))
        if new_mod_single_lin is not None:
            new_mod_single = copy.copy(mod_single)
            new_mod_single.lin = new_mod_single_lin

    top_level = (new_img_in, new_txt_in, new_time_in, new_vector_in,
                 new_guidance_in, new_final_layer, new_mod_img, new_mod_txt, new_mod_single)
    applied += sum(1 for v in top_level if v is not None)

    if not touched_double and not touched_single and not any(top_level):
        print("[ASDX] LoRA: no matching weights found")
        return transformer

    new_transformer = copy.copy(transformer)
    if touched_double:
        new_transformer.double_blocks = double_container
    if touched_single:
        new_transformer.single_blocks = single_container
    if new_img_in is not None:
        new_transformer.img_in = new_img_in
    if new_txt_in is not None:
        new_transformer.txt_in = new_txt_in
    if new_time_in is not None:
        new_transformer.time_in = new_time_in
    if new_vector_in is not None:
        new_transformer.vector_in = new_vector_in
    if new_guidance_in is not None:
        new_transformer.guidance_in = new_guidance_in
    if new_final_layer is not None:
        new_transformer.final_layer = new_final_layer
    if new_mod_img is not None:
        new_transformer.double_stream_modulation_img = new_mod_img
    if new_mod_txt is not None:
        new_transformer.double_stream_modulation_txt = new_mod_txt
    if new_mod_single is not None:
        new_transformer.single_stream_modulation = new_mod_single
    print(f"[ASDX] LoRA (residual): attached {applied}/{total} adapters")
    return new_transformer


# ── Forward-time-residual LoRA (Phase 3: Z-Image) ────────────────────────
#
# See canon Phase 3. Z-Image's `NextDiT` has THREE independent block lists
# (`context_refiner`, `noise_refiner`, `layers`, all `JointTransformerBlock`)
# instead of Krea2's single `blocks` -- otherwise structurally identical to
# Phase 1: every real target (`attention.qkv`/`attention.out`/
# `feed_forward.w1`/`w2`/`w3`) is a whole, un-fused Linear, so a BFL/comfy-
# native LoRA file's raw key IS the target -- pure low-rank residual, zero
# full-size materialization. No diffusers/PEFT fused-weight fallback exists
# for Z-Image in this codebase (same as the merge path this replaces, which
# also has none -- see `detect_lora_family`'s zimage signature comment
# about `.attention.to_*` naming spotted in the wild but never resolved).

_ZIMAGE_RESIDUAL_TARGETS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("attention",), "qkv"), (("attention",), "out"),
    (("feed_forward",), "w1"), (("feed_forward",), "w2"), (("feed_forward",), "w3"),
    # `adaLN_modulation` is an nn.Sequential holding one Linear, so the leaf
    # lives at `.layers.0` -- see `_normalize_native_lora_key`, which rewrites
    # the flat `adaLN_modulation.0` form every real LoRA file ships into this
    # path. Present in `layers`/`noise_refiner` only; `context_refiner` blocks
    # are built with `modulation=False` and have no such attribute, which
    # `_adapt_leaf` already handles by returning None.
    (("adaLN_modulation",), "0"),
)

# Leaf renames between a Z-Image HF-diffusers LoRA and the native module tree.
# `attention.qkv` is deliberately absent: it is the one FUSED native weight and
# needs three separate diffusers deltas assembled, handled explicitly in
# `_resolve_zimage_diffusers_lora` rather than as a 1:1 rename.
_ZIMAGE_DIFFUSERS_LEAF_RENAME: dict[str, str] = {
    "attention.out.weight": "attention.to_out.0.weight",
    "feed_forward.w1.weight": "feed_forward.w1.weight",
    "feed_forward.w2.weight": "feed_forward.w2.weight",
    "feed_forward.w3.weight": "feed_forward.w3.weight",
}

_ZIMAGE_BLOCK_RE = re.compile(
    r"^(layers|context_refiner|noise_refiner)\.(\d+)\.(.+)$"
)


def _materialize_delta_native_or_kohya(key: str, lora: "LoRAAdapter") -> mx.array | None:
    """`_materialize_delta`, but also trying the kohya-flat spelling of `key`.

    Needed because real Z-Image LoRAs mix the two axes independently: naming
    convention (dotted vs. kohya-flat) and module vocabulary (native `qkv` vs.
    diffusers `to_q`/`to_k`/`to_v`). A real file on this machine
    (`AerithGainsborough_SoloLoRA_Zv1`) is kohya-flat AND diffusers-named
    (`lora_unet_layers_0_attention_to_q`), so it matches neither
    `_lookup_native_or_kohya` (wrong vocabulary) nor a plain diffusers lookup
    (wrong naming) -- it resolved 90/210 until this was added.
    """
    pair, delta = _lookup_native_or_kohya(lora, key)
    if delta is not None:
        return delta
    if pair is not None:
        return _delta_from_factors(*pair)
    return None


def _resolve_zimage_diffusers_lora(
    native_key: str, lora: "LoRAAdapter", qkv_widths: tuple[int, int, int] | None
) -> tuple[tuple[mx.array, mx.array] | None, mx.array | None, int]:
    """Resolve a native Z-Image key against an HF-diffusers/PEFT-style LoRA.

    Every real Z-Image LoRA on this machine (9/9 files) uses diffusers module
    names. Unlike Krea2, the loss here was PARTIAL and therefore far easier to
    miss: `feed_forward.w1/w2/w3` happen to be spelled identically in both
    conventions and always resolved, so each file applied ~90/240 targets --
    its MLP but none of its attention. The visible result is a LoRA that looks
    weak, not one that looks broken, which is why it was never reported.

    `attention.to_q`/`to_k`/`to_v` are three separate diffusers weights that
    must be concatenated into the native FUSED `attention.qkv` (see
    `native/zimage/model.py::JointAttention`, which splits qkv in q,k,v output
    order). That is the same situation FLUX.1's fused `qkv` is in, so it reuses
    `_assemble_fused_delta` and, like FLUX, must materialize one full-size
    delta for that target -- a fused weight cannot be represented as a single
    low-rank residual when each component carries its own rank.

    Returns `(pair, delta, consumed)`; `consumed` counts raw file entries
    absorbed, which is >1 for an assembled qkv.
    """
    m = _ZIMAGE_BLOCK_RE.match(native_key)
    if m is None:
        return None, None, 0
    container, idx, leaf = m.group(1), m.group(2), m.group(3)

    if leaf == "attention.qkv.weight":
        if qkv_widths is None:
            return None, None, 0
        pieces = [
            _materialize_delta_native_or_kohya(f"{container}.{idx}.attention.{name}.weight", lora)
            for name in ("to_q", "to_k", "to_v")
        ]
        delta, consumed = _assemble_fused_delta(pieces, list(qkv_widths))
        return None, delta, consumed

    renamed = _ZIMAGE_DIFFUSERS_LEAF_RENAME.get(leaf)
    if renamed is None:
        return None, None, 0
    pair, delta = _lookup_native_or_kohya(lora, f"{container}.{idx}.{renamed}")
    if pair is not None:
        return pair, None, 1
    return None, delta, (1 if delta is not None else 0)


def _apply_lora_residual_to_zimage(transformer: Any, lora: "LoRAAdapter") -> Any:
    """Z-Image-only Phase 3 forward-time-residual LoRA attach -- see the
    module comment above. Returns a NEW transformer object whose untouched
    blocks/layers are the exact same objects as the input's.
    """
    containers = {
        name: list(getattr(transformer, name))
        for name in ("context_refiner", "noise_refiner", "layers")
    }
    touched: dict[str, dict[int, Any]] = {name: {} for name in containers}
    applied = 0
    total = (len(lora.factors) + len(lora.deltas) + len(lora.lokr_factors)
             + len(lora.loha_factors))

    # q/k/v output widths of the native fused `attention.qkv`, needed to
    # assemble a diffusers LoRA's three separate deltas into it. Read off a
    # live JointAttention rather than hard-coded, so a GQA variant
    # (n_kv_heads < n_heads) stays correct -- the class is already written
    # to be generic over that even though Z-Image itself is full MHA.
    qkv_widths: tuple[int, int, int] | None = None
    for _c in containers.values():
        if _c:
            _attn = getattr(_c[0], "attention", None)
            if _attn is not None:
                qkv_widths = (
                    _attn.n_heads * _attn.head_dim,
                    _attn.n_kv_heads * _attn.head_dim,
                    _attn.n_kv_heads * _attn.head_dim,
                )
            break

    for container_name, container in containers.items():
        for idx in range(len(container)):
            for path, leaf_name in _ZIMAGE_RESIDUAL_TARGETS:
                native_key = f"{container_name}.{idx}." + "".join(f"{p}." for p in path) + f"{leaf_name}.weight"
                pair, delta = _lookup_native_or_kohya(lora, native_key)
                consumed = 1 if (pair is not None or delta is not None) else 0
                if consumed == 0:
                    # Native-dotted and kohya-flat both missed -- try the
                    # HF-diffusers convention every real Z-Image LoRA uses.
                    pair, delta, consumed = _resolve_zimage_diffusers_lora(
                        native_key, lora, qkv_widths
                    )
                if pair is None and delta is None:
                    continue
                leaf = _adapt_leaf(container, touched[container_name], idx, path, leaf_name)
                if leaf is None:
                    continue
                if pair is not None:
                    a, b = pair
                    _upsert_lora_factor(leaf, a, b, lora.scale)
                else:
                    leaf.merge_delta(delta, lora.scale)
                applied += consumed

    if not any(touched.values()):
        print("[ASDX] LoRA: no matching weights found")
        return transformer

    new_transformer = copy.copy(transformer)
    for name, touched_idx in touched.items():
        if touched_idx:
            setattr(new_transformer, name, containers[name])
    print(f"[ASDX] LoRA (residual): attached {applied}/{total} adapters")
    return new_transformer


def _apply_lora_to_minimax_h3(transformer: Any, lora: "LoRAAdapter", config: Any) -> Any:
    """Merge LoRA deltas into MiniMax H3's DiT, dequantizing/requantizing
    the four big per-block linears (`attn.qkv_proj`, `attn.out_proj`,
    `mlp.fc1`, `mlp.fc2`) that `load_minimax_h3_from_gguf` keeps in MLX's
    native quantized format (see `quantized_linear.py`'s module docstring
    for why -- the dense weight doesn't fit in 64GB).

    The generic merge loop below (`_apply_lora_to_transformer`'s fallthrough
    path) assumes every targeted parameter is a dense `[out, in]` array
    matching the LoRA delta's shape 1:1. For a quantized linear, `.weight`
    is instead a packed array whose last dimension is narrower by the bit
    width's pack factor (4-bit -> 8x narrower), so `value + delta * scale`
    broadcasts two differently-shaped matrices -- this is what produced the
    `[broadcast_shapes] (21504,672) and (21504,5376)` crash. Fixed by
    dequantizing the current packed weight, adding the dense delta, and
    requantizing with the module's own `group_size`/`bits` (read off the
    live module rather than assumed, in case a future loader call changes
    them) -- the same dequantize/operate/requantize shape this project
    already uses to build these weights in the first place
    (`weight_map.py::load_minimax_h3_from_gguf`).
    """
    from mlx.utils import tree_flatten, tree_unflatten

    from .native.minimax_h3.weight_map import _is_big_linear

    model_flat = dict(tree_flatten(transformer.parameters()))
    quantized_modules = {
        name: module for name, module in transformer.named_modules()
        if isinstance(module, nn.QuantizedLinear)
    }

    new_flat = dict(model_flat)
    applied = 0
    for stem, module in quantized_modules.items():
        weight_key = f"{stem}.weight"
        delta = _materialize_delta(weight_key, lora)
        if delta is None:
            continue
        dense = mx.dequantize(
            model_flat[weight_key], model_flat[f"{stem}.scales"], model_flat[f"{stem}.biases"],
            group_size=module.group_size, bits=module.bits,
        )
        merged_dense = dense.astype(mx.float32) + delta.astype(mx.float32) * lora.scale
        w_q, scales, biases = mx.quantize(merged_dense, group_size=module.group_size, bits=module.bits)
        new_flat[weight_key] = w_q
        new_flat[f"{stem}.scales"] = scales
        new_flat[f"{stem}.biases"] = biases
        applied += 1
        mx.eval(w_q, scales, biases)
        mx.clear_cache()

    for flat_key, value in model_flat.items():
        stem = flat_key[: -len(".weight")] if flat_key.endswith(".weight") else None
        if stem in quantized_modules or flat_key.endswith((".scales", ".biases")):
            continue  # handled above, or belongs to an already-handled quantized triple
        delta = _materialize_delta(flat_key, lora)
        if delta is not None:
            new_flat[flat_key] = value + delta.astype(value.dtype) * lora.scale
            applied += 1

    if applied == 0:
        print("[ASDX] LoRA: no matching weights found")
        return transformer

    # A fresh `type(transformer)(config)` is dense (plain `nn.Linear`), so its
    # module tree wouldn't have the `.scales`/`.biases` leaves `new_flat`
    # carries for the four quantized linears -- re-quantize the skeleton
    # with the SAME group_size/bits the source model actually used (read off
    # a live quantized module, not assumed) before `.update()`, exactly like
    # `load_minimax_h3_from_gguf` builds it.
    sample_module = next(iter(quantized_modules.values()))
    new_transformer = type(transformer)(config)
    nn.quantize(new_transformer, group_size=sample_module.group_size, bits=sample_module.bits,
                class_predicate=_is_big_linear)
    new_transformer.update(tree_unflatten(list(new_flat.items())))
    mx.eval(new_transformer.parameters())
    print(f"[ASDX] LoRA: applied {applied}/{len(lora.deltas) + len(lora.factors)} deltas (MiniMax H3)")
    return new_transformer


# ── MiniMax H3 turbo step-distill LoRA recipes ──────────────────────────
#
# lightx2v/Minimax-h3-Turbo publishes step-distilled adapters that each need
# their OWN {steps, video shift, audio shift} recipe -- ported from
# SceneWorks' measured manifest (config/manifests/builtin.loras.jsonc,
# epic 17137/sc-18725, MEASURED on real weights: 12.6min vs 2.42h at
# 1344x768/124 frames, 11.57x, with a 7.05x floor at 576x320). The four
# published files disagree with each other (768p trains at shift 6, the
# other two 4-/8-step files at shift 12), so "one recipe per family" doesn't
# work -- this table is keyed per file, matching SceneWorks' own approach.
#
# Values are (steps, shift_video, shift_audio). Keys are the lightx2v
# DIFFUSERS-format filename stems (lowercase, no extension) -- the repo also
# publishes each adapter as a `_comfyui_` twin (`diffusion_model.*` prefix,
# block-diagonal `attn.qkv_proj` B with alpha x3, `mlp.fc1` packed
# `[gate | value]`). That initially looked incompatible with this project's
# own fused representation (going by SceneWorks' warning that ITS OWN Rust
# DiT uses `[value | gate]` and refuses the `_comfyui_` files for that
# reason) -- but this project's `MLP.__call__`
# (`native/minimax_h3/model.py:109-124`) splits `fc1`'s output the same way
# ComfyUI does, gate first, and a real local file's header confirms the same
# target convention (`__metadata__.swi_glu_mapping`:
# "Diffusers [value;gate] -> ComfyUI [gate;value]", `qkv_fusion`: "block
# diagonal B; concat A; alpha multiplied by 3") -- SceneWorks' caveat was
# about ITS OWN engine's differing convention, not a defect in the
# `_comfyui_` files themselves. The block-diagonal qkv construction folds
# correctly through a plain `B @ A` (verified: stored alpha 384 against
# per-sub-block rank 128 keeps the alpha/rank scale ratio at 1.0, matching
# `mlp.fc1`'s own 128/128). Matching strips a `_comfyui` infix so both twins
# resolve to the same entry.
_MINIMAX_H3_TURBO_LORA_RECIPES: dict[str, tuple[int, float, float]] = {
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16": (4, 6.0, 3.0),
    "minimax_h3_fl2v_turbo_8step_v1.0_bf16": (8, 12.0, 3.0),
    "minimax_h3_fl2v_turbo_4step_v0.1": (4, 12.0, 3.0),
    "minimax_h3_ref2v_turbo_4step_v0.1_bf16": (4, 12.0, 3.0),
}


def _maybe_apply_minimax_h3_turbo_recipe(lora_name: str, model: dict) -> dict:
    """If `lora_name` matches a known MiniMax H3 turbo LoRA (diffusers or
    `_comfyui_` export), auto-apply its measured sigma shifts and print the
    matching step count -- `steps` lives on `ASDX_MiniMaxH3Sampler`, a
    separate node this function can't reach, so it can only recommend that
    value rather than set it. No-op for anything that isn't a recognized
    MiniMax H3 turbo file (returns `model` unchanged).
    """
    if model.get("family") != "minimax_h3":
        return model
    stem = Path(lora_name).stem.lower().replace("_comfyui", "")
    recipe = _MINIMAX_H3_TURBO_LORA_RECIPES.get(stem)
    if recipe is None:
        return model

    steps, shift_video, shift_audio = recipe
    from .minimax_h3_nodes import _replace_sigma_shifts

    new_config = _replace_sigma_shifts(model["config"], shift_video, shift_audio)
    print(
        f"[ASDX] MiniMax H3 turbo LoRA recipe matched for '{lora_name}': "
        f"shift_video={shift_video}, shift_audio={shift_audio} applied automatically -- "
        f"set ASDX_MiniMaxH3Sampler's steps to {steps} to match (SceneWorks-measured recipe)."
    )
    return {**model, "config": new_config}


# ── Diffusers/PEFT FLUX.1 / Flux.2 LoRA key mapping ─────────────────────
#
# A diffusers-trained FLUX.1 or Flux.2/Klein LoRA (e.g. ai-toolkit,
# SimpleTuner, kohya's sd-scripts diffusers mode) uses HF's
# `transformer_blocks`/`single_transformer_blocks` naming with separate
# `to_q`/`to_k`/`to_v` projections, not BFL's fused `double_blocks`/
# `single_blocks` `qkv`/`linear1`. Ported directly from comfy's own
# `comfy/utils.py::flux_to_diffusers` (the exact table comfy uses for this
# same file convention, and which already covers both families -- Flux.2's
# single-block entries are just extra keys appended to the same
# `block_map`) rather than guessed. comfy applies each split component as
# an independent sliced patch via `weight.narrow(dim, start, length)`
# because each of q/k/v/mlp keeps its own (possibly different) LoRA rank
# and can't be concatenated at the A/B stage -- `_assemble_fused_delta`
# below reproduces that by zero-padding each already-computed (B @ A)
# delta to the fused weight's full output width and concatenating.
#
# Double-stream blocks (`transformer_blocks.{i}`) are handled identically
# for both families by `_resolve_flux_double_diffusers_lora`: FLUX.1 and
# Flux.2/Klein share the exact same `double_blocks` module tree (see
# `native/flux2/config.py`'s docstring), and comfy's own key_map uses one
# unmodified `block_map` for both. Single-stream blocks differ: FLUX.1's
# diffusers port keeps `to_q`/`to_k`/`to_v`/`proj_mlp` split (needs the
# same assembly as the double-block qkv), while Flux.2's diffusers port
# already fuses them into one `attn.to_qkv_mlp_proj` weight matching
# native `linear1` exactly (a straight rename, no assembly) and
# `attn.to_out` covers the full `linear2` -- see comfy's
# `flux_to_diffusers`, the `# Flux 2` entries in its single-block
# `block_map`. Confirmed against a real Civitai Flux.2 Klein 9B diffusers
# LoRA's actual warning keys (8 `transformer_blocks.*` + 24
# `single_transformer_blocks.*`, matching Klein 9B's block counts).
#
# Native attribute names (see native/__init__.py's DoubleBlock/SingleBlock,
# and native/flux2/model.py's DoubleBlock/SingleBlock) match their BFL
# checkpoint 1:1 EXCEPT the MLP layers, which are flat `img_mlp_0`/
# `img_mlp_2` attributes (underscore) instead of a dotted Sequential
# `img_mlp.0`/`img_mlp.2` -- see native/weight_map.py's own docstring for
# the same rename applied at checkpoint-load time.
_FLUX_DOUBLE_QKV_RE = re.compile(r"^double_blocks\.(\d+)\.(img_attn|txt_attn)\.qkv\.weight$")
_FLUX_SINGLE_LINEAR1_RE = re.compile(r"^single_blocks\.(\d+)\.linear1\.weight$")
_FLUX_DOUBLE_RENAME_RE = re.compile(r"^double_blocks\.(\d+)\.(.+)$")
_FLUX_SINGLE_RENAME_RE = re.compile(r"^single_blocks\.(\d+)\.(.+)$")

# Native MLP suffix -> candidate diffusers suffixes, tried in order. Two
# real-world naming variants exist for the same weight (diffusers' own
# `ff.net.0.proj`/`ff.net.2` vs. the LyCoris/LoKr export convention's
# `ff.linear_in`/`ff.linear_out`, seen on the actual Flux.2 Klein LoRA this
# was verified against) -- comfy's own key_map maps both onto the same
# native target, so both are tried here too.
_FLUX_DOUBLE_DIFFUSERS_RENAME: dict[str, tuple[str, ...]] = {
    "img_attn.proj.weight": ("attn.to_out.0.weight",),
    "txt_attn.proj.weight": ("attn.to_add_out.weight",),
    "img_mod.lin.weight": ("norm1.linear.weight",),
    "txt_mod.lin.weight": ("norm1_context.linear.weight",),
    "img_mlp_0.weight": ("ff.net.0.proj.weight", "ff.linear_in.weight"),
    "img_mlp_2.weight": ("ff.net.2.weight", "ff.linear_out.weight"),
    "txt_mlp_0.weight": ("ff_context.net.0.proj.weight", "ff_context.linear_in.weight"),
    "txt_mlp_2.weight": ("ff_context.net.2.weight", "ff_context.linear_out.weight"),
}
_FLUX_SINGLE_DIFFUSERS_RENAME: dict[str, str] = {
    "modulation.lin.weight": "norm.linear.weight",
    "linear2.weight": "proj_out.weight",
}


def _assemble_fused_delta(
    pieces: list[mx.array | None], lengths: list[int]
) -> tuple[mx.array | None, int]:
    """Concatenate disjoint per-slice LoRA deltas (q/k/v[/mlp]) into one
    fused-weight delta. `pieces`/`lengths` are in output-axis order and
    span the fused weight's full output width contiguously -- a missing
    slice (LoRA doesn't target that component) becomes zeros so the
    concatenation still lines up. Returns (delta, num_pieces_consumed) --
    the count lets the caller's applied/total log line reflect actual raw
    delta entries consumed, since one fused native key here can absorb
    multiple raw file entries (e.g. 3 for a qkv projection).
    """
    consumed = sum(p is not None for p in pieces)
    if consumed == 0:
        return None, 0
    in_features = next(p.shape[1] for p in pieces if p is not None)
    dtype = next(p.dtype for p in pieces if p is not None)
    blocks = [
        p.astype(mx.float32) if p is not None else mx.zeros((length, in_features), dtype=mx.float32)
        for p, length in zip(pieces, lengths)
    ]
    return mx.concatenate(blocks, axis=0).astype(dtype), consumed


def _resolve_flux_double_diffusers_lora(
    flat_key: str, lora: "LoRAAdapter", hidden_dim: int
) -> tuple[mx.array | None, int]:
    """Resolve a native `double_blocks.*` flat parameter key against a
    diffusers/PEFT-style LoRA's per-component deltas, materialized lazily
    via `_materialize_delta`. Shared by both FLUX.1 and Flux.2/Klein -- see
    module-level comment above. Returns (None, 0) if `flat_key` isn't a
    `double_blocks.*` key or has no diffusers counterpart in `lora`.
    """
    m = _FLUX_DOUBLE_QKV_RE.match(flat_key)
    if m:
        idx, stream = m.group(1), m.group(2)
        sub_names = ("to_q", "to_k", "to_v") if stream == "img_attn" else \
            ("add_q_proj", "add_k_proj", "add_v_proj")
        pieces = [_materialize_delta(f"transformer_blocks.{idx}.attn.{name}.weight", lora) for name in sub_names]
        return _assemble_fused_delta(pieces, [hidden_dim, hidden_dim, hidden_dim])

    m = _FLUX_DOUBLE_RENAME_RE.match(flat_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        for diffusers_suffix in _FLUX_DOUBLE_DIFFUSERS_RENAME.get(rest, ()):
            delta = _materialize_delta(f"transformer_blocks.{idx}.{diffusers_suffix}", lora)
            if delta is not None:
                return delta, 1
        return None, 0

    return None, 0


def _resolve_flux_diffusers_lora(
    flat_key: str, lora: "LoRAAdapter", hidden_dim: int, mlp_dim: int
) -> tuple[mx.array | None, int]:
    """Resolve a native FLUX.1 flat parameter key against a diffusers/PEFT-
    style LoRA's per-component deltas, materialized lazily via
    `_materialize_delta` (see module-level comment above). Returns (None, 0)
    if `flat_key` has no diffusers counterpart in `lora` -- callers fall
    through to leaving the parameter unchanged.
    """
    delta, consumed = _resolve_flux_double_diffusers_lora(flat_key, lora, hidden_dim)
    if consumed:
        return delta, consumed

    m = _FLUX_SINGLE_LINEAR1_RE.match(flat_key)
    if m:
        idx = m.group(1)
        pieces = [
            _materialize_delta(f"single_transformer_blocks.{idx}.attn.to_q.weight", lora),
            _materialize_delta(f"single_transformer_blocks.{idx}.attn.to_k.weight", lora),
            _materialize_delta(f"single_transformer_blocks.{idx}.attn.to_v.weight", lora),
            _materialize_delta(f"single_transformer_blocks.{idx}.proj_mlp.weight", lora),
        ]
        return _assemble_fused_delta(pieces, [hidden_dim, hidden_dim, hidden_dim, mlp_dim])

    m = _FLUX_SINGLE_RENAME_RE.match(flat_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        diffusers_suffix = _FLUX_SINGLE_DIFFUSERS_RENAME.get(rest)
        if diffusers_suffix is not None:
            delta = _materialize_delta(f"single_transformer_blocks.{idx}.{diffusers_suffix}", lora)
            return delta, (1 if delta is not None else 0)
        return None, 0

    return None, 0


def _resolve_flux2_diffusers_lora(
    flat_key: str, lora: "LoRAAdapter", hidden_size: int
) -> tuple[mx.array | None, int]:
    """Resolve a native Flux.2/Klein flat parameter key against a
    diffusers/PEFT-style LoRA's per-component deltas, materialized lazily
    via `_materialize_delta` (see module-level comment above). Double-stream
    blocks reuse `_resolve_flux_double_diffusers_lora`; single-stream blocks
    are direct renames (`attn.to_qkv_mlp_proj` -> `linear1`, `attn.to_out`
    -> `linear2`), unlike FLUX.1 which needs 4-way assembly for `linear1`.
    """
    delta, consumed = _resolve_flux_double_diffusers_lora(flat_key, lora, hidden_size)
    if consumed:
        return delta, consumed

    m = _FLUX_SINGLE_LINEAR1_RE.match(flat_key)
    if m:
        idx = m.group(1)
        delta = _materialize_delta(f"single_transformer_blocks.{idx}.attn.to_qkv_mlp_proj.weight", lora)
        return delta, (1 if delta is not None else 0)

    m = _FLUX_SINGLE_RENAME_RE.match(flat_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        if rest == "linear2.weight":
            delta = _materialize_delta(f"single_transformer_blocks.{idx}.attn.to_out.weight", lora)
            return delta, (1 if delta is not None else 0)
        return None, 0

    return None, 0


def _apply_lora_to_clip(clip: Any, lora_path: Path, strength_clip: float) -> Any:
    """Patch a CLIP text encoder with a LoRA file.

    `mlx_clip` (see conditioning.py) is a real `comfy.sd.CLIP` -- a PyTorch/
    ModelPatcher object, not our own MLX reimplementation -- so unlike
    `_apply_lora_to_transformer` there is no reason to hand-roll key
    matching here. `comfy.sd.load_lora_for_models(model, clip, ...)` already
    does this correctly (alpha/rank normalization included); passing
    `model=None` skips the model-side patch we already handle ourselves in
    MLX. Needed for SDXL, where (unlike FLUX/Flux2/Krea2/Z-Image, whose
    LoRAs are trained on the transformer only) real CLIP-side LoRA deltas
    are common (kohya `lora_te1_`/`lora_te2_` keys) and silently dropping
    them under-applies the LoRA's intended effect.

    `raw` is filtered to drop transformer-only keys before the comfy call:
    with `model=None`, comfy's internal key_map only covers CLIP, so every
    transformer-side key in the file (already applied separately by
    `_apply_lora_to_transformer`) would otherwise log a spurious
    "lora key not loaded" warning -- for a real kohya SDXL LoRA (e.g.
    Illustrious) that is thousands of warning lines burying any genuine
    CLIP-side mismatch. Covers the kohya `lora_unet_*` convention, the
    diffusers/PEFT FLUX/Flux2 convention -- bare `transformer_blocks.*`/
    `single_transformer_blocks.*`, NOT wrapped in a `transformer.` prefix,
    confirmed against a real diffusers Flux.2 Klein LoRA whose raw keys
    have no prefix at all -- and the BFL-native convention
    (`diffusion_model.double_blocks./single_blocks.*`, confirmed to
    otherwise flood the log with one warning per raw tensor on a real
    Flux.2 Klein LoRA) -- real CLIP LoRA keys use `lora_te*_`/`text_model.`
    naming, never any of these prefixes, so this can't drop a genuine
    CLIP-side key.
    """
    if clip is None or strength_clip == 0:
        return clip
    import comfy.sd
    import comfy.utils

    raw = comfy.utils.load_torch_file(str(lora_path), safe_load=True)
    raw = {k: v for k, v in raw.items()
           if not k.startswith("lora_unet_") and not k.startswith("transformer.")
           and not k.startswith("diffusion_model.")
           and not k.startswith("transformer_blocks.")
           and not k.startswith("single_transformer_blocks.")}
    _, new_clip = comfy.sd.load_lora_for_models(None, clip, raw, 0.0, strength_clip)
    return new_clip if new_clip is not None else clip


# ── LoRA Loader Node ─────────────────────────────────────────────────

class ASDX_LoraLoader(io.ComfyNode):
    """Load a LoRA adapter and apply it to a model.

    Supports standard LoRA (A/B matrices) and ComfyUI diff format.
    Multiple LoRAs can be stacked by chaining LoRA loaders.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_LoraLoader",
            display_name="🍏 ASDX LoRA Loader",
            category="ASDX/LoRA",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Combo.Input("lora_name", options=cls._get_loras()),
                io.Float.Input("strength_model", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Custom("mlx_clip").Input("clip", optional=True),
                io.Float.Input("strength_clip", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
            ],
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
                io.Custom("mlx_clip").Output(display_name="clip"),
            ],
        )

    @staticmethod
    def _get_loras() -> list[str]:
        """Get list of available LoRA files."""
        try:
            import folder_paths
            loras = []
            for folder in ("loras",):
                try:
                    loras.extend(folder_paths.get_filename_list(folder))
                except Exception:
                    pass
            if loras:
                return loras
        except Exception:
            pass
        return ["example_lora.safetensors"]

    @classmethod
    def execute(
        cls,
        model: dict,
        lora_name: str,
        strength_model: float,
        clip: Any = None,
        strength_clip: float = 1.0,
    ) -> io.NodeOutput:
        """Load and apply a LoRA adapter to the model (and, if connected, the CLIP)."""
        t0 = time.perf_counter()
        # Diagnostic: confirms whether ComfyUI actually cache-hits this node
        # across repeated queues of an unchanged graph, or re-executes it
        # every run (which would re-merge the LoRA and transiently hold the
        # previous cached output's transformer alongside the new one).
        print("[ASDX] LoraLoader.execute() running (cache miss)")

        transformer = model["transformer"]
        lora_path = cls._resolve_lora_path(lora_name)
        _check_lora_compatibility(lora_path, model)

        # Load LoRA weights (alpha comes from the file itself, see
        # _load_lora_file -- matches real ComfyUI's LoraLoader, which has no
        # user-facing alpha widget either).
        lora = cls._load_lora_file(lora_path)

        # Apply scale
        lora.scale = base_lora_scale(lora.alpha, lora.rank) * strength_model

        # Apply to transformer — returns a NEW transformer, the cached base
        # model (model["transformer"]) is never mutated (see
        # _apply_lora_to_transformer docstring).
        new_transformer = cls._apply_lora_to_transformer(transformer, lora, model["config"])
        new_model = {**model, "transformer": new_transformer}
        new_model = _maybe_apply_minimax_h3_turbo_recipe(lora_name, new_model)

        # The raw delta/factor arrays are already merged into new_transformer
        # and never read again below -- `lora` otherwise stays referenced by
        # this frame right through the clear_mlx_cache() call below, so
        # gc.collect() finds it still live and can't reclaim it (confirmed
        # via ASDX_RECORD_MEMORY_EVIDENCE: MPS driver memory stayed pinned
        # near its post-merge peak instead of dropping back toward the base
        # transformer's footprint). `lora.factors` (Phase 0) keeps this small
        # by design even before this clear -- only the un-diffed A/B pairs
        # ever touched it -- but it's still worth dropping the reference. Do
        # not apply this to ASDX_LoraSchedule's `lora` -- that one is stored
        # on the model and its deltas/factors are needed again every
        # sampling step.
        lora.deltas = {}
        lora.factors = {}

        # The old (now-orphaned) transformer's arrays are unreferenced after
        # this reassignment, but MLX's allocator won't return that memory to
        # the OS on its own -- trim it now instead of letting it sit as idle
        # cache through however many more nodes run before something else
        # happens to clear it (loader.py does the same after every load).
        bridge.clear_mlx_cache()

        # CLIP-side LoRA -- only meaningful when a clip is actually connected
        # (SDXL); see _apply_lora_to_clip. Left as None when no clip input is
        # wired, matching this node's pre-existing FLUX/Krea2/etc. workflows
        # that never connect one.
        new_clip = _apply_lora_to_clip(clip, lora_path, strength_clip)

        elapsed = time.perf_counter() - t0
        print(f"[ASDX] LoRA '{lora_name}' applied: rank={lora.rank}, "
              f"scale={lora.scale:.4f}, {elapsed:.2f}s")

        return io.NodeOutput(new_model, new_clip)

    @staticmethod
    def _load_lora_file(path: Path) -> LoRAAdapter:
        """Load a LoRA file and extract delta weights."""
        t0 = time.perf_counter()
        name = path.stem

        if path.suffix == ".safetensors":
            import torch
            import safetensors.torch
            state = safetensors.torch.load_file(path, device="cpu")
            raw = {}
            for k, v in state.items():
                if v.dtype == torch.bfloat16:
                    v = v.float()
                raw[k] = v.cpu().numpy()
        elif path.suffix == ".pt" or path.suffix == ".bin":
            import torch
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict):
                raw = {k: v.numpy() for k, v in state.items()}
            else:
                raw = {}
        else:
            raise ValueError(f"Unsupported LoRA format: {path.suffix}")

        # Extract deltas from raw weights
        lora = LoRAAdapter(name=name)
        deltas: dict[str, tuple[mx.array, mx.array]] = {}  # key -> (A, B) or diff
        unsupported_lokr: list[str] = []
        unsupported_loha: list[str] = []
        alpha_value: float | None = None

        for key, weight in raw.items():
            if key.endswith(".alpha"):
                # Real per-file training alpha (comfy/lora.py's convention:
                # alpha_name = "{x}.alpha", same prefix as lora_A/B/up/down
                # minus any .weight suffix). Every target in a real LoRA
                # file shares the same value in practice -- keep the first
                # one found rather than tracking one per target.
                #
                # A full-w1/w2 LoKr target's alpha is NOT a scale: comfy uses
                # it only to rebuild low-rank `lokr_w*_a/_b` factors
                # (`weight_adapter/lokr.py`), and a real file on this machine
                # carries a sentinel (9999220736.0). Reading it as the
                # application scale made every LoKr generation overflow to a
                # black image -- the delta itself was correct, the scale it
                # was multiplied by was ~1e10.
                stem = key[: -len(".alpha")]
                if f"{stem}.lokr_w1" in raw and f"{stem}.lokr_w1_a" not in raw:
                    continue
                if alpha_value is None:
                    try:
                        alpha_value = float(weight)
                    except (TypeError, ValueError):
                        pass
                continue

            weight_arr = mx.array(weight if isinstance(weight, mx.array) else weight)

            # Standard LoRA format: {prefix}.lora_A.{param} / {prefix}.lora_B.{param}
            if ".lora_A." in key:
                # PEFT (HF diffusers-trained) files insert an adapter name
                # between lora_A/B and the param, e.g. "...lora_A.default.
                # weight" instead of comfy/kohya's "...lora_A.weight" -- a
                # plain string replace would leave "default.weight" stuck to
                # the prefix and never match anything downstream (confirmed:
                # a real diffusers Flux.2 LoRA applied 0/144 deltas this
                # way). lora_A/B always target a weight matrix, so drop
                # whatever follows and hard-code ".weight" back on.
                prefix = key.partition(".lora_A.")[0] + ".weight"
                # Strip common ComfyUI prefixes (diffusion_model., model., etc.)
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if prefix.startswith(pfx):
                        prefix = prefix[len(pfx):]
                        break
                # Native module-tree renames (Krea2 attn.gate -> attn.gate_proj,
                # FLUX/Flux2 img_mlp./txt_mlp. -> img_mlp_/txt_mlp_) -- see
                # _normalize_native_lora_key.
                prefix = _normalize_native_lora_key(prefix)
                if prefix not in deltas:
                    deltas[prefix] = (None, None)
                deltas[prefix] = (weight_arr, deltas[prefix][1])
            elif ".lora_B." in key:
                prefix = key.partition(".lora_B.")[0] + ".weight"
                # Strip common ComfyUI prefixes
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if prefix.startswith(pfx):
                        prefix = prefix[len(pfx):]
                        break
                prefix = _normalize_native_lora_key(prefix)
                if prefix not in deltas:
                    deltas[prefix] = (deltas[prefix][0], None)
                deltas[prefix] = (deltas[prefix][0], weight_arr)
            elif ".lora_up." in key:
                # kohya-style format: {prefix}.lora_up.weight / {prefix}.lora_down.weight
                down_key = key.replace(".lora_up.", ".lora_down.")
                if down_key not in raw:
                    continue
                prefix = key.replace(".lora_up.", ".")
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if prefix.startswith(pfx):
                        prefix = prefix[len(pfx):]
                        break
                prefix = _normalize_native_lora_key(prefix)
                down_raw = raw[down_key]
                down_arr = down_raw if isinstance(down_raw, mx.array) else mx.array(down_raw)
                # Route through the same (a, b) pairing dict as lora_A/lora_B so the
                # shared "delta = b @ a" conversion loop below handles it uniformly
                # (up=B [out,rank], down=A [rank,in] -- comfy/weight_adapter/lora.py
                # computes `diff = lora_up.weight @ lora_down.weight`, i.e. B @ A).
                deltas[prefix] = (down_arr, weight_arr)
            elif key.endswith(".lokr_w1"):
                # LoKr (LyCORIS Kronecker): delta = scale * kron(w1, w2). A
                # different algebra from LoRA -- it cannot live in `factors`.
                # Each Kronecker factor is stored either FULL (`lokr_w1`) or as
                # a low-rank product (`lokr_w1_a @ lokr_w1_b`); rebuilding the
                # low-rank form yields the same SMALL factor, so both stay
                # deferrable through `_kron_matmul` and neither materializes
                # the `[out, in]` delta.
                #
                # Math ported from mlx-gen's `reconstruct_lokr_delta_scaled`
                # (which mirrors LyCORIS `LokrModule.get_weight`/`make_kron`)
                # and cross-checked bit-exact against comfy's
                # `weight_adapter/lokr.py`.
                #
                # Scale follows LyCORIS exactly via `_lycoris_scale`: a rank
                # only exists when a factor is decomposed, and both-full
                # therefore applies at 1.0 -- see that helper for why using
                # this file's `alpha` instead produced black images.
                stem = key[: -len(".lokr_w1")]
                factors = _lokr_factor_pair(stem, raw)
                if factors is None:
                    unsupported_lokr.append(stem)
                    continue
                w1_arr, w2_arr, rank = factors
                lora.lokr_factors[_strip_and_normalize_key(f"{stem}.weight")] = (
                    w1_arr, w2_arr * _lycoris_scale(_alpha_of(stem, raw), rank),
                )
            elif key.endswith((".lokr_w2", ".lokr_w1_a", ".lokr_w1_b",
                               ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2")):
                continue  # consumed alongside its `.lokr_w1` sibling above
            elif key.endswith(".hada_w1_a"):
                # LoHa (LyCORIS Hadamard): delta = scale * ((w1_a@w1_b) *
                # (w2_a@w2_b)) -- an ELEMENTWISE product of two low-rank
                # products, so unlike LoKr the result is inherently full-size
                # and has no structured deferred form. Materialized lazily,
                # one target at a time, like any other full delta.
                stem = key[: -len(".hada_w1_a")]
                need = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
                if any(f"{stem}.{n}" not in raw for n in need):
                    unsupported_loha.append(stem)
                    continue
                if f"{stem}.hada_t1" in raw or f"{stem}.hada_t2" in raw:
                    unsupported_loha.append(stem)   # tucker/CP, conv-only
                    continue
                parts = [_as_mx(raw[f"{stem}.{n}"]) for n in need]
                if any(p.ndim != 2 for p in parts):
                    unsupported_loha.append(stem)
                    continue
                # LyCORIS rank is the inner dim of the low-rank product.
                rank = parts[0].shape[1]
                lora.loha_factors[_strip_and_normalize_key(f"{stem}.weight")] = (
                    *parts, _lycoris_scale(_alpha_of(stem, raw), rank),
                )
            elif key.endswith((".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
                               ".hada_t1", ".hada_t2")):
                continue  # consumed alongside its `.hada_w1_a` sibling above
            elif ".diff_b" in key:
                # ComfyUI diff format (bias delta). Real key is "{x}.diff_b" with
                # NO .weight/.bias suffix on x (comfy/lora.py maps it to
                # "{x}.bias") -- append .bias, don't just strip the suffix.
                diff_key = key.replace(".diff_b", ".bias")
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if diff_key.startswith(pfx):
                        diff_key = diff_key[len(pfx):]
                        break
                diff_key = _normalize_native_lora_key(diff_key)
                lora.deltas[diff_key] = weight_arr
            elif ".diff" in key:
                # ComfyUI diff format (weight delta). Real key is "{x}.diff" with
                # NO .weight suffix on x (comfy/lora.py maps it to "{x}.weight")
                # -- append .weight, don't just strip the suffix.
                diff_key = key.replace(".diff", ".weight")
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if diff_key.startswith(pfx):
                        diff_key = diff_key[len(pfx):]
                        break
                diff_key = _normalize_native_lora_key(diff_key)
                lora.deltas[diff_key] = weight_arr

        if unsupported_loha:
            print(f"[ASDX] LoRA: {len(unsupported_loha)} LoHa target(s) use the "
                  f"tucker/CP variant or are missing a factor -- those targets are "
                  f"UNCHANGED (e.g. {unsupported_loha[0]})")
        if unsupported_lokr:
            # Never drop these silently: a LoKr whose factors need rebuilding
            # would otherwise load "successfully" and apply nothing, the exact
            # failure mode that hid the diffusers-key gap for so long.
            print(f"[ASDX] LoRA: {len(unsupported_lokr)} LoKr target(s) use the "
                  f"low-rank-rebuild or Tucker variant, which is not implemented "
                  f"-- those targets are UNCHANGED (e.g. {unsupported_lokr[0]})")

        # Route each (A, B) pair into `lora.factors` as its raw low-rank
        # pair, UNMATERIALIZED -- the old code computed `delta = B @ A` for
        # every target right here, so the full set of full-size deltas
        # stayed alive for the whole merge on top of the whole-model rebuild
        # _apply_lora_to_transformer also builds (duplication source #1, see
        # the canon). Rank is inferred from A/B's shape alone, which needs
        # no matmul. Actual delta computation now happens lazily, one target
        # at a time, via _materialize_delta -- called from
        # _apply_lora_to_transformer right where the result is consumed.
        for key, (a, b) in deltas.items():
            if a is not None and b is not None:
                lora.factors[key] = (a, b)
                if lora.rank == 0:
                    lora.rank = a.shape[0]
            elif a is not None:
                lora.deltas[key] = a
                if lora.rank == 0:
                    lora.rank = a.shape[0]
            elif b is not None:
                lora.deltas[key] = b
                if lora.rank == 0:
                    lora.rank = b.shape[-1]

        # Clean up unused pairs
        lora.deltas = {k: v for k, v in lora.deltas.items()
                       if not (isinstance(v, tuple) and v[0] is None and v[1] is None)}

        lora.alpha = alpha_value
        mx.eval(*lora.deltas.values(),
                *(t for pair in lora.factors.values() for t in pair))
        return lora

    @staticmethod
    def _apply_lora_to_transformer(
        transformer: Any,
        lora: LoRAAdapter,
        config: Any,
    ) -> Any:
        """Apply LoRA delta weights to transformer parameters and return a
        NEW transformer instance — never mutates `transformer` in place.

        `transformer` may be the same object cached in loader.py's
        `_MODEL_CACHE` and shared across separate ComfyUI executions; an
        in-place `setattr` here would permanently bake the LoRA into that
        cached base model, so a later cache hit (e.g. re-running the same
        workflow after only changing an unrelated downstream widget) would
        re-apply the delta on top of an already-modified transformer and
        silently compound the LoRA's effect. Every sibling MLX project
        checked (mflux, SDMLX) applies LoRA non-destructively for this same
        reason.

        Untouched parameters are carried over by reference (no copy), so
        this is cheap relative to a real checkpoint reload — only the
        LoRA-targeted arrays are freshly computed.
        """
        from mlx.utils import tree_flatten, tree_unflatten

        if not lora.deltas and not lora.factors and not lora.lokr_factors \
                and not lora.loha_factors:
            print("[ASDX] LoRA: no matching weights found")
            return transformer

        # ASDX_LoraSchedule re-applies the SAME LoRA every sampling step, where
        # all that changes is the scale -- rescale the already-attached
        # adapters in place instead of re-cloning and re-resolving everything.
        # Returns None (and falls through) whenever that isn't provably safe.
        rescaled = _rescale_attached_lora(transformer, lora)
        if rescaled is not None:
            return rescaled

        if isinstance(transformer, SingleStreamDiT):
            # Krea2 -- Phase 1 forward-time residual (see canon and the
            # module comment above `AdaptableLinear`), not the merge below.
            return _apply_lora_residual_to_krea2(transformer, lora)
        if isinstance(transformer, FluxTransformer):
            # BFL-native FLUX.1 (and Krea2's non-existent overlap aside --
            # SingleStreamDiT is checked above and never reaches here) --
            # Phase 2 forward-time residual, see the module comment above
            # `_apply_lora_residual_to_flux`.
            return _apply_lora_residual_to_flux(transformer, lora, config, is_flux2=False)
        if isinstance(transformer, Flux2Transformer):
            return _apply_lora_residual_to_flux(transformer, lora, config, is_flux2=True)
        if isinstance(transformer, NextDiT):
            # Z-Image -- Phase 3 forward-time residual, see canon and the
            # module comment above `_apply_lora_residual_to_zimage`.
            return _apply_lora_residual_to_zimage(transformer, lora)
        if isinstance(transformer, MiniMaxH3Model):
            # MiniMax H3's DiT keeps its 4 big per-block linears in MLX's
            # quantized format (memory, not a Phase 1-3 residual choice) --
            # the merge below assumes dense `[out, in]` weights and breaks on
            # a quantized target's packed shape, see
            # `_apply_lora_to_minimax_h3`'s docstring.
            return _apply_lora_to_minimax_h3(transformer, lora, config)

        # key could be something like "double_blocks.0.img_attn.qkv.weight"
        # or "single_blocks.5.mlp_0.weight" — same dotted-string convention
        # tree_flatten uses for checkpoint keys (see native/*/model.py's
        # load_*_transformer), so deltas match model_flat keys directly.
        model_flat = tree_flatten(transformer.parameters())

        # Only SDXL's native module tree diverges from its real checkpoint
        # keys (MLX Sequential's `.layers.` insertion, see weight_map.py) --
        # FLUX/Krea2's native tree matches its checkpoint 1:1, so no un-
        # mapping is needed there (or if a future architecture needs one,
        # add it to that architecture's own weight_map.py, not here).
        # isinstance, not type(...).__module__ string matching: ComfyUI's custom
        # node loader imports this package under a name derived from its
        # install path (`nodes.py::load_custom_node`'s `sys_module_name =
        # module_path.replace(".", "_x_")`, using the FULL filesystem path, not
        # "apple_silicon_nodes") -- a `__module__` string comparison silently
        # never matches in a real ComfyUI install, only in a standalone script
        # importing this package by its repo name. isinstance is immune to
        # whatever name the outer package ends up loaded under, since the
        # imported class objects here and the ones used to build `transformer`
        # both resolve through the same relative-import chain.
        checkpoint_stem_fn = None
        if isinstance(transformer, SDXLUNetModel):
            from .native.sdxl.weight_map import native_key_to_checkpoint_stem
            checkpoint_stem_fn = native_key_to_checkpoint_stem

        # FLUX.1/Flux.2's diffusers/PEFT fallback (see
        # `_resolve_flux_diffusers_lora`/`_resolve_flux2_diffusers_lora`) no
        # longer applies here -- both `FluxTransformer` and `Flux2Transformer`
        # are intercepted above by `_apply_lora_residual_to_flux` (Phase 2)
        # before reaching this merge loop, which now only ever runs for
        # SDXL (and any future family that hasn't gotten a residual port
        # yet).

        new_flat = []
        applied = 0
        # Evaluated incrementally in chunks below (not all at once at the end)
        # -- deferring every one of the up-to-256 touched-key delta computations
        # to a single final mx.eval() forces MLX's lazy graph engine to schedule
        # and hold temporaries for all of them concurrently, inflating the
        # transient peak well past the final merged size. Evaluating in small
        # batches as they're built lets each batch's temporaries be freed
        # before the next one is computed.
        _EVAL_CHUNK = 32
        _pending_eval = []
        for flat_key, value in model_flat:
            delta = _materialize_delta(flat_key, lora)
            if delta is None:
                # kohya-ss (sd-scripts) flat naming -- used by essentially all SD/
                # SDXL LoRAs (e.g. Illustrious): dots replaced by underscores and
                # prefixed "lora_unet_", e.g. "input_blocks.4.1.proj_in.weight" ->
                # "lora_unet_input_blocks_4_1_proj_in.weight". `_load_lora_file`
                # only strips a few known *dotted* prefixes, so these flat kohya
                # keys land in `lora.deltas` unchanged and never match `flat_key`
                # directly. Build the candidate the same direction comfy's real
                # `comfy/lora.py::model_lora_keys_unet` does (FROM the known real
                # key, not reverse-engineered from the flat one, which is
                # ambiguous whenever a module name itself contains an underscore,
                # e.g. "proj_in"/"ff_net_0_proj").
                #
                # `flat_key` is OUR native module's tree_flatten key, not the
                # real checkpoint key -- SDXL's weight_map.py inserts `.layers.`
                # (and restructures ff.net/label_emb) to address MLX's
                # nn.Sequential, so any target weight inside one of those
                # wrappers must be un-mapped back to its real checkpoint stem
                # before the kohya string is built, or it silently never
                # matches (this was the cause of the 442/986 partial match).
                stem, _, suffix = flat_key.rpartition(".")
                if checkpoint_stem_fn is not None:
                    stem = checkpoint_stem_fn(stem)
                delta = _materialize_delta(f"lora_unet_{stem.replace('.', '_')}.{suffix}", lora)
            consumed = 1 if delta is not None else 0
            if delta is not None:
                delta_mapped = delta.astype(value.dtype)
                merged = value + delta_mapped * lora.scale
                new_flat.append((flat_key, merged))
                _pending_eval.append(merged)
                applied += consumed
                if len(_pending_eval) >= _EVAL_CHUNK:
                    mx.eval(_pending_eval)
                    # mx.eval() alone materializes the chunk's results but
                    # leaves their now-unneeded scratch buffers (e.g. the
                    # matmul intermediates behind each delta) sitting in
                    # MLX's own allocator cache pool rather than releasing
                    # them to the driver -- mx.clear_cache() is what actually
                    # returns that memory, same call clear_mlx_cache() in
                    # bridge.py makes. Without it, chunked eval alone doesn't
                    # lower the observed transient peak (confirmed: driver
                    # peaks were unchanged before this was added).
                    mx.clear_cache()
                    _pending_eval = []
            else:
                new_flat.append((flat_key, value))
        if _pending_eval:
            mx.eval(_pending_eval)
            mx.clear_cache()

        new_transformer = type(transformer)(config)
        new_transformer.update(tree_unflatten(new_flat))
        mx.eval(new_transformer.parameters())
        print(f"[ASDX] LoRA: applied {applied}/{len(lora.deltas) + len(lora.factors)} deltas")
        return new_transformer

    @staticmethod
    def _resolve_lora_path(name: str) -> Path:
        """Resolve LoRA name to file path."""
        try:
            import folder_paths
            for folder in ("loras",):
                try:
                    full = folder_paths.get_full_path(folder, name)
                    if full:
                        return Path(full)
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback
        for candidate in (
            Path.home() / "ComfyUI" / "models" / "loras" / name,
            Path(name),
        ):
            if candidate.exists():
                return candidate
        return Path(name)


# ── Multi LoRA Loader ────────────────────────────────────────────────

class ASDX_MultiLoraLoader(io.ComfyNode):
    """Load up to 5 LoRA adapters at once, each with its own on/off toggle
    and independent model/clip strengths.

    Functional parity with LoraManager's LoraLoaderLM and rgthree's Power
    Lora Loader (per-entry active toggle, separate strength_model/
    strength_clip) -- implemented with plain ComfyUI widgets rather than
    their dynamic add/remove JS/Vue frontends, since this project has no
    web/ frontend infrastructure at all.

    Applies all LoRAs in a single pass for better performance.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        loras = ASDX_LoraLoader._get_loras()
        inputs: list = [io.Custom("asdx_model").Input("model")]
        for i in range(1, 6):
            inputs.append(io.Boolean.Input(f"lora{i}_enabled", default=False))
            inputs.append(io.Combo.Input(f"lora{i}_name", options=loras))
            inputs.append(io.Float.Input(f"lora{i}_strength_model", default=1.0, min=-10.0, max=10.0, step=0.01))
            inputs.append(io.Float.Input(f"lora{i}_strength_clip", default=1.0, min=-10.0, max=10.0, step=0.01))
        inputs.append(io.Custom("mlx_clip").Input("clip", optional=True))
        return io.Schema(
            node_id="ASDX_MultiLoraLoader",
            display_name="🍏 ASDX Multi LoRA Loader",
            category="ASDX/LoRA",
            inputs=inputs,
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
                io.Custom("mlx_clip").Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: dict,
        lora1_enabled: bool, lora1_name: str, lora1_strength_model: float, lora1_strength_clip: float,
        lora2_enabled: bool, lora2_name: str, lora2_strength_model: float, lora2_strength_clip: float,
        lora3_enabled: bool, lora3_name: str, lora3_strength_model: float, lora3_strength_clip: float,
        lora4_enabled: bool, lora4_name: str, lora4_strength_model: float, lora4_strength_clip: float,
        lora5_enabled: bool, lora5_name: str, lora5_strength_model: float, lora5_strength_clip: float,
        clip: Any = None,
    ) -> io.NodeOutput:
        """Apply up to 5 active LoRA adapters at once (and, if connected, the CLIP)."""
        # Diagnostic: see matching comment in ASDX_LoraLoader.execute().
        print("[ASDX] MultiLoraLoader.execute() running (cache miss)")
        entries = [
            (lora1_enabled, lora1_name, lora1_strength_model, lora1_strength_clip),
            (lora2_enabled, lora2_name, lora2_strength_model, lora2_strength_clip),
            (lora3_enabled, lora3_name, lora3_strength_model, lora3_strength_clip),
            (lora4_enabled, lora4_name, lora4_strength_model, lora4_strength_clip),
            (lora5_enabled, lora5_name, lora5_strength_model, lora5_strength_clip),
        ]

        transformer = model["transformer"]
        config = model["config"]
        applied_any = False

        for enabled, lora_name, strength_model, strength_clip in entries:
            if not enabled or not lora_name or lora_name == "None":
                continue
            if strength_model == 0 and strength_clip == 0:
                continue

            lora_path = ASDX_LoraLoader._resolve_lora_path(lora_name)
            _check_lora_compatibility(lora_path, model)

            if strength_model != 0:
                lora = ASDX_LoraLoader._load_lora_file(lora_path)
                # Same alpha/rank normalization as ASDX_LoraLoader.load_lora,
                # so a rank-N LoRA doesn't apply at N times its intended
                # strength.
                lora.scale = base_lora_scale(lora.alpha, lora.rank) * strength_model

                # Reuse the (non-destructive, tree-flatten-based) apply
                # logic — do not duplicate it here. Each call returns a NEW
                # transformer; chaining `transformer =` here stacks LoRAs
                # onto that new object, never onto the cached base model.
                transformer = ASDX_LoraLoader._apply_lora_to_transformer(transformer, lora, config)
                # Drop the now-unneeded raw delta/factor arrays before
                # cleanup -- see the matching comment in
                # ASDX_LoraLoader.execute().
                lora.deltas = {}
                lora.factors = {}
                # Each iteration orphans the previous intermediate
                # transformer (up to 5 full-size copies chained here) --
                # trim MLX's idle buffer cache between iterations instead of
                # letting them stack.
                bridge.clear_mlx_cache()

            clip = _apply_lora_to_clip(clip, lora_path, strength_clip)

            updated = _maybe_apply_minimax_h3_turbo_recipe(lora_name, {**model, "config": config})
            config = updated["config"]

            applied_any = True
            print(f"[ASDX] MultiLoRA: '{lora_name}' "
                  f"(model={strength_model:.2f}, clip={strength_clip:.2f})")

        if not applied_any:
            print("[ASDX] MultiLoRA: no LoRAs to apply")
            return io.NodeOutput(model, clip)

        return io.NodeOutput({**model, "transformer": transformer, "config": config}, clip)


# ── LoRA Schedule (per-step strength modulation) ─────────────────────

class ASDX_LoraSchedule(io.ComfyNode):
    """Schedule LoRA strength across sampling steps.

    Allows LoRA strength to vary per step (e.g., stronger at start, weaker at end).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_LoraSchedule",
            display_name="🍏 ASDX LoRA Schedule",
            category="ASDX/Advanced",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Combo.Input("lora_name", options=ASDX_LoraLoader._get_loras()),
                io.Float.Input("strength_start", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("strength_end", default=0.5, min=0.0, max=10.0, step=0.01),
                io.Float.Input("strength_middle", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Combo.Input("strength_curve", options=["linear", "cosine", "ease_in_out"], default="linear"),
                io.Custom("mlx_clip").Input("clip", optional=True),
                io.Float.Input("strength_clip", default=1.0, min=0.0, max=10.0, step=0.01, optional=True),
            ],
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
                io.Custom("mlx_clip").Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: dict,
        lora_name: str,
        strength_start: float,
        strength_end: float,
        strength_middle: float,
        strength_curve: str,
        clip: Any = None,
        strength_clip: float = 1.0,
    ) -> io.NodeOutput:
        """Attach LoRA schedule metadata to the model dict.

        The sampler reads this metadata and adjusts LoRA strength per step.
        """
        lora_path = ASDX_LoraLoader._resolve_lora_path(lora_name)
        _check_lora_compatibility(lora_path, model)
        lora = ASDX_LoraLoader._load_lora_file(lora_path)

        # Apply with start strength (same alpha/rank normalization as
        # ASDX_LoraLoader.load_lora — a rank-N LoRA must not apply at N times
        # its intended strength). Returns a NEW transformer — the cached
        # base model is never mutated. `sampler/core.py::_update_lora_schedule`
        # then keeps adjusting THIS new (private, non-cached) transformer
        # in place per step, which is safe since it's no longer shared.
        lora.scale = base_lora_scale(lora.alpha, lora.rank) * strength_start
        new_transformer = ASDX_LoraLoader._apply_lora_to_transformer(
            model["transformer"], lora, model["config"]
        )
        bridge.clear_mlx_cache()
        new_model = {**model, "transformer": new_transformer}

        # CLIP-side LoRA is applied once at strength_clip (not scheduled):
        # text conditioning is computed once before the sampling loop, not
        # per step, so a step-varying CLIP strength has no sampler hook to
        # attach to -- see _apply_lora_to_clip.
        new_clip = _apply_lora_to_clip(clip, lora_path, strength_clip)

        # Store schedule info on the model — the sampler reads this and
        # adjusts LoRA strength per step (see sampler/core.py).
        new_model["lora_schedule"] = {
            "name": lora_name,
            "lora": lora,
            "strength_start": strength_start,
            "strength_end": strength_end,
            "strength_middle": strength_middle,
            "strength_curve": strength_curve,
        }

        print(f"[ASDX] LoRA schedule: '{lora_name}' "
              f"start={strength_start:.2f} middle={strength_middle:.2f} end={strength_end:.2f}")

        return io.NodeOutput(new_model, new_clip)

    @staticmethod
    def _compute_schedule_strength(
        step: int,
        total_steps: int,
        start: float,
        end: float,
        middle: float,
        curve: str,
    ) -> float:
        """Compute LoRA strength for a given step."""
        if total_steps <= 1:
            return start

        progress = step / total_steps

        if curve == "linear":
            # Start -> Middle (0-0.5) -> End (0.5-1.0)
            if progress <= 0.5:
                return start + (middle - start) * (progress * 2)
            else:
                return middle + (end - middle) * ((progress - 0.5) * 2)

        elif curve == "cosine":
            # Smooth cosine interpolation
            mid = (start + middle) / 2
            end_val = (middle + end) / 2
            if progress <= 0.5:
                return mid + (start - mid) * 0.5 * (1 - mx.cos(mx.array(progress * 2 * 3.14159)))
            else:
                return end_val + (end - end_val) * 0.5 * (1 - mx.cos(mx.array((progress - 0.5) * 2 * 3.14159)))

        elif curve == "ease_in_out":
            # S-curve: slow start, fast middle, slow end
            t = 3 * progress ** 2 - 2 * progress ** 3
            if progress <= 0.5:
                return start + (middle - start) * t
            else:
                return middle + (end - middle) * t

        return start  # default: constant


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_LIST = [
    ASDX_LoraLoader,
    ASDX_MultiLoraLoader,
    ASDX_LoraSchedule,
]
