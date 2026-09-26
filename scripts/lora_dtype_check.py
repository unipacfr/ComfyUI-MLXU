"""Per-family real-weight verification of the LoRA factor-dtype change
(Task 1, commits 73b7123/cb321e9: LoRA factors keep their file dtype
instead of being force-upcast to float32 on load; `AdaptableLinear`'s
residual no longer casts `x` to the factor dtype).

Run (uv, project venv -- ComfyUI is not installed there, so this script
uses the same `tests/support/comfy_stub.py` trick `scripts/anima_parity.py`
uses to import `apple_silicon_nodes.loader`/`.lora` standalone):

    uv run python scripts/lora_dtype_check.py
    uv run python scripts/lora_dtype_check.py --families anima,krea2
    uv run python scripts/lora_dtype_check.py --rounds 5

For each family this loads the real base checkpoint ONCE, at the family's
usual host precision (bf16 for Anima; float16 -- `ASDX_DiffusionLoader`'s
default -- for every other family), then for each of that family's real
LoRA files:

  1. Loads it via `ASDX_LoraLoader._load_lora_file` and attaches it via
     `_apply_lora_to_transformer` TWICE from the same base transformer +
     LoRAAdapter -- this produces two independently-wrapped
     `AdaptableLinear` trees whose `.weight`/`.bias` arrays are the SAME
     underlying base-model arrays (never duplicated), but whose attached
     `_lora_factors`/`_lokr_factors` are independent.
  2. Leaves one tree ("new") exactly as `_apply_lora_to_transformer` built
     it -- factors in file dtype, current `AdaptableLinear.__call__`.
  3. Upcasts the other tree's ("old") attached factors to float32 (`.astype`
     returns a new array, so this can't affect the "new" tree) and, only
     for the duration of that tree's forward calls, monkeypatches
     `AdaptableLinear.__call__` to the pre-Task-1 body (`x.astype(a.dtype)`
     before the residual matmul -- see `git show
     48185b7:apple_silicon_nodes/lora.py`). This reproduces the exact old
     runtime behavior without reverting any code. The LoKr loop's own code
     is textually unchanged by Task 1 (`x.astype(w1.dtype)` either way) --
     what changes is w1/w2's dtype itself, which is exactly the case under
     test for the Krea2 LoKr files.
  4. Runs one forward each for base (no LoRA)/old/new on the same fixed
     random inputs, computes d_old = out_old - out_base and
     d_new = out_new - out_base, and reports cosine(d_new, d_old), the
     norm ratio, and whether both are finite.
  5. Times base/old/new interleaved over `--rounds` rounds (default 5) and
     reports medians -- this machine's thermal drift is up to 20% within a
     run, so a single-shot timing is not trustworthy on its own.

Pass criterion: cosine(d_new, d_old) > 0.995 and norm ratio within 1%.
A family/LoRA pair below that is reported as DONE_WITH_CONCERNS with its
numbers, not silently tuned away.

Resolutions are intentionally smaller than each family's production
default (a 512px-equivalent token grid, not 1024px) to keep multiple
real multi-GB checkpoints loadable and forwardable in one process in a
reasonable amount of wall-clock time on this machine. ponytail: this is a
deliberate ceiling on script runtime, not a claim about image quality --
raise `_GRID` per family if a future run needs the production resolution.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

MODELS_A = Path("/Volumes/X10Pro/Images/models")
MODELS_B = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/models")

SEED = 0
_GRID = 32  # token-grid side for the packed families -> 1024 tokens (~512px)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _norm_ratio(new: np.ndarray, old: np.ndarray) -> float:
    n_old = np.linalg.norm(old.reshape(-1).astype(np.float64))
    n_new = np.linalg.norm(new.reshape(-1).astype(np.float64))
    if n_old == 0:
        return float("nan")
    return float(n_new / n_old)


# ── Old-path emulation ──────────────────────────────────────────────────

def _install_old_call(lora_mod):
    """Return (patch(), restore()) that swap `AdaptableLinear.__call__` to
    the pre-Task-1 body (`git show 48185b7:apple_silicon_nodes/lora.py`)."""
    AdaptableLinear = lora_mod.AdaptableLinear
    nn = lora_mod.nn
    _kron_matmul = lora_mod._kron_matmul
    new_call = AdaptableLinear.__call__

    def old_call(self, x):
        y = nn.Linear.__call__(self, x)
        for a, b, scale in self._lora_factors:
            residual = (x.astype(a.dtype) @ a.T) @ b.T
            y = y + (scale * residual).astype(y.dtype)
        for w1, w2, scale in self._lokr_factors:
            y = y + (scale * _kron_matmul(x.astype(w1.dtype), w1, w2)).astype(y.dtype)
        return y

    def patch():
        AdaptableLinear.__call__ = old_call

    def restore():
        AdaptableLinear.__call__ = new_call

    return patch, restore


def _upcast_factors_to_f32(transformer, lora_mod):
    import mlx.core as mx

    for leaf in lora_mod._iter_adaptable_leaves(transformer):
        leaf._lora_factors = [
            (a.astype(mx.float32), b.astype(mx.float32), s) for a, b, s in leaf._lora_factors
        ]
        leaf._lokr_factors = [
            (w1.astype(mx.float32), w2.astype(mx.float32), s) for w1, w2, s in leaf._lokr_factors
        ]


def _factor_dtypes(transformer, lora_mod) -> set[str]:
    dtypes: set[str] = set()
    for leaf in lora_mod._iter_adaptable_leaves(transformer):
        for a, b, _ in leaf._lora_factors:
            dtypes.add(str(a.dtype))
            dtypes.add(str(b.dtype))
        for w1, w2, _ in leaf._lokr_factors:
            dtypes.add(str(w1.dtype))
            dtypes.add(str(w2.dtype))
    return dtypes


def _rng_array(rng, shape, dtype):
    import mlx.core as mx
    arr = rng.standard_normal(shape).astype(np.float32)
    return mx.array(arr).astype(dtype)


# ── Family setup: base checkpoint + a forward callable ──────────────────
# Each returns (transformer, config, forward) built ONCE per family, reused
# across every LoRA file listed for that family in FAMILY_LORAS below.

def setup_anima(loader_mod):
    ckpt = MODELS_A / "diffusion_models/Anima/anime/WaiHassakuAnima.safetensors"
    dtype = "bfloat16"
    transformer, config = loader_mod._load_transformer_for_type(ckpt, "anima", dtype)

    import mlx.core as mx
    rng = np.random.default_rng(SEED)
    hw = 64
    dtype = config.mlx_dtype
    x = _rng_array(rng, (1, 16, hw, hw), dtype)
    qwen = _rng_array(rng, (1, 20, 1024), dtype)
    ids = mx.array(rng.integers(0, 32128, size=(1, 12)).astype(np.int32))
    weights = mx.ones((1, 12), dtype=dtype)
    sigma = mx.array([0.8], dtype=mx.float32)

    def forward(model):
        ctx = model.encode_context(qwen, ids, weights)
        out = model(x, sigma, ctx)
        mx.eval(out)
        return out

    return transformer, config, forward, f"anima bf16, [1,16,{hw},{hw}] latent"


def setup_krea2(loader_mod):
    ckpt = MODELS_A / "diffusion_models/Krea 2/base model/krea2_turbo_bf16.safetensors"
    dtype = "float16"
    transformer, config = loader_mod._load_transformer_for_type(ckpt, "krea2", dtype)

    import mlx.core as mx
    rng = np.random.default_rng(SEED)
    h = w = _GRID
    img = _rng_array(rng, (1, h * w, 64), transformer.dtype)
    txt = _rng_array(rng, (1, 8, config.text_layers * config.text_dim), transformer.dtype)
    t = mx.array([0.8], dtype=mx.float32)

    def forward(model):
        out = model(img, txt=txt, t=t, img_h=h, img_w=w)
        mx.eval(out)
        return out

    return transformer, config, forward, f"krea2 f16, {h}x{w} token grid"


def setup_flux1(loader_mod):
    ckpt = MODELS_A / "diffusion_models/Flux.1 D/base model/flux1-dev.safetensors"
    dtype = "float16"
    model_type = loader_mod._detect_model_type(ckpt)
    transformer, config = loader_mod._load_transformer_for_type(ckpt, model_type, dtype)

    import mlx.core as mx
    rng = np.random.default_rng(SEED)
    h = w = _GRID
    img = _rng_array(rng, (1, h * w, transformer.config.in_channels), transformer.dtype)
    txt = _rng_array(rng, (1, 8, 4096), transformer.dtype)
    t = mx.array([0.8], dtype=mx.float32)
    guidance = mx.array([3.5], dtype=mx.float32) if config.guidance_embed else None
    pooled = _rng_array(rng, (1, 768), transformer.dtype)
    rope = transformer.get_rope(h, w, txt.shape[1])

    def forward(model):
        out = model(img, txt, t, guidance=guidance, pooled=pooled, rope=rope)
        mx.eval(out)
        return out

    return transformer, config, forward, f"flux1-dev f16, {h}x{w} token grid"


def setup_flux2(loader_mod):
    ckpt = MODELS_A / "diffusion_models/Flux.2 Klein 9B/base model/flux2Klein_9b.safetensors"
    dtype = "float16"
    transformer, config = loader_mod._load_transformer_for_type(ckpt, "flux2", dtype)

    import mlx.core as mx
    rng = np.random.default_rng(SEED)
    h = w = _GRID
    img = _rng_array(rng, (1, h * w, config.in_channels), transformer.dtype)
    txt = _rng_array(rng, (1, 8, config.context_in_dim), transformer.dtype)
    t = mx.array([0.8], dtype=mx.float32)
    rope = transformer.get_rope(h, w, txt.shape[1])

    def forward(model):
        out = model(img, txt, t, rope=rope)
        mx.eval(out)
        return out

    return transformer, config, forward, f"flux2/klein f16, {h}x{w} token grid"


def setup_zimage(loader_mod):
    ckpt = MODELS_A / "diffusion_models/ZImageTurbo/base model/z_image_turbo_bf16.safetensors"
    dtype = "float16"
    model_type = loader_mod._detect_model_type(ckpt)
    transformer, config = loader_mod._load_transformer_for_type(ckpt, model_type, dtype)

    import mlx.core as mx
    rng = np.random.default_rng(SEED)
    h = w = _GRID
    img = _rng_array(
        rng, (1, h * w, config.patch_size * config.patch_size * config.in_channels), transformer.dtype
    )
    context = _rng_array(rng, (1, 8, config.cap_feat_dim), transformer.dtype)
    t = mx.array([0.8], dtype=mx.float32)

    def forward(model):
        out = model(img, context, t, h, w)
        mx.eval(out)
        return out

    return transformer, config, forward, f"z-image-turbo f16, {h}x{w} token grid"


FAMILY_SETUP = {
    "anima": setup_anima,
    "krea2": setup_krea2,
    "flux1": setup_flux1,
    "flux2": setup_flux2,
    "zimage": setup_zimage,
}

FAMILY_LORAS = {
    "anima": [MODELS_A / "loras/Anima/character/tifa-anima.safetensors"],
    "krea2": [
        MODELS_A / "loras/Krea 2/realistic/ultra_real_krea2_v2.safetensors",
        MODELS_B / "loras/Krea 2/style/Greg Capullo Krea2.safetensors",
        MODELS_A / "loras/Krea 2/concept/snofs_krea_v1_4.safetensors",
    ],
    "flux1": [
        MODELS_A / "loras/Flux.1 D/realistic/Hand v2.safetensors",
        MODELS_A / "loras/Flux.1 D/character/lora.TA_trained.safetensors",
    ],
    "flux2": [
        MODELS_A / "loras/Flux.2 Klein 9B/style/OctaneRenderKlein9b.safetensors",
        MODELS_A / "loras/Flux.2 Klein 9B/concept/klein_snofs_v1_4.safetensors",
    ],
    "zimage": [MODELS_A / "loras/ZImageTurbo/character/Tifa_Lockhart_v1.safetensors"],
}


def run_lora(transformer, config, forward, lora_path: Path, lora_mod, rounds: int) -> dict:
    import mlx.core as mx

    lora = lora_mod.ASDX_LoraLoader._load_lora_file(lora_path)
    lora.scale = lora_mod.base_lora_scale(lora.alpha, lora.rank)
    lora_kind = "LoKr" if (lora.lokr_factors or lora.unsupported_lokr) else "LoRA"

    transformer_new = lora_mod.ASDX_LoraLoader._apply_lora_to_transformer(transformer, lora, config)
    transformer_old = lora_mod.ASDX_LoraLoader._apply_lora_to_transformer(transformer, lora, config)

    if transformer_new is transformer:
        # `_apply_lora_to_transformer` returned the base object unchanged --
        # this family's residual-attach function never matched any of this
        # LoRA's keys (e.g. FLUX.1/Flux2's `_apply_lora_residual_to_flux`
        # only ever looks up `lora.factors`/`.deltas`/`.loha_factors` via
        # `_lookup_native_or_kohya`, never `lora.lokr_factors` -- a
        # pre-existing LoKr-routing gap for those two families, unrelated
        # to Task 1's factor-dtype change). Report as N/A rather than a
        # misleading nan-cosine DONE_WITH_CONCERNS.
        return dict(
            lora_file=lora_path.name, lora_kind=lora_kind, new_factor_dtypes=[],
            cosine=float("nan"), norm_ratio=float("nan"), finite=True,
            passed=None, not_applicable="no keys matched (see comment in script)",
            s_step_base=float("nan"), s_step_old=float("nan"), s_step_new=float("nan"),
        )

    new_dtypes = _factor_dtypes(transformer_new, lora_mod)
    _upcast_factors_to_f32(transformer_old, lora_mod)

    patch, restore = _install_old_call(lora_mod)

    # Correctness (single shot).
    out_base = np.array(forward(transformer).astype(mx.float32))
    out_new = np.array(forward(transformer_new).astype(mx.float32))
    patch()
    try:
        out_old = np.array(forward(transformer_old).astype(mx.float32))
    finally:
        restore()

    d_new = out_new - out_base
    d_old = out_old - out_base
    finite = bool(np.all(np.isfinite(out_new)) and np.all(np.isfinite(out_old)))
    cosine = _cosine(d_new, d_old)
    ratio = _norm_ratio(d_new, d_old)
    passed = finite and cosine > 0.995 and abs(ratio - 1.0) <= 0.01

    # Timing: interleave base/old/new over `rounds`, take medians.
    t_base, t_new, t_old = [], [], []
    for _ in range(rounds):
        t0 = time.perf_counter(); forward(transformer); t_base.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); forward(transformer_new); t_new.append(time.perf_counter() - t0)
        patch()
        try:
            t0 = time.perf_counter(); forward(transformer_old); t_old.append(time.perf_counter() - t0)
        finally:
            restore()

    return dict(
        lora_file=lora_path.name,
        lora_kind=lora_kind,
        new_factor_dtypes=sorted(new_dtypes),
        cosine=cosine,
        norm_ratio=ratio,
        finite=finite,
        passed=passed,
        s_step_base=statistics.median(t_base),
        s_step_old=statistics.median(t_old),
        s_step_new=statistics.median(t_new),
    )


# Per ruling #2, interleaved multi-round timing is required for Anima and
# Krea2; "others optional if cheap". Flux.1/Flux2/Z-Image forwards on this
# machine measured ~0.5-70s/step at this script's resolution (not cheap), so
# they default to fewer rounds -- override with --rounds if a full 5-round
# timing is wanted anyway.
_DEFAULT_ROUNDS = {"anima": 5, "krea2": 5, "flux1": 1, "flux2": 1, "zimage": 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILY_SETUP))
    ap.add_argument("--rounds", type=int, default=None,
                     help="Override every family's round count (default: per-family, see _DEFAULT_ROUNDS)")
    args = ap.parse_args()

    from support.comfy_stub import install_comfy_stubs, load_node_module

    install_comfy_stubs()
    loader_mod = load_node_module("loader")
    lora_mod = load_node_module("lora")

    import mlx.core as mx

    for family in args.families.split(","):
        family = family.strip()
        if not family:
            continue
        print(f"\n=== {family} ===")
        try:
            transformer, config, forward, resolution = FAMILY_SETUP[family](loader_mod)
        except Exception as e:  # noqa: BLE001
            print(f"[lora_dtype_check] {family} base load FAILED: {e!r}")
            continue

        for lora_path in FAMILY_LORAS[family]:
            if not lora_path.exists():
                print(f"[lora_dtype_check] {family}: {lora_path} missing, skipping")
                continue
            rounds = args.rounds if args.rounds is not None else _DEFAULT_ROUNDS.get(family, 5)
            try:
                result = run_lora(transformer, config, forward, lora_path, lora_mod, rounds)
            except Exception as e:  # noqa: BLE001
                print(f"[lora_dtype_check] {family}/{lora_path.name} FAILED: {e!r}")
                continue
            if result.get("not_applicable"):
                print(
                    f"[N/A] {family}/{result['lora_file']} ({result['lora_kind']}): "
                    f"{result['not_applicable']}"
                )
                continue
            status = "PASS" if result["passed"] else "DONE_WITH_CONCERNS"
            print(
                f"[{status}] {family}/{result['lora_file']} ({result['lora_kind']}) "
                f"resolution={resolution} factor_dtypes={result['new_factor_dtypes']}\n"
                f"  cosine(d_new,d_old)={result['cosine']:.6f} norm_ratio={result['norm_ratio']:.4f} "
                f"finite={result['finite']}\n"
                f"  s/step base={result['s_step_base']:.3f} old={result['s_step_old']:.3f} "
                f"new={result['s_step_new']:.3f}"
            )

        del transformer
        mx.clear_cache()


if __name__ == "__main__":
    main()
