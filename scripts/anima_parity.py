"""Anima MLX vs ComfyUI DiT parity check on real checkpoint weights.

Each side loads an ~8 GB float32 copy of the checkpoint, so run the two
halves as SEPARATE processes sharing a scratch dir rather than one process
holding both:

    /Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/.venv/bin/python scripts/anima_parity.py \
        --only comfy --scratch /tmp/anima_parity
    uv run python scripts/anima_parity.py --only mlx --scratch /tmp/anima_parity

The second invocation finds both saved outputs in --scratch and prints the
comparison. Re-running either half alone re-generates fresh random inputs
unless the other half already wrote them to --scratch first, so always do
the `--only comfy` pass first (it also writes the shared inputs).

Add `--lora PATH --strength S` to BOTH invocations (same PATH/strength on
each side) to run the LoRA leg instead of (in addition to) the base-model
leg: ComfyUI applies the LoRA via `comfy.sd.load_lora_for_models` +
`model_patcher.patch_model()` (proven applied: its output must differ from
the unpatched base), MLX via `ASDX_LoraLoader._load_lora_file` +
`_apply_lora_to_transformer` at the same strength. Pass criterion (fp32,
CPU stream both sides): cosine of the LoRA-induced difference
`(v_lora - v_base)` between the two sides > 0.999, norm ratio within 1%.
Harness self-check still runs first on each side's own v_lora.

Pass criterion: cosine similarity of the two velocity outputs > 0.9999 in
float32 -- ComfyUI on CPU float32 vs MLX float32 under `mx.stream(mx.cpu)`.
The MLX CPU stream is required for the comparison: float32 matmul on the
Metal/GPU stream carries ~1e-2 relative error vs CPU (see
tests/native/anima/test_model.py), which would swamp a real divergence.

A secondary, informational-only number reports MLX bfloat16 on the default
GPU stream against the same ComfyUI float32 reference -- no pass threshold,
expected cosine lower than the fp32/CPU number.

Before trusting the cross-framework number, each side is run twice on the
same weights and inputs and must self-match exactly (diff 0, cosine 1.0) --
if not, the harness itself is broken, not the model.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = Path("/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI")
DEFAULT_CHECKPOINT = "/Volumes/X10Pro/Images/models/diffusion_models/Anima/anime/WaiHassakuAnima.safetensors"

SEED = 0
LATENT_HW = 64
QWEN_LEN = 20
IDS_LEN = 12
SIGMA = 0.8

_INPUT_NAMES = ("x", "qwen", "ids", "weights", "sigma")


def _assert_self_match(diff: float, cos: float, label: str) -> None:
    """Same weights, same inputs, run twice -- expect exact determinism.
    Torch's threaded CPU scaled_dot_product_attention can reduce in a
    slightly different order between calls, so allow float32-noise-floor
    slack (1e-6) instead of requiring bit-identical output."""
    assert diff < 1e-6 and cos > 1.0 - 1e-9, (
        f"{label} harness is non-deterministic beyond float32 noise (diff={diff!r}, cosine={cos!r}) "
        "-- fix before trusting the comparison"
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _relative_l2(ref: np.ndarray, got: np.ndarray) -> float:
    ref64, got64 = ref.astype(np.float64), got.astype(np.float64)
    return float(np.linalg.norm(got64 - ref64) / np.linalg.norm(ref64))


def _per_channel_min_cosine(ref: np.ndarray, got: np.ndarray) -> float:
    """ref/got are [B,C,H,W] -- cosine per output channel, minimum across channels."""
    cosines = [_cosine(ref[:, c], got[:, c]) for c in range(ref.shape[1])]
    return float(min(cosines))


def _make_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return dict(
        x=rng.standard_normal((1, 16, LATENT_HW, LATENT_HW)).astype(np.float32),
        qwen=rng.standard_normal((1, QWEN_LEN, 1024)).astype(np.float32),
        ids=rng.integers(0, 32128, size=(1, IDS_LEN)).astype(np.int64),
        weights=np.ones((1, IDS_LEN), dtype=np.float32),
        sigma=np.array([SIGMA], dtype=np.float32),
    )


def _inputs(scratch: Path) -> dict[str, np.ndarray]:
    """Load shared inputs from --scratch if a previous run already saved
    them, else generate fresh ones (fixed seed) and save them there."""
    if all((scratch / f"input_{n}.npy").exists() for n in _INPUT_NAMES):
        return {n: np.load(scratch / f"input_{n}.npy") for n in _INPUT_NAMES}
    scratch.mkdir(parents=True, exist_ok=True)
    inputs = _make_inputs()
    for n, arr in inputs.items():
        np.save(scratch / f"input_{n}.npy", arr)
    return inputs


def run_comfy(checkpoint: str, scratch: Path, lora_path: str | None = None, strength: float = 1.0) -> None:
    import torch

    sys.path.insert(0, str(COMFYUI_ROOT))
    import comfy.sd  # noqa: E402  (path insert must happen first)
    import comfy.utils  # noqa: E402

    inputs = _inputs(scratch)
    x = torch.from_numpy(inputs["x"])[:, :, None]  # [1,16,64,64] -> [1,16,1,64,64]
    sigma = torch.from_numpy(inputs["sigma"])
    qwen = torch.from_numpy(inputs["qwen"])
    ids = torch.from_numpy(inputs["ids"])
    weights = torch.from_numpy(inputs["weights"])[..., None]  # [1,12] -> [1,12,1]

    def forward(dit) -> np.ndarray:
        dit.eval()
        with torch.no_grad():
            return dit(x, sigma, qwen, t5xxl_ids=ids, t5xxl_weights=weights).numpy()[:, :, 0]

    model = comfy.sd.load_diffusion_model(checkpoint, model_options={"dtype": torch.float32})

    out_a = forward(model.model.diffusion_model)
    out_b = forward(model.model.diffusion_model)
    diff = float(np.max(np.abs(out_a - out_b)))
    cos = _cosine(out_a, out_b)
    print(f"[comfy] harness self-check (base): max abs diff={diff!r}, cosine={cos!r}")
    _assert_self_match(diff, cos, "ComfyUI base")
    np.save(scratch / "out_comfy_fp32.npy", out_a)
    print(f"[comfy] saved fp32 base output -> {scratch / 'out_comfy_fp32.npy'}")

    if lora_path is None:
        return

    lora_sd = comfy.utils.load_torch_file(lora_path)
    patched = comfy.sd.load_lora_for_models(model, None, lora_sd, strength, 0)[0]
    dit_lora = patched.patch_model()  # applies patches in place, returns the patched torch model
    try:
        out_lora_a = forward(dit_lora.diffusion_model)
        out_lora_b = forward(dit_lora.diffusion_model)
    finally:
        patched.unpatch_model()

    diff_lora = float(np.max(np.abs(out_lora_a - out_lora_b)))
    cos_lora = _cosine(out_lora_a, out_lora_b)
    print(f"[comfy] harness self-check (lora): max abs diff={diff_lora!r}, cosine={cos_lora!r}")
    _assert_self_match(diff_lora, cos_lora, "ComfyUI lora")

    # Prove the patch was actually applied: the LoRA-ed output must differ
    # from the unpatched base output on the SAME inputs.
    applied_diff = float(np.max(np.abs(out_lora_a - out_a)))
    print(f"[comfy] LoRA patch applied: max abs diff vs base={applied_diff!r} "
          f"({'OK' if applied_diff > 0 else 'NO-OP -- patch not applied!'})")
    assert applied_diff > 0, "ComfyUI LoRA patch had no effect -- patch_model() did not apply it"

    np.save(scratch / "out_comfy_lora_fp32.npy", out_lora_a)
    print(f"[comfy] saved fp32 lora output -> {scratch / 'out_comfy_lora_fp32.npy'}")


def run_mlx(checkpoint: str, scratch: Path, lora_path: str | None = None, strength: float = 1.0) -> None:
    import mlx.core as mx

    inputs = _inputs(scratch)

    def forward(model, stream) -> np.ndarray:
        with stream:
            ctx = model.encode_context(
                mx.array(inputs["qwen"]), mx.array(inputs["ids"].astype(np.int32)), mx.array(inputs["weights"])
            )
            out = model(mx.array(inputs["x"]), mx.array(inputs["sigma"]), ctx)
            out = out.astype(mx.float32)  # bfloat16 has no numpy dtype; upcast before np.array()
            mx.eval(out)
            return np.array(out)

    cpu = mx.stream(mx.cpu)

    if lora_path is not None:
        # Same comfy-stub loader tests/test_lora_anima_real.py uses, so
        # ASDX_LoraLoader._load_lora_file/_apply_lora_to_transformer see the
        # exact code path a real workflow's LoRA Loader node runs.
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from support.comfy_stub import install_comfy_stubs, load_node_module  # noqa: E402

        install_comfy_stubs()
        lora_mod = load_node_module("lora")
        weight_map = load_node_module("native.anima.weight_map")

        model_fp32 = weight_map.load_anima_checkpoint(checkpoint, dtype="float32")
        out_a = forward(model_fp32, cpu)
        np.save(scratch / "out_mlx_fp32.npy", out_a)
        print(f"[mlx] saved fp32/cpu base output -> {scratch / 'out_mlx_fp32.npy'}")

        lora = lora_mod.ASDX_LoraLoader._load_lora_file(Path(lora_path))
        lora.scale = lora_mod.base_lora_scale(lora.alpha, lora.rank) * strength
        with cpu:
            lora_model = lora_mod.ASDX_LoraLoader._apply_lora_to_transformer(model_fp32, lora, None)
        out_lora_a = forward(lora_model, cpu)
        out_lora_b = forward(lora_model, cpu)

        diff_lora = float(np.max(np.abs(out_lora_a - out_lora_b)))
        cos_lora = _cosine(out_lora_a, out_lora_b)
        print(f"[mlx] harness self-check (lora, fp32/cpu): max abs diff={diff_lora!r}, cosine={cos_lora!r}")
        _assert_self_match(diff_lora, cos_lora, "MLX lora")

        applied_diff = float(np.max(np.abs(out_lora_a - out_a)))
        print(f"[mlx] LoRA patch applied: max abs diff vs base={applied_diff!r} "
              f"({'OK' if applied_diff > 0 else 'NO-OP -- patch not applied!'})")
        assert applied_diff > 0, "MLX LoRA apply had no effect"

        np.save(scratch / "out_mlx_lora_fp32.npy", out_lora_a)
        print(f"[mlx] saved fp32/cpu lora output -> {scratch / 'out_mlx_lora_fp32.npy'}")
        return

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from support.anima_module_loader import load_native_module  # noqa: E402

    weight_map = load_native_module("anima.weight_map")

    model_fp32 = weight_map.load_anima_checkpoint(checkpoint, dtype="float32")
    out_a = forward(model_fp32, cpu)
    out_b = forward(model_fp32, cpu)

    diff = float(np.max(np.abs(out_a - out_b)))
    cos = _cosine(out_a, out_b)
    print(f"[mlx] harness self-check (fp32/cpu): max abs diff={diff!r}, cosine={cos!r}")
    _assert_self_match(diff, cos, "MLX")

    np.save(scratch / "out_mlx_fp32.npy", out_a)
    print(f"[mlx] saved fp32/cpu output -> {scratch / 'out_mlx_fp32.npy'}")

    del model_fp32
    mx.clear_cache()

    model_bf16 = weight_map.load_anima_checkpoint(checkpoint, dtype="bfloat16")
    out_bf16 = forward(model_bf16, mx.stream(mx.gpu))
    np.save(scratch / "out_mlx_bf16.npy", out_bf16)
    print(f"[mlx] saved bf16/gpu output -> {scratch / 'out_mlx_bf16.npy'} (informational only)")

    del model_bf16
    mx.clear_cache()

    model_fp16 = weight_map.load_anima_checkpoint(checkpoint, dtype="float16")
    out_fp16 = forward(model_fp16, mx.stream(mx.gpu))
    np.save(scratch / "out_mlx_fp16.npy", out_fp16)
    print(f"[mlx] saved fp16/gpu output -> {scratch / 'out_mlx_fp16.npy'} (informational only)")


def compare(scratch: Path) -> None:
    ref = np.load(scratch / "out_comfy_fp32.npy")
    got = np.load(scratch / "out_mlx_fp32.npy")
    diff = float(np.max(np.abs(ref - got)))
    cos = _cosine(ref, got)
    status = "PASS" if cos > 0.9999 else "FAIL"
    print(f"[parity] fp32 comfy-vs-mlx: max abs diff={diff:.3e}, cosine={cos:.8f} ({status}, threshold 0.9999)")

    bf16_path = scratch / "out_mlx_bf16.npy"
    if bf16_path.exists():
        bf16 = np.load(bf16_path)
        cos_bf16 = _cosine(ref, bf16)
        min_cos_bf16 = _per_channel_min_cosine(ref, bf16)
        rel_l2_bf16 = _relative_l2(ref, bf16)
        print(
            f"[parity] bf16 comfy-vs-mlx (informational, no threshold): cosine={cos_bf16:.8f}, "
            f"per-channel min cosine={min_cos_bf16:.8f}, relative L2={rel_l2_bf16:.6f}, "
            f"isfinite={bool(np.isfinite(bf16).all())}"
        )

    fp16_path = scratch / "out_mlx_fp16.npy"
    if fp16_path.exists():
        fp16 = np.load(fp16_path)
        finite = bool(np.isfinite(fp16).all())
        if finite:
            cos_fp16 = _cosine(ref, fp16)
            min_cos_fp16 = _per_channel_min_cosine(ref, fp16)
            rel_l2_fp16 = _relative_l2(ref, fp16)
            print(
                f"[parity] fp16 comfy-vs-mlx (informational, no threshold): cosine={cos_fp16:.8f}, "
                f"per-channel min cosine={min_cos_fp16:.8f}, relative L2={rel_l2_fp16:.6f}, isfinite={finite}"
            )
        else:
            print("[parity] fp16 comfy-vs-mlx: output is NOT finite (NaN/Inf present)")


def compare_lora(scratch: Path) -> None:
    """The real check for a LoRA leg: compare the LoRA-INDUCED DIFFERENCE
    `(v_lora - v_base)` between ComfyUI and MLX, not just the full outputs
    -- two frameworks can agree closely on the (dominant) base signal while
    disagreeing on a LoRA effect that is a small fraction of its magnitude.
    Pass: difference cosine > 0.999 and norm ratio within 1%."""
    base_comfy = np.load(scratch / "out_comfy_fp32.npy")
    base_mlx = np.load(scratch / "out_mlx_fp32.npy")
    lora_comfy = np.load(scratch / "out_comfy_lora_fp32.npy")
    lora_mlx = np.load(scratch / "out_mlx_lora_fp32.npy")

    cos_full = _cosine(lora_comfy, lora_mlx)
    print(f"[parity] lora full-output comfy-vs-mlx: cosine={cos_full:.8f}")

    d_comfy = (lora_comfy - base_comfy).astype(np.float64)
    d_mlx = (lora_mlx - base_mlx).astype(np.float64)
    cos_diff = float(np.dot(d_comfy.reshape(-1), d_mlx.reshape(-1))
                      / (np.linalg.norm(d_comfy) * np.linalg.norm(d_mlx)))
    norm_comfy = float(np.linalg.norm(d_comfy))
    norm_mlx = float(np.linalg.norm(d_mlx))
    norm_ratio = norm_mlx / norm_comfy if norm_comfy else float("nan")
    within_1pct = abs(norm_ratio - 1.0) <= 0.01
    status = "PASS" if (cos_diff > 0.999 and within_1pct) else "FAIL"
    print(
        f"[parity] lora-induced-diff comfy-vs-mlx: cosine={cos_diff:.8f} (threshold >0.999), "
        f"norm ratio (mlx/comfy)={norm_ratio:.6f} (threshold within 1%), "
        f"|comfy diff|={norm_comfy:.6e}, |mlx diff|={norm_mlx:.6e} ({status})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scratch", default=str(Path(tempfile.gettempdir()) / "anima_parity"))
    parser.add_argument("--only", choices=["comfy", "mlx"], default=None,
                         help="Run only one half (use when the other interpreter lacks torch+comfy or mlx).")
    parser.add_argument("--lora", default=None, help="LoRA .safetensors path -- run the LoRA leg instead of the base-model-only leg.")
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    scratch = Path(args.scratch)

    if args.only in (None, "comfy"):
        run_comfy(args.checkpoint, scratch, args.lora, args.strength)
    if args.only in (None, "mlx"):
        run_mlx(args.checkpoint, scratch, args.lora, args.strength)

    if args.lora is not None:
        needed = ("out_comfy_fp32.npy", "out_mlx_fp32.npy", "out_comfy_lora_fp32.npy", "out_mlx_lora_fp32.npy")
        if all((scratch / n).exists() for n in needed):
            compare_lora(scratch)
        else:
            print(f"[parity] only one half has run so far; rerun with --only <the other half> --scratch {scratch} --lora {args.lora} --strength {args.strength}")
        return

    if (scratch / "out_comfy_fp32.npy").exists() and (scratch / "out_mlx_fp32.npy").exists():
        compare(scratch)
    else:
        print(f"[parity] only one half has run so far; rerun with --only <the other half> --scratch {scratch}")


if __name__ == "__main__":
    main()
