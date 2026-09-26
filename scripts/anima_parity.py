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


def run_comfy(checkpoint: str, scratch: Path) -> None:
    import torch

    sys.path.insert(0, str(COMFYUI_ROOT))
    import comfy.sd  # noqa: E402  (path insert must happen first)

    inputs = _inputs(scratch)
    x = torch.from_numpy(inputs["x"])[:, :, None]  # [1,16,64,64] -> [1,16,1,64,64]
    sigma = torch.from_numpy(inputs["sigma"])
    qwen = torch.from_numpy(inputs["qwen"])
    ids = torch.from_numpy(inputs["ids"])
    weights = torch.from_numpy(inputs["weights"])[..., None]  # [1,12] -> [1,12,1]

    model = comfy.sd.load_diffusion_model(checkpoint, model_options={"dtype": torch.float32})
    dit = model.model.diffusion_model
    dit.eval()

    with torch.no_grad():
        out_a = dit(x, sigma, qwen, t5xxl_ids=ids, t5xxl_weights=weights).numpy()[:, :, 0]
        out_b = dit(x, sigma, qwen, t5xxl_ids=ids, t5xxl_weights=weights).numpy()[:, :, 0]

    diff = float(np.max(np.abs(out_a - out_b)))
    cos = _cosine(out_a, out_b)
    print(f"[comfy] harness self-check: max abs diff={diff!r}, cosine={cos!r}")
    _assert_self_match(diff, cos, "ComfyUI")

    np.save(scratch / "out_comfy_fp32.npy", out_a)
    print(f"[comfy] saved fp32 output -> {scratch / 'out_comfy_fp32.npy'}")


def run_mlx(checkpoint: str, scratch: Path) -> None:
    import mlx.core as mx

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from support.anima_module_loader import load_native_module  # noqa: E402

    weight_map = load_native_module("anima.weight_map")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scratch", default=str(Path(tempfile.gettempdir()) / "anima_parity"))
    parser.add_argument("--only", choices=["comfy", "mlx"], default=None,
                         help="Run only one half (use when the other interpreter lacks torch+comfy or mlx).")
    args = parser.parse_args()
    scratch = Path(args.scratch)

    if args.only in (None, "comfy"):
        run_comfy(args.checkpoint, scratch)
    if args.only in (None, "mlx"):
        run_mlx(args.checkpoint, scratch)

    if (scratch / "out_comfy_fp32.npy").exists() and (scratch / "out_mlx_fp32.npy").exists():
        compare(scratch)
    else:
        print(f"[parity] only one half has run so far; rerun with --only <the other half> --scratch {scratch}")


if __name__ == "__main__":
    main()
