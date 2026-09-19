"""VisionTower parity vs the real ComfyUI Qwen3VLVisionModel on a tiny
random-weight config (architecture, not weights), plus a metric sanity check."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_qwen3vl
from support.minimax_h3_module_loader import load_native_module

tower_mod = load_native_module("minimax_h3.vision_tower")
map_mod = load_native_module("minimax_h3.vision_weight_map")

TINY = dict(
    hidden_size=64, intermediate_size=128, depth=4, num_heads=4, patch_size=4,
    temporal_patch_size=2, in_channels=3, spatial_merge_size=2,
    num_position_embeddings=16, deepstack_visual_indexes=(0, 1, 2), out_hidden_size=32,
)
# MLX fp32 matmul on the GPU carries ~1e-3 absolute error vs float64 (measured: patch-embed
# alone is off by 3e-3 vs a float64 reference, while torch CPU is off by 4e-7). Parity tests
# therefore run on the MLX CPU stream, where the architecture matches ComfyUI to ~5e-7.
CPU = mx.stream(mx.cpu)
GRIDS = [(1, 4, 6), (1, 2, 4), (2, 4, 6)]  # two images plus a t>1 grid: exercises per-image spans and temporal frames


def _reference(seed: int = 0):
    import torch

    qwen3vl, _, ops = load_real_comfy_qwen3vl()
    cfg = dict(TINY, deepstack_visual_indexes=list(TINY["deepstack_visual_indexes"]))
    ref = qwen3vl.Qwen3VLVisionModel(cfg, device="cpu", dtype=torch.float32, ops=ops.disable_weight_init)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ref.parameters():  # disable_weight_init leaves weights uninitialized
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
    return ref


def _patches(seed: int = 1):
    n = sum(t * h * w for t, h, w in GRIDS)
    cfg = tower_mod.VisionConfig(**TINY)
    return np.random.default_rng(seed).standard_normal((n, cfg.patch_dim)).astype(np.float32)


def _mlx_from_ref(ref):
    model = tower_mod.VisionTower(tower_mod.VisionConfig(**TINY))
    tensors = {k: mx.array(v.detach().numpy()) for k, v in ref.state_dict().items()}
    map_mod.assign_vision_weights(model, tensors)
    return model


def test_matches_comfyui_on_random_weights():
    import torch

    ref = _reference()
    patches = _patches()
    with torch.no_grad():
        ref_merged, ref_ds = ref(torch.from_numpy(patches), torch.tensor(GRIDS))
    with CPU:
        model = _mlx_from_ref(ref)
        merged, ds = model(mx.array(patches), GRIDS)
        mx.eval(merged, ds)

    assert merged.shape == tuple(ref_merged.shape)
    assert len(ds) == len(ref_ds) == 3
    assert np.abs(np.array(merged) - ref_merged.numpy()).max() < 1e-3
    for got, want in zip(ds, ref_ds):
        assert np.abs(np.array(got) - want.numpy()).max() < 1e-3


def test_metric_separates_different_weights():
    """Null case: the parity metric must see a changed weight (else it measures nothing)."""
    import torch

    ref = _reference()
    patches = _patches()
    with torch.no_grad():
        ref_merged, _ = ref(torch.from_numpy(patches), torch.tensor(GRIDS))
    with CPU:
        other = _mlx_from_ref(_reference(seed=99))
        merged, _ = other(mx.array(patches), GRIDS)
        mx.eval(merged)
    assert np.abs(np.array(merged) - ref_merged.numpy()).max() > 1e-2


def test_output_row_counts_follow_merge():
    model = tower_mod.VisionTower(tower_mod.VisionConfig(**TINY))
    merged, ds = model(mx.array(_patches()), GRIDS)
    total = sum(t * h * w for t, h, w in GRIDS) // 4
    assert merged.shape[0] == total and all(d.shape[0] == total for d in ds)


def _seeded_tower(seed: int):
    """Tower with deterministic numpy weights (no ComfyUI needed)."""
    from mlx.utils import tree_flatten

    model = tower_mod.VisionTower(tower_mod.VisionConfig(**TINY))
    rng = np.random.default_rng(seed)
    tensors = {
        k: mx.array((rng.standard_normal(v.shape) * 0.1).astype(np.float32))
        for k, v in tree_flatten(model.parameters())
    }
    map_mod.assign_vision_weights(model, tensors)
    return model


def _run(model):
    merged, ds = model(mx.array(_patches()), GRIDS)
    mx.eval(merged, ds)
    return [np.array(merged)] + [np.array(d) for d in ds]


def test_gpu_stream_agrees_with_cpu_stream():
    """The default (GPU) stream must track the CPU stream within a relative bound,
    and the bound must be far below the gap to an unrelated tower (null case)."""
    model = _seeded_tower(0)
    gpu = _run(model)
    with CPU:
        cpu = _run(model)
        other = _run(_seeded_tower(1))
    bound = 1e-2
    for g, c, o in zip(gpu, cpu, other):
        scale = np.abs(c).max()
        assert np.abs(g - c).max() / scale < bound
        assert np.abs(o - c).max() / scale > 10 * bound  # null case: would fail if GPU output were unrelated
