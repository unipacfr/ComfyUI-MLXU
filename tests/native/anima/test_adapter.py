from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.anima_module_loader import load_native_module
from support.comfyui_reference_loader import load_real_comfy_anima

adapter_mod = load_native_module("anima.adapter")
config_mod = load_native_module("anima.config")

# MLX fp32 matmul on the GPU carries ~1e-2 relative error vs torch CPU float32 (measured:
# a lone 1024x1024 float32 matmul already diverges by ~0.01). Parity tests therefore run on
# the MLX CPU stream, matching the established pattern in
# tests/native/minimax_h3/test_vision_tower.py.
CPU = mx.stream(mx.cpu)


def _copy_torch_to_mlx(torch_module, mlx_module):
    from mlx.utils import tree_flatten, tree_unflatten
    sd = {k: mx.array(v.detach().float().numpy()) for k, v in torch_module.state_dict().items()}
    names = [k for k, _ in tree_flatten(mlx_module.parameters())]
    assert sorted(names) == sorted(sd), set(names) ^ set(sd)
    mlx_module.update(tree_unflatten([(k, sd[k]) for k in names]))


def _init_ref(ref, seed=0):
    """N(1, 0.02) for norm weights (so RMSNorm isn't ~0), N(0, 0.08) for
    everything else (so the reference output std is well above the noise
    floor -- see task-2 ruling 1)."""
    import torch
    torch.manual_seed(seed)
    for name, p in ref.named_parameters():
        if "norm" in name:
            torch.nn.init.normal_(p, mean=1.0, std=0.02)
        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.08)


@pytest.mark.parametrize("target_len", [5, 600])
def test_adapter_matches_comfy(target_len):
    import torch
    anima_model, ops = load_real_comfy_anima()
    ref = anima_model.LLMAdapter(operations=ops.disable_weight_init, dtype=torch.float32)
    _init_ref(ref)
    ours = adapter_mod.LLMAdapter(config_mod.AnimaConfig(dtype="float32"))
    _copy_torch_to_mlx(ref, ours)

    torch.manual_seed(1)
    src = torch.randn(1, 9, 1024)
    ids = torch.randint(0, 32128, (1, target_len))
    with torch.no_grad():
        want = ref(src, ids).numpy()

    assert want.std() > 1e-3, f"reference output std too small to be a meaningful check: {want.std()}"

    with CPU:
        got = np.array(ours(mx.array(src.numpy()), mx.array(ids.numpy().astype(np.int32))))
    assert got.shape == (1, target_len, 1024)  # no truncation for > 512 tokens
    max_abs_diff = np.max(np.abs(got - want))
    np.testing.assert_allclose(got, want, atol=2e-4, rtol=2e-4), f"max abs diff {max_abs_diff}"


def test_perturbed_weight_is_detected():
    """Sanity check that the parity comparison above can actually detect a
    divergence (task-2 ruling 1) -- perturb one MLX out_proj weight and
    confirm the comparison then fails."""
    import torch
    anima_model, ops = load_real_comfy_anima()
    ref = anima_model.LLMAdapter(operations=ops.disable_weight_init, dtype=torch.float32)
    _init_ref(ref)
    ours = adapter_mod.LLMAdapter(config_mod.AnimaConfig(dtype="float32"))
    _copy_torch_to_mlx(ref, ours)

    torch.manual_seed(2)
    src = torch.randn(1, 9, 1024)
    ids = torch.randint(0, 32128, (1, 5))
    with torch.no_grad():
        want = ref(src, ids).numpy()

    ours.out_proj.weight = ours.out_proj.weight.at[0, 0].add(0.1)
    with CPU:
        got = np.array(ours(mx.array(src.numpy()), mx.array(ids.numpy().astype(np.int32))))

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(got, want, atol=2e-4, rtol=2e-4)
