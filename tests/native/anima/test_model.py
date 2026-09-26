from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.anima_module_loader import load_native_module
from support.comfyui_reference_loader import load_real_comfy_anima

model_mod = load_native_module("anima.model")
config_mod = load_native_module("anima.config")

WIDTH, HEADS, BLOCKS = 256, 2, 2

# MLX fp32 matmul on the GPU carries ~1e-2 relative error vs torch CPU float32 (measured in
# tests/native/minimax_h3/test_vision_tower.py). Parity tests therefore run on the MLX CPU
# stream, matching the established pattern used by tests/native/anima/test_adapter.py.
CPU = mx.stream(mx.cpu)


def _ref_model(torch, anima_model, ops):
    return anima_model.Anima(
        max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=WIDTH, num_blocks=BLOCKS, num_heads=HEADS, crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=256, extra_per_block_abs_pos_emb=False,
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
        image_model="anima", operations=ops.disable_weight_init, dtype=torch.float32,
    )


def _ours():
    cfg = config_mod.AnimaConfig(dtype="float32", model_channels=WIDTH, num_blocks=BLOCKS, num_heads=HEADS)
    return model_mod.AnimaTransformer(cfg)


def _init_ref(ref, seed=0):
    """N(1, 0.02) for norm weights (RMSNorm stays near-identity), N(0, 0.08) for
    everything else -- reference output std must stay well above the noise floor
    (task-2 ruling 1)."""
    import torch
    torch.manual_seed(seed)
    for name, p in ref.named_parameters():
        if "norm" in name:
            torch.nn.init.normal_(p, mean=1.0, std=0.02)
        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.08)


def _copy(ref, ours):
    from mlx.utils import tree_flatten, tree_unflatten
    sd = {k: mx.array(v.detach().float().numpy()) for k, v in ref.state_dict().items()}
    names = [k for k, _ in tree_flatten(ours.parameters())]
    assert sorted(names) == sorted(sd), set(names) ^ set(sd)
    ours.update(tree_unflatten([(k, sd[k]) for k in names]))


@pytest.mark.parametrize("hw", [(8, 6), (12, 10)])
def test_forward_matches_comfy(hw):
    import torch
    anima_model, ops = load_real_comfy_anima()
    ref = _ref_model(torch, anima_model, ops)
    _init_ref(ref)
    ours = _ours()
    _copy(ref, ours)

    h, w = hw
    torch.manual_seed(1)
    x = torch.randn(1, 16, 1, h, w)
    sigma = torch.tensor([0.7])
    qwen = torch.randn(1, 9, 1024)
    ids = torch.randint(0, 32128, (1, 6))
    weights = torch.rand(6) + 0.5
    with torch.no_grad():
        want = ref(x, sigma, qwen, t5xxl_ids=ids, t5xxl_weights=weights.view(1, -1, 1)).numpy()[:, :, 0]

    assert want.std() > 1e-3, f"reference output std too small to be a meaningful check: {want.std()}"

    with CPU:
        ctx = ours.encode_context(mx.array(qwen.numpy()), mx.array(ids.numpy().astype(np.int32)),
                                  mx.array(weights.numpy()[None]))
        assert ctx.shape == (1, 512, 1024)
        got = np.array(ours(mx.array(x.numpy()[:, :, 0]), mx.array(sigma.numpy()), ctx))
    assert got.shape == want.shape
    max_abs_diff = np.max(np.abs(got - want))
    np.testing.assert_allclose(got, want, atol=5e-4, rtol=5e-4), f"max abs diff {max_abs_diff}"


def test_perturbed_weight_is_detected():
    """Sanity check that the parity comparison above can actually detect a
    divergence (task-2 ruling 1) -- perturb one MLX weight and confirm the
    comparison then fails."""
    import torch
    anima_model, ops = load_real_comfy_anima()
    ref = _ref_model(torch, anima_model, ops)
    _init_ref(ref, seed=2)
    ours = _ours()
    _copy(ref, ours)

    torch.manual_seed(3)
    x = torch.randn(1, 16, 1, 8, 6)
    sigma = torch.tensor([0.3])
    qwen = torch.randn(1, 9, 1024)
    ids = torch.randint(0, 32128, (1, 6))
    with torch.no_grad():
        want = ref(x, sigma, qwen, t5xxl_ids=ids).numpy()[:, :, 0]

    ours.blocks[0].adaln_modulation_self_attn[2].weight = (
        ours.blocks[0].adaln_modulation_self_attn[2].weight.at[0, 0].add(0.5)
    )
    with CPU:
        ctx = ours.encode_context(mx.array(qwen.numpy()), mx.array(ids.numpy().astype(np.int32)))
        got = np.array(ours(mx.array(x.numpy()[:, :, 0]), mx.array(sigma.numpy()), ctx))

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(got, want, atol=5e-4, rtol=5e-4)


def test_rejects_odd_latent_grid():
    ours = _ours()
    with pytest.raises(ValueError, match="even"):
        ours(mx.zeros((1, 16, 7, 6)), mx.array([0.5]), mx.zeros((1, 512, 1024)))
