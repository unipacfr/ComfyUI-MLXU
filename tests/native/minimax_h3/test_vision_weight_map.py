"""assign_vision_weights / verify_vision_shapes / load_vision_tower."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_qwen3vl
from support.minimax_h3_module_loader import load_native_module

tower_mod = load_native_module("minimax_h3.vision_tower")
map_mod = load_native_module("minimax_h3.vision_weight_map")
source_mod = load_native_module("minimax_h3.checkpoint_source")
pre_mod = load_native_module("minimax_h3.vision_preprocess")

TINY = tower_mod.VisionConfig(
    hidden_size=64, intermediate_size=128, depth=2, num_heads=4, patch_size=4,
    num_position_embeddings=16, deepstack_visual_indexes=(0,), out_hidden_size=32,
)


def _tensors_for(model, patch_5d: bool = False):
    flat = dict(tree_flatten(model.parameters()))
    out = {k: mx.array(np.random.default_rng(0).standard_normal(v.shape).astype(np.float32)) for k, v in flat.items()}
    if patch_5d:  # checkpoint stores the Conv3d weight 5-D
        c = model.config
        out["patch_embed.proj.weight"] = out["patch_embed.proj.weight"].reshape(
            c.hidden_size, c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size
        )
    return out


def test_assign_reshapes_5d_and_gguf_4d_patch_embed():
    c = TINY
    for shape in [
        (c.hidden_size, c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size),
        (c.hidden_size * c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size),  # GGUF layout
    ]:
        model = tower_mod.VisionTower(c)
        tensors = _tensors_for(model)
        tensors["patch_embed.proj.weight"] = tensors["patch_embed.proj.weight"].reshape(shape)
        assert map_mod.assign_vision_weights(model, tensors) == len(dict(tree_flatten(model.parameters())))
        got = dict(tree_flatten(model.parameters()))["patch_embed.proj.weight"]
        assert got.shape == (c.hidden_size, c.patch_dim)


def test_assign_raises_on_missing_key():
    model = tower_mod.VisionTower(TINY)
    tensors = _tensors_for(model)
    del tensors["blocks.1.attn.qkv.bias"]
    with pytest.raises(KeyError, match="blocks.1.attn.qkv.bias"):
        map_mod.assign_vision_weights(model, tensors)


def test_assign_raises_on_wrong_size():
    model = tower_mod.VisionTower(TINY)
    tensors = _tensors_for(model)
    tensors["pos_embed.weight"] = mx.zeros((3, 3))
    with pytest.raises(ValueError, match="pos_embed.weight"):
        map_mod.assign_vision_weights(model, tensors)


def test_assign_raises_on_same_size_wrong_layout():
    model = tower_mod.VisionTower(TINY)
    tensors = _tensors_for(model)
    tensors["blocks.0.attn.qkv.weight"] = tensors["blocks.0.attn.qkv.weight"].T
    with pytest.raises(ValueError, match="blocks.0.attn.qkv.weight"):
        map_mod.assign_vision_weights(model, tensors)


def test_verify_shapes_accepts_real_dims_and_rejects_other():
    cfg = tower_mod.VisionConfig()
    real = {
        "patch_embed.proj.weight": (1152, 3, 2, 16, 16), "pos_embed.weight": (2304, 1152),
        "merger.linear_fc2.weight": (5120, 4608), "blocks.0.mlp.linear_fc1.weight": (4304, 1152),
        **{f"blocks.{i}.norm1.weight": (1152,) for i in range(27)},
        **{f"deepstack_merger_list.{i}.norm.weight": (4608,) for i in range(3)},
    }
    map_mod.verify_vision_shapes(real, cfg)
    with pytest.raises(ValueError, match="blocks"):
        map_mod.verify_vision_shapes({k: v for k, v in real.items() if k != "blocks.26.norm1.weight"}, cfg)


_ST = Path("/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
_GGUF = Path("/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf")
_FULL = pytest.mark.skipif(os.environ.get("ASDX_FULL_GGUF_TEST") != "1", reason="reads real checkpoints; set ASDX_FULL_GGUF_TEST=1")


@_FULL
def test_gguf_and_safetensors_visual_weights_are_identical():
    if not (_ST.exists() and _GGUF.exists()):
        pytest.skip("real TE checkpoints not present")
    st, gg = source_mod.open_checkpoint(_ST), source_mod.open_checkpoint(_GGUF)
    names = [n for n in st.shapes() if n.startswith("visual.")]
    assert len(names) == 351 and set(names) == {n for n in gg.shapes() if n.startswith("visual.")}
    for name in names:
        a, b = st.get(name), gg.get(name)
        assert a.size == b.size, name
        assert bool(mx.array_equal(a.reshape(-1), b.reshape(-1))), name


@_FULL
def test_real_weights_match_comfyui_on_a_real_image():
    if not _ST.exists():
        pytest.skip("real TE checkpoint not present")
    import torch
    from safetensors import safe_open

    qwen3vl, _, ops = load_real_comfy_qwen3vl()
    cfg = {**qwen3vl.QWEN3VL_VISION_COMMON, **qwen3vl.QWEN3VL_VISION["qwen3vl_32b"], "out_hidden_size": 5120}
    ref = qwen3vl.Qwen3VLVisionModel(cfg, device="cpu", dtype=torch.float32, ops=ops.disable_weight_init)
    with safe_open(str(_ST), framework="pt") as f:
        state = {k[len("visual."):]: f.get_tensor(k).float() for k in f.keys() if k.startswith("visual.")}
    ref.load_state_dict(state, strict=True)

    image = np.random.default_rng(3).random((224, 336, 3), dtype=np.float32)
    patches, grid = pre_mod.preprocess_image(image)
    with torch.no_grad():
        ref_merged, ref_ds = ref(torch.from_numpy(patches), torch.tensor([grid]))
    model = map_mod.load_vision_tower(_ST)
    merged, ds = model(mx.array(patches), [grid])  # default (GPU) stream, as in production
    mx.eval(merged, ds)

    def cos(a, b):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    values = [cos(merged, ref_merged.numpy())] + [cos(g, w.numpy()) for g, w in zip(ds, ref_ds)]
    print(f"\nREAL-WEIGHTS COSINE (gpu): merged={values[0]:.7f} deepstack={[f'{v:.7f}' for v in values[1:]]}")
    assert len(values) == 4
    assert all(v > 0.9999 for v in values), values
