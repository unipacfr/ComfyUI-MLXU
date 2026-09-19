"""Unfolding the MiniMax H3 presentation into ids + vision rows, token tags, and the
end-to-end encode (tiny random-weight tower + encoder; real weights behind ASDX_FULL_GGUF_TEST=1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.comfyui_reference_loader import load_real_comfy_text_encoders
from support.minimax_h3_module_loader import load_native_module

cond = load_native_module("minimax_h3.vision_conditioning")
enc_mod = load_native_module("minimax_h3.text_encoder")
config_mod = load_native_module("minimax_h3.text_encoder_config")
tower_mod = load_native_module("minimax_h3.vision_tower")

VISION_START, VISION_END = 151652, 151653
TINY_TOWER = dict(
    hidden_size=64, intermediate_size=128, depth=4, num_heads=4, patch_size=16,
    temporal_patch_size=2, in_channels=3, spatial_merge_size=2,
    num_position_embeddings=16, deepstack_visual_indexes=(0, 1, 2), out_hidden_size=32,
)


def _seed(module, seed):
    rng = np.random.default_rng(seed)
    flat = dict(tree_flatten(module.parameters()))
    module.update(tree_unflatten([(k, mx.array((rng.standard_normal(v.shape) * 0.1).astype(np.float32))) for k, v in flat.items()]))
    return module


def _tower(seed=0):
    return _seed(tower_mod.VisionTower(tower_mod.VisionConfig(**TINY_TOWER)), seed)


def _encoder(seed=1):
    cfg = config_mod.Qwen3TextEncoderConfig(
        vocab_size=200000, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, dtype="float32",
    )
    return _seed(enc_mod.Qwen3TextEncoder(cfg), seed)


def _image(seed, h=64, w=96):
    return np.random.default_rng(seed).random((1, h, w, 3), dtype=np.float32)


def _entries(image=None, video=None):
    e = [(9707, 1.0), (11, 1.0)]
    if image is not None:
        e += [(VISION_START, 1.0), ({"type": "image", "data": image, "original_type": "image"}, 1.0), (VISION_END, 1.0)]
    if video is not None:
        e += [(VISION_START, 1.0),
              ({"type": "image", "data": video, "original_type": "image", "minimax_video_block": True}, 1.0),
              (VISION_END, 1.0)]
    return e + [(1879, 1.0), (0, 1.0)]


def test_text_only_entries_give_no_vision():
    ids, vision, tags = cond.build_vision_conditioning(_entries(), _tower())
    assert vision is None
    assert ids.tolist() == [9707, 11, 1879, 0] and tags.tolist() == [1, 1, 1, 1]


def test_image_is_unfolded_between_the_markers():
    ids, vision, tags = cond.build_vision_conditioning(_entries(image=_image(0)), _tower())
    size = (64 // 16) * (96 // 16) // 4  # merged rows: gh*gw/4
    assert ids.shape[0] == 2 + 1 + size + 1 + 2
    assert ids[2] == VISION_START and ids[2 + 1 + size] == VISION_END
    assert vision.row_indices.tolist() == list(range(3, 3 + size))
    assert vision.rows.shape == (size, 32) and len(vision.deepstack) == 3
    assert vision.position_ids.shape == (3, ids.shape[0])
    # whole block incl. the flanking markers carries the video tag 0
    assert tags.tolist() == [1, 1] + [0] * (1 + size + 1) + [1, 1]


def test_token_tags_match_comfyui():
    minimax, _, _, _ = load_real_comfy_text_encoders()
    infos = [dict(index=3, size=4, grid=(1, 4, 4)), dict(index=12, size=2, grid=(1, 2, 4)), dict(index=0, size=1, grid=(1, 2, 2))]
    ref = minimax.token_tags_from_embeds_info(
        20, [{"type": "image", "index": i["index"], "size": i["size"]} for i in infos]
    )
    assert np.array_equal(cond.token_tags(20, infos), ref.numpy())


def test_video_block_and_image_rows_are_ordered():
    ids, vision, _ = cond.build_vision_conditioning(
        _entries(image=_image(0), video=np.concatenate([_image(1, 48, 48), _image(2, 48, 48)])), _tower()
    )
    assert np.all(np.diff(vision.row_indices) > 0)
    assert vision.rows.shape[0] == len(vision.row_indices) == vision.deepstack[0].shape[0]


def test_encode_is_finite_and_vision_matters():
    tower, enc = _tower(), _encoder()
    with_img, tags = cond.encode_with_vision(enc, tower, _entries(image=_image(0)), rope_dims=(2, 1, 1))
    other, _ = cond.encode_with_vision(enc, tower, _entries(image=_image(5)), rope_dims=(2, 1, 1))
    mx.eval(with_img, other)
    assert bool(mx.all(mx.isfinite(with_img)).item())
    assert with_img.shape[0] == tags.shape[0]
    assert np.abs(np.array(with_img) - np.array(other)).max() > 1e-3  # null case: a different image must change it


def test_image_without_a_tower_is_an_error():
    with pytest.raises(ValueError, match="vision tower"):
        cond.build_vision_conditioning(_entries(image=_image(0)), None)


def test_real_tokenizer_presentation_is_unfolded():
    minimax, _, _, _ = load_real_comfy_text_encoders()
    import torch

    try:
        tokenizer = minimax.MiniMaxH3Tokenizer()
    except Exception as e:  # tokenizer files unavailable
        pytest.skip(f"MiniMaxH3Tokenizer not constructible here: {e}")
    img = torch.from_numpy(_image(0))
    entries = tokenizer.tokenize_with_weights("a cat", images=[img])["qwen3vl_32b"][0]
    ids, vision, tags = cond.build_vision_conditioning(entries, _tower())
    assert vision is not None and int(tags.sum()) < len(tags)
    assert ids[np.where(tags == 0)[0][0]] == VISION_START


_ST = Path("/Volumes/X10Pro/Images/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")


@pytest.mark.skipif(os.environ.get("ASDX_FULL_GGUF_TEST") != "1", reason="loads the real 26GB text encoder; set ASDX_FULL_GGUF_TEST=1")
def test_real_text_encoder_with_a_real_image():
    if not _ST.exists():
        pytest.skip("real TE checkpoint not present")
    minimax, _, _, _ = load_real_comfy_text_encoders()
    import torch

    enc_wm = load_native_module("minimax_h3.text_encoder_weight_map")
    vis_wm = load_native_module("minimax_h3.vision_weight_map")
    tokenizer = minimax.MiniMaxH3Tokenizer()
    encoder = enc_wm.load_qwen3_text_encoder_checkpoint(_ST, dtype="float16")
    tower = vis_wm.load_vision_tower(_ST)
    img = torch.from_numpy(np.random.default_rng(0).random((1, 224, 336, 3), dtype=np.float32))
    mx.reset_peak_memory()
    with_img, tags = cond.encode_with_vision(encoder, tower, tokenizer.tokenize_with_weights("a cat", images=[img])["qwen3vl_32b"][0])
    text_only, _ = cond.encode_with_vision(encoder, tower, tokenizer.tokenize_with_weights("a cat")["qwen3vl_32b"][0])
    mx.eval(with_img, text_only)
    print(f"[real] with image: {with_img.shape}, text only: {text_only.shape}, peak {mx.get_peak_memory() / 1e9:.2f} GB")
    assert bool(mx.all(mx.isfinite(with_img)).item()) and with_img.shape[1] == 5120
    assert with_img.shape[0] == tags.shape[0] > text_only.shape[0]
    # Teeth: the trailing prompt tokens are the same text ids in both runs, so only the image can move them.
    k = text_only.shape[0]
    zeros = np.flatnonzero(tags == 0)
    # one contiguous vision span (begin marker + rows + end marker), followed by exactly the k prompt tokens
    assert zeros.size > 2 and np.array_equal(zeros, np.arange(zeros[0], zeros[-1] + 1)) and zeros[-1] == with_img.shape[0] - k - 1
    assert bool((tags[-k:] == 1).all())
    delta_text = float(mx.abs(with_img[-k:] - text_only[-k:]).max().item())
    img2 = torch.from_numpy(np.random.default_rng(1).random((1, 224, 336, 3), dtype=np.float32))
    other, _ = cond.encode_with_vision(encoder, tower, tokenizer.tokenize_with_weights("a cat", images=[img2])["qwen3vl_32b"][0])
    mx.eval(other)
    assert other.shape == with_img.shape
    delta_img = float(mx.abs(other[-k:] - with_img[-k:]).max().item())
    print(f"[real] max|with_img - text_only| on trailing {k} rows: {delta_text:.4f}; image1 vs image2: {delta_img:.4f}; max|text_only| {float(mx.abs(text_only).max().item()):.4f}")
    assert delta_text > 1000.0 and delta_img > 0.1  # measured 15806 and 1.30
