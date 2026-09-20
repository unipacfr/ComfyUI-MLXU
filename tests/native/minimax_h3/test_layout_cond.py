"""PackedLayout with keyframe/reference segments vs the real ComfyUI PackedLayout."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support import minimax_h3_dit_reference as R
from support.comfyui_reference_loader import load_real_comfy_minimax_model
from support.minimax_h3_module_loader import load_native_module

layout_mod = load_native_module("minimax_h3.layout")

TARGET = dict(latent_t=2, latent_h=4, latent_w=4, audio_t=3)


@pytest.mark.parametrize("name", list(R.CASES))
def test_layout_matches_comfyui(name):
    mm = load_real_comfy_minimax_model()
    ours, theirs = R.payload_pair(name)
    ref = mm.PackedLayout(
        R.TEXT_LEN, TARGET["latent_t"], TARGET["latent_h"], TARGET["latent_w"], TARGET["audio_t"],
        keyframes=theirs.get("keyframes"), refs=theirs.get("refs"),
    )
    mine = layout_mod.PackedLayout(
        R.TEXT_LEN, TARGET["latent_t"], TARGET["latent_h"], TARGET["latent_w"], TARGET["audio_t"],
        keyframes=ours.keyframes, refs=ours.refs,
    )
    assert mine.seq_len == ref.seq_len
    assert mine.segments == [(a, b, k) for a, b, k in ref.segments]
    assert np.abs(np.array(mine.position_ids) - ref.position_ids.numpy()).max() < 1e-9


def test_target_streams_stay_last():
    ours, _ = R.payload_pair("combined")
    mine = layout_mod.PackedLayout(R.TEXT_LEN, 2, 4, 4, 3, keyframes=ours.keyframes, refs=ours.refs)
    kinds = [k for _, _, k in mine.segments]
    assert kinds[:1] == ["text"] and kinds[-2:] == ["audio", "video"]
    assert set(kinds) >= {"cond", "cond_audio", "ref_img", "ref_audio"}


def test_ref_t_span():
    RB = load_native_module("minimax_h3.condition").RefBlock
    assert layout_mod.ref_t_span(RB(kind="image")) == 1.0
    assert layout_mod.ref_t_span(RB(kind="audio", ref_audio_t=4)) == 4.0
    video = layout_mod.ref_t_span(RB(kind="video_audio", latent_t=2, ref_audio_t=3))
    assert video == max(3.0, sum(layout_mod.video_t_spans(2)))


def test_ref_video_with_other_resolution_matches_comfyui():
    """Reference audio takes its w extremes from the reference frame grid, not the target's
    (identical only when both share a resolution, as in R.CASES)."""
    mm = load_real_comfy_minimax_model()
    RB = load_native_module("minimax_h3.condition").RefBlock
    ref = mm.PackedLayout(
        R.TEXT_LEN, 2, 4, 4, 3,
        refs=[dict(kind="video_audio", latent_t=2, latent_h=8, latent_w=4, ref_audio_t=3)],
    )
    mine = layout_mod.PackedLayout(
        R.TEXT_LEN, 2, 4, 4, 3,
        refs=[RB(kind="video_audio", latent_t=2, latent_h=8, latent_w=4, ref_audio_t=3)],
    )
    assert mine.segments == [(a, b, k) for a, b, k in ref.segments]
    assert np.abs(np.array(mine.position_ids) - ref.position_ids.numpy()).max() < 1e-9


# ---- fail-closed cross-checks between the condition payload and the layout ----

_COND = load_native_module("minimax_h3.condition")


def _z(*shape):
    import mlx.core as mx

    return mx.zeros(shape)


def _build(keyframes=(), refs=()):
    return layout_mod.PackedLayout(R.TEXT_LEN, 2, 4, 4, 3, keyframes=keyframes, refs=refs)


def test_image_ref_latent_grid_must_match_declared_dims():
    blk = _COND.RefBlock(kind="image", latent=_z(1, 4, 1, 8, 8), latent_t=1, latent_h=4, latent_w=4)
    with pytest.raises(ValueError, match=r"ASDX.*reference block 1.*latent"):
        _build(refs=[_COND.RefBlock(kind="image"), blk])


def test_audio_ref_latent_length_must_match_ref_audio_t():
    blk = _COND.RefBlock(kind="audio", audio_latent=_z(1, 6, 2, 3), ref_audio_t=2)
    with pytest.raises(ValueError, match=r"ASDX.*reference block 0.*audio"):
        _build(refs=[blk])


def test_audio_ref_with_latent_but_zero_ref_audio_t_is_rejected():
    first = _COND.RefBlock(kind="audio", audio_latent=_z(1, 6, 2, 3), ref_audio_t=0)
    second = _COND.RefBlock(kind="audio", audio_latent=_z(1, 6, 2, 2), ref_audio_t=2)
    with pytest.raises(ValueError, match=r"ASDX.*reference block 0.*audio"):
        _build(refs=[first, second])


def test_empty_audio_latent_is_rejected():
    blk = _COND.RefBlock(kind="audio", audio_latent=_z(1, 6, 2, 0), ref_audio_t=0)
    with pytest.raises(ValueError, match=r"ASDX.*reference block 0.*audio"):
        _build(refs=[blk])


def test_unknown_ref_kind_is_rejected():
    with pytest.raises(ValueError, match=r"ASDX.*reference block 1.*kind 'imgae'"):
        _build(refs=[_COND.RefBlock(kind="image"), _COND.RefBlock(kind="imgae")])
    with pytest.raises(ValueError, match=r"ASDX.*kind 'imgae'"):
        layout_mod.ref_t_span(_COND.RefBlock(kind="imgae"))


def test_keyframe_latent_must_share_the_target_grid():
    kf = _COND.KeyframeCond(resolved_frame_index=0, latent=_z(1, 4, 1, 4, 8))
    with pytest.raises(ValueError, match=r"ASDX.*keyframe 1.*grid"):
        _build(keyframes=[_COND.KeyframeCond(0), kf])


def test_consistent_and_latentless_blocks_still_build():
    refs = [
        _COND.RefBlock(kind="image", latent=_z(1, 4, 1, 8, 4), latent_t=1, latent_h=8, latent_w=4),
        _COND.RefBlock(kind="audio", audio_latent=_z(1, 6, 2, 2), ref_audio_t=2),
        _COND.RefBlock(kind="video_audio", latent_t=2, latent_h=4, latent_w=4, ref_audio_t=3),  # text-only
    ]
    kfs = [_COND.KeyframeCond(0, latent=_z(1, 4, 1, 4, 4)), _COND.KeyframeCond(3)]
    assert _build(keyframes=kfs, refs=refs).seq_len > 0
