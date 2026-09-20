"""Geometry helpers of minimax_h3_conditioning, checked against the real ComfyUI
`comfy_extras.nodes_minimax_h3` values (skipped cleanly when it is not importable)
and against hand-computed cases from that source."""

from __future__ import annotations

import importlib

import pytest
import torch

from tests.support.comfy_stub import install_comfy_stubs, load_node_module
from tests.support.comfyui_reference_loader import assert_no_comfyui_leak, real_comfy_isolated

install_comfy_stubs()
geo = load_node_module("minimax_h3_conditioning")


@pytest.fixture(autouse=True)
def _resize_without_comfy(monkeypatch):
    """`resize_image` uses comfy.utils.common_upscale (real lanczos, ComfyUI's own numerics);
    the geometry checks here only need the output shape, so use torch bilinear."""
    import torch.nn.functional as F

    def fake_resize(image, width, height, crop):
        x = F.interpolate(image[..., :3].movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False)
        return x.movedim(1, -1)

    monkeypatch.setattr(geo, "resize_image", fake_resize)


def _reference_adapt_canvas(w: int, h: int) -> tuple[int, int]:
    """Real `comfy_extras.nodes_minimax_h3.adapt_canvas`; the real ComfyUI import is confined
    to `real_comfy_isolated` (sys.modules / sys.path restored on exit so the comfy stubs come
    back for later tests in the same session)."""
    with real_comfy_isolated():
        try:
            module = importlib.import_module("comfy_extras.nodes_minimax_h3")
        except Exception as e:  # heavy runtime imports may fail outside a ComfyUI process
            pytest.skip(f"comfy_extras.nodes_minimax_h3 not importable here: {e}")
        return module.adapt_canvas(w, h)


@pytest.mark.parametrize("wh", [(1344, 768), (768, 1344), (1920, 1080), (640, 640), (100, 4000), (32, 32)])
def test_adapt_canvas_hand_computed_and_reference(wh):
    w, h = wh
    got = geo.adapt_canvas(w, h)
    assert got[0] % 32 == 0 and got[1] % 32 == 0 and got[0] >= 32 and got[1] >= 32
    assert got[0] * got[1] <= 768 * 1344 * 1.03  # rounding to 32 may exceed the cap slightly
    assert geo.adapt_canvas(1344, 768) == (1344, 768)
    assert got == _reference_adapt_canvas(w, h)


@pytest.mark.parametrize("mode", ["match", "max"])
@pytest.mark.parametrize("hw", [(768, 1344), (4000, 3000), (256, 256), (1080, 1920)])
def test_ref_image_size(mode, hw):
    h, w = hw
    tw, th = geo.ref_image_size(h, w, 1344, 768, mode)
    assert tw % 32 == 0 and th % 32 == 0 and tw >= 32 and th >= 32
    if mode == "match" and hw == (768, 1344):
        assert (tw, th) == (1344, 768)  # already the generation area: unchanged
    if hw == (256, 256):
        assert (tw, th) == (256, 256)  # never upscaled, in either mode
    if mode == "max" and hw == (4000, 3000):
        assert min(tw, th) <= 2048 + 32  # short edge capped near 2048


def test_ref_image_size_match_scales_down_to_generation_area():
    tw, th = geo.ref_image_size(3000, 4000, 1344, 768, "match")
    assert tw * th <= 1344 * 768 * 1.06 and tw > th


def test_ref_image_size_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        geo.ref_image_size(100, 100, 1344, 768, "huge")


def test_prepare_ref_video_grid_and_limits():
    frames = torch.rand(30, 64, 96, 3)
    out, cw, ch = geo.prepare_ref_video(frames, frame_count=124)
    assert out.shape[0] == 22 and out.shape[0] % 17 == 5  # 30 -> 22 (17k+5)
    assert out.shape[1:] == (ch, cw, 3) and cw % 32 == 0 and ch % 32 == 0
    short, _, _ = geo.prepare_ref_video(torch.rand(200, 64, 96, 3), frame_count=39)
    assert short.shape[0] == 39  # truncated to the target's frame count, already on the grid
    with pytest.raises(ValueError, match="at least 5"):
        geo.prepare_ref_video(torch.rand(4, 64, 96, 3), frame_count=124)


def test_qwen_video_frames_two_fps():
    frames = torch.arange(39, dtype=torch.float32).view(39, 1, 1, 1).expand(39, 4, 4, 3)
    sampled, stamps = geo.qwen_video_frames(frames)
    assert [int(v) for v in sampled[:, 0, 0, 0].tolist()] == [0, 12, 24, 36]
    assert stamps == [0.0, 0.5, 1.0, 1.5]


def test_estimate_packed_rows_matches_layout_arithmetic():
    # 1344x768: 42*24 = 1008 rows per latent frame; latent_t 37 (124 frames); audio 2*T
    base = geo.estimate_packed_rows(1344, 768, 37, 207)
    assert base == 37 * 1008 + 2 * 207
    with_kf = geo.estimate_packed_rows(1344, 768, 37, 207, keyframe_frames=1)
    assert with_kf == base + 1008
    with_ref = geo.estimate_packed_rows(1344, 768, 37, 207, ref_image_sizes=((1344, 768),))
    assert with_ref == base + 1008
    with_video = geo.estimate_packed_rows(1344, 768, 37, 207, ref_videos=((2, 640, 352, 3),))
    assert with_video == base + 2 * (20 * 11) + 2 * 3


@pytest.mark.parametrize(
    "wh, expected",
    [
        ((1344, 768), (1344, 768)),
        ((768, 1344), (768, 1344)),
        ((1920, 1080), (1344, 768)),  # 768*1.7778 = 1365.3 -> area cap scales to 1344x756 -> 32-rounded
        ((640, 640), (768, 768)),  # short edge lifted to 768, square, under the cap
        ((32, 32), (768, 768)),
        ((100, 4000), (160, 6432)),  # extreme aspect: cap scaling then round, values from the reference source
    ],
)
def test_adapt_canvas_pinned_values_independent_of_comfyui_import(wh, expected):
    assert geo.adapt_canvas(*wh) == expected


@pytest.mark.parametrize(
    "hw, mode, expected",
    [
        # match: scale = sqrt(1344*768 / (w*h)); 3000x4000 (h x w) -> 0.29329;
        # w 1173.2/32 = 36.66 -> 37 -> 1184 ; h 879.9/32 = 27.50 -> 27 -> 864
        ((3000, 4000), "match", (1184, 864)),
        # 1000x2000 -> sqrt(0.516096) = 0.71840; w 1436.8/32 = 44.9 -> 45 -> 1440 ; h 718.4/32 = 22.45 -> 22 -> 704
        ((1000, 2000), "match", (1440, 704)),
        # never upscaled (scale 1); ties go to the even multiple (Python banker's rounding):
        # w 80/32 = 2.5 -> 2 -> 64 (half-up would give 96), h 112/32 = 3.5 -> 4 -> 128
        ((112, 80), "match", (64, 128)),
        ((112, 80), "max", (64, 128)),
        # 10x10: 10/32 = 0.31 -> 0 -> floored to 32 in both modes
        ((10, 10), "match", (32, 32)),
        ((10, 10), "max", (32, 32)),
        # max: 2048 / min(4000, 3000) = 0.68267; w 2730.7/32 = 85.3 -> 85 -> 2720 ; h 2048 -> 2048
        ((3000, 4000), "max", (2720, 2048)),
        # max, small: scale 1; 1000/32 = 31.25 -> 31 -> 992
        ((1000, 1000), "max", (992, 992)),
    ],
)
def test_ref_image_size_pinned_from_reference_source(hw, mode, expected):
    """Values worked out by hand from nodes_minimax_h3.py lines 300-306."""
    assert geo.ref_image_size(hw[0], hw[1], 1344, 768, mode) == expected


def test_prepare_ref_video_frame_count_below_five_raises():
    with pytest.raises(ValueError, match="at least 5"):
        geo.prepare_ref_video(torch.rand(30, 64, 96, 3), frame_count=4)  # truncated to 4 frames


def test_prepare_ref_video_tiny_reference_is_never_upscaled():
    # 10x10 < its adapted 768x768 canvas -> 32-floored own size; 6 frames -> 5 (17k+5 grid)
    out, cw, ch = geo.prepare_ref_video(torch.rand(6, 10, 10, 3), frame_count=124)
    assert (cw, ch) == (32, 32) and out.shape == (5, 32, 32, 3)


def test_real_comfy_parity_import_leaves_sys_modules_and_path_untouched():
    """The parity test's real-ComfyUI window must leave no ComfyUI-namespace module and an
    identical `sys.path` behind. Must pass alone."""
    import sys

    before_modules, before_path = dict(sys.modules), list(sys.path)
    _reference_adapt_canvas(1344, 768)
    assert_no_comfyui_leak(before_modules, before_path)
