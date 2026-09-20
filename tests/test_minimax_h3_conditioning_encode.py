"""Encoding of keyframes and references through fake VAEs (shape/contract level), the
packed-row estimate against the real `PackedLayout`, and `resize_image` against the real
ComfyUI `common_upscale` (skipped cleanly when ComfyUI is absent)."""

from __future__ import annotations

import sys

import mlx.core as mx
import pytest
import torch

from tests.support.comfy_stub import install_comfy_stubs, load_node_module
from tests.support.comfyui_reference_loader import assert_no_comfyui_leak, real_comfy_isolated
from tests.support.minimax_h3_module_loader import load_native_module

install_comfy_stubs()
cond = load_node_module("minimax_h3_conditioning")
_REAL_RESIZE = cond.resize_image
_REAL_NODES = load_node_module("minimax_h3_nodes")


def _lat_t(frames: int) -> int:
    return max(1, (frames - 1) // 17 * 5 + 2) if frames > 1 else 1


class FakeVAE:
    """Mimics comfy.sd.VAE.encode on [T, H, W, 3] for the MiniMax H3 video VAE."""

    def __init__(self):
        self.calls = []

    def encode(self, pixels):
        self.calls.append(tuple(pixels.shape))
        t, h, w, _ = pixels.shape
        return torch.full((1, 24, _lat_t(t), h // 16, w // 16), 0.5)


class FakeAudioVAE:
    audio_sample_rate = 32000

    def __init__(self):
        self.calls = []
        self.waves = []

    def encode(self, wave):  # wave: [1, L, C]
        self.calls.append(tuple(wave.shape))
        self.waves.append(wave)
        return torch.full((1, 32, 2, wave.shape[1] // 800), 0.25)


@pytest.fixture
def fake_resize(monkeypatch):
    """`resize_image` uses comfy.utils.common_upscale; use torch bilinear here (numerics of
    the lanczos are ComfyUI's own; these tests check the contracts)."""
    import torch.nn.functional as F

    def _resize(image, width, height, crop):
        x = image[..., :3].movedim(-1, 1)
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return x.movedim(1, -1)

    monkeypatch.setattr(cond, "resize_image", _resize)


def test_encode_video_latent_returns_mlx_latent():
    z = cond.encode_video_latent(FakeVAE(), torch.rand(5, 64, 96, 3))
    assert isinstance(z, mx.array) and z.shape == (1, 24, 2, 4, 6) and z.dtype == mx.float32


def test_encode_video_latent_rejects_wrong_shape():
    class Bad:
        def encode(self, pixels):
            return torch.zeros(1, 16, 2, 4, 6)

    with pytest.raises(ValueError, match="ASDX"):
        cond.encode_video_latent(Bad(), torch.rand(5, 64, 96, 3))


class TiledVAE:
    """`encode` fails with `error`; `encode_tiled` returns a marker latent (or fails with `tiled_error`)."""

    def __init__(self, error: Exception, tiled_error: Exception | None = None):
        self.error, self.tiled_error = error, tiled_error
        self.encode_calls = 0
        self.tiled_calls = []

    def encode(self, pixels):
        self.encode_calls += 1
        raise self.error

    def encode_tiled(self, pixels):
        self.tiled_calls.append(tuple(pixels.shape))
        if self.tiled_error is not None:
            raise self.tiled_error
        return torch.full((1, 24, 2, 4, 6), 0.75)


@pytest.fixture
def cache_spy(monkeypatch):
    """The comfy stub has no `soft_empty_cache`; record calls to it."""
    import comfy.model_management as mm

    calls = []
    monkeypatch.setattr(mm, "soft_empty_cache", lambda *a, **k: calls.append(1), raising=False)
    return calls


@pytest.mark.parametrize("message", ["MPS backend out of memory (MPS allocated: 40 GB)", "Placeholder tensor is larger than INT_MAX"])
def test_encode_video_latent_retries_tiled_on_mps_limits(message, capsys, cache_spy):
    vae = TiledVAE(RuntimeError(message))
    z = cond.encode_video_latent(vae, torch.rand(5, 64, 96, 3))
    assert vae.encode_calls == 1 and vae.tiled_calls == [(5, 64, 96, 3)]
    assert z.shape == (1, 24, 2, 4, 6) and float(z[0, 0, 0, 0, 0]) == 0.75
    assert "[ASDX]" in capsys.readouterr().out and cache_spy == [1]


@pytest.mark.parametrize("error", [RuntimeError("something else"), ValueError("out of memory")])
def test_encode_video_latent_other_errors_propagate_without_tiling(error):
    vae = TiledVAE(error)
    with pytest.raises(type(error), match=str(error)):
        cond.encode_video_latent(vae, torch.rand(5, 64, 96, 3))
    assert vae.tiled_calls == []


def test_encode_video_latent_tiled_failure_propagates(cache_spy):
    vae = TiledVAE(RuntimeError("MPS backend out of memory"), tiled_error=RuntimeError("tiled boom"))
    with pytest.raises(RuntimeError, match="tiled boom"):
        cond.encode_video_latent(vae, torch.rand(5, 64, 96, 3))


def test_encode_video_latent_tiled_result_is_shape_validated(cache_spy):
    class BadTiled(TiledVAE):
        def encode_tiled(self, pixels):
            return torch.zeros(1, 16, 2, 4, 6)

    with pytest.raises(ValueError, match="ASDX"):
        cond.encode_video_latent(BadTiled(RuntimeError("MPS backend out of memory")), torch.rand(5, 64, 96, 3))


def test_encode_video_latent_untiled_path_never_touches_tiled():
    class Spy(FakeVAE):
        def encode_tiled(self, pixels):
            raise AssertionError("tiled path must not run")

    assert cond.encode_video_latent(Spy(), torch.rand(5, 64, 96, 3)).shape == (1, 24, 2, 4, 6)


def test_encode_audio_latent_no_resample_keeps_channels_and_first_batch():
    wave = torch.zeros(2, 2, 32000)  # [B, C, L]: batch 0 = (0, 1) per channel, batch 1 = (5, 6)
    wave[0, 1] = 1.0
    wave[1, 0], wave[1, 1] = 5.0, 6.0
    vae = FakeAudioVAE()
    z, t = cond.encode_audio_latent(vae, {"waveform": wave, "sample_rate": 32000})
    assert z.shape == (1, 32, 2, 40) and t == 40
    assert vae.calls == [(1, 32000, 2)]  # first batch item only, channels moved last
    assert torch.equal(vae.waves[0][0, :, 0], torch.zeros(32000)) and torch.equal(vae.waves[0][0, :, 1], torch.ones(32000))


def test_encode_audio_latent_rejects_wrong_rank():
    class Bad:
        def encode(self, wave):
            return torch.zeros(1, 32, 40)

    with pytest.raises(ValueError, match="ASDX"):
        cond.encode_audio_latent(Bad(), {"waveform": torch.zeros(1, 2, 3200), "sample_rate": 32000})


def test_keyframes_first_stretched_last_at_final_frame(fake_resize):
    vae = FakeVAE()
    first, last = torch.rand(1, 100, 200, 3), torch.rand(1, 300, 200, 3)
    images, kfs = cond.build_keyframes(vae, 96, 64, 124, first, last)
    assert [k.resolved_frame_index for k in kfs] == [0, 123]
    assert len(images) == 2 and images[0].shape == (1, 64, 96, 3)
    assert vae.calls == [(1, 64, 96, 3), (1, 64, 96, 3)]
    assert kfs[0].latent.shape == (1, 24, 1, 4, 6)


def test_keyframes_crop_mode_and_index_per_frame(monkeypatch):
    """First frame: plain stretch at index 0. Last frame: cover crop at the final frame."""
    seen = []

    def spy(image, width, height, crop):
        seen.append((float(image[0, 0, 0, 0]), crop))
        return torch.zeros(1, height, width, 3)

    monkeypatch.setattr(cond, "resize_image", spy)
    first, last = torch.full((1, 8, 8, 3), 1.0), torch.full((1, 8, 8, 3), 2.0)
    _, kfs = cond.build_keyframes(FakeVAE(), 96, 64, 124, first, last)
    assert seen == [(1.0, "disabled"), (2.0, "center")]
    assert [k.resolved_frame_index for k in kfs] == [0, 123]


def test_keyframes_last_only_and_none(fake_resize):
    images, kfs = cond.build_keyframes(FakeVAE(), 96, 64, 124, None, torch.rand(1, 64, 96, 3))
    assert [k.resolved_frame_index for k in kfs] == [123] and len(images) == 1
    assert cond.build_keyframes(FakeVAE(), 96, 64, 124, None, None) == ([], [])


def test_references_order_labels_and_blocks(fake_resize):
    vae, avae = FakeVAE(), FakeAudioVAE()
    img = torch.rand(1, 64, 96, 3)
    vid = torch.rand(22, 64, 96, 3)
    snd = {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
    items, refs = cond.build_references(
        vae, avae, 96, 64, 124, "match",
        ref_images={"ref_image_1": img}, ref_videos={"ref_video_1": vid},
        ref_video_audios={"ref_video_audio_1": snd}, ref_audios={"ref_audio_1": snd},
    )
    assert [i["type"] for i in items] == ["image", "audio", "video", "audio"]  # audio label BEFORE its video
    assert [r.kind for r in refs] == ["image", "video_audio", "audio"]
    assert refs[0].latent_h == 4 and refs[0].latent_w == 6 and refs[0].latent.shape[2] == refs[0].latent_t == 1
    assert refs[1].ref_audio_t == 40 and refs[1].latent_t == refs[1].latent.shape[2]
    assert refs[2].latent is None and refs[2].ref_audio_t == 40


def test_references_without_vae_are_text_only(fake_resize):
    items, refs = cond.build_references(
        None, None, 96, 64, 124, "match",
        ref_images={"ref_image_1": torch.rand(1, 64, 96, 3)}, ref_videos={}, ref_video_audios={},
        ref_audios={"ref_audio_1": {"waveform": torch.zeros(1, 2, 100), "sample_rate": 32000}},
    )
    assert [i["type"] for i in items] == ["image", "audio"] and refs == []


def test_references_without_vae_video_is_text_only(fake_resize):
    items, refs = cond.build_references(
        None, None, 96, 64, 124, "match", ref_images={},
        ref_videos={"ref_video_1": torch.rand(22, 64, 96, 3)}, ref_video_audios={}, ref_audios={},
    )
    assert [i["type"] for i in items] == ["video"] and refs == []


def test_reference_caps_and_short_video(fake_resize):
    with pytest.raises(ValueError, match="at most 9"):
        cond.build_references(None, None, 96, 64, 124, "match",
                              ref_images={f"ref_image_{i}": torch.rand(1, 32, 32, 3) for i in range(10)},
                              ref_videos={}, ref_video_audios={}, ref_audios={})
    with pytest.raises(ValueError, match="at most 3"):
        cond.build_references(None, None, 96, 64, 124, "match", ref_images={},
                              ref_videos={f"ref_video_{i}": torch.rand(22, 64, 96, 3) for i in range(4)},
                              ref_video_audios={}, ref_audios={})
    with pytest.raises(ValueError, match="at most 3"):
        cond.build_references(None, None, 96, 64, 124, "match", ref_images={}, ref_videos={}, ref_video_audios={},
                              ref_audios={f"ref_audio_{i}": {"waveform": torch.zeros(1, 2, 10), "sample_rate": 32000}
                                          for i in range(4)})
    with pytest.raises(ValueError, match="at least 5"):
        cond.build_references(FakeVAE(), None, 96, 64, 124, "match", ref_images={},
                              ref_videos={"ref_video_1": torch.rand(4, 64, 96, 3)}, ref_video_audios={}, ref_audios={})


def test_caps_allow_the_maximum(fake_resize):
    items, _ = cond.build_references(
        None, None, 96, 64, 124, "match",
        ref_images={f"ref_image_{i}": torch.rand(1, 32, 32, 3) for i in range(9)},
        ref_videos={f"ref_video_{i}": torch.rand(22, 64, 96, 3) for i in range(3)},
        ref_video_audios={},
        ref_audios={f"ref_audio_{i}": {"waveform": torch.zeros(1, 2, 10), "sample_rate": 32000} for i in range(3)},
    )
    assert len(items) == 15


def test_video_audio_pairing_is_by_index(fake_resize):
    items, refs = cond.build_references(
        FakeVAE(), FakeAudioVAE(), 96, 64, 124, "match", ref_images={},
        ref_videos={"ref_video_1": torch.rand(22, 64, 96, 3), "ref_video_2": torch.rand(22, 64, 96, 3)},
        ref_video_audios={"ref_video_audio_2": {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}},
        ref_audios={},
    )
    assert [i["type"] for i in items] == ["video", "audio", "video"]  # only video 2 has a soundtrack
    assert [r.kind for r in refs] == ["video", "video_audio"]


def test_video_reference_encodes_full_frames_not_the_2fps_sample(fake_resize):
    vae = FakeVAE()
    _, refs = cond.build_references(
        vae, None, 96, 64, 124, "match", ref_images={},
        ref_videos={"ref_video_1": torch.rand(22, 64, 96, 3)}, ref_video_audios={}, ref_audios={},
    )
    assert vae.calls == [(22, 64, 96, 3)] and refs[0].latent_t == 7 == refs[0].latent.shape[2]


def test_video_with_soundtrack_but_no_audio_vae_is_a_plain_video(fake_resize):
    snd = {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
    items, refs = cond.build_references(
        FakeVAE(), None, 96, 64, 124, "match", ref_images={},
        ref_videos={"ref_video_1": torch.rand(22, 64, 96, 3)},
        ref_video_audios={"ref_video_audio_1": snd}, ref_audios={},
    )
    assert [i["type"] for i in items] == ["audio", "video"]  # label still emitted
    assert [r.kind for r in refs] == ["video"] and refs[0].ref_audio_t == 0 and refs[0].audio_latent is None


def test_reference_batches_reduce_to_the_first_item(fake_resize):
    vae, avae = FakeVAE(), FakeAudioVAE()
    cond.build_references(
        vae, avae, 96, 64, 124, "match", ref_images={"ref_image_1": torch.rand(2, 64, 96, 3)},
        ref_videos={}, ref_video_audios={},
        ref_audios={"ref_audio_1": {"waveform": torch.zeros(2, 2, 32000), "sample_rate": 32000}},
    )
    assert vae.calls == [(1, 64, 96, 3)] and avae.calls == [(1, 32000, 2)]


def test_reference_images_are_stretched_not_cropped(monkeypatch):
    seen = []

    def spy(image, width, height, crop):
        seen.append(crop)
        return torch.zeros(1, height, width, 3)

    monkeypatch.setattr(cond, "resize_image", spy)
    cond.build_references(
        None, None, 96, 64, 124, "match", ref_images={"ref_image_1": torch.rand(1, 64, 96, 3)},
        ref_videos={}, ref_video_audios={}, ref_audios={},
    )
    assert seen == ["disabled"]


def test_reference_image_max_mode_uses_max_size(fake_resize):
    img = torch.rand(1, 3000, 4000, 3)  # match -> 1184x864, max -> 2720x2048: they differ
    _, refs = cond.build_references(
        FakeVAE(), None, 1344, 768, 124, "max", ref_images={"ref_image_1": img},
        ref_videos={}, ref_video_audios={}, ref_audios={},
    )
    tw, th = cond.ref_image_size(3000, 4000, 1344, 768, "max")
    assert (tw, th) == (2720, 2048) != cond.ref_image_size(3000, 4000, 1344, 768, "match")
    assert (refs[0].latent_w, refs[0].latent_h) == (tw // 16, th // 16)
    assert tuple(refs[0].latent.shape[3:]) == (th // 16, tw // 16)


def test_none_reference_entries_are_skipped(fake_resize):
    vae = FakeVAE()
    items, refs = cond.build_references(
        vae, None, 96, 64, 124, "match", ref_images={"ref_image_1": None},
        ref_videos={"ref_video_1": None}, ref_video_audios={}, ref_audios={"ref_audio_1": None},
    )
    assert items == [] and refs == [] and vae.calls == []


def test_estimate_packed_rows_matches_real_packed_layout(fake_resize):
    """Every term of `estimate_packed_rows` against `PackedLayout.seq_len`: keyframe video and
    audio, an image ref, a video_audio ref and a standalone audio ref."""
    layout = load_native_module("minimax_h3.layout")
    width, height, latent_t, audio_t, text_len = 96, 64, 2, 3, 7
    vae, avae = FakeVAE(), FakeAudioVAE()
    snd = {"waveform": torch.zeros(1, 2, 4000), "sample_rate": 32000}  # 5 audio latent frames
    kf = cond.KeyframeCond(
        resolved_frame_index=0,
        latent=cond.encode_video_latent(vae, torch.rand(1, height, width, 3)),
        audio_latent=cond.encode_audio_latent(avae, snd)[0],
    )
    items, refs = cond.build_references(
        vae, avae, width, height, 124, "match",
        ref_images={"ref_image_1": torch.rand(1, 64, 64, 3)},
        ref_videos={"ref_video_1": torch.rand(22, height, width, 3)},
        ref_video_audios={"ref_video_audio_1": {"waveform": torch.zeros(1, 2, 8000), "sample_rate": 32000}},
        ref_audios={"ref_audio_1": {"waveform": torch.zeros(1, 2, 12000), "sample_rate": 32000}},
    )
    assert [r.kind for r in refs] == ["image", "video_audio", "audio"]
    real = layout.PackedLayout(text_len, latent_t, height // 16, width // 16, audio_t, keyframes=(kf,), refs=tuple(refs))
    assert {k for _, _, k in real.segments} >= {"cond", "cond_audio", "ref_img", "ref_audio"}
    est = cond.estimate_packed_rows(
        width, height, latent_t, audio_t,
        keyframe_frames=1, keyframe_audio_t=5,
        ref_image_sizes=((64, 64),),
        ref_videos=((refs[1].latent_t, width, height, refs[1].ref_audio_t),),
        ref_audio_t=refs[2].ref_audio_t,
    )
    assert est == real.seq_len - text_len
    # each audio term is really exercised (non-zero) so dropping one cannot pass silently
    assert refs[1].ref_audio_t > 0 and refs[2].ref_audio_t > 0


@pytest.mark.parametrize("crop", ["disabled", "center"])
def test_real_resize_shape_and_channel_slice(crop):
    with real_comfy_isolated():
        img = torch.zeros(2, 40, 60, 4)
        img[..., 0], img[..., 1], img[..., 2], img[..., 3] = 0.2, 0.4, 0.6, 9.0
        out = _REAL_RESIZE(img, 32, 48, crop)
    assert out.shape == (2, 48, 32, 3)
    assert torch.allclose(out[0, 24, 16], torch.tensor([0.2, 0.4, 0.6]), atol=1e-3)  # alpha (9.0) dropped


def test_real_resize_disabled_stretches_and_center_crops_to_the_middle():
    with real_comfy_isolated():
        # wide image (H 40, W 120), columns in thirds 0 / 0.5 / 1
        wide = torch.zeros(1, 40, 120, 3)
        wide[:, :, 40:80] = 0.5
        wide[:, :, 80:] = 1.0
        stretched = _REAL_RESIZE(wide, 32, 96, "disabled")  # tall canvas
        cropped = _REAL_RESIZE(wide, 32, 96, "center")
        # tall image (H 120, W 40), rows in thirds 0 / 0.5 / 1 -> wide canvas
        tall = torch.zeros(1, 120, 40, 3)
        tall[:, 40:80] = 0.5
        tall[:, 80:] = 1.0
        stretched_t = _REAL_RESIZE(tall, 96, 32, "disabled")
        cropped_t = _REAL_RESIZE(tall, 96, 32, "center")
    # stretch keeps the whole stripe structure along the squeezed axis
    assert stretched[0, 48, 0, 0] < 0.05 and abs(stretched[0, 48, 16, 0] - 0.5) < 0.05 and stretched[0, 48, -1, 0] > 0.95
    assert stretched_t[0, 0, 48, 0] < 0.05 and abs(stretched_t[0, 16, 48, 0] - 0.5) < 0.05 and stretched_t[0, -1, 48, 0] > 0.95
    # cover crop keeps only the central third: a single uniform value
    assert torch.allclose(cropped, torch.full_like(cropped, 0.5), atol=0.05)
    assert torch.allclose(cropped_t, torch.full_like(cropped_t, 0.5), atol=0.05)


def test_real_comfy_import_leaves_sys_modules_and_path_untouched():
    """The real-ComfyUI window must leave no ComfyUI-namespace module and an identical
    `sys.path` behind (third-party caches such as torch/PIL/transformers are not covered).
    Must pass alone, independent of what earlier tests left in `sys.modules`."""
    before_modules, before_path = dict(sys.modules), list(sys.path)
    _call_real_resize()
    assert_no_comfyui_leak(before_modules, before_path)


def _call_real_resize() -> None:
    with real_comfy_isolated():
        _REAL_RESIZE(torch.rand(1, 8, 8, 3), 32, 32, "disabled")


def test_encode_audio_latent_resamples_with_real_torchaudio():
    """48 kHz -> 32 kHz through the ComfyUI venv's torchaudio (absent from the project venv)."""
    vae = FakeAudioVAE()
    with real_comfy_isolated():
        pytest.importorskip("torchaudio")
        cond.encode_audio_latent(vae, {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000})
    assert vae.calls == [(1, 32000, 2)]


_TOWER = {"type": "asdx_minimax_h3_text_encoder", "vision_tower": object()}  # placeholder: the prompt is stubbed


def _stub_prompt(monkeypatch, rows=10):
    """Replace encode_minimax_h3_prompt so the builders can be tested without an encoder."""
    import types

    seen = {}
    nodes = types.ModuleType("apple_silicon_nodes.minimax_h3_nodes")

    def fake_encode(text_encoder, prompt, *, images=None, ref_items=None):
        seen.update(prompt=prompt, images=images, ref_items=ref_items)
        return {"type": "minimax_h3", "hidden_states": mx.zeros((rows, 4)), "token_tags": mx.ones((rows,), dtype=mx.int32), "text": prompt}

    nodes.encode_minimax_h3_prompt = fake_encode
    nodes._temporal_shape = _REAL_NODES._temporal_shape  # the real 17k+5 snap: 120 -> 124, 41 -> 56, 5 -> 5
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.minimax_h3_nodes", nodes)
    return seen


def test_empty_av_latents_shapes(monkeypatch):
    _stub_prompt(monkeypatch)
    video, audio, frame_count, latent_t, audio_t = cond.empty_av_latents(1344, 768, 124)
    assert tuple(video["samples"].shape) == (1, 24, 37, 48, 84) and tuple(audio["samples"].shape) == (1, 32, 2, 207)
    assert (frame_count, latent_t, audio_t) == (124, 37, 207)


def test_i2v_conditioning_carries_keyframes_and_images(monkeypatch, fake_resize):
    seen = _stub_prompt(monkeypatch)
    conditioning, video, audio = cond.build_i2v_conditioning(
        _TOWER, FakeVAE(), "a cat", 96, 64, 124, torch.rand(1, 64, 96, 3), None)
    assert [k.resolved_frame_index for k in conditioning["keyframes"]] == [0] and "refs" not in conditioning
    assert len(seen["images"]) == 1 and seen["ref_items"] is None
    assert tuple(video["samples"].shape)[3:] == (4, 6)


def test_i2v_without_frames_is_plain_t2v(monkeypatch):
    seen = _stub_prompt(monkeypatch)
    conditioning, _, _ = cond.build_i2v_conditioning({}, FakeVAE(), "a cat", 96, 64, 124, None, None)
    assert "keyframes" not in conditioning and "refs" not in conditioning and not seen["images"]


def test_i2v_needs_a_vae_when_frames_are_given(monkeypatch):
    _stub_prompt(monkeypatch)
    with pytest.raises(ValueError, match="vae"):
        cond.build_i2v_conditioning({}, None, "a cat", 96, 64, 124, torch.rand(1, 64, 96, 3), None)
    with pytest.raises(ValueError, match="vae"):
        cond.build_i2v_conditioning({}, None, "a cat", 96, 64, 124, None, torch.rand(1, 64, 96, 3))


def test_reference_conditioning_forwards_ref_items_in_order(monkeypatch, fake_resize):
    seen = _stub_prompt(monkeypatch)
    snd = {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
    conditioning, _, _ = cond.build_reference_conditioning(
        _TOWER, FakeVAE(), FakeAudioVAE(), "a cat", 96, 64, 124, "match",
        ref_images={"ref_image_1": torch.rand(1, 64, 96, 3)}, ref_videos={}, ref_video_audios={}, ref_audios={"ref_audio_1": snd})
    assert [i["type"] for i in seen["ref_items"]] == ["image", "audio"] and seen["images"] is None
    assert [r.kind for r in conditioning["refs"]] == ["image", "audio"] and "keyframes" not in conditioning


def test_reference_conditioning_without_refs_has_no_refs_key(monkeypatch):
    seen = _stub_prompt(monkeypatch)
    conditioning, _, _ = cond.build_reference_conditioning(
        {}, None, None, "a cat", 96, 64, 124, "match", ref_images={}, ref_videos={}, ref_video_audios={}, ref_audios={})
    assert "refs" not in conditioning and "keyframes" not in conditioning and seen["ref_items"] is None


def test_row_warning_only_above_threshold(monkeypatch, capsys):
    _stub_prompt(monkeypatch, rows=10)
    args = ({}, None, None, "a cat", 1344, 768, 124, "match")
    kwargs = dict(ref_images={}, ref_videos={}, ref_video_audios={}, ref_audios={})
    cond.build_reference_conditioning(*args, **kwargs)
    quiet = capsys.readouterr().out
    assert "packed sequence" in quiet and "WARNING" not in quiet  # 10 text + 37296 + 414 rows: under the threshold
    monkeypatch.setattr(cond, "ROW_WARN_THRESHOLD", 1000)
    cond.build_reference_conditioning(*args, **kwargs)
    assert "WARNING" in capsys.readouterr().out


def test_reported_total_is_text_plus_video_audio_rows(monkeypatch, capsys):
    _stub_prompt(monkeypatch, rows=10)
    cond.build_reference_conditioning({}, None, None, "a cat", 1344, 768, 124, "match",
                                      ref_images={}, ref_videos={}, ref_video_audios={}, ref_audios={})
    assert "37720 rows (10 text + 37710 video/audio/conditions)" in capsys.readouterr().out


def test_i2v_report_counts_keyframe_rows(monkeypatch, capsys, fake_resize):
    _stub_prompt(monkeypatch, rows=10)
    frame = torch.rand(1, 64, 96, 3)
    cond.build_i2v_conditioning(_TOWER, FakeVAE(), "a cat", 96, 64, 124, frame, frame)
    # 6 rows per frame: target 37*6 + 2*207 = 636, two keyframes 12 -> 648, + 10 text
    assert "658 rows (10 text + 648 video/audio/conditions)" in capsys.readouterr().out


def test_reported_totals_match_real_packed_layout(monkeypatch, capsys, fake_resize):
    """The printed total equals `text_rows + PackedLayout.seq_len - text_len`: for the ref2va builder
    with an image ref, a video_audio ref and a standalone audio ref, and for the i2v builder with a keyframe."""
    import re

    layout = load_native_module("minimax_h3.layout")
    text_rows, width, height = 10, 96, 64
    _stub_prompt(monkeypatch, rows=text_rows)  # _temporal_shape -> (124, 37, 207)

    def printed_total() -> int:
        return int(re.search(r"(\d+) rows", capsys.readouterr().out).group(1))

    conditioning, _, _ = cond.build_reference_conditioning(
        _TOWER, FakeVAE(), FakeAudioVAE(), "a cat", width, height, 124, "match",
        ref_images={"ref_image_1": torch.rand(1, 32, 64, 3)},  # non-square: a w/h swap changes the rows
        ref_videos={"ref_video_1": torch.rand(22, height, width, 3)},
        ref_video_audios={"ref_video_audio_1": {"waveform": torch.zeros(1, 2, 8000), "sample_rate": 32000}},
        ref_audios={"ref_audio_1": {"waveform": torch.zeros(1, 2, 12000), "sample_rate": 32000}})
    refs = tuple(conditioning["refs"])
    assert [r.kind for r in refs] == ["image", "video_audio", "audio"] and refs[1].ref_audio_t > 0 and refs[2].ref_audio_t > 0
    assert (refs[0].latent_w, refs[0].latent_h) == (4, 2)
    real = layout.PackedLayout(text_rows, 37, height // 16, width // 16, 207, refs=refs)
    assert printed_total() == text_rows + (real.seq_len - real.segments[0][1])

    conditioning, _, _ = cond.build_i2v_conditioning(
        _TOWER, FakeVAE(), "a cat", width, height, 124, torch.rand(1, height, width, 3), None)
    real = layout.PackedLayout(text_rows, 37, height // 16, width // 16, 207, keyframes=tuple(conditioning["keyframes"]))
    assert printed_total() == text_rows + (real.seq_len - real.segments[0][1])


def test_row_warning_boundary_is_strictly_above_threshold(monkeypatch, capsys):
    _stub_prompt(monkeypatch, rows=10)
    args = ({}, None, None, "a cat", 96, 64, 124, "match", {}, {}, {}, {})
    monkeypatch.setattr(cond, "ROW_WARN_THRESHOLD", 646 + 1)  # only 10 + 636 rows exist here
    cond.build_reference_conditioning(*args)
    assert "WARNING" not in capsys.readouterr().out
    monkeypatch.setattr(cond, "ROW_WARN_THRESHOLD", 646)
    cond.build_reference_conditioning(*args)
    assert "WARNING" not in capsys.readouterr().out  # equal to the threshold: quiet
    monkeypatch.setattr(cond, "ROW_WARN_THRESHOLD", 645)
    cond.build_reference_conditioning(*args)
    assert "WARNING" in capsys.readouterr().out


def _check_latents(video, audio, latent_t, audio_t, h=4, w=6):
    assert tuple(video["samples"].shape) == (1, 24, latent_t, h, w)
    assert tuple(audio["samples"].shape) == (1, 32, 2, audio_t)
    for latent in (video, audio):
        t = latent["samples"]
        assert t.dtype == torch.float32 and t.device.type == "cpu" and not t.any()


def test_i2v_latents_follow_length_and_are_zero_float32_cpu(monkeypatch):
    _stub_prompt(monkeypatch)
    _, video, audio = cond.build_i2v_conditioning({}, None, "a cat", 96, 64, 41, None, None)
    _check_latents(video, audio, 17, 93)  # 41 snaps to 56 frames


def test_reference_latents_follow_length_and_are_zero_float32_cpu(monkeypatch):
    _stub_prompt(monkeypatch)
    _, video, audio = cond.build_reference_conditioning(
        {}, None, None, "a cat", 96, 64, 41, "match", ref_images={}, ref_videos={}, ref_video_audios={}, ref_audios={})
    _check_latents(video, audio, 17, 93)


def test_last_keyframe_index_uses_the_snapped_frame_count(monkeypatch, fake_resize):
    _stub_prompt(monkeypatch)
    frame = torch.rand(1, 64, 96, 3)
    conditioning, _, _ = cond.build_i2v_conditioning(_TOWER, FakeVAE(), "a cat", 96, 64, 120, frame, frame)
    assert [k.resolved_frame_index for k in conditioning["keyframes"]] == [0, 123]  # not 119


def test_reference_video_is_capped_at_the_snapped_frame_count(monkeypatch, fake_resize):
    _stub_prompt(monkeypatch)
    for length, video_frames, expected in ((5, 40, 5), (120, 150, 124)):
        vae = FakeVAE()
        cond.build_reference_conditioning(
            _TOWER, vae, None, "a cat", 96, 64, length, "match", ref_images={},
            ref_videos={"ref_video_1": torch.rand(video_frames, 64, 96, 3)}, ref_video_audios={}, ref_audios={})
        # length 120 snaps to 124 (=17*7+5, kept whole); capping at 120 would land on 107 frames
        assert vae.calls == [(expected, 64, 96, 3)]


def test_no_vae_message_only_when_references_are_connected_without_vae(monkeypatch, capsys, fake_resize):
    _stub_prompt(monkeypatch)
    msg = "references only condition the text encoder"
    img = {"ref_image_1": torch.rand(1, 64, 96, 3)}
    empty = dict(ref_videos={}, ref_video_audios={}, ref_audios={})
    cond.build_reference_conditioning(_TOWER, None, None, "a cat", 96, 64, 124, "match", ref_images=img, **empty)
    assert msg in capsys.readouterr().out
    cond.build_reference_conditioning({}, None, None, "a cat", 96, 64, 124, "match", ref_images={}, **empty)
    assert msg not in capsys.readouterr().out
    cond.build_reference_conditioning(_TOWER, FakeVAE(), None, "a cat", 96, 64, 124, "match", ref_images=img, **empty)
    assert msg not in capsys.readouterr().out


def test_reference_videos_are_stretched_not_cropped(monkeypatch):
    seen = []

    def spy(image, width, height, crop):
        seen.append(crop)
        return torch.zeros(image.shape[0], height, width, 3)

    monkeypatch.setattr(cond, "resize_image", spy)
    cond.prepare_ref_video(torch.rand(22, 64, 96, 3), 124)
    assert seen == ["disabled"]


def test_audio_only_references_with_audio_vae_and_no_video_vae(fake_resize):
    """A legitimate graph: audio references need neither the tower nor the video VAE."""
    avae = FakeAudioVAE()
    snd = {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
    items, refs = cond.build_references(
        None, avae, 96, 64, 124, "match", ref_images={}, ref_videos={}, ref_video_audios={}, ref_audios={"ref_audio_1": snd})
    assert items == [{"type": "audio"}]
    assert [r.kind for r in refs] == ["audio"] and refs[0].ref_audio_t > 0 and refs[0].latent is None


@pytest.mark.parametrize("which", ["first", "last"])
def test_keyframe_batches_reduce_to_the_first_image(fake_resize, which):
    vae = FakeVAE()
    batch = torch.rand(3, 100, 200, 3)
    first, last = (batch, None) if which == "first" else (None, batch)
    images, kfs = cond.build_keyframes(vae, 96, 64, 124, first, last)
    assert vae.calls == [(1, 64, 96, 3)] and images[0].shape == (1, 64, 96, 3) and len(kfs) == 1


def test_reference_video_cap_window_just_above_frame_count(fake_resize):
    """22 frames at frame_count 5: capped to 5 (a cap that only starts 17 frames later leaves 22)."""
    vae = FakeVAE()
    cond.build_references(
        vae, None, 96, 64, 5, "match", ref_images={},
        ref_videos={"ref_video_1": torch.rand(22, 64, 96, 3)}, ref_video_audios={}, ref_audios={})
    assert vae.calls == [(5, 64, 96, 3)]


def test_encode_audio_latent_rejects_mono_latent():
    class Mono:
        def encode(self, wave):
            return torch.zeros(1, 32, 1, 40)

    with pytest.raises(ValueError, match="ASDX.*stereo"):
        cond.encode_audio_latent(Mono(), {"waveform": torch.zeros(1, 1, 32000), "sample_rate": 32000})


_NO_TOWER = {"type": "asdx_minimax_h3_text_encoder", "vision_tower": None}
_NO_KEY = {"type": "asdx_minimax_h3_text_encoder"}


@pytest.mark.parametrize("te", [_NO_TOWER, _NO_KEY])
@pytest.mark.parametrize("which", ["first", "last"])
def test_i2v_without_vision_tower_fails_before_any_encode(monkeypatch, fake_resize, te, which):
    _stub_prompt(monkeypatch)
    vae = FakeVAE()
    frame = torch.rand(1, 64, 96, 3)
    first, last = (frame, None) if which == "first" else (None, frame)
    with pytest.raises(ValueError, match="load_vision"):
        cond.build_i2v_conditioning(te, vae, "a cat", 96, 64, 124, first, last)
    assert vae.calls == []


def test_i2v_without_frames_needs_no_vision_tower(monkeypatch):
    _stub_prompt(monkeypatch)
    cond.build_i2v_conditioning(_NO_TOWER, FakeVAE(), "a cat", 96, 64, 124, None, None)


@pytest.mark.parametrize("te", [_NO_TOWER, _NO_KEY])
@pytest.mark.parametrize("kind", ["image", "video"])
def test_references_with_visual_refs_and_no_tower_fail_before_any_encode(monkeypatch, fake_resize, te, kind):
    _stub_prompt(monkeypatch)
    vae, avae = FakeVAE(), FakeAudioVAE()
    refs = dict(ref_images={}, ref_videos={}, ref_video_audios={}, ref_audios={})
    refs["ref_images" if kind == "image" else "ref_videos"] = {
        f"ref_{kind}_1": torch.rand(1 if kind == "image" else 22, 64, 96, 3)}
    with pytest.raises(ValueError, match="load_vision"):
        cond.build_reference_conditioning(te, vae, avae, "a cat", 96, 64, 124, "match", **refs)
    assert vae.calls == [] and avae.calls == []


def test_audio_only_references_need_no_vision_tower(monkeypatch):
    _stub_prompt(monkeypatch)
    avae = FakeAudioVAE()
    conditioning, _, _ = cond.build_reference_conditioning(
        _NO_TOWER, None, avae, "a cat", 96, 64, 124, "match", ref_images={"ref_image_1": None},
        ref_videos={}, ref_video_audios={},
        ref_audios={"ref_audio_1": {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}})
    assert [r.kind for r in conditioning["refs"]] == ["audio"] and avae.calls
