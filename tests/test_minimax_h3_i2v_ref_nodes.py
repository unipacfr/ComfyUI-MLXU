"""The two MiniMax H3 nodes (image-to-video, reference-to-video): registration, schema and
delegation of execute() to the conditioning builders.

The schema test runs against the REAL `comfy_api.latest.io` (loaded inside
`real_comfy_isolated()`), by substituting the node module's `io` for the duration of a
`define_schema()` call; it is skipped, not faked, when the ComfyUI install is absent."""

from __future__ import annotations

import sys
import types

import pytest

from tests.support.comfy_stub import install_comfy_stubs, load_node_module
from tests.support.comfyui_reference_loader import real_comfy_isolated

install_comfy_stubs()
nodes = load_node_module("minimax_h3_nodes")

ENCODER = {"type": "asdx_minimax_h3_text_encoder"}
OUT = ({"type": "minimax_h3"}, {"samples": 1}, {"samples": 2})


def test_nodes_are_registered():
    ids = [n.__name__ for n in nodes.NODE_LIST]
    assert "ASDX_MiniMaxH3ImageToVideo" in ids and "ASDX_MiniMaxH3ReferenceToVideo" in ids


def _install_fake_builders(monkeypatch):
    """Fake builders with NAMED parameters, so a value is checked against the parameter it
    must reach (a swapped vae/audio_vae or a misplaced positional arg cannot pass)."""
    calls = {}
    mod = types.ModuleType("apple_silicon_nodes.minimax_h3_conditioning")
    mod.MAX_REF_IMAGES, mod.MAX_REF_VIDEOS, mod.MAX_REF_AUDIOS = 9, 3, 3

    def build_i2v_conditioning(text_encoder, vae, prompt, width, height, length, first_frame, last_frame):
        calls["i2v"] = dict(text_encoder=text_encoder, vae=vae, prompt=prompt, width=width, height=height,
                            length=length, first_frame=first_frame, last_frame=last_frame)
        return OUT

    def build_reference_conditioning(text_encoder, vae, audio_vae, prompt, width, height, length,
                                    ref_image_size_mode, ref_images, ref_videos, ref_video_audios, ref_audios):
        calls["ref"] = dict(text_encoder=text_encoder, vae=vae, audio_vae=audio_vae, prompt=prompt, width=width,
                            height=height, length=length, ref_image_size_mode=ref_image_size_mode,
                            ref_images=ref_images, ref_videos=ref_videos, ref_video_audios=ref_video_audios,
                            ref_audios=ref_audios)
        return OUT

    mod.build_i2v_conditioning = build_i2v_conditioning
    mod.build_reference_conditioning = build_reference_conditioning
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.minimax_h3_conditioning", mod)
    return calls


def test_i2v_execute_delegates_each_value_to_its_parameter(monkeypatch):
    calls = _install_fake_builders(monkeypatch)
    out = nodes.ASDX_MiniMaxH3ImageToVideo.execute(ENCODER, "a cat", 96, 64, 124, vae="VAE",
                                                   first_frame="F", last_frame="L")
    assert tuple(out.values) == OUT
    assert calls["i2v"] == dict(text_encoder=ENCODER, vae="VAE", prompt="a cat", width=96, height=64,
                                length=124, first_frame="F", last_frame="L")


def test_i2v_optional_inputs_default_to_none(monkeypatch):
    calls = _install_fake_builders(monkeypatch)
    nodes.ASDX_MiniMaxH3ImageToVideo.execute(ENCODER, "a cat", 96, 64, 124)
    assert calls["i2v"]["vae"] is None and calls["i2v"]["first_frame"] is None and calls["i2v"]["last_frame"] is None


@pytest.mark.parametrize("cls_name", ["ASDX_MiniMaxH3ImageToVideo", "ASDX_MiniMaxH3ReferenceToVideo"])
@pytest.mark.parametrize("bad", [{"type": "other"}, {}, None, "encoder"])
def test_wrong_encoder_type_is_rejected_before_delegating(monkeypatch, cls_name, bad):
    calls = _install_fake_builders(monkeypatch)
    with pytest.raises(RuntimeError, match="ASDX_MiniMaxH3TextEncoderLoader"):
        getattr(nodes, cls_name).execute(bad, "a cat", 96, 64, 124)
    assert calls == {}


def test_ref_execute_delegates_each_value_to_its_parameter(monkeypatch):
    calls = _install_fake_builders(monkeypatch)
    out = nodes.ASDX_MiniMaxH3ReferenceToVideo.execute(
        ENCODER, "a cat", 96, 64, 124, ref_image_size="max", vae="V", audio_vae="A",
        ref_images={"ref_image_1": "I"}, ref_videos={"ref_video_1": "VID"},
        ref_video_audios={"ref_video_audio_1": "S"}, ref_audios={"ref_audio_1": "AU"})
    assert tuple(out.values) == OUT
    assert calls["ref"] == dict(
        text_encoder=ENCODER, vae="V", audio_vae="A", prompt="a cat", width=96, height=64, length=124,
        ref_image_size_mode="max", ref_images={"ref_image_1": "I"}, ref_videos={"ref_video_1": "VID"},
        ref_video_audios={"ref_video_audio_1": "S"}, ref_audios={"ref_audio_1": "AU"})


def test_ref_execute_defaults(monkeypatch):
    calls = _install_fake_builders(monkeypatch)
    nodes.ASDX_MiniMaxH3ReferenceToVideo.execute(ENCODER, "a cat", 96, 64, 124)
    c = calls["ref"]
    assert c["ref_image_size_mode"] == "match" and c["vae"] is None and c["audio_vae"] is None
    assert c["ref_images"] == {} and c["ref_videos"] == {} and c["ref_video_audios"] == {} and c["ref_audios"] == {}


def _schemas_with_real_io():
    with real_comfy_isolated():
        import comfy_api.latest as latest

        real_io = latest.io
        saved = nodes.io
        nodes.io = real_io
        try:
            return real_io, {n: getattr(nodes, n).define_schema()
                             for n in ("ASDX_MiniMaxH3ImageToVideo", "ASDX_MiniMaxH3ReferenceToVideo")}
        finally:
            nodes.io = saved


def _inputs(schema) -> dict:
    return {i.id: i for i in schema.inputs}


# (id, io_type, optional, numeric spec or None)
_INT_W = dict(default=1344, min=32, max=4096, step=32)
_INT_H = dict(default=768, min=32, max=4096, step=32)
_INT_L = dict(default=124, min=5, max=3600, step=17)
_COMMON = [
    ("text_encoder", "asdx_minimax_h3_text_encoder", False, None),
    ("vae", "VAE", True, None),
]
_I2V_INPUTS = _COMMON + [
    ("prompt", "STRING", False, None),
    ("width", "INT", False, _INT_W),
    ("height", "INT", False, _INT_H),
    ("length", "INT", False, _INT_L),
    ("first_frame", "IMAGE", True, None),
    ("last_frame", "IMAGE", True, None),
]
_REF_INPUTS = _COMMON + [
    ("audio_vae", "VAE", True, None),
    ("prompt", "STRING", False, None),
    ("width", "INT", False, _INT_W),
    ("height", "INT", False, _INT_H),
    ("length", "INT", False, _INT_L),
    ("ref_image_size", "COMBO", False, None),
    ("ref_images", "COMFY_AUTOGROW_V3", True, None),
    ("ref_videos", "COMFY_AUTOGROW_V3", True, None),
    ("ref_video_audios", "COMFY_AUTOGROW_V3", True, None),
    ("ref_audios", "COMFY_AUTOGROW_V3", True, None),
]


def _assert_inputs(schema, expected) -> None:
    actual = list(schema.inputs)
    assert [i.id for i in actual] == [e[0] for e in expected]
    for inp, (name, io_type, optional, spec) in zip(actual, expected):
        assert inp.io_type == io_type, name
        assert bool(inp.optional) is optional, name
        if spec is not None:
            assert {k: getattr(inp, k) for k in spec} == spec, name


def test_schemas_build_with_the_real_comfy_api():
    """Real `comfy_api.latest.io` (not the stub): node ids, category, every input's id, order,
    socket type and optional flag, numeric widgets, output layout/types, the ref_image_size Combo
    and the Autogrow prefixes, caps and item types."""
    try:
        real_io, schemas = _schemas_with_real_io()
    except ImportError as e:
        pytest.skip(f"real comfy_api not importable: {e}")
    i2v = schemas["ASDX_MiniMaxH3ImageToVideo"]
    ref = schemas["ASDX_MiniMaxH3ReferenceToVideo"]
    assert i2v.node_id == "ASDX_MiniMaxH3ImageToVideo"
    assert ref.node_id == "ASDX_MiniMaxH3ReferenceToVideo"
    for schema in (i2v, ref):
        assert schema.category == "ASDX/Conditioning"
        assert [o.display_name for o in schema.outputs] == ["conditioning", "video_latent", "audio_latent"]
        assert [o.io_type for o in schema.outputs] == ["asdx_minimax_h3_conditioning", "LATENT", "LATENT"]
    _assert_inputs(i2v, _I2V_INPUTS)
    _assert_inputs(ref, _REF_INPUTS)
    ins = _inputs(ref)
    combo = ins["ref_image_size"]
    assert isinstance(combo, real_io.Combo.Input)
    assert list(combo.options) == ["match", "max"] and combo.default == "match"
    for name, prefix, cap, item_type in (
        ("ref_images", "ref_image_", 9, "IMAGE"), ("ref_videos", "ref_video_", 3, "IMAGE"),
        ("ref_video_audios", "ref_video_audio_", 3, "AUDIO"), ("ref_audios", "ref_audio_", 3, "AUDIO"),
    ):
        grow = ins[name]
        assert isinstance(grow, real_io.Autogrow.Input)
        assert grow.template.prefix == prefix and grow.template.max == cap and grow.template.min == 0
        assert grow.template.input.io_type == item_type
