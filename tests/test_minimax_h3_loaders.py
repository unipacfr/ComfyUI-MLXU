"""Tests for ASDX_MiniMaxH3ModelLoader / ASDX_MiniMaxH3TextEncoderLoader /
ASDX_MiniMaxH3TextEncode.

Runs in the bare project venv via ``comfy_stub.load_node_module``. Real
weight loading (``load_minimax_h3_checkpoint`` /
``load_qwen3_text_encoder_checkpoint``) is monkeypatched out -- those
functions already have their own dedicated, real-checkpoint-verified tests
(``test_weight_map.py`` / ``test_text_encoder_weight_map.py``); this file
only exercises the node-level plumbing (file resolution, caching, the
tokenizer bridge, error handling).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
nodes_module = load_node_module("minimax_h3_nodes")
ASDX_MiniMaxH3ModelLoader = nodes_module.ASDX_MiniMaxH3ModelLoader
ASDX_MiniMaxH3TextEncoderLoader = nodes_module.ASDX_MiniMaxH3TextEncoderLoader
ASDX_MiniMaxH3TextEncode = nodes_module.ASDX_MiniMaxH3TextEncode


def _fake_folder_paths(files: dict[str, str]):
    """`files`: {name: full_path}. Installed as sys.modules["folder_paths"]."""
    mod = types.ModuleType("folder_paths")

    def get_filename_list(folder):
        return list(files.keys())

    def get_full_path(folder, name):
        return files.get(name)

    mod.get_filename_list = get_filename_list
    mod.get_full_path = get_full_path
    return mod


_REAL_SUPPORTED_PT_EXTENSIONS = {".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft"}


def _realistic_fake_folder_paths(disk_files: dict[str, tuple[str, str]]):
    """A closer simulation of the real `folder_paths` module than
    `_fake_folder_paths` above: `get_filename_list` actually filters by each
    folder key's registered extension set, the same way the real ComfyUI
    module does (and the same way it silently hid MiniMax H3's own .gguf
    checkpoints before `_register_gguf_extension` was added -- this is the
    regression test for exactly that bug).

    `disk_files`: {name: (folder_key, full_path)} -- every file physically
    "on disk", regardless of whether its extension is currently registered
    for its folder.
    """
    mod = types.ModuleType("folder_paths")
    mod.folder_names_and_paths = {
        "diffusion_models": (["/models/diffusion_models", "/models/unet"], set(_REAL_SUPPORTED_PT_EXTENSIONS)),
        "text_encoders": (["/models/text_encoders", "/models/clip"], set(_REAL_SUPPORTED_PT_EXTENSIONS)),
    }

    def get_filename_list(folder_key):
        # Mirrors the real folder_paths.get_filename_list: once a folder key's
        # scan is cached, it is reused verbatim regardless of a later
        # extension-set change -- only filename_list_cache.pop() clears it.
        cached = mod.filename_list_cache.get(folder_key)
        if cached is not None:
            return cached
        _, extensions = mod.folder_names_and_paths.get(folder_key, ([], set()))
        result = [
            name for name, (key, _path) in disk_files.items()
            if key == folder_key and Path(name).suffix.lower() in extensions
        ]
        mod.filename_list_cache[folder_key] = result
        return result

    def get_full_path(folder_key, name):
        entry = disk_files.get(name)
        if entry is None or entry[0] != folder_key:
            return None
        return entry[1]

    mod.get_filename_list = get_filename_list
    mod.get_full_path = get_full_path
    mod.filename_list_cache = {}
    return mod


@pytest.fixture(autouse=True)
def _clear_caches():
    nodes_module._DIT_CACHE.clear()
    nodes_module._TEXT_ENCODER_CACHE.clear()
    yield
    nodes_module._DIT_CACHE.clear()
    nodes_module._TEXT_ENCODER_CACHE.clear()


def test_model_loader_raises_when_file_not_found(monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", _fake_folder_paths({}))
    with pytest.raises(RuntimeError, match="could not find"):
        ASDX_MiniMaxH3ModelLoader.execute("missing.gguf")


def test_model_loader_calls_weight_map_and_caches(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "folder_paths", _fake_folder_paths({"h3.gguf": "/models/unet/h3.gguf"})
    )
    fake_model = Mock()
    fake_model.config = Mock()
    load_calls = []

    def fake_load(path, dtype):
        load_calls.append((str(path), dtype))
        return fake_model

    weight_map_stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.weight_map")
    weight_map_stub.load_minimax_h3_checkpoint = fake_load
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.weight_map", weight_map_stub)

    result = ASDX_MiniMaxH3ModelLoader.execute("h3.gguf", precision="float16")
    model_desc = result.values[0]
    assert model_desc["type"] == "asdx_model"
    assert model_desc["family"] == "minimax_h3"
    assert model_desc["transformer"] is fake_model
    assert load_calls == [("/models/unet/h3.gguf", "float16")]

    # second call with the same path+precision hits the cache -- no second load
    result2 = ASDX_MiniMaxH3ModelLoader.execute("h3.gguf", precision="float16")
    assert result2.values[0] is model_desc
    assert len(load_calls) == 1


def test_text_encoder_loader_calls_weight_map(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "folder_paths", _fake_folder_paths({"qwen.gguf": "/models/text_encoders/qwen.gguf"})
    )
    fake_encoder = Mock()
    load_calls = []

    def fake_load(path, dtype):
        load_calls.append((str(path), dtype))
        return fake_encoder

    stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.text_encoder_weight_map")
    stub.load_qwen3_text_encoder_checkpoint = fake_load
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.text_encoder_weight_map", stub)

    result = ASDX_MiniMaxH3TextEncoderLoader.execute("qwen.gguf")
    encoder_desc = result.values[0]
    assert encoder_desc["type"] == "asdx_minimax_h3_text_encoder"
    assert encoder_desc["encoder"] is fake_encoder
    assert load_calls == [("/models/text_encoders/qwen.gguf", "float16")]


def test_text_encode_rejects_wrong_input_type():
    with pytest.raises(RuntimeError, match="ASDX_MiniMaxH3TextEncoderLoader"):
        ASDX_MiniMaxH3TextEncode.execute({"type": "something_else"}, "a prompt")


def _install_fake_minimax_tokenizer(monkeypatch, token_ids, record=None):
    minimax_mod = types.ModuleType("comfy.text_encoders.minimax")

    class FakeTokenizer:
        def tokenize_with_weights(self, text, images=(), minimax_ref_items=None):
            if record is not None:
                record.append({"images": images, "minimax_ref_items": minimax_ref_items})
            return {"qwen3vl_32b": [[(tid, 1.0) for tid in token_ids]]}

    minimax_mod.MiniMaxH3Tokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "comfy.text_encoders.minimax", minimax_mod)
    text_encoders_mod = sys.modules["comfy.text_encoders"]
    monkeypatch.setattr(text_encoders_mod, "minimax", minimax_mod, raising=False)


def test_text_encode_runs_tokenizer_and_encoder(monkeypatch):
    import mlx.core as mx

    token_ids = [64, 8251, 11699, 389, 264, 5517]
    _install_fake_minimax_tokenizer(monkeypatch, token_ids)

    captured_input_ids = {}

    def fake_encoder(input_ids, vision=None):
        captured_input_ids["ids"] = input_ids
        return mx.ones((len(token_ids), 8))

    text_encoder = {"type": "asdx_minimax_h3_text_encoder", "encoder": fake_encoder}
    result = ASDX_MiniMaxH3TextEncode.execute(text_encoder, "a cat sitting on a mat")
    cond = result.values[0]

    assert cond["type"] == "minimax_h3"
    assert cond["text"] == "a cat sitting on a mat"
    assert cond["hidden_states"].shape == (len(token_ids), 8)
    assert captured_input_ids["ids"].tolist() == token_ids


def test_text_encode_raises_on_non_finite_output(monkeypatch):
    import mlx.core as mx

    _install_fake_minimax_tokenizer(monkeypatch, [1, 2, 3])
    text_encoder = {
        "type": "asdx_minimax_h3_text_encoder",
        "encoder": lambda input_ids, vision=None: mx.array([[float("nan")] * 4] * 3),
    }
    with pytest.raises(RuntimeError, match="non-finite"):
        ASDX_MiniMaxH3TextEncode.execute(text_encoder, "prompt")


# ---------------------------------------------------------------------------
# Regression tests for the real bug reported by the user: .gguf files were
# invisible in both loader dropdowns because ComfyUI's real
# folder_paths.supported_pt_extensions (what "diffusion_models"/
# "text_encoders" use) does not include ".gguf" -- confirmed against the
# real folder_paths.py, not assumed. _fake_folder_paths above (used by the
# tests before this point) doesn't simulate extension filtering at all, so
# it could not have caught this; _realistic_fake_folder_paths does.
# ---------------------------------------------------------------------------


def test_gguf_files_invisible_without_extension_registered():
    # Documents the bug itself: querying the unmodified real extension set
    # never returns a .gguf name, regardless of what our own code filters for.
    fp = _realistic_fake_folder_paths({
        "minimax_h3.gguf": ("diffusion_models", "/models/unet/minimax_h3.gguf"),
    })
    assert fp.get_filename_list("diffusion_models") == []


def test_register_gguf_extension_makes_model_gguf_visible(monkeypatch):
    fp = _realistic_fake_folder_paths({
        "minimax_h3.gguf": ("diffusion_models", "/models/unet/minimax_h3.gguf"),
        "some_model.safetensors": ("diffusion_models", "/models/diffusion_models/some_model.safetensors"),
    })
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    nodes_module._register_gguf_extension("diffusion_models")

    names = ASDX_MiniMaxH3ModelLoader._get_models()
    assert names == ["minimax_h3.gguf"]  # only .gguf, the safetensors file is a different loader's job
    # the non-.gguf file must still be visible to other code querying the same key
    assert set(fp.get_filename_list("diffusion_models")) == {"minimax_h3.gguf", "some_model.safetensors"}


def test_register_gguf_extension_makes_text_encoder_gguf_visible(monkeypatch):
    fp = _realistic_fake_folder_paths({
        "qwen3vl.gguf": ("text_encoders", "/models/text_encoders/qwen3vl.gguf"),
    })
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    nodes_module._register_gguf_extension("text_encoders")

    assert ASDX_MiniMaxH3TextEncoderLoader._get_encoders() == ["qwen3vl.gguf"]


def test_register_gguf_extension_is_idempotent(monkeypatch):
    fp = _realistic_fake_folder_paths({})
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    nodes_module._register_gguf_extension("diffusion_models")
    nodes_module._register_gguf_extension("diffusion_models")

    _, extensions = fp.folder_names_and_paths["diffusion_models"]
    assert extensions == _REAL_SUPPORTED_PT_EXTENSIONS | {".gguf"}


def test_register_gguf_extension_busts_a_cache_populated_before_it_ran(monkeypatch):
    """Real-ComfyUI ordering: core loaders (UNETLoader/CLIPLoader) query
    "diffusion_models"/"text_encoders" during core node registration, which
    happens before custom_nodes -- including this module -- are imported.
    That first query caches the scan without .gguf in folder_paths'
    process-lifetime filename_list_cache, which is keyed only on directory
    mtimes and is never invalidated by an extension-set change on its own.
    Without popping that cache entry, `_register_gguf_extension` has no
    visible effect -- this is the bug the user actually hit."""
    fp = _realistic_fake_folder_paths({
        "minimax_h3.gguf": ("diffusion_models", "/models/unet/minimax_h3.gguf"),
    })
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    # simulate a core loader querying the list first, before .gguf is registered
    stale = fp.get_filename_list("diffusion_models")
    assert stale == []

    nodes_module._register_gguf_extension("diffusion_models")

    assert ASDX_MiniMaxH3ModelLoader._get_models() == ["minimax_h3.gguf"]


def test_text_encoder_loader_loads_vision_tower_only_on_request(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "folder_paths", _fake_folder_paths({"qwen.safetensors": "/models/text_encoders/qwen.safetensors"})
    )
    nodes_module._TEXT_ENCODER_CACHE.clear()
    encoder, tower = Mock(), Mock()
    tower_calls = []

    enc_stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.text_encoder_weight_map")
    enc_stub.load_qwen3_text_encoder_checkpoint = lambda path, dtype: encoder
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.text_encoder_weight_map", enc_stub)
    vis_stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.vision_weight_map")
    vis_stub.load_vision_tower = lambda path: tower_calls.append(str(path)) or tower
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.vision_weight_map", vis_stub)

    plain = ASDX_MiniMaxH3TextEncoderLoader.execute("qwen.safetensors").values[0]
    assert plain["vision_tower"] is None and tower_calls == []

    with_vision = ASDX_MiniMaxH3TextEncoderLoader.execute("qwen.safetensors", load_vision=True).values[0]
    assert with_vision["vision_tower"] is tower and tower_calls == ["/models/text_encoders/qwen.safetensors"]
    assert with_vision is not plain  # the flag is part of the cache key


def test_encode_prompt_helper_rejects_wrong_input_type():
    with pytest.raises(RuntimeError, match="ASDX_MiniMaxH3TextEncoderLoader"):
        nodes_module.encode_minimax_h3_prompt({"type": "something_else"}, "a prompt")


def test_encode_prompt_helper_returns_hidden_states_and_tags(monkeypatch):
    import mlx.core as mx
    import numpy as np

    _install_fake_minimax_tokenizer(monkeypatch, [11, 22, 33])
    hidden = mx.ones((3, 8))
    stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.vision_conditioning")
    stub.encode_with_vision = lambda enc, tower, entries: (hidden, np.array([1, 1, 1]))
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.vision_conditioning", stub)

    desc = {"type": "asdx_minimax_h3_text_encoder", "encoder": Mock(), "vision_tower": None}
    out = nodes_module.encode_minimax_h3_prompt(desc, "a prompt")
    assert out["type"] == "minimax_h3" and out["text"] == "a prompt"
    assert out["hidden_states"] is hidden and out["token_tags"].tolist() == [1, 1, 1]


def test_encode_prompt_helper_aborts_on_non_finite(monkeypatch):
    import mlx.core as mx
    import numpy as np

    _install_fake_minimax_tokenizer(monkeypatch, [11])
    stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.vision_conditioning")
    stub.encode_with_vision = lambda enc, tower, entries: (mx.array([[float("nan")]]), np.array([1]))
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.vision_conditioning", stub)
    desc = {"type": "asdx_minimax_h3_text_encoder", "encoder": Mock(), "vision_tower": None}
    with pytest.raises(RuntimeError, match="non-finite"):
        nodes_module.encode_minimax_h3_prompt(desc, "x")


def test_encode_prompt_helper_forwards_images_and_ref_items(monkeypatch):
    import mlx.core as mx
    import numpy as np

    calls: list = []
    _install_fake_minimax_tokenizer(monkeypatch, [11], record=calls)
    stub = types.ModuleType("apple_silicon_nodes.native.minimax_h3.vision_conditioning")
    stub.encode_with_vision = lambda enc, tower, entries: (mx.ones((1, 4)), np.array([1]))
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.vision_conditioning", stub)
    desc = {"type": "asdx_minimax_h3_text_encoder", "encoder": Mock(), "vision_tower": None}

    images, ref_items = [object(), object()], [{"kind": "ref"}]
    nodes_module.encode_minimax_h3_prompt(desc, "p", images=images, ref_items=ref_items)
    assert calls[-1]["images"] is images and calls[-1]["minimax_ref_items"] is ref_items

    nodes_module.encode_minimax_h3_prompt(desc, "p")
    assert calls[-1] == {"images": [], "minimax_ref_items": None}
