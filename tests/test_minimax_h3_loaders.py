"""Tests for ASDX_MiniMaxH3ModelLoader / ASDX_MiniMaxH3TextEncoderLoader /
ASDX_MiniMaxH3TextEncode.

Runs in the bare project venv via ``comfy_stub.load_node_module``. Real
weight loading (``load_minimax_h3_from_gguf`` /
``load_qwen3_text_encoder_from_gguf``) is monkeypatched out -- those
functions already have their own dedicated, real-checkpoint-verified tests
(``test_weight_map.py`` / ``test_text_encoder_weight_map.py``); this file
only exercises the node-level plumbing (file resolution, caching, the
tokenizer bridge, error handling).
"""

from __future__ import annotations

import sys
import types
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
    weight_map_stub.load_minimax_h3_from_gguf = fake_load
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
    stub.load_qwen3_text_encoder_from_gguf = fake_load
    monkeypatch.setitem(sys.modules, "apple_silicon_nodes.native.minimax_h3.text_encoder_weight_map", stub)

    result = ASDX_MiniMaxH3TextEncoderLoader.execute("qwen.gguf")
    encoder_desc = result.values[0]
    assert encoder_desc["type"] == "asdx_minimax_h3_text_encoder"
    assert encoder_desc["encoder"] is fake_encoder
    assert load_calls == [("/models/text_encoders/qwen.gguf", "float16")]


def test_text_encode_rejects_wrong_input_type():
    with pytest.raises(RuntimeError, match="ASDX_MiniMaxH3TextEncoderLoader"):
        ASDX_MiniMaxH3TextEncode.execute({"type": "something_else"}, "a prompt")


def _install_fake_minimax_tokenizer(monkeypatch, token_ids):
    minimax_mod = types.ModuleType("comfy.text_encoders.minimax")

    class FakeTokenizer:
        def tokenize_with_weights(self, text):
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

    def fake_encoder(input_ids):
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
        "encoder": lambda input_ids: mx.array([[float("nan")] * 4] * 3),
    }
    with pytest.raises(RuntimeError, match="non-finite"):
        ASDX_MiniMaxH3TextEncode.execute(text_encoder, "prompt")
