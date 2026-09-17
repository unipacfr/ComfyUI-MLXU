"""Tests for `_maybe_apply_minimax_h3_turbo_recipe` (lora.py).

Runs in the bare project venv via `comfy_stub.load_node_module`, same
pattern as `test_minimax_h3_nodes.py`.
"""

from __future__ import annotations

import dataclasses

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
lora_module = load_node_module("lora")
_maybe_apply_minimax_h3_turbo_recipe = lora_module._maybe_apply_minimax_h3_turbo_recipe


@dataclasses.dataclass(frozen=True)
class _FakeConfig:
    sigma_shift_video: float = 12.0
    sigma_shift_audio: float = 3.0


def _model(family: str = "minimax_h3") -> dict:
    return {"family": family, "config": _FakeConfig()}


def test_matches_768p_4step_recipe():
    model = _model()
    new_model = _maybe_apply_minimax_h3_turbo_recipe(
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors", model
    )
    assert new_model["config"].sigma_shift_video == 6.0
    assert new_model["config"].sigma_shift_audio == 3.0


def test_matches_8step_recipe():
    model = _model()
    new_model = _maybe_apply_minimax_h3_turbo_recipe(
        "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors", model
    )
    assert new_model["config"].sigma_shift_video == 12.0
    assert new_model["config"].sigma_shift_audio == 3.0


def test_matches_comfyui_export_twin():
    # The file actually shipped locally (Civitai) is the `_comfyui_` twin,
    # not the diffusers-named file the table is keyed on.
    model = _model()
    new_model = _maybe_apply_minimax_h3_turbo_recipe(
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", model
    )
    assert new_model["config"].sigma_shift_video == 6.0
    assert new_model["config"].sigma_shift_audio == 3.0


def test_matches_8step_comfyui_export_twin():
    model = _model()
    new_model = _maybe_apply_minimax_h3_turbo_recipe(
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", model
    )
    assert new_model["config"].sigma_shift_video == 12.0
    assert new_model["config"].sigma_shift_audio == 3.0


def test_matching_is_case_insensitive():
    model = _model()
    new_model = _maybe_apply_minimax_h3_turbo_recipe(
        "MiniMax_H3_FL2V_Turbo_4Step_v0.1.safetensors", model
    )
    assert new_model["config"].sigma_shift_video == 12.0


def test_unrecognized_lora_is_a_noop():
    model = _model()
    new_model = _maybe_apply_minimax_h3_turbo_recipe("some_other_lora.safetensors", model)
    assert new_model is model


def test_noop_for_non_minimax_h3_family():
    model = _model(family="flux1")
    new_model = _maybe_apply_minimax_h3_turbo_recipe(
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors", model
    )
    assert new_model is model
