"""qwen_image21's sigma schedule: same flux_time_shift/ModelSamplingFlux family as Krea2,
fixed shift=0.69 (comfy/supported_models.py::QwenImage21.sampling_settings)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
scheduling_mod = load_node_module("sampler.scheduling")


def test_generate_sigmas_matches_flux_fixed_shift_directly():
    steps = 10
    expected = scheduling_mod._flux_fixed_shift_sigmas(0.69, steps)
    got = scheduling_mod.generate_sigmas(steps, "qwen_image21")
    assert got == expected


def test_generate_sigmas_differs_from_krea2_shift():
    # Confirms it's really using shift=0.69, not accidentally reusing Krea2's 1.15
    # or falling through to a different branch entirely.
    steps = 10
    qwen = scheduling_mod.generate_sigmas(steps, "qwen_image21")
    krea2 = scheduling_mod.generate_sigmas(steps, "krea2")
    assert qwen != krea2


def test_flow_shift_fn_matches_flux_time_shift_at_shift_0_69():
    fn = scheduling_mod._flow_shift_fn("qwen_image21")
    expected = scheduling_mod.flux_time_shift(0.69, 1.0, 0.5)
    assert abs(fn(0.5) - expected) < 1e-9


def test_ends_at_zero():
    sigmas = scheduling_mod.generate_sigmas(10, "qwen_image21")
    assert sigmas[-1] == 0.0
    assert len(sigmas) == 11
