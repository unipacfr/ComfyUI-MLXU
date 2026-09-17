"""Tests for `_gate_minimax_h3_component` (minimax_h3_nodes.py).

Runs in the bare project venv via `comfy_stub.load_node_module`, same
pattern as `test_minimax_h3_loaders.py`. `available_unified_memory_bytes` is
monkeypatched to make the total/available split deterministic instead of
depending on the machine actually running the tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
nodes_module = load_node_module("minimax_h3_nodes")
memory_calibration = load_node_module("memory_calibration")
_gate_minimax_h3_component = nodes_module._gate_minimax_h3_component

GB = 1024 ** 3


@pytest.fixture
def gguf_file(tmp_path: Path) -> Path:
    path = tmp_path / "component.gguf"
    path.write_bytes(b"\0" * 1024)  # size only matters -- content is never read
    return path


def _patch_total_available(monkeypatch, total_gb: float, available_gb: float) -> None:
    monkeypatch.setattr(
        memory_calibration,
        "available_unified_memory_bytes",
        lambda: (int(total_gb * GB), int(available_gb * GB)),
    )


def test_no_other_resident_delegates_to_single_component_gate(gguf_file, monkeypatch):
    _patch_total_available(monkeypatch, total_gb=64, available_gb=64)
    estimate = _gate_minimax_h3_component("dit", gguf_file, "float16", other_cache={})
    assert estimate is not None
    assert estimate.status == "heuristic"


def test_missing_file_returns_none_without_raising(tmp_path, monkeypatch):
    _patch_total_available(monkeypatch, total_gb=64, available_gb=64)
    missing = tmp_path / "does_not_exist.gguf"
    estimate = _gate_minimax_h3_component("dit", missing, "float16", other_cache={})
    assert estimate is None


def test_combined_footprint_within_total_does_not_evict(gguf_file, monkeypatch):
    _patch_total_available(monkeypatch, total_gb=64, available_gb=64)
    other_cache = {"key": {"_predicted_peak_bytes": 1 * GB}}

    estimate = _gate_minimax_h3_component("dit", gguf_file, "float16", other_cache)

    assert estimate is not None
    assert other_cache  # not evicted -- combined footprint fits


def test_combined_footprint_exceeding_total_evicts_other_cache(gguf_file, monkeypatch):
    # A tiny total (well under any real heuristic estimate) forces the
    # "exceeds total memory" branch regardless of the heuristic's exact
    # multiplier.
    _patch_total_available(monkeypatch, total_gb=0.0000001, available_gb=0.0000001)
    other_cache = {"key": {"_predicted_peak_bytes": 40 * GB}}

    estimate = _gate_minimax_h3_component("dit", gguf_file, "float16", other_cache)

    assert estimate is not None
    assert other_cache == {}  # sequential offload: the other component was evicted
