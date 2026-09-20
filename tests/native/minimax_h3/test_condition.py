"""condition.prepare_condition vs the real ComfyUI `_cond_video_rows` / `_cond_audio_rows`."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support import minimax_h3_dit_reference as R
from support.minimax_h3_module_loader import load_native_module

cond = load_native_module("minimax_h3.condition")
PATCH = (1, 2, 2)


@pytest.mark.parametrize("name", ["keyframe_first", "keyframes_first_last", "keyframe_with_audio", "ref_image", "ref_video_audio", "ref_audio", "combined"])
def test_rows_match_comfyui(name):
    ref = R.build_reference_model()
    ours, theirs = R.payload_pair(name, seed=7)
    prepared = cond.prepare_condition(ours, PATCH, noise_fn=R.torch_noise)
    want_v = ref._cond_video_rows(theirs, "cpu")
    want_a = ref._cond_audio_rows(theirs, "cpu")
    if want_v is None:
        assert prepared.video_rows is None
    else:
        assert np.abs(np.array(prepared.video_rows) - want_v.numpy()).max() < 1e-6
    if want_a is None:
        assert prepared.audio_rows is None
    else:
        assert np.abs(np.array(prepared.audio_rows) - want_a.numpy()).max() < 1e-6


def test_no_conditions_gives_no_rows():
    prepared = cond.prepare_condition(cond.ConditionPayload(), PATCH)
    assert prepared.video_rows is None and prepared.audio_rows is None


def test_default_noise_is_deterministic_and_seed_dependent():
    a, b, c = cond.default_noise((4, 3), 7), cond.default_noise((4, 3), 7), cond.default_noise((4, 3), 8)
    assert np.array_equal(np.array(a), np.array(b)) and np.abs(np.array(a) - np.array(c)).max() > 0.1


def test_noise_actually_augments_the_rows():
    """Null case: with aug 0.999 the rows differ slightly from the raw patchified latent; with aug 1.0 they equal it."""
    ours, _ = R.payload_pair("keyframe_first", seed=7)
    noisy = cond.prepare_condition(ours, PATCH, noise_fn=R.torch_noise).video_rows
    clean = cond.prepare_condition(
        cond.ConditionPayload(keyframes=ours.keyframes, seed=7, visual_cond_noise_aug=1.0), PATCH, noise_fn=R.torch_noise
    ).video_rows
    delta = float(mx.max(mx.abs(noisy - clean)).item())
    assert 1e-5 < delta < 0.1


def test_same_stream_restarts_for_every_condition():
    ours, _ = R.payload_pair("keyframes_first_last", seed=7)
    calls = []
    cond.prepare_condition(ours, PATCH, noise_fn=lambda shape, seed: calls.append((tuple(shape), seed)) or R.torch_noise(shape, seed))
    assert [s for _, s in calls] == [7, 7]  # every visual condition draws from the same seed


@pytest.mark.parametrize("name", ["keyframe_with_audio", "ref_audio", "combined"])
def test_audio_noise_augmentation_matches_comfyui(name):
    """The default audio aug is 1.0 (clean), which never draws noise; force aug < 1 so the seed + 1 stream is compared."""
    import dataclasses

    ref = R.build_reference_model()
    ours, theirs = R.payload_pair(name, seed=7)
    ours = dataclasses.replace(ours, audio_cond_noise_aug=0.9)
    theirs["audio_cond_noise_aug"] = 0.9
    prepared = cond.prepare_condition(ours, PATCH, noise_fn=R.torch_noise)
    want = ref._cond_audio_rows(theirs, "cpu").numpy()
    clean = cond.prepare_condition(dataclasses.replace(ours, audio_cond_noise_aug=1.0), PATCH, noise_fn=R.torch_noise)
    assert np.abs(np.array(prepared.audio_rows) - np.array(clean.audio_rows)).max() > 1e-3  # noise really applied
    assert np.abs(np.array(prepared.audio_rows) - want).max() < 1e-6
