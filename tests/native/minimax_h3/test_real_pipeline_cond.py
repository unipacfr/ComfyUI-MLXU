"""End to end on real files (i2v): TE + vision tower + REAL video VAE -> conditioning -> real fl2va DiT,
1 step, tiny canvas (256x384, length 5). The VAE is a real `comfy.sd.VAE` built from the real
checkpoint (no fake VAE). Stages run sequentially with everything freed in between (64 GB machine)."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from tests.support.comfy_stub import install_comfy_stubs, load_node_module
from tests.support.comfyui_reference_loader import real_comfy_isolated
from tests.support.minimax_h3_module_loader import load_native_module

MODELS = Path("/Volumes/X10Pro/Images/models")
TE = MODELS / "text_encoders" / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE = MODELS / "vae" / "minimax_h3_video_vae_fp16.safetensors"
DIT = MODELS / "diffusion_models" / "MiniMax H3" / "base model" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"


def _mem(tag: str) -> str:
    """Format MLX active/cache/peak memory in GB."""
    return f"[mem {tag}] active {mx.get_active_memory() / 1e9:.2f} GB, cache {mx.get_cache_memory() / 1e9:.2f} GB, peak {mx.get_peak_memory() / 1e9:.2f} GB"


def _free() -> None:
    gc.collect()
    mx.clear_cache()


@pytest.mark.skipif(os.environ.get("ASDX_FULL_GGUF_TEST") != "1", reason="loads real TE, VAE and DiT; set ASDX_FULL_GGUF_TEST=1")
def test_i2v_pipeline_on_real_files() -> None:
    for p in (TE, VAE, DIT):
        if not p.exists():
            pytest.skip(f"{p.name} not present")
    import torch

    # Import mechanism only: the package `__init__` needs comfy_api, so the node modules are
    # loaded with the project stubs; the real ComfyUI is put on sys.path (isolated) for the
    # stage that needs the real tokenizer and the real `comfy.sd.VAE`.
    install_comfy_stubs()
    nodes = load_node_module("minimax_h3_nodes")
    cond_builders = load_node_module("minimax_h3_conditioning")
    te_wm = load_native_module("minimax_h3.text_encoder_weight_map")
    vis_wm = load_native_module("minimax_h3.vision_weight_map")
    dit_wm = load_native_module("minimax_h3.weight_map")
    sampling = load_native_module("minimax_h3.sampling")

    mx.clear_cache()
    mx.reset_peak_memory()
    prompt = "a cat walks"
    frame = torch.from_numpy(np.random.default_rng(0).random((1, 256, 384, 3), dtype=np.float32))

    with real_comfy_isolated():
        import comfy.sd
        import comfy.utils

        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(VAE)))
        text_encoder = {"type": "asdx_minimax_h3_text_encoder",
                        "encoder": te_wm.load_qwen3_text_encoder_checkpoint(TE, dtype="float16"),
                        "vision_tower": vis_wm.load_vision_tower(TE)}
        print(_mem("stage 1 after TE + tower + VAE load"))
        conditioning, video_latent, audio_latent = cond_builders.build_i2v_conditioning(
            text_encoder, vae, prompt, 384, 256, 5, frame, None)
        text_only = nodes.encode_minimax_h3_prompt(text_encoder, prompt)
        mx.eval(conditioning["hidden_states"], text_only["hidden_states"])
        print(_mem("stage 1 after i2v conditioning + text-only encode"))
        kf_latent = conditioning["keyframes"][0].latent
        kf_shape = tuple(kf_latent.shape)
        kf_finite = bool(mx.all(mx.isfinite(kf_latent)).item())
        text_rows = int(text_only["hidden_states"].shape[0])
        del text_only, text_encoder, vae, kf_latent
        _free()
        torch.mps.empty_cache()
    print(_mem("stage 2 after freeing TE + tower + VAE"))

    assert kf_shape == (1, 24, 1, 16, 24) and kf_finite
    assert [k.resolved_frame_index for k in conditioning["keyframes"]] == [0]
    hidden = conditioning["hidden_states"]
    tags = np.array(conditioning["token_tags"])
    assert hidden.shape[1] == 5120 and tags.shape[0] == hidden.shape[0]
    assert hidden.shape[0] > text_rows, "the image must add vision rows over a text-only encode"
    assert (tags == 0).any(), "vision rows must be tagged 0"

    model = dit_wm.load_minimax_h3_checkpoint(DIT, dtype="float16")
    print(_mem("stage 3 after DiT load"))
    video = mx.random.normal(tuple(video_latent["samples"].shape))
    audio = mx.random.normal(tuple(audio_latent["samples"].shape))
    payload = nodes.payload_from_conditioning(conditioning, seed=1)
    with_kf = sampling.run_minimax_h3_sampling(model, video, audio, hidden, 1, payload=payload)
    plain = sampling.run_minimax_h3_sampling(model, video, audio, hidden, 1)
    mx.eval(*with_kf, *plain)
    delta = float(mx.max(mx.abs(with_kf[0] - plain[0])).item())
    print(f"[real i2v] peak {mx.get_peak_memory() / 1e9:.2f} GB, keyframe delta {delta:.3f}")
    print(_mem("stage 3 after 1 sampling step x2"))
    assert bool(mx.all(mx.isfinite(with_kf[0])).item()) and bool(mx.all(mx.isfinite(with_kf[1])).item())
    assert delta > 1e-3
