"""End to end on real files (ref2va): TE + vision tower + REAL video VAE + REAL audio VAE -> reference
conditioning -> real ref2va DiT, 1 step, tiny canvas (384x256, length 5). One image reference, one
5-frame video reference and one ~1 s stereo audio reference. Both VAEs are real `comfy.sd.VAE`s built
from the real checkpoints (no fakes). Stages run sequentially with everything freed in between (64 GB)."""

from __future__ import annotations

import gc
import os
import re
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
AUDIO_VAE = MODELS / "vae" / "minimax_h3_audio_vae_fp32.safetensors"
DIT = MODELS / "diffusion_models" / "MiniMax H3" / "base model" / "minimax_h3_ref2va_pruned_int8_convrot.safetensors"


def _mem(tag: str) -> str:
    """Format MLX active/cache/peak memory in GB."""
    return f"[mem {tag}] active {mx.get_active_memory() / 1e9:.2f} GB, cache {mx.get_cache_memory() / 1e9:.2f} GB, peak {mx.get_peak_memory() / 1e9:.2f} GB"


def _free() -> None:
    gc.collect()
    mx.clear_cache()


@pytest.mark.skipif(os.environ.get("ASDX_FULL_GGUF_TEST") != "1", reason="loads real TE, VAEs and DiT; set ASDX_FULL_GGUF_TEST=1")
def test_ref2va_pipeline_on_real_files(capsys) -> None:
    for p in (TE, VAE, AUDIO_VAE, DIT):
        if not p.exists():
            pytest.skip(f"{p.name} not present")
    import torch

    install_comfy_stubs()
    nodes = load_node_module("minimax_h3_nodes")
    cond_builders = load_node_module("minimax_h3_conditioning")
    te_wm = load_native_module("minimax_h3.text_encoder_weight_map")
    vis_wm = load_native_module("minimax_h3.vision_weight_map")
    dit_wm = load_native_module("minimax_h3.weight_map")
    sampling = load_native_module("minimax_h3.sampling")
    layout_mod = load_native_module("minimax_h3.layout")

    mx.clear_cache()
    mx.reset_peak_memory()
    width, height, length = 384, 256, 5
    prompt = "a cat walks"
    rng = np.random.default_rng(0)
    image = torch.from_numpy(rng.random((1, 256, 384, 3), dtype=np.float32))
    video = torch.from_numpy(rng.random((5, 256, 384, 3), dtype=np.float32))
    stereo = {"waveform": torch.from_numpy(rng.standard_normal((1, 2, 32000)).astype(np.float32)) * 0.1, "sample_rate": 32000}

    with real_comfy_isolated():
        import comfy.sd
        import comfy.text_encoders.minimax as tok_mod
        import comfy.utils

        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(VAE)))
        audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(AUDIO_VAE)))
        audio_rate = getattr(audio_vae, "audio_sample_rate", None)
        print(f"[real ref] audio VAE audio_sample_rate = {audio_rate}")
        text_encoder = {"type": "asdx_minimax_h3_text_encoder",
                        "encoder": te_wm.load_qwen3_text_encoder_checkpoint(TE, dtype="float16"),
                        "vision_tower": vis_wm.load_vision_tower(TE)}
        print(_mem("stage 1 after TE + tower + video VAE + audio VAE load"))

        # image reference alone through the real video VAE: latent_t must be 1
        single = cond_builders.encode_video_latent(vae, image)
        print(f"[real ref] single-image latent shape {tuple(single.shape)}")

        # record the entries the real tokenizer produced for the reference prompt
        captured: list = []
        original = tok_mod.MiniMaxH3Tokenizer.tokenize_with_weights

        def spy(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            captured.append((self, out["qwen3vl_32b"][0]))
            return out

        tok_mod.MiniMaxH3Tokenizer.tokenize_with_weights = spy
        try:
            print(capsys.readouterr().out, end="")  # keep what was printed so far; only the builder output is parsed
            conditioning, video_latent, audio_latent = cond_builders.build_reference_conditioning(
                text_encoder, vae, audio_vae, prompt, width, height, length, "match",
                ref_images={"ref_image_1": image}, ref_videos={"ref_video_1": video}, ref_video_audios={},
                ref_audios={"ref_audio_1": stereo})
            builder_out = capsys.readouterr().out
        finally:
            tok_mod.MiniMaxH3Tokenizer.tokenize_with_weights = original
        print(builder_out, end="")
        tokenizer, entries = captured[-1]
        ids = [e[0] for e in entries if isinstance(e[0], int)]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        labels = re.findall(r"<(Picture|Video|Audio) (\d+)>", decoded)

        text_only = nodes.encode_minimax_h3_prompt(text_encoder, prompt)
        mx.eval(conditioning["hidden_states"], text_only["hidden_states"])
        print(_mem("stage 1 after ref conditioning + text-only encode"))
        text_rows = int(text_only["hidden_states"].shape[0])

        # audio path at 44.1 kHz, when torchaudio is available in the isolated window
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            print("[real ref] torchaudio not available inside the isolated ComfyUI window: 44.1 kHz path NOT checked")
            z441_shape = None
        else:
            wave441 = {"waveform": torch.from_numpy(rng.standard_normal((1, 2, 44100)).astype(np.float32)) * 0.1, "sample_rate": 44100}
            z441, t441 = cond_builders.encode_audio_latent(audio_vae, wave441)
            z441_shape = (tuple(z441.shape), t441)
            print(f"[real ref] 44.1 kHz audio latent {z441_shape}")
        refs = tuple(conditioning["refs"])
        z_audio_shape = tuple(refs[2].audio_latent.shape)
        del text_only, text_encoder, vae, audio_vae, single
        _free()
        torch.mps.empty_cache()
    print(_mem("stage 2 after freeing TE + tower + VAEs"))

    print(f"[real ref] labels {labels}")
    print(f"[real ref] ref kinds {[r.kind for r in refs]}")
    for r in refs:
        print(f"[real ref] {r.kind}: latent {None if r.latent is None else tuple(r.latent.shape)}, audio_latent "
              f"{None if r.audio_latent is None else tuple(r.audio_latent.shape)}, latent_t {r.latent_t}, "
              f"h {r.latent_h}, w {r.latent_w}, ref_audio_t {r.ref_audio_t}")

    assert audio_rate == 32000
    assert [r.kind for r in refs] == ["image", "video", "audio"]
    assert refs[0].latent.shape[2] == refs[0].latent_t == 1  # the real VAE gives one latent frame for one image
    for r in refs[:2]:
        assert tuple(r.latent.shape[3:]) == (r.latent_h, r.latent_w) and r.latent.shape[2] == r.latent_t
    assert refs[2].ref_audio_t == z_audio_shape[-1] > 0 and z_audio_shape[1:3] == (32, 2)
    assert all(bool(mx.all(mx.isfinite(x)).item()) for r in refs for x in (r.latent, r.audio_latent) if x is not None)
    assert labels == [("Picture", "1"), ("Video", "1"), ("Audio", "1")]

    hidden = conditioning["hidden_states"]
    tags = np.array(conditioning["token_tags"])
    assert hidden.shape[1] == 5120 and tags.shape[0] == hidden.shape[0]
    assert hidden.shape[0] > text_rows, "the image and video must add vision rows over a text-only encode"
    assert (tags == 0).any(), "vision rows must be tagged 0"

    latent_t, audio_t = video_latent["samples"].shape[2], audio_latent["samples"].shape[3]
    real = layout_mod.PackedLayout(int(hidden.shape[0]), latent_t, height // 16, width // 16, audio_t, refs=refs)
    reported = int(re.search(r"packed sequence: (\d+) rows", builder_out).group(1))
    text_len = int(hidden.shape[0])
    assert reported == text_len + (real.seq_len - text_len)
    print(f"[real ref] reported rows {reported}, layout seq_len {real.seq_len}, text rows {hidden.shape[0]}")

    model = dit_wm.load_minimax_h3_checkpoint(DIT, dtype="float16")
    print(_mem("stage 3 after DiT load"))
    video_noise = mx.random.normal(tuple(video_latent["samples"].shape))
    audio_noise = mx.random.normal(tuple(audio_latent["samples"].shape))
    payload = nodes.payload_from_conditioning(conditioning, seed=1)
    with_refs = sampling.run_minimax_h3_sampling(model, video_noise, audio_noise, hidden, 1, payload=payload)
    plain = sampling.run_minimax_h3_sampling(model, video_noise, audio_noise, hidden, 1)
    mx.eval(*with_refs, *plain)
    delta = float(mx.max(mx.abs(with_refs[0] - plain[0])).item())
    print(f"[real ref] peak {mx.get_peak_memory() / 1e9:.2f} GB, reference delta {delta:.3f}")
    print(_mem("stage 3 after 1 sampling step x2"))
    assert bool(mx.all(mx.isfinite(with_refs[0])).item()) and bool(mx.all(mx.isfinite(with_refs[1])).item())
    assert delta > 1e-3
