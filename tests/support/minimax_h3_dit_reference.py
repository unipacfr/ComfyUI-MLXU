"""A tiny real ComfyUI `MiniMaxH3Model` (random weights, fp32, CPU) and the
matching MLX model + payloads, so parity tests compare the SAME forward."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support.comfyui_reference_loader import load_real_comfy_minimax_model
from support.minimax_h3_module_loader import load_native_module

condition = load_native_module("minimax_h3.condition")
config_mod = load_native_module("minimax_h3.config")
model_mod = load_native_module("minimax_h3.model")

TINY = dict(
    num_layers=2, token_refiner_num_layers=1, hidden_size=16, latents_dim=4, audio_latents_dim=6,
    attention_head_dim=16, num_attention_heads=2, ffn_hidden_size=32, text_dim=10,
    patch_size=(1, 2, 2), rope_inv_freq_len=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
    sigma_shift_video=12.0, sigma_shift_audio=3.0, gate_compress=False,
    adaln_curve_grid=17, time_embed_dim=6,
)


def build_reference_model(seed: int = 0):
    """Real `comfy.ldm.minimax.model.MiniMaxH3Model`, tiny, initialized (the
    `disable_weight_init` ops leave weights and buffers uninitialized)."""
    mm = load_real_comfy_minimax_model()
    import comfy.ops
    import torch

    ref = mm.MiniMaxH3Model(dtype=torch.float32, operations=comfy.ops.disable_weight_init, **TINY)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ref.parameters():
            # matrices x4 vs 0.1: at 0.1 the conditions moved the output by ~1e-3, too little
            # for whole-model parity to separate subtle mutations from the fp32 residual
            p.copy_(torch.randn(p.shape, generator=gen) * (0.4 if p.dim() >= 2 else 0.1))
        ref.adaln_t_table.copy_(torch.randn(ref.adaln_t_table.shape, generator=gen))
        n = ref.rope.inv_freq.shape[0]
        ref.rope.inv_freq.copy_(1.0 / (10000.0 ** (torch.arange(n, dtype=torch.float32) / n)))
    # inference-only: comfy_kitchen's in-place RoPE refuses tensors that require grad
    return ref.eval().requires_grad_(False)


# Persistent reference state keys our model deliberately does not consume. Empty: the
# reference and our model have exactly the same 1:1 key set (checked when this was written).
REFERENCE_KEYS_NOT_USED: frozenset[str] = frozenset()


def ours_from_reference(ref):
    """Our MLX model with the reference's exact weights. Strict: the key sets must match
    (modulo `REFERENCE_KEYS_NOT_USED`) and every shape must be identical (no reshape, so a
    transposed same-numel weight fails loudly)."""
    model = model_mod.MiniMaxH3Model(config_mod.MiniMaxH3Config(dtype="float32", **TINY))
    state = {k: mx.array(v.detach().numpy()) for k, v in ref.state_dict().items()}
    ours = dict(tree_flatten(model.parameters()))
    missing = sorted(set(ours) - set(state))
    assert not missing, f"reference state_dict lacks {missing[:5]}"
    unused = sorted(set(state) - set(ours) - REFERENCE_KEYS_NOT_USED)
    assert not unused, f"reference state keys not consumed by our model: {unused[:5]}"
    bad = [(k, tuple(state[k].shape), tuple(ours[k].shape)) for k in ours if state[k].shape != ours[k].shape]
    assert not bad, f"shape mismatch (name, reference, ours): {bad[:5]}"
    model.update(tree_unflatten([(k, state[k]) for k in ours]))
    mx.eval(model.parameters())
    return model


def torch_noise(shape, seed: int) -> mx.array:
    """The reference's own noise draw (fresh CPU generator per condition)."""
    import torch

    gen = torch.Generator("cpu").manual_seed(int(seed))
    return mx.array(torch.randn(tuple(shape), generator=gen, dtype=torch.float32).numpy())


def _rand(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# name -> spec. Latent shapes: video [1, 4, T, h, w] (h, w even), audio [1, 6, 2, T].
CASES: dict[str, dict] = {
    "t2v": {},
    "keyframe_first": {"keyframes": [dict(idx=0, video=(1, 4, 4))]},
    "keyframes_first_last": {"keyframes": [dict(idx=0, video=(1, 4, 4)), dict(idx=7, video=(1, 4, 4))]},
    "keyframe_with_audio": {"keyframes": [dict(idx=0, video=(1, 4, 4), audio=3)]},
    "ref_image": {"refs": [dict(kind="image", video=(1, 4, 4))]},
    "ref_video_audio": {"refs": [dict(kind="video_audio", video=(2, 4, 4), audio=3)]},
    "ref_audio": {"refs": [dict(kind="audio", audio=2)]},
    "ref_audio_then_image": {"refs": [dict(kind="audio", audio=2), dict(kind="image", video=(1, 4, 4))]},
    "ref_image_other_res": {"refs": [dict(kind="image", video=(1, 4, 8))]},
    "ref_video_other_res": {"refs": [dict(kind="video_audio", video=(2, 8, 4), audio=3)]},
    "tags": {"tags": [1, 1, 0, 0, 0, 1]},
    "combined": {
        "keyframes": [dict(idx=0, video=(1, 4, 4), audio=2)],
        "refs": [dict(kind="image", video=(1, 4, 4)), dict(kind="video_audio", video=(2, 4, 4), audio=3)],
        "tags": [1, 1, 0, 0, 0, 1],
    },
}
TEXT_LEN = 6  # matches the tags above; every case uses a 6-row context


def payload_pair(name: str, seed: int = 7):
    """`(our ConditionPayload, ComfyUI minimax_payload dict)` for a named case."""
    import torch

    spec = CASES[name]
    kfs, refs = [], []
    ref_kfs, ref_refs = [], []
    n = 0
    for kf in spec.get("keyframes", []):
        n += 1
        video = _rand((1, TINY["latents_dim"], *kf["video"]), n) if "video" in kf else None
        audio = _rand((1, TINY["audio_latents_dim"], 2, kf["audio"]), n + 100) if "audio" in kf else None
        kfs.append(condition.KeyframeCond(kf["idx"], None if video is None else mx.array(video), None if audio is None else mx.array(audio)))
        ref_kfs.append({"resolved_frame_index": kf["idx"],
                        **({"latent": torch.from_numpy(video)} if video is not None else {}),
                        **({"audio_latent": torch.from_numpy(audio)} if audio is not None else {})})
    for r in spec.get("refs", []):
        n += 1
        video = _rand((1, TINY["latents_dim"], *r["video"]), n) if "video" in r else None
        audio = _rand((1, TINY["audio_latents_dim"], 2, r["audio"]), n + 100) if "audio" in r else None
        t, h, w = r.get("video", (0, 0, 0))
        refs.append(condition.RefBlock(kind=r["kind"], latent=None if video is None else mx.array(video),
                                       audio_latent=None if audio is None else mx.array(audio),
                                       latent_t=t, latent_h=h, latent_w=w, ref_audio_t=r.get("audio", 0)))
        blk = {"kind": r["kind"], "latent_t": t, "latent_h": h, "latent_w": w, "ref_audio_t": r.get("audio", 0)}
        if video is not None:
            blk["latent"] = torch.from_numpy(video)
        if audio is not None:
            blk["audio_latent"] = torch.from_numpy(audio)
        ref_refs.append(blk)
    tags = spec.get("tags")
    ours = condition.ConditionPayload(
        text_token_tags=None if tags is None else np.asarray(tags, dtype=np.int64),
        keyframes=tuple(kfs), refs=tuple(refs), seed=seed,
    )
    theirs: dict = {"seed": seed}
    if ref_kfs:
        theirs["keyframes"] = ref_kfs
        theirs["cond_video_latents"] = [k["latent"] for k in ref_kfs if "latent" in k]
        theirs["cond_audio_latents"] = [k["audio_latent"] for k in ref_kfs if "audio_latent" in k]
    if ref_refs:
        theirs["refs"] = ref_refs
        theirs["cond_video_latents"] = theirs.get("cond_video_latents", []) + [r["latent"] for r in ref_refs if "latent" in r]
        theirs["cond_audio_latents"] = theirs.get("cond_audio_latents", []) + [r["audio_latent"] for r in ref_refs if "audio_latent" in r]
    if tags is not None:
        theirs["text_token_tags"] = torch.tensor(tags, dtype=torch.long)[None]
    return ours, theirs
