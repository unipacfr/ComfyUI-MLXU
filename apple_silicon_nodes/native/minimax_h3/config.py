"""
MiniMax H3 audio-video DiT architecture configuration.

Defaults verified against the real checkpoints on this machine (2026-09-14/15
investigation, safetensors header + GGUF header via `native/gguf/reader.py`):
`diffusion_models/MiniMax H3/base model/minimax_h3_fl2va_pruned_int8_convrot.safetensors`
and `unet/MiniMax H3/minimax_h3_fl2va_pruned-Q5_0.gguf` (byte-identical
architecture in both formats) — and against comfy's own config-detection
branch for `video_patch_proj.weight`/`audio_patch_proj.weight`
(`comfy/model_detection.py:390-418`), which `detect_minimax_h3_config` below
mirrors exactly rather than assuming this checkpoint's shape is universal.

Single-stream packed text+audio+video DiT (see `comfy/ldm/minimax/model.py`'s
module docstring for the packed-sequence layout `[text | cond | audio |
video]`) — NOT a two-tower MMDiT like FLUX/SDXL. `patch_size`/`norm_eps`/
`qk_norm_eps`/`final_norm_eps`/`sigma_shift_*` are project-wide constants (not
shape-derived; comfy hardcodes them identically for every MiniMax H3
checkpoint) — `sigma_shift_video`/`sigma_shift_audio` also match
`comfy/supported_models.py::MiniMaxH3.sampling_settings`.

Curve-form adaln, not the plain time-embedder: the real checkpoint has an
`adaln_t_table` buffer (shape `[1025, 8]`) and NO `time_embedder.*` keys.
Per `comfy/ldm/minimax/model.py::MiniMaxH3Model.__init__`, this means the
"curve" variant is in play (`adaln_curve_grid=1025, time_embed_dim=8`) —
a small shared basis of the time-embedding curve replaces the sinusoidal
`TimeEmbedder` module and shrinks the adaln linears' input from a full
`time_embed_dim=2688` (the class default, for the OTHER variant) down to 8.
A checkpoint using the other (non-curve) variant is a real possibility this
project has not seen yet; `detect_minimax_h3_config` still branches on it
(mirroring comfy exactly) so it fails loudly instead of silently
misdetecting rather than being asserted to work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..common import to_mlx_dtype


@dataclass(frozen=True)
class MiniMaxH3Config:
    """MiniMax H3 DiT architecture configuration.

    Attributes:
        num_layers: Main DiT blocks (50 on the checkpoints verified here).
        token_refiner_num_layers: Text-token refiner blocks preceding the
            main stack (2 — separate, unquantized weights even on an
            otherwise-quantized checkpoint).
        hidden_size: Model hidden dimension (5376).
        latents_dim: Video VAE latent channels (24).
        audio_latents_dim: Audio VAE latent channels (32).
        attention_head_dim: Per-head dimension (128).
        num_attention_heads: Attention heads (56 — hidden_size=5376 does not
            evenly split by a "nice" head_dim like 64; comfy derives this
            from `qkv_proj.weight.shape[0] // (3 * attention_head_dim)`
            rather than hardcoding it).
        ffn_hidden_size: MLP intermediate size before the SwiGLU split
            (14336 — `fc1` projects to `2 * ffn_hidden_size` since `MLP.forward`
            uses `comfy.ops.linear_input_act(fc2, fc1(x), "swiglu")`).
        text_dim: Qwen3-VL hidden size the text encoder hands over (5120),
            projected to `hidden_size` by `condition_proj` before the token
            refiner.
        patch_size: Video patchify `(t, h, w)` — always `(1, 2, 2)` for
            MiniMax H3 (not shape-derived: `video_patch_proj`'s input dim is
            `latents_dim * prod(patch_size)`, so patch_size can only be
            inferred jointly with `latents_dim`, and comfy hardcodes it).
        rope_inv_freq_len: Half the 3-axis (t, h, w) RoPE frequency count per
            axis (16 — `rope.inv_freq` is `[16]`; `rope_freqs` tiles this
            across 3 axes then duplicates for the rotation's cos/sin halves,
            giving a 96-dim rotation per token: `16 * 3 * 2 = 96`).
        norm_eps: RMSNorm epsilon for `norm1`/`norm2` (1e-5).
        qk_norm_eps: RMSNorm epsilon for the per-head `q_norm`/`k_norm` (1e-5).
        final_norm_eps: RMSNorm epsilon for the token refiner's `final_norm`
            and the main model's `final_layer.norm` (1e-5).
        sigma_shift_video: Flow-matching sigma shift for the video stream
            (12.0 — matches `comfy/supported_models.py::MiniMaxH3.
            sampling_settings["shift"]`).
        sigma_shift_audio: Flow-matching sigma shift for the audio stream,
            on its own schedule derived from the video sigma in closed form
            (3.0 — matches `sampling_settings["audio_shift"]`).
        gate_compress: Whether attention blocks carry a VSA
            (sparse-attention) gate projection (`to_gate_compress.weight`).
            False on every checkpoint seen so far; VSA-trained checkpoints
            would need it for compatibility with comfy's sparse-attention
            patches, but the dense forward path never reads it either way.
        adaln_curve_grid: Number of rows in the `adaln_t_table` curve basis
            (1025), or `None` if this checkpoint uses the plain sinusoidal
            `TimeEmbedder` instead (see module docstring). Mutually exclusive
            with `timestep_input_dim`/`time_embed_hidden_size` below.
        time_embed_dim: Width of the per-token modulation input vector fed to
            every `AdalnProj` (8 in the curve form seen here; `2688` is
            comfy's class default for the non-curve form, derived from that
            variant's `time_embedder.proj_out.weight` shape when present).
        timestep_input_dim: Only set when `adaln_curve_grid is None` — the
            sinusoidal timestep embedding's frequency count before
            `TimeEmbedder.proj_in`.
        time_embed_hidden_size: Only set when `adaln_curve_grid is None` —
            `TimeEmbedder`'s hidden width between `proj_in` and `proj_out`.
        dtype: MLX dtype string ("float16" or "bfloat16").
    """

    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    latents_dim: int = 24
    audio_latents_dim: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 56
    ffn_hidden_size: int = 14336
    text_dim: int = 5120
    patch_size: tuple[int, int, int] = (1, 2, 2)
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    sigma_shift_video: float = 12.0
    sigma_shift_audio: float = 3.0
    gate_compress: bool = False
    adaln_curve_grid: int | None = 1025
    time_embed_dim: int = 8
    timestep_input_dim: int | None = None
    time_embed_hidden_size: int | None = None
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.adaln_curve_grid is None:
            if self.timestep_input_dim is None or self.time_embed_hidden_size is None:
                raise ValueError(
                    "ASDX: MiniMaxH3Config without adaln_curve_grid needs "
                    "timestep_input_dim and time_embed_hidden_size (the "
                    "non-curve TimeEmbedder variant) -- got neither."
                )
        if self.num_attention_heads <= 0 or self.attention_head_dim <= 0:
            raise ValueError(
                f"ASDX: MiniMaxH3Config has non-positive num_attention_heads="
                f"{self.num_attention_heads} or attention_head_dim={self.attention_head_dim}."
            )

    @property
    def video_patch_dim(self) -> int:
        """`video_patch_proj`'s input width: latent channels x patch volume."""
        pt, ph, pw = self.patch_size
        return self.latents_dim * pt * ph * pw

    @property
    def attention_inner_dim(self) -> int:
        """`qkv_proj`'s per-stream width (heads x head_dim) -- distinct from
        `hidden_size` for MiniMax H3 (7168 vs 5376 on the checkpoints seen)."""
        return self.num_attention_heads * self.attention_head_dim

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return to_mlx_dtype(self.dtype, "MiniMax H3")


def detect_minimax_h3_config(state_dict: dict[str, Any], dtype: str = "float16") -> MiniMaxH3Config:
    """Derive a MiniMaxH3Config from a (normalized) checkpoint state dict.

    Mirrors `comfy/model_detection.py`'s `video_patch_proj.weight`/
    `audio_patch_proj.weight` detection branch (lines 390-418) exactly:
    every field below that is shape-derived there is shape-derived here,
    from the same tensor. `patch_size`/the three `*_eps` fields/
    `sigma_shift_*` are NOT shape-derived by comfy either -- they're the
    dataclass defaults above, same as comfy hardcodes them for this branch.

    Required rather than a single hardcoded config because MiniMax H3 ships
    at least two structurally different variants (curve-form adaln vs the
    plain `TimeEmbedder`) distinguishable only by which keys are present.
    """
    video_patch_w = state_dict.get("video_patch_proj.weight")
    audio_patch_w = state_dict.get("audio_patch_proj.weight")
    video_out_w = state_dict.get("final_layer.video_out.weight")
    audio_out_w = state_dict.get("final_layer.audio_out.weight")
    condition_proj_w = state_dict.get("condition_proj.weight")
    if video_patch_w is None or audio_patch_w is None or video_out_w is None or condition_proj_w is None:
        raise ValueError(
            "ASDX: cannot detect MiniMax H3 config -- checkpoint is missing "
            "video_patch_proj.weight/audio_patch_proj.weight/"
            "final_layer.video_out.weight/condition_proj.weight after key normalization."
        )

    def _block_indices(prefix: str) -> set[int]:
        indices = set()
        for key in state_dict:
            if key.startswith(prefix):
                rest = key[len(prefix) :]
                indices.add(int(rest.split(".", 1)[0]))
        return indices

    main_blocks = _block_indices("blocks.")
    refiner_blocks = _block_indices("token_refiner.blocks.")
    if not main_blocks:
        raise ValueError("ASDX: cannot detect MiniMax H3 config -- no blocks.N.* keys found.")

    hidden_size = int(video_patch_w.shape[0])
    # patch 1x2x2 is the only variant comfy's own detection branch assumes
    # (`// 4`); latents_dim is derived from it rather than hardcoded.
    latents_dim = int(video_out_w.shape[0]) // 4
    audio_latents_dim = int(audio_out_w.shape[0]) if audio_out_w is not None else 32
    attention_head_dim = int(state_dict["blocks.0.attn.q_norm.weight"].shape[0])
    qkv_out = int(state_dict["blocks.0.attn.qkv_proj.weight"].shape[0])
    num_attention_heads = qkv_out // (3 * attention_head_dim)
    ffn_hidden_size = int(state_dict["blocks.0.mlp.fc1.weight"].shape[0]) // 2
    text_dim = int(condition_proj_w.shape[1])
    rope_inv_freq_len = int(state_dict["rope.inv_freq"].shape[0])
    gate_compress = "blocks.0.attn.to_gate_compress.weight" in state_dict

    kwargs: dict[str, Any] = {}
    if "adaln_t_table" in state_dict:
        table_shape = state_dict["adaln_t_table"].shape
        kwargs["adaln_curve_grid"] = int(table_shape[0])
        kwargs["time_embed_dim"] = int(table_shape[1])
    else:
        te_in_w = state_dict.get("time_embedder.proj_in.weight")
        te_out_w = state_dict.get("time_embedder.proj_out.weight")
        if te_in_w is None or te_out_w is None:
            raise ValueError(
                "ASDX: cannot detect MiniMax H3 config -- checkpoint has neither "
                "adaln_t_table nor time_embedder.proj_in/proj_out.weight."
            )
        kwargs["adaln_curve_grid"] = None
        kwargs["timestep_input_dim"] = int(te_in_w.shape[1])
        kwargs["time_embed_hidden_size"] = int(te_in_w.shape[0])
        kwargs["time_embed_dim"] = int(te_out_w.shape[0])

    return MiniMaxH3Config(
        num_layers=len(main_blocks),
        token_refiner_num_layers=len(refiner_blocks),
        hidden_size=hidden_size,
        latents_dim=latents_dim,
        audio_latents_dim=audio_latents_dim,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=ffn_hidden_size,
        text_dim=text_dim,
        rope_inv_freq_len=rope_inv_freq_len,
        gate_compress=gate_compress,
        dtype=dtype,
        **kwargs,
    )
