"""
Z-Image (NextDiT / Lumina2 family) transformer.

Ported line-by-line from comfy's real NextDiT implementation
(`comfy/ldm/lumina/model.py`): `JointAttention`, `FeedForward`,
`JointTransformerBlock`, `FinalLayer`, `NextDiT.__init__`/`embed_cap`/
`embed_all`/`patchify_and_embed`/`_forward`.

Architecture reference:
  https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/lumina/model.py

Scope: base single-image txt2img path only (`ref_latents=[]`, no SigLIP,
no pixel-space variant `NextDiTPixelSpace`) — the "omni" multi-reference
path and `timestep_zero_index` splitting are not implemented.

Encapsulation note: unlike the FLUX/Krea2 sampler loop (which precomputes
RoPE once outside the step loop), this model recomputes cap embedding,
context/noise refiners, and position tables on every `__call__` — the same
simplicity tradeoff `Krea2Transformer.__call__` already makes by re-running
`txtfusion` every step. Keeps `_run_zimage()` a thin per-step loop.

RoPE reuses the exact paired-interleave convention already ported natively
in this project (`native/__init__.py::apply_rope`/`embed_nd`, duplicated
here per the Krea2 precedent to avoid a circular import between `native/`
and `native/zimage/`) — Z-Image's reference imports the identical functions
from `comfy.ldm.flux.{math,layers}`.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import ZImageConfig


# ── RoPE (paired-interleave, matches native/__init__.py exactly) ────────

def _rope_freqs_axis(pos: mx.array, dim: int, theta: float) -> mx.array:
    """[N, dim/2, 2, 2] rotation matrices for one RoPE axis."""
    assert dim % 2 == 0
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta ** scale)
    out = pos.astype(mx.float32)[:, None] * omega[None, :]
    cos, sin = mx.cos(out), mx.sin(out)
    return mx.stack([cos, -sin, sin, cos], axis=-1).reshape(*out.shape, 2, 2)


def embed_nd(ids: mx.array, axes_dim: tuple[int, ...], theta: float) -> mx.array:
    """3-axis RoPE embedding table. ids: [N, 3] -> [N, head_dim/2, 2, 2]."""
    parts = [_rope_freqs_axis(ids[:, i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return mx.concatenate(parts, axis=-3)


def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply paired-interleave RoPE. x: [B,H,N,D], freqs: [N,D/2,2,2]."""
    B, H, N, D = x.shape
    x_pairs = x.reshape(B, H, N, D // 2, 1, 2)
    f = freqs[None, None]
    out = (f[..., 0] * x_pairs[..., 0]) + (f[..., 1] * x_pairs[..., 1])
    return out.reshape(B, H, N, D)


def timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    """Sinusoidal embedding matching comfy's `timestep_embedding` (repeat_only=False)."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


# ── JointAttention (full MHA for Z-Image: n_kv_heads == n_heads) ────────

class JointAttention(nn.Module):
    """Fused-QKV attention with per-head QK-RMSNorm and paired-interleave RoPE.

    Generic over n_kv_heads < n_heads (GQA) even though Z-Image itself uses
    full MHA (30 == 30) — matches the reference `JointAttention` class shape.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, qk_norm: bool):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, (n_heads + 2 * n_kv_heads) * head_dim, bias=False)
        self.out = nn.Linear(n_heads * head_dim, dim, bias=False)

        if qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(self, x: mx.array, freqs: mx.array) -> mx.array:
        B, N, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(
            qkv,
            [self.n_heads * self.head_dim, (self.n_heads + self.n_kv_heads) * self.head_dim],
            axis=-1,
        )
        q = q.reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q * scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.n_heads * self.head_dim)
        return self.out(out)


# ── SwiGLU FeedForward ───────────────────────────────────────────────────

class FeedForward(nn.Module):
    """SwiGLU: w2(silu(w1(x)) * w3(x)). Matches comfy's w1=gate/w2=down/w3=up naming."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# ── JointTransformerBlock ────────────────────────────────────────────────

def _modulate(x: mx.array, scale: mx.array) -> mx.array:
    """x * (1 + scale), scale broadcast over the sequence dim."""
    return x * (1.0 + scale[:, None, :])


class JointTransformerBlock(nn.Module):
    """Pre/post-RMSNorm block with tanh-gated adaLN modulation (or plain
    self-attn+FF when `modulation=False`, used by `context_refiner`).

    z_image_modulation: `adaLN_modulation` is a SINGLE Linear (no SiLU
    prefix) from a 256-dim timestep vector -> 4*dim, chunked into
    (scale_msa, gate_msa, scale_mlp, gate_mlp) — scale+gate only, no shift
    (contrast FLUX/Krea2's 6-way shift+scale+gate). Gates are tanh'd, not
    raw or sigmoid.
    """

    def __init__(self, config: ZImageConfig, modulation: bool = True):
        super().__init__()
        self.modulation = modulation
        self.attention = JointAttention(
            config.dim, config.n_heads, config.n_kv_heads, config.head_dim, config.qk_norm,
        )
        self.feed_forward = FeedForward(config.dim, config.ffn_hidden_dim)
        self.attention_norm1 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm1 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm2 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm2 = nn.RMSNorm(config.dim, eps=config.norm_eps)

        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(config.dim, 256), 4 * config.dim),
            )

    def __call__(self, x: mx.array, freqs: mx.array, adaln_input: mx.array | None = None) -> mx.array:
        if self.modulation:
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(mod, 4, axis=-1)

            attn_out = self.attention_norm2(
                self.attention(_modulate(self.attention_norm1(x), scale_msa), freqs)
            )
            x = x + mx.tanh(gate_msa)[:, None, :] * attn_out

            ff_out = self.ffn_norm2(self.feed_forward(_modulate(self.ffn_norm1(x), scale_mlp)))
            x = x + mx.tanh(gate_mlp)[:, None, :] * ff_out
        else:
            x = x + self.attention_norm2(self.attention(self.attention_norm1(x), freqs))
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


# ── FinalLayer ────────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    """LayerNorm(no affine) -> modulate(scale-only) -> Linear.

    `adaLN_modulation` keeps the SiLU+Linear pair regardless of
    z_image_modulation (only the block-level modulation drops SiLU).
    """

    def __init__(self, dim: int, patch_size: int, out_channels: int, min_mod: int = 256):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, min_mod), dim),
        )

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = self.adaLN_modulation(c)
        x = _modulate(self.norm_final(x), scale)
        return self.linear(x)


# ── NextDiT ───────────────────────────────────────────────────────────────

class NextDiT(nn.Module):
    """Z-Image transformer: context_refiner + noise_refiner (separate) -> layers (joint)."""

    def __init__(self, config: ZImageConfig | None = None):
        super().__init__()
        config = config or ZImageConfig()
        self.config = config
        self.dtype = config.mlx_dtype

        self.x_embedder = nn.Linear(config.patch_size * config.patch_size * config.in_channels, config.dim)
        self.cap_embedder = nn.Sequential(
            nn.RMSNorm(config.cap_feat_dim, eps=config.norm_eps),
            nn.Linear(config.cap_feat_dim, config.dim),
        )
        self.t_embedder = nn.Sequential(
            nn.Linear(256, min(config.dim, 1024)),
            nn.SiLU(),
            nn.Linear(min(config.dim, 1024), 256),
        )

        self.context_refiner = [
            JointTransformerBlock(config, modulation=False) for _ in range(config.n_refiner_layers)
        ]
        self.noise_refiner = [
            JointTransformerBlock(config, modulation=True) for _ in range(config.n_refiner_layers)
        ]
        self.layers = [
            JointTransformerBlock(config, modulation=True) for _ in range(config.n_layers)
        ]

        self.final_layer = FinalLayer(config.dim, config.patch_size, config.in_channels)

        if config.pad_tokens_multiple is not None:
            self.x_pad_token = mx.zeros((1, config.dim))
            self.cap_pad_token = mx.zeros((1, config.dim))

    @staticmethod
    def _text_pos_ids(n: int) -> mx.array:
        """[n, 3]: axis0 = arange(n)+1 (sequential), axes1/2 = 0."""
        ids = mx.zeros((n, 3), dtype=mx.float32)
        ids[:, 0] = mx.arange(n, dtype=mx.float32) + 1.0
        return ids

    @staticmethod
    def _image_pos_ids(h: int, w: int, start_t: float) -> mx.array:
        """[h*w, 3]: axis0 = constant start_t, axes1/2 = (row, col) grid."""
        ids = mx.zeros((h * w, 3), dtype=mx.float32)
        ids[:, 0] = start_t
        rows = mx.arange(h, dtype=mx.float32)[:, None]
        cols = mx.arange(w, dtype=mx.float32)[None, :]
        ids[:, 1] = mx.broadcast_to(rows, (h, w)).reshape(-1)
        ids[:, 2] = mx.broadcast_to(cols, (h, w)).reshape(-1)
        return ids

    def _pad_tokens(self, feats: mx.array, pad_token: mx.array) -> tuple[mx.array, int]:
        """Append `pad_token` up to the next multiple of `pad_tokens_multiple`."""
        multiple = self.config.pad_tokens_multiple
        n = feats.shape[1]
        pad_extra = (-n) % multiple
        if pad_extra == 0:
            return feats, 0
        B = feats.shape[0]
        pad = mx.broadcast_to(pad_token[None].astype(feats.dtype), (B, pad_extra, feats.shape[-1]))
        return mx.concatenate([feats, pad], axis=1), pad_extra

    def __call__(
        self,
        img: mx.array,       # [B, N_img, patch^2*in_channels] packed image patches
        context: mx.array,   # [B, T, cap_feat_dim] text embeddings (single-layer Qwen3-4B)
        t: mx.array,         # [B] timestep, flow-matching sigma convention
        img_h: int,
        img_w: int,
    ) -> mx.array:
        """Forward pass.

        Args:
            img: packed image tokens [B, img_h*img_w, 64].
            context: text embeddings [B, T, cap_feat_dim].
            t: sigma value(s) in [0, 1] (flow-matching convention, same as FLUX/Krea2).
            img_h, img_w: target image token grid (in patches).

        Returns:
            [B, img_h*img_w, 64] noise prediction — NEGATED to match comfy's
            `NextDiT._forward` (`return -img`), a load-bearing sign flip: the
            Euler update integrates in the wrong direction without it.
        """
        img_tokens_real = img_h * img_w
        if img.shape[1] != img_tokens_real:
            raise ValueError(
                f"ASDX: img_h*img_w ({img_h}*{img_w}={img_tokens_real}) does not "
                f"match img token count ({img.shape[1]})"
            )

        # ── Text: embed, pad, position, refine ──────────────────────────
        cap_feats = self.cap_embedder(context.astype(self.dtype))
        if self.config.pad_tokens_multiple is not None:
            cap_feats, _ = self._pad_tokens(cap_feats, self.cap_pad_token)
        cap_len = cap_feats.shape[1]
        cap_pos_ids = self._text_pos_ids(cap_len)
        cap_freqs = embed_nd(cap_pos_ids, self.config.axes_dims, self.config.rope_theta)

        for block in self.context_refiner:
            cap_feats = block(cap_feats, cap_freqs)

        # ── Timestep: shared adaln_input for noise_refiner + main layers ──
        # t_input = (1 - sigma) * time_scale, matching comfy's `t = 1.0 - timesteps`
        # then `t_embedder(t * self.time_scale, ...)`.
        t_input = (1.0 - t) * self.config.time_scale
        adaln_input = self.t_embedder(timestep_embedding(t_input, 256).astype(self.dtype))

        # ── Image: embed, position (real grid, before padding), pad ──────
        img_emb = self.x_embedder(img.astype(self.dtype))
        img_pos_ids = self._image_pos_ids(img_h, img_w, start_t=float(cap_len + 1))
        if self.config.pad_tokens_multiple is not None:
            img_emb, pad_extra = self._pad_tokens(img_emb, self.x_pad_token)
            if pad_extra > 0:
                img_pos_ids = mx.concatenate(
                    [img_pos_ids, mx.zeros((pad_extra, 3), dtype=mx.float32)], axis=0
                )
        img_freqs = embed_nd(img_pos_ids, self.config.axes_dims, self.config.rope_theta)

        for block in self.noise_refiner:
            img_emb = block(img_emb, img_freqs, adaln_input=adaln_input)

        # ── Joint stage: concatenate refined [text, image], run main layers ──
        combined = mx.concatenate([cap_feats, img_emb], axis=1)
        combined_freqs = mx.concatenate([cap_freqs, img_freqs], axis=0)

        for block in self.layers:
            combined = block(combined, combined_freqs, adaln_input=adaln_input)

        out = self.final_layer(combined, adaln_input)

        # Drop text tokens and any image-side padding: keep exactly the
        # real image tokens, matching comfy's `unpatchify` slice
        # `x[i][begin:end]` with `end = begin + (H//pH)*(W//pW)` (the REAL
        # grid size, not the padded length).
        img_out = out[:, cap_len:cap_len + img_tokens_real, :]

        return -img_out

    def predict(
        self,
        img: mx.array,
        context: mx.array,
        timestep: float,
        img_h: int,
        img_w: int,
    ) -> mx.array:
        """Convenience wrapper for one denoising step."""
        t = mx.array([timestep], dtype=mx.float32)
        return self(img, context, t, img_h, img_w)


# ── Loader ───────────────────────────────────────────────────────────────

def load_zimage_transformer(path, dtype: str = "float16") -> NextDiT:
    """Load a Z-Image checkpoint into a native MLX NextDiT.

    Same 6-step recipe as `load_transformer`/`load_krea2_transformer`/
    `load_sdxl_unet`.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    from .. import _load_safetensors, _check_weight_match
    from .weight_map import normalize_zimage_keys, map_zimage_to_native

    state_dict = _load_safetensors(path)
    state_dict = normalize_zimage_keys(state_dict)
    state_dict = map_zimage_to_native(state_dict)

    config = ZImageConfig(dtype=dtype)
    transformer = NextDiT(config)

    model_flat = tree_flatten(transformer.parameters())
    new_flat = []
    matched = 0
    for flat_key, value in model_flat:
        if flat_key in state_dict:
            new_flat.append((flat_key, state_dict.pop(flat_key).astype(config.mlx_dtype)))
            matched += 1
        else:
            new_flat.append((flat_key, value))
    new_nested = tree_unflatten(new_flat)
    transformer.update(new_nested)
    mx.eval(transformer.parameters())

    print(f"[ASDX] Z-Image transformer: matched {matched}/{len(model_flat)} params from checkpoint")
    _check_weight_match(matched, len(model_flat), "Z-Image transformer", path)
    return transformer
