"""Anima DiT: Cosmos-Predict2 `MiniTrainDIT` (image-only: T=1, no extra
per-block pos-emb, no padding-mask resize) + `LLMAdapter`.

Port of `comfy/ldm/cosmos/predict2.py` and `comfy/ldm/anima/model.py`.
List-valued attributes (`t_embedder`, `x_embedder.proj`, `adaln_modulation_*`)
keep the checkpoint's `nn.Sequential` indices so parameter names equal the
prefix-stripped checkpoint keys."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..common import timestep_embedding
from .adapter import LLMAdapter
from .config import AnimaConfig
from .rope import apply_rope_split_half, cosmos_rope_3d

_EPS = 1e-6


def _layer_norm(x: mx.array) -> mx.array:
    return mx.fast.layer_norm(x, None, None, _EPS)


def _run(seq: list, x: mx.array) -> mx.array:
    for layer in seq:
        x = layer(x)
    return x


class _Attention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, query_dim // heads
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.output_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=_EPS)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=_EPS)

    def __call__(self, x, context=None, rope=None):
        ctx = x if context is None else context
        b, lq, _ = x.shape
        lk = ctx.shape[1]
        q = self.q_norm(self.q_proj(x).reshape(b, lq, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(ctx).reshape(b, lk, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(ctx).reshape(b, lk, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        if rope is not None:  # self-attn only (predict2.py:184)
            q = apply_rope_split_half(q, *rope)
            k = apply_rope_split_half(k, *rope)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        return self.output_proj(o.transpose(0, 2, 1, 3).reshape(b, lq, -1))


def _adaln(d: int, lora: int, chunks: int) -> list:
    return [nn.SiLU(), nn.Linear(d, lora, bias=False), nn.Linear(lora, chunks * d, bias=False)]


class _FeedForward(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.layer1 = nn.Linear(d, hidden, bias=False)
        self.layer2 = nn.Linear(hidden, d, bias=False)

    def __call__(self, x):
        return self.layer2(nn.gelu(self.layer1(x)))


class _Block(nn.Module):
    def __init__(self, cfg: AnimaConfig):
        super().__init__()
        d = cfg.model_channels
        self.self_attn = _Attention(d, d, cfg.num_heads)
        self.cross_attn = _Attention(d, cfg.crossattn_emb_channels, cfg.num_heads)
        self.mlp = _FeedForward(d, int(d * cfg.mlp_ratio))
        self.adaln_modulation_self_attn = _adaln(d, cfg.adaln_lora_dim, 3)
        self.adaln_modulation_cross_attn = _adaln(d, cfg.adaln_lora_dim, 3)
        self.adaln_modulation_mlp = _adaln(d, cfg.adaln_lora_dim, 3)

    def __call__(self, x, emb, context, rope, adaln_lora):
        res_dtype, compute_dtype = x.dtype, emb.dtype

        def mod(seq):
            return mx.split(_run(seq, emb) + adaln_lora, 3, axis=-1)  # shift, scale, gate

        shift, scale, gate = mod(self.adaln_modulation_self_attn)
        h = (_layer_norm(x) * (1 + scale) + shift).astype(compute_dtype)
        x = x + gate.astype(res_dtype) * self.self_attn(h, None, rope).astype(res_dtype)

        shift, scale, gate = mod(self.adaln_modulation_cross_attn)
        h = (_layer_norm(x) * (1 + scale) + shift).astype(compute_dtype)
        x = x + gate.astype(res_dtype) * self.cross_attn(h, context).astype(res_dtype)

        shift, scale, gate = mod(self.adaln_modulation_mlp)
        h = (_layer_norm(x) * (1 + scale) + shift).astype(compute_dtype)
        return x + gate.astype(res_dtype) * self.mlp(h).astype(res_dtype)


class _TimestepEmbedding(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear_1 = nn.Linear(d, d, bias=False)
        self.linear_2 = nn.Linear(d, 3 * d, bias=False)

    def __call__(self, x):
        return self.linear_2(nn.silu(self.linear_1(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.proj = [nn.Identity(), nn.Linear(in_dim, d, bias=False)]

    def __call__(self, x):
        return self.proj[1](x)


class _FinalLayer(nn.Module):
    def __init__(self, cfg: AnimaConfig):
        super().__init__()
        d = cfg.model_channels
        self.linear = nn.Linear(d, cfg.patch_spatial ** 2 * cfg.out_channels, bias=False)
        self.adaln_modulation = _adaln(d, cfg.adaln_lora_dim, 2)

    def __call__(self, x, emb, adaln_lora):
        d = x.shape[-1]
        shift, scale = mx.split(_run(self.adaln_modulation, emb) + adaln_lora[..., : 2 * d], 2, axis=-1)
        return self.linear(_layer_norm(x) * (1 + scale) + shift)


class AnimaTransformer(nn.Module):
    def __init__(self, config: AnimaConfig):
        super().__init__()
        self.config = config
        d, p = config.model_channels, config.patch_spatial
        self.t_embedder = [nn.Identity(), _TimestepEmbedding(d)]
        self.t_embedding_norm = nn.RMSNorm(d, eps=_EPS)
        self.x_embedder = _PatchEmbed((config.in_channels + 1) * p * p, d)
        self.blocks = [_Block(config) for _ in range(config.num_blocks)]
        self.final_layer = _FinalLayer(config)
        self.llm_adapter = LLMAdapter(config)

    def encode_context(self, qwen_hidden, t5_ids, t5_weights=None):
        """`Anima.preprocess_text_embeds`: adapter, per-token weights, right-pad to 512."""
        dtype = self.config.mlx_dtype
        out = self.llm_adapter(qwen_hidden.astype(dtype), t5_ids)
        if t5_weights is not None:
            out = out * t5_weights[..., None].astype(dtype)
        pad = self.config.min_context_len - out.shape[1]
        if pad > 0:
            out = mx.pad(out, [(0, 0), (0, pad), (0, 0)])
        return out

    def __call__(self, x, timestep, context):
        cfg = self.config
        b, c, h, w = x.shape
        p = cfg.patch_spatial
        if h % p or w % p:
            raise ValueError(f"ASDX: Anima latent grid {h}x{w} must be even (patch {p}); use a width/height multiple of 16.")
        dtype = cfg.mlx_dtype
        gh, gw = h // p, w // p

        # concat_padding_mask (all zeros) -> patchify "b c (h m) (w n) -> b (h w) (c m n)".
        x = mx.concatenate([x.astype(dtype), mx.zeros((b, 1, h, w), dtype=dtype)], axis=1)
        x = x.reshape(b, c + 1, gh, p, gw, p).transpose(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, (c + 1) * p * p)
        x = self.x_embedder(x)

        rope = cosmos_rope_3d(cfg.head_dim, 1, gh, gw,
                              h_ratio=cfg.rope_ratios[1], w_ratio=cfg.rope_ratios[2], t_ratio=cfg.rope_ratios[0])

        sincos = timestep_embedding(timestep, cfg.model_channels, time_factor=1.0).astype(dtype)[:, None, :]
        adaln_lora = self.t_embedder[1](sincos)
        emb = self.t_embedding_norm(sincos)

        context = context.astype(dtype)
        if context.shape[0] != b:
            if context.shape[0] != 1:
                raise ValueError(
                    f"ASDX: Anima context batch {context.shape[0]} does not match latent batch {b} "
                    "(must be 1 or match the latent batch)."
                )
            context = mx.broadcast_to(context, (b,) + context.shape[1:])
        if dtype == mx.float16:  # predict2.py:907-912: fp32 residual stream under fp16 compute
            x = x.astype(mx.float32)
        for block in self.blocks:
            x = block(x, emb, context, rope, adaln_lora)

        x = self.final_layer(x.astype(context.dtype), emb, adaln_lora)
        # unpatchify "B (H W) (p1 p2 C) -> B C (H p1) (W p2)" -- channel-minor, unlike patchify.
        x = x.reshape(b, gh, gw, p, p, cfg.out_channels).transpose(0, 5, 1, 3, 2, 4)
        return x.reshape(b, cfg.out_channels, h, w)
