"""Anima LLM adapter: T5 token ids (learned query tokens) cross-attend into
Qwen3-0.6B hidden states. Port of `comfy/ldm/anima/model.py::LLMAdapter`
(inference path: no attention masks, `preprocess_text_embeds` passes none)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import AnimaConfig
from .rope import adapter_rope, apply_rope_split_half


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def __call__(self, x, context, rope_q, rope_k):
        b, lq, _ = x.shape
        lk = context.shape[1]
        q = self.q_norm(self.q_proj(x).reshape(b, lq, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(context).reshape(b, lk, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(context).reshape(b, lk, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        # RotaryEmbedding.forward casts cos/sin to the activation dtype before applying.
        q = apply_rope_split_half(q, rope_q[0].astype(x.dtype), rope_q[1].astype(x.dtype))
        k = apply_rope_split_half(k, rope_k[0].astype(x.dtype), rope_k[1].astype(x.dtype))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(b, lq, -1))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm_self_attn = nn.RMSNorm(dim, eps=1e-6)
        self.self_attn = _Attention(dim, heads)
        self.norm_cross_attn = nn.RMSNorm(dim, eps=1e-6)
        self.cross_attn = _Attention(dim, heads)
        self.norm_mlp = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = [nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)]

    def __call__(self, x, context, rope_x, rope_ctx):
        h = self.norm_self_attn(x)
        x = x + self.self_attn(h, h, rope_x, rope_x)
        x = x + self.cross_attn(self.norm_cross_attn(x), context, rope_x, rope_ctx)
        h = self.norm_mlp(x)
        for layer in self.mlp:
            h = layer(h)
        return x + h


class LLMAdapter(nn.Module):
    def __init__(self, cfg: AnimaConfig):
        super().__init__()
        d = cfg.adapter_dim
        self.head_dim = d // cfg.adapter_heads
        self.embed = nn.Embedding(cfg.adapter_vocab, d)
        self.blocks = [_Block(d, cfg.adapter_heads) for _ in range(cfg.adapter_layers)]
        self.out_proj = nn.Linear(d, d)
        self.norm = nn.RMSNorm(d, eps=1e-6)

    def __call__(self, source_hidden: mx.array, target_ids: mx.array) -> mx.array:
        x = self.embed(target_ids).astype(source_hidden.dtype)
        rope_x = adapter_rope(x.shape[1], self.head_dim)
        rope_ctx = adapter_rope(source_hidden.shape[1], self.head_dim)
        for block in self.blocks:
            x = block(x, source_hidden, rope_x, rope_ctx)
        return self.norm(self.out_proj(x))
