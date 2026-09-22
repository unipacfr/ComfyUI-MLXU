"""Qwen3-VL-8B text-only backbone -- Qwen Image 2.1's T2I text conditioning
encoder (text-only scope; no vision tower, no reference-image conditioning
in this brick -- see text_encoder_config.py's module docstring).

Ported from `comfy/text_encoders/llama.py`'s shared `Attention`/`MLP`/
`TransformerBlock`/`Llama2_` classes (Qwen3-VL uses this same generic
backbone, configured via `Qwen3VL_8BConfig` -- there is no Qwen3-VL-specific
transformer block class). The real checkpoint's `model.norm`/`lm_head` are
deliberately not built here: Qwen Image 2.1's T2I path consumes the raw
last-block hidden state (`comfy/text_encoders/qwen_image21.py`:
`layer_norm_hidden_state=False`, `layer="hidden"`, `layer_idx=-1`).

Causal masking: `Llama2_.forward` always builds a causal mask for
`seq_len > 1`; this port does too (`mx.fast.scaled_dot_product_attention(...,
mask="causal")`).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .rope import qwen3_rope_cos_sin, rms_norm, rms_norm_rope_split_half
from .text_encoder_config import Qwen3VL8BTextEncoderConfig


class RMSNorm(nn.Module):
    """Plain RMSNorm (`x / sqrt(mean(x^2) + eps) * weight`), matching
    `torch.nn.functional.rms_norm`'s direct-multiply convention."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    """GQA self-attention with per-head RMSNorm + full-rotation RoPE
    (`rot_dim == head_dim`). `mx.fast.scaled_dot_product_attention` handles
    GQA natively -- `k`/`v` are kept at their own (smaller) head count."""

    def __init__(self, config: Qwen3VL8BTextEncoderConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.eps = config.rms_norm_eps
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = RMSNorm(self.head_dim, eps=self.eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        s = x.shape[0]
        q = self.q_proj(x).reshape(s, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(s, self.kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(s, self.kv_heads, self.head_dim)

        cos_b, sin_b = cos[:, None, :], sin[:, None, :]
        q = rms_norm_rope_split_half(q, self.q_norm.weight, self.eps, self.head_dim, cos_b, sin_b)
        k = rms_norm_rope_split_half(k, self.k_norm.weight, self.eps, self.head_dim, cos_b, sin_b)

        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        scale = 1.0 / (self.head_dim**0.5)
        mask = "causal" if s > 1 else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out = out[0].transpose(1, 0, 2).reshape(s, self.heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """Standard SwiGLU: separate `gate_proj`/`up_proj` (confirmed from the
    real checkpoint header -- not fused)."""

    def __init__(self, config: Qwen3VL8BTextEncoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Standard pre-norm block: `x += attn(norm1(x))`, `x += mlp(norm2(x))`."""

    def __init__(self, config: Qwen3VL8BTextEncoderConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class _Qwen3VL8BBackbone(nn.Module):
    """Embedding + `num_hidden_layers` `TransformerBlock`s. No final norm, no
    lm_head: the output is the raw hidden state after the last block."""

    def __init__(self, config: Qwen3VL8BTextEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.embed_tokens(input_ids)
        cos, sin = qwen3_rope_cos_sin(x.shape[0], self.config.head_dim, self.config.rope_theta)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


class Qwen3VL8BTextEncoder(nn.Module):
    """Wraps `_Qwen3VL8BBackbone` under a `model` attribute, matching the
    checkpoint's own key prefix (`model.embed_tokens.weight`,
    `model.layers.N....weight`) -- checkpoint-key-compatible attribute names
    throughout this project, so weight loading needs no translation table."""

    def __init__(self, config: Qwen3VL8BTextEncoderConfig):
        super().__init__()
        self.config = config
        self.model = _Qwen3VL8BBackbone(config)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """`input_ids`: `[S]` int32 token ids (batch already squeezed).
        Returns `[S, hidden_size]`."""
        return self.model(input_ids)
