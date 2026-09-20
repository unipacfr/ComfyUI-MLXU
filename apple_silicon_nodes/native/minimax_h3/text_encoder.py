"""Qwen3-VL-32B text-only backbone -- MiniMax H3's text conditioning
encoder (text-only scope; see text_encoder_config.py's module docstring for
why the vision tower is out of scope entirely).

Ported from `comfy/text_encoders/llama.py`'s shared `Attention`/`MLP`/
`TransformerBlock`/`Llama2_` classes (Qwen3-VL uses this same generic
backbone, configured via `Qwen3VL_32BConfig` -- there is no
Qwen3-VL-specific transformer block class, only a specific config + the
vision tower this project doesn't implement).

Causal masking: this checkpoint is a causal-LM backbone used as an encoder
(hidden states after layer 50 are the conditioning signal, per
`comfy/ldm/minimax/model.py`'s module docstring); `Llama2_.forward` always
builds a causal mask for `seq_len > 1`, so this port does too
(`mx.fast.scaled_dot_product_attention(..., mask="causal")`) -- no
bidirectional-attention option exists in the reference to omit.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import RMSNorm
from .rope import rms_norm_rope_split_half
from .text_encoder_config import Qwen3TextEncoderConfig
from .text_encoder_rope import qwen3_rope_cos_sin
from .vision_rope import interleaved_mrope_cos_sin


class Attention(nn.Module):
    """GQA self-attention with per-head RMSNorm + full-rotation RoPE
    (`rot_dim == head_dim`, see `text_encoder_rope.py`'s module docstring).
    `mx.fast.scaled_dot_product_attention` handles GQA natively -- `k`/`v`
    are kept at their own (smaller) head count, never tiled up to match `q`."""

    def __init__(self, config: Qwen3TextEncoderConfig):
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

        q = q.transpose(1, 0, 2)[None]  # [1, heads, S, head_dim]
        k = k.transpose(1, 0, 2)[None]  # [1, kv_heads, S, head_dim]
        v = v.transpose(1, 0, 2)[None]
        scale = 1.0 / (self.head_dim**0.5)
        mask = "causal" if s > 1 else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out = out[0].transpose(1, 0, 2).reshape(s, self.heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """Standard (non-fused) SwiGLU: separate `gate_proj`/`up_proj` tensors
    (unlike MiniMax H3's DiT, which fuses them into one `fc1` -- this
    checkpoint's `mlp.gate_proj`/`mlp.up_proj` are genuinely separate
    weights, confirmed from the real header)."""

    def __init__(self, config: Qwen3TextEncoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Standard pre-norm block: `x += attn(norm1(x))`, `x += mlp(norm2(x))`."""

    def __init__(self, config: Qwen3TextEncoderConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


@dataclass(frozen=True)
class VisionInputs:
    """Vision-grounded conditioning for one prompt (fl2va / ref2va): rows that
    replace text embeddings, DeepStack features added after the first layers,
    and the M-RoPE position ids. See `vision_conditioning.py`."""

    rows: mx.array                  # [Nv, hidden], merged vision embeddings
    row_indices: np.ndarray         # [Nv] int, positions in the unfolded sequence
    deepstack: list[mx.array]       # each [Nv, hidden]; layer i gets deepstack[i] for i < len
    position_ids: np.ndarray        # [3, S] M-RoPE ids (float32)
    rope_dims: tuple[int, int, int] = (24, 20, 20)


def refuse_if_degenerate(x: mx.array, producer: str) -> None:
    """Raise `RuntimeError` if `x` is all exactly zero or holds NaN/Inf.

    A failed lazy weight read can yield an all-zero token-embedding table, and
    vision rows spliced over the pad positions then mask it, so a NaN/Inf-only
    guard never fires. Forces evaluation of the reductions it needs."""
    xf = x.astype(mx.float32)
    finite, peak = mx.all(mx.isfinite(xf)), mx.max(mx.abs(xf))
    mx.eval(finite, peak)
    if not bool(finite.item()):
        defect = "non-finite (NaN/Inf) values"
    elif float(peak.item()) == 0.0:
        defect = "all-zero values"
    else:
        return
    raise RuntimeError(
        f"ASDX MiniMax H3: degenerate conditioning from {producer}: {defect}, "
        f"shape={tuple(x.shape)}, max|x|={float(peak.item())}"
    )


class _Qwen3Backbone(nn.Module):
    """Embedding + `num_hidden_layers` `TransformerBlock`s. No final norm, no
    lm_head (matches `final_norm=False`/`lm_head=False` in the reference
    config for this truncated 50-layer variant) -- the output is the raw
    hidden state after the last block, which is what MiniMax H3 conditions on."""

    def __init__(self, config: Qwen3TextEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, input_ids: mx.array, vision: VisionInputs | None = None) -> mx.array:
        x = self.embed_tokens(input_ids)
        # Screen the text-position embeddings BEFORE vision rows can mask a bad table.
        if vision is None:
            refuse_if_degenerate(x, "token embedding")
        else:
            text_idx = np.setdiff1d(np.arange(x.shape[0]), np.asarray(vision.row_indices))
            if text_idx.size:
                refuse_if_degenerate(x[mx.array(text_idx.astype(np.int32))], "token embedding")
        if vision is None:
            cos, sin = qwen3_rope_cos_sin(x.shape[0], self.config.head_dim, self.config.rope_theta)
        else:
            idx = mx.array(np.asarray(vision.row_indices, dtype=np.int32))
            x[idx] = vision.rows.astype(x.dtype)
            cos, sin = interleaved_mrope_cos_sin(
                vision.position_ids, self.config.head_dim, self.config.rope_theta, vision.rope_dims
            )
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            if vision is not None and i < len(vision.deepstack):
                x = x.at[idx].add(vision.deepstack[i].astype(x.dtype))
        return x


class Qwen3TextEncoder(nn.Module):
    """Wraps `_Qwen3Backbone` under a `model` attribute, matching the
    checkpoint's own key prefix exactly (`model.embed_tokens.weight`,
    `model.layers.N....weight`) -- this project's convention throughout is
    checkpoint-key-compatible attribute names so weight loading needs no
    translation table."""

    def __init__(self, config: Qwen3TextEncoderConfig):
        super().__init__()
        self.config = config
        self.model = _Qwen3Backbone(config)

    def __call__(self, input_ids: mx.array, vision: VisionInputs | None = None) -> mx.array:
        """`input_ids`: `[S]` int32 token ids (batch already squeezed, this
        port's convention throughout); with `vision`, the ids at
        `vision.row_indices` are placeholders that get replaced. Returns
        `[S, hidden_size]`."""
        return self.model(input_ids, vision)
