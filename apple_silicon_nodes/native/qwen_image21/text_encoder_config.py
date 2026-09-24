"""Qwen3-VL-8B text-only backbone configuration -- Qwen Image 2.1's text
conditioning encoder.

Ported from `comfy/text_encoders/llama.py::Qwen3VL_8BConfig` (extends
`Qwen3_8BConfig`) -- verified against the real checkpoint's own header
(`qwen3vl_8b_bf16.safetensors`, 750 tensors, inspected directly on
2026-09-22): 36 text layers, hidden_size 4096, 32 attention heads / 8 KV
heads (GQA) at head_dim 128, intermediate_size 12288, rope_theta 5000000.0.

Unlike MiniMax H3's truncated 50-of-64-layer Qwen3-VL-32B checkpoint (which
has no `model.norm.weight`/`lm_head.weight`, `final_norm=False`/
`lm_head=False` in the reference config), this checkpoint is the FULL
Qwen3-VL-8B: it DOES contain `model.norm.weight` and `lm_head.weight`, plus a
complete `model.visual.*` vision tower (496 tensors). This module builds
neither the final norm, the lm_head, nor the vision tower -- Qwen Image
2.1's T2I path consumes the raw last-block hidden state
(`comfy/text_encoders/qwen_image21.py::QwenImage21Qwen3VLClipModel` sets
`layer_norm_hidden_state=False`, `layer="hidden"`, `layer_idx=-1`), and this
brick is text-only (no reference-image conditioning in scope). The weight
loader (`text_encoder_weight_map.py`) explicitly skips these extra keys
rather than silently accepting any unrecognized key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..common import to_mlx_dtype


@dataclass(frozen=True)
class Qwen3VL8BTextEncoderConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"ASDX: num_attention_heads={self.num_attention_heads} not divisible by "
                f"num_key_value_heads={self.num_key_value_heads} -- invalid kv groups."
            )

    @property
    def kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return to_mlx_dtype(self.dtype, "Qwen3-VL-8B text encoder")


def detect_qwen3vl_8b_text_encoder_config(
    state_dict: dict[str, Any], dtype: str = "float16"
) -> Qwen3VL8BTextEncoderConfig:
    """Derive config from a checkpoint's own tensor shapes, same rationale as
    MiniMax H3's `detect_qwen3_text_encoder_config`: read layer count and
    dims from the real weights rather than assuming the released 8B is
    exactly what the reference config class default says."""
    embed_w = state_dict.get("model.embed_tokens.weight")
    q_proj_w = state_dict.get("model.layers.0.self_attn.q_proj.weight")
    k_proj_w = state_dict.get("model.layers.0.self_attn.k_proj.weight")
    q_norm_w = state_dict.get("model.layers.0.self_attn.q_norm.weight")
    gate_proj_w = state_dict.get("model.layers.0.mlp.gate_proj.weight")
    if embed_w is None or q_proj_w is None or k_proj_w is None or q_norm_w is None or gate_proj_w is None:
        raise ValueError(
            "ASDX: cannot detect Qwen3-VL-8B text encoder config -- checkpoint is missing "
            "model.embed_tokens.weight / model.layers.0.self_attn.{q_proj,k_proj,q_norm}.weight / "
            "model.layers.0.mlp.gate_proj.weight after key normalization."
        )

    layer_indices = set()
    for key in state_dict:
        if key.startswith("model.layers."):
            layer_indices.add(int(key.split(".")[2]))
    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 0

    vocab_size, hidden_size = embed_w.shape
    head_dim = q_norm_w.shape[0]
    num_attention_heads = q_proj_w.shape[0] // head_dim
    num_key_value_heads = k_proj_w.shape[0] // head_dim
    intermediate_size = gate_proj_w.shape[0]

    return Qwen3VL8BTextEncoderConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
