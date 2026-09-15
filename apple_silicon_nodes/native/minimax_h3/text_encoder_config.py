"""Qwen3-VL-32B text-only backbone configuration -- MiniMax H3's text
conditioning encoder.

Text-only scope only (see text_encoder.py's module docstring): the vision
tower (`visual.*`, 27 blocks of Qwen3-VL's naViT-style ViT + DeepStack
mergers) is never used for MiniMax H3's t2va/fl2va path without keyframes,
and is not implemented here.

Ported from `comfy/text_encoders/llama.py::Qwen3VL_32BConfig` (which extends
`Qwen3VL_8BConfig(Qwen3_8BConfig)`) -- verified against the real checkpoint's
tensor shapes (`qwen3vl_32b_minimax_h3-Q4_K_M.gguf`, `..._int8_convrot.safetensors`):
50 layers (truncated from the real Qwen3-VL-32B's 64 -- `final_norm=False`,
`lm_head=False` in the reference config, matching: no `model.norm.weight`
key, no `lm_head.weight` key in either real checkpoint), hidden_size 5120,
64 attention heads / 8 KV heads (GQA) at head_dim 128, intermediate_size
25600, `rope_theta=5000000.0`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class Qwen3TextEncoderConfig:
    vocab_size: int = 151936
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = 50
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"ASDX: num_attention_heads={self.num_attention_heads} is not a multiple of "
                f"num_key_value_heads={self.num_key_value_heads} -- GQA requires an integer group size."
            )

    @property
    def kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def mlx_dtype(self) -> mx.Dtype:
        dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
        if self.dtype not in dtype_map:
            raise ValueError(f"ASDX: unknown Qwen3 text encoder dtype {self.dtype!r}")
        return dtype_map[self.dtype]


def detect_qwen3_text_encoder_config(state_dict: dict[str, Any], dtype: str = "float16") -> Qwen3TextEncoderConfig:
    """Derive config from a checkpoint's own tensor shapes -- same rationale
    as `config.py::detect_minimax_h3_config`: this checkpoint is a truncated
    (50-of-64-layer) variant of the public Qwen3-VL-32B, not the full model,
    so layer count and any other divergence should be read from the real
    weights rather than assumed."""
    embed_w = state_dict.get("model.embed_tokens.weight")
    q_proj_w = state_dict.get("model.layers.0.self_attn.q_proj.weight")
    k_proj_w = state_dict.get("model.layers.0.self_attn.k_proj.weight")
    q_norm_w = state_dict.get("model.layers.0.self_attn.q_norm.weight")
    gate_proj_w = state_dict.get("model.layers.0.mlp.gate_proj.weight")
    if embed_w is None or q_proj_w is None or k_proj_w is None or q_norm_w is None or gate_proj_w is None:
        raise ValueError(
            "ASDX: cannot detect Qwen3 text encoder config -- checkpoint is missing "
            "model.embed_tokens.weight / model.layers.0.self_attn.{q_proj,k_proj,q_norm}.weight / "
            "model.layers.0.mlp.gate_proj.weight after key normalization."
        )

    layer_indices = set()
    for key in state_dict:
        if key.startswith("model.layers."):
            layer_indices.add(int(key.split(".")[2]))
    if not layer_indices:
        raise ValueError("ASDX: cannot detect Qwen3 text encoder config -- no model.layers.N.* keys found.")

    vocab_size, hidden_size = int(embed_w.shape[0]), int(embed_w.shape[1])
    head_dim = int(q_norm_w.shape[0])
    num_attention_heads = int(q_proj_w.shape[0]) // head_dim
    num_key_value_heads = int(k_proj_w.shape[0]) // head_dim
    intermediate_size = int(gate_proj_w.shape[0])

    return Qwen3TextEncoderConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=len(layer_indices),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
