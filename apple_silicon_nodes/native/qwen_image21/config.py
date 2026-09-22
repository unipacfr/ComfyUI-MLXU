"""Qwen Image 2.1 DiT (QwenImage21Transformer2DModel) configuration.

Ported from `comfy/ldm/qwen_image21/model.py::QwenImage21Transformer2DModel.__init__`'s
defaults -- verified against the real checkpoint's own header (`qwen_image_2.1_bf16.
safetensors`, 265 tensors, inspected directly on 2026-09-22): no `model.diffusion_model.`
prefix, bare keys already matching this module's own attribute names 1:1, no bias anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class QwenImage21Config:
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    eps: float = 1e-6
    dtype: str = "float16"

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def mlx_dtype(self) -> mx.Dtype:
        dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
        if self.dtype not in dtype_map:
            raise ValueError(f"ASDX: unknown Qwen Image 2.1 DiT dtype {self.dtype!r}")
        return dtype_map[self.dtype]


def detect_qwen_image21_config(state_dict: dict[str, Any], dtype: str = "float16") -> QwenImage21Config:
    """Derive config from a checkpoint's own tensor shapes."""
    img_in_w = state_dict.get("img_in.weight")
    txt_in_w = state_dict.get("txt_in.in_layer.weight")
    proj_out_w = state_dict.get("proj_out.weight")
    q_w = state_dict.get("transformer_blocks.0.attn.to_q.weight")
    q_norm_w = state_dict.get("transformer_blocks.0.attn.norm_q.weight")
    gate_up_w = state_dict.get("transformer_blocks.0.img_mlp.gate_up.weight")
    if (img_in_w is None or txt_in_w is None or proj_out_w is None or q_w is None
            or q_norm_w is None or gate_up_w is None):
        raise ValueError(
            "ASDX: cannot detect Qwen Image 2.1 DiT config -- checkpoint is missing "
            "img_in.weight / txt_in.in_layer.weight / proj_out.weight / "
            "transformer_blocks.0.attn.{to_q,norm_q}.weight / "
            "transformer_blocks.0.img_mlp.gate_up.weight after key normalization."
        )

    layer_indices = set()
    for key in state_dict:
        if key.startswith("transformer_blocks."):
            layer_indices.add(int(key.split(".")[1]))
    num_layers = max(layer_indices) + 1 if layer_indices else 0

    inner_dim, in_channels = img_in_w.shape
    context_in_dim = txt_in_w.shape[1]
    out_channels = proj_out_w.shape[0]
    attention_head_dim = q_norm_w.shape[0]
    num_attention_heads = inner_dim // attention_head_dim
    # gate_up fuses [gate; up] (SwiGLUFeedForward, model.py), so its output dim is
    # 2 * mlp_hidden_dim; mlp_hidden_dim = inner_dim * mlp_ratio.
    if gate_up_w.shape[0] % 2 != 0:
        raise ValueError(
            f"ASDX: transformer_blocks.0.img_mlp.gate_up.weight has an odd output dim "
            f"({gate_up_w.shape[0]}) -- cannot split into [gate; up] halves."
        )
    mlp_hidden_dim = gate_up_w.shape[0] // 2
    if mlp_hidden_dim % inner_dim != 0:
        raise ValueError(
            f"ASDX: cannot derive an integer mlp_ratio -- mlp_hidden_dim ({mlp_hidden_dim}) "
            f"is not a multiple of inner_dim ({inner_dim})."
        )
    mlp_ratio = mlp_hidden_dim // inner_dim

    return QwenImage21Config(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        mlp_ratio=mlp_ratio,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        context_in_dim=context_in_dim,
        dtype=dtype,
    )
