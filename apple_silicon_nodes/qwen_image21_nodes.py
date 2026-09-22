"""ComfyUI nodes for Qwen Image 2.1's text encoder (Qwen3-VL-8B, MLX native).

Dedicated loader + encode nodes, NOT a clip_type on ASDX_CLIPLoader -- that
node routes through comfy.sd.CLIPType/comfy.sd.load_clip (ComfyUI's real
PyTorch CLIP pipeline), which would load ComfyUI's own reference Qwen3-VL
implementation instead of this project's native MLX port. Same pattern as
MiniMax H3's ASDX_MiniMaxH3TextEncoderLoader/encode_minimax_h3_prompt
(minimax_h3_nodes.py) -- a dedicated node pair returning this project's own
conditioning dict, not the standard ComfyUI CONDITIONING type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_api.latest import io

import mlx.core as mx

_TEXT_ENCODER_CACHE: dict[str, Any] = {}


class ASDX_QwenImage21TextEncoderLoader(io.ComfyNode):
    """Load Qwen Image 2.1's Qwen3-VL-8B text encoder (bf16 safetensors, brick 1)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_QwenImage21TextEncoderLoader",
            display_name="🍏 ASDX Qwen Image 2.1 Text Encoder Loader",
            category="ASDX/Loaders",
            inputs=[
                io.Combo.Input("encoder_name", options=cls._get_encoders()),
                io.Combo.Input("precision", options=["float16", "bfloat16", "float32"], default="float16"),
            ],
            outputs=[
                io.Custom("asdx_qwen_image21_text_encoder").Output(display_name="text_encoder"),
            ],
        )

    @staticmethod
    def _get_encoders() -> list[str]:
        try:
            import folder_paths
            return [n for n in folder_paths.get_filename_list("text_encoders") if "qwen3vl_8b" in n.lower()]
        except Exception:
            return []

    @classmethod
    def execute(cls, encoder_name: str, precision: str = "float16") -> io.NodeOutput:
        import folder_paths

        found = folder_paths.get_full_path("text_encoders", encoder_name)
        if not found:
            raise RuntimeError(f"ASDX Qwen Image 2.1 Text Encoder Loader: could not find '{encoder_name}'.")
        path = Path(found)

        cache_key = f"{path}:{precision}"
        if cache_key in _TEXT_ENCODER_CACHE:
            print(f"[ASDX] Qwen Image 2.1 text encoder cache hit: {encoder_name}")
            return io.NodeOutput(_TEXT_ENCODER_CACHE[cache_key])

        from .native.qwen_image21.text_encoder_weight_map import load_qwen_image21_text_encoder_checkpoint

        _TEXT_ENCODER_CACHE.clear()
        encoder = load_qwen_image21_text_encoder_checkpoint(path, dtype=precision)
        result = {
            "type": "asdx_qwen_image21_text_encoder",
            "name": encoder_name,
            "path": str(path),
            "encoder": encoder,
            "precision": precision,
        }
        _TEXT_ENCODER_CACHE[cache_key] = result
        return io.NodeOutput(result)


def encode_qwen_image21_prompt(text_encoder: dict, prompt: str) -> dict:
    """Tokenize (ComfyUI's real, weight-free QwenImage21Tokenizer) and encode
    with the native MLX Qwen3VL8BTextEncoder. Returns the conditioning dict
    `conditioning_qwen_image21_to_mlx` (bridge.py) consumes."""
    if not isinstance(text_encoder, dict) or text_encoder.get("type") != "asdx_qwen_image21_text_encoder":
        raise RuntimeError(
            "ASDX Qwen Image 2.1 Text Encode: expected the output of ASDX_QwenImage21TextEncoderLoader."
        )

    import comfy.text_encoders.qwen_image21

    tokenizer = comfy.text_encoders.qwen_image21.QwenImage21Tokenizer()
    tokens = tokenizer.tokenize_with_weights(prompt)["qwen3vl_8b"][0]
    input_ids = mx.array([t[0] for t in tokens], dtype=mx.int32)

    hidden_states = text_encoder["encoder"](input_ids)
    print(f"[ASDX] Qwen Image 2.1 Text Encode: {len(prompt)} chars, {hidden_states.shape[0]} rows")
    return {"type": "qwen_image21", "hidden_states": hidden_states, "text": prompt}


class ASDX_QwenImage21TextEncode(io.ComfyNode):
    """Encode a T2I prompt with Qwen Image 2.1's Qwen3-VL-8B text encoder."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_QwenImage21TextEncode",
            display_name="🍏 ASDX Qwen Image 2.1 Text Encode",
            category="ASDX/Conditioning",
            inputs=[
                io.Custom("asdx_qwen_image21_text_encoder").Input("text_encoder"),
                io.String.Input("prompt", multiline=True),
            ],
            outputs=[
                io.Custom("mlx_conditioning").Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder: dict, prompt: str) -> io.NodeOutput:
        return io.NodeOutput(encode_qwen_image21_prompt(text_encoder, prompt))


NODE_LIST = [
    ASDX_QwenImage21TextEncoderLoader,
    ASDX_QwenImage21TextEncode,
]
