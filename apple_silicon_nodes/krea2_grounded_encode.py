"""ASDX_Krea2GroundedEncode — Krea2 image-grounded prompt encoding as a focused node.

Extracts the Krea2-specific image-grounded encode out of ``ASDX_CLIPTextEncode``
(a node whose stated purpose is generic FLUX/SD text encoding) into its own
upstream node. The whole point of this node is grounding: it encodes the prompt
*and* a source image through the Krea2 CLIP's vision tower, using the trained
conditioning template (``_krea2_grounding_template``) and the training-default
system prompt (``_KREA2_DEFAULT_GROUNDING_SYSTEM``) — the exact path the
krea2_edit LoRA was trained on.

A non-Krea2 ``mlx_clip`` raises (rather than warn-and-ignore) because grounding
is this node's only job: there is no "generic" fallback to fall through to.
"""

from __future__ import annotations

from typing import Any

import torch

import comfy.sd
import comfy.text_encoders.krea2
import comfy.utils
from comfy_api.latest import io

from . import metadata_extractors


# Krea2 Identity Edit grounded-encode template: the trained conditioning
# template (`comfy/text_encoders/krea2.py::KREA2_TEMPLATE`) with a vision
# block inserted before the user text -- ported from comfyui-krea2edit's
# `Krea2EditGroundedEncode` (Apache-2.0). Required because Krea2Tokenizer only
# overrides the *no-image* template; passing `images=` alone falls back to the
# base Qwen3VLTokenizer's generic vision template (no system prompt), silently
# mismatching what the krea2_edit LoRA was trained on.
# `_KREA2_DEFAULT_GROUNDING_SYSTEM` is the training-default system prompt --
# generic ("objects and background", no explicit person/face wording).
# `system_prompt` on the node overrides it, e.g. to steer the vision encoder
# toward facial identity detail, matching the reference's own override input.
_KREA2_DEFAULT_GROUNDING_SYSTEM = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)


def _krea2_grounding_template(system_prompt: str) -> str:
    sp = system_prompt.strip() or _KREA2_DEFAULT_GROUNDING_SYSTEM
    return (
        "<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    )


def _prep_grounding_image(image: torch.Tensor, grounding_px: int) -> torch.Tensor:
    """Resize an IMAGE for the vision tower, matching comfyui-krea2edit's
    Krea2EditGroundedEncode._prep -- the krea2_edit LoRA was trained with
    384-768px jitter, so capping here keeps inference in-distribution.
    """
    samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    h, w = samples.shape[2], samples.shape[3]
    if grounding_px and max(h, w) > grounding_px:
        s = grounding_px / max(h, w)
        samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
    return samples.movedim(1, -1)[:, :, :, :3]


class ASDX_Krea2GroundedEncode(io.ComfyNode):
    """Encode a prompt grounded on a source image through Krea2's vision tower.

    Requires a Krea2 ``mlx_clip`` (raises otherwise) -- grounding is this
    node's only job, so there is no generic fallback to warn-and-ignore into.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_Krea2GroundedEncode",
            display_name="🍏 ASDX Krea2 Grounded Encode",
            category="ASDX/Conditioning",
            inputs=[
                io.Custom("mlx_clip").Input("mlx_clip"),
                io.String.Input("text", multiline=True, default=""),
                io.Image.Input(
                    "image",
                    tooltip="source image, encoded through the CLIP's vision "
                            "tower alongside the prompt (image-grounded "
                            "conditioning).",
                ),
                io.Int.Input(
                    "grounding_px", default=768, min=0, max=4096, step=64, optional=True,
                    tooltip="cap longest side fed to the vision tower; 0 = native resolution",
                ),
                io.String.Input(
                    "system_prompt", multiline=True, default="", optional=True,
                    tooltip="advanced (optional): override the "
                            "grounding system prompt (empty = "
                            "training default). Steers what the "
                            "vision encoder attends to, e.g. "
                            "facial identity detail.",
                ),
            ],
            outputs=[
                io.Custom("mlx_conditioning").Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mlx_clip: Any,
        text: str,
        image: torch.Tensor,
        grounding_px: int = 768,
        system_prompt: str = "",
    ) -> io.NodeOutput:
        metadata_extractors.ensure_registered()
        if not isinstance(mlx_clip, comfy.sd.CLIP):
            raise RuntimeError("ASDX: mlx_clip must be a Comfy CLIP object.")
        if not isinstance(mlx_clip.tokenizer, comfy.text_encoders.krea2.Krea2Tokenizer):
            raise RuntimeError(
                "ASDX: ASDX_Krea2GroundedEncode requires a Krea2 mlx_clip -- "
                "image-grounded encoding is only implemented for Krea2 CLIPs. "
                "Use ASDX_CLIPTextEncode for non-Krea2 checkpoints."
            )

        tokens = mlx_clip.tokenize(
            text,
            images=[_prep_grounding_image(image, grounding_px)],
            llama_template=_krea2_grounding_template(system_prompt),
        )
        conditioning = mlx_clip.encode_from_tokens_scheduled(tokens)
        # Fail fast on a corrupt (NaN/Inf) embedding rather than letting it
        # silently ride through ~7min of diffusion sampling and VAE decode to
        # surface only as a black output image. The vision tower is where this
        # has been seen in practice: comfy's default text encoder dtype is
        # float16 (`model_management.text_encoder_dtype`), which overflows
        # more easily than bf16 in vision-transformer attention.
        for cond, _ in conditioning:
            if not torch.isfinite(cond).all():
                raise RuntimeError(
                    "ASDX: Krea2 Grounded Encode produced a non-finite (NaN/Inf) "
                    "embedding -- aborting before the expensive sampling pass."
                )

        result = {
            "type": "clip",
            "conditioning": conditioning,
            "text": text,
        }
        print(f"[ASDX] Text encoded (Krea2 grounded): {len(text)} chars, "
              f"grounding_px={grounding_px}")
        return io.NodeOutput(result)


NODE_LIST = [ASDX_Krea2GroundedEncode]
