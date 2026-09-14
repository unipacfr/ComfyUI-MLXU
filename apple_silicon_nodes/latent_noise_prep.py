"""ASDX_LatentNoisePrep — img2img/inpaint/fill mode routing as a focused node.

Extracts the sampler's mode/image/mask/image_strength/mask_blur/mask_padding
inputs into their own upstream node, following the model-dict-priority
pattern already used for `lora_schedule`/`identity_edit`/`depth_cond`/
`kontext` -- with one difference from those: the actual noise blend
(`_SamplerCore._prepare_img2img_noise`/`_prepare_inpainting_noise`) needs the
SEEDED noise tensor the sampler derives from `latent_image` + `seed`
(`bridge.prepare_noise_from_latent`), which is not available at this node's
`execute()` time. So this node only validates and mode-tags the raw
ingredients -- the actual blend-with-noise step stays in `_SamplerCore`,
where the noise tensor already lives.

"depth" is deliberately NOT one of this node's `mode` options: depth-control
routing is `ASDX_DepthConditioning`'s job (Phase D) and doesn't go through
noise-blending at all (it concatenates a channel partner, it never blends
into the noise -- see `_SamplerCore._prepare_depth_noise`).
"""

from __future__ import annotations

from typing import Any

import torch

from comfy_api.latest import io

_MODES = ["auto", "text2img", "img2img", "inpaint", "fill"]


class ASDX_LatentNoisePrep(io.ComfyNode):
    """Validate and mode-tag img2img/inpaint/fill ingredients for the sampler.

    Emits the model dict with a ``latent_prep`` entry the sampler consumes in
    priority over its own same-named mode/image/mask/image_strength/
    mask_blur/mask_padding inputs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_LatentNoisePrep",
            display_name="🍏 ASDX Latent Noise Prep",
            category="ASDX/Conditioning",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Combo.Input("mode", options=_MODES, default="auto"),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Float.Input("image_strength", default=0.8, min=0.0, max=1.0, step=0.01, optional=True),
                io.Int.Input("mask_blur", default=0, min=0, max=64, optional=True),
                io.Int.Input("mask_padding", default=48, min=32, max=256, optional=True),
            ],
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: dict,
        mode: str,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        image_strength: float = 0.8,
        mask_blur: int = 0,
        mask_padding: int = 48,
    ) -> io.NodeOutput:
        if mode == "inpaint" and mask is None:
            raise RuntimeError(
                "ASDX_LatentNoisePrep: mode='inpaint' requires a mask input."
            )

        latent_prep = {
            "mode": mode,
            "image": image,
            "mask": mask,
            "image_strength": image_strength,
            "mask_blur": mask_blur,
            "mask_padding": mask_padding,
        }
        # Shallow copy -- never mutate the input dict (it may be the cached
        # model shared across executions), same rule as ASDX_LoraLoader /
        # ASDX_Krea2Edit / ASDX_DepthConditioning / ASDX_KontextReference.
        new_model = {**model, "latent_prep": latent_prep}
        return io.NodeOutput(new_model)


NODE_LIST = [ASDX_LatentNoisePrep]
