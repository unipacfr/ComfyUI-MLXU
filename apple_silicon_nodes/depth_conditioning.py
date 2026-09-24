"""ASDX_DepthConditioning — FLUX depth-control as a focused, composable node.

Extracts the depth-control inputs (`depth_image`/`depth_strength`) out of
`ASDX_MLXSampler` into their own upstream node, following the same
model-dict-priority pattern already used for `lora_schedule`/`identity_edit`:
this node validates and stores the raw ingredients (`depth_image`, `vae`,
`strength`) in `model["depth_cond"]`; the actual VAE-encode + 2x2 pack stays
in `_SamplerCore._prepare_depth_noise` (it needs `self.noise`'s shape to
resize against, so encoding can't happen upstream in this node).

Only meaningful for `flux1_depth` checkpoints (Task C1's dynamically
detected wide `img_in`) — a non-depth model is refused with an explicit
error, same convention as `ASDX_Krea2Edit`.
"""

from __future__ import annotations

from typing import Any

import torch

from comfy_api.latest import io


class ASDX_DepthConditioning(io.ComfyNode):
    """Prepare FLUX depth-control ingredients for the sampler.

    Emits the model dict with a ``depth_cond`` entry the sampler consumes in
    priority over its legacy ``depth_image``/``depth_strength`` inputs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_DepthConditioning",
            display_name="🍏 ASDX Depth Conditioning",
            category="ASDX/Conditioning",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Image.Input("depth_image"),
                io.Vae.Input("vae"),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.01),
            ],
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: dict,
        depth_image: torch.Tensor,
        vae: Any,
        strength: float,
    ) -> io.NodeOutput:
        family = model["capability"].family
        if family != "flux1_depth":
            raise RuntimeError(
                f"ASDX_DepthConditioning: requires a flux1_depth model, got "
                f"family '{family}'. Depth control needs a checkpoint with a "
                f"depth-conditioned img_in (e.g. flux1-depth-dev)."
            )

        depth_cond = {"depth_image": depth_image, "vae": vae, "strength": strength}
        # Shallow copy -- never mutate the input dict (it may be the cached
        # model shared across executions), same rule as ASDX_LoraLoader /
        # ASDX_Krea2Edit.
        new_model = {**model, "depth_cond": depth_cond}
        return io.NodeOutput(new_model)


NODE_LIST = [ASDX_DepthConditioning]
