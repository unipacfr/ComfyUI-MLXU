"""ASDX_KontextReference — Kontext reference-image conditioning as a focused node.

Extracts the Kontext inputs (`kontext`/`kontext_reference_latent`/
`kontext_reference_strength`) out of `ASDX_MLXSampler` into their own
upstream node, following the same model-dict-priority pattern already used
for `lora_schedule`/`identity_edit`/`depth_cond`: this node just validates
and stores the reference latent + strength in `model["kontext"]`; the
sampler still does the actual RoPE-table extension and per-generation
packing (`_SamplerCore._prepare_kontext_reference`) since that needs
`self.noise`'s grid and the shared `rope` table it builds alongside it.
"""

from __future__ import annotations

from comfy_api.latest import io


class ASDX_KontextReference(io.ComfyNode):
    """Attach a Kontext reference latent to the model dict.

    Emits the model dict with a ``kontext`` entry the sampler consumes in
    priority over its legacy ``kontext``/``kontext_reference_latent``/
    ``kontext_reference_strength`` inputs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_KontextReference",
            display_name="🍏 ASDX Kontext Reference",
            category="ASDX/Conditioning",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Latent.Input("reference_latent"),
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
        reference_latent: dict,
        strength: float,
    ) -> io.NodeOutput:
        kontext = {"reference_latent": reference_latent, "strength": strength}
        # Shallow copy -- never mutate the input dict (it may be the cached
        # model shared across executions), same rule as ASDX_LoraLoader /
        # ASDX_Krea2Edit / ASDX_DepthConditioning.
        new_model = {**model, "kontext": kontext}
        return io.NodeOutput(new_model)


NODE_LIST = [ASDX_KontextReference]
