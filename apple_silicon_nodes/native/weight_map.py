"""State dict weight mapping for FLUX.1 (BFL-native) checkpoints.

FLUX.1 checkpoints (as distributed by Black Forest Labs, and as used by
ComfyUI's own `comfy.ldm.flux.model.Flux`) already use almost exactly our
internal naming — verified against `comfy/utils.py::flux_to_diffusers`'s
`block_map`, which lists the checkpoint's own (non-diffusers) key names on
the right-hand side of each mapping:

  double_blocks.{i}.img_mod.lin.weight/bias
  double_blocks.{i}.txt_mod.lin.weight/bias
  double_blocks.{i}.img_attn.qkv.weight/bias
  double_blocks.{i}.img_attn.proj.weight/bias
  double_blocks.{i}.img_attn.norm.query_norm.weight   (QKNorm)
  double_blocks.{i}.img_attn.norm.key_norm.weight
  double_blocks.{i}.txt_attn.{qkv,proj,norm.*}         (same shape, txt_ prefix)
  double_blocks.{i}.img_mlp.0.weight/bias
  double_blocks.{i}.img_mlp.2.weight/bias
  double_blocks.{i}.txt_mlp.0/2.weight/bias
  single_blocks.{i}.modulation.lin.weight/bias
  single_blocks.{i}.linear1.weight/bias                (fused qkv + mlp_in)
  single_blocks.{i}.linear2.weight/bias                (fused proj + mlp_out)
  single_blocks.{i}.norm.query_norm.weight/key_norm.weight
  img_in.weight/bias, txt_in.weight/bias
  time_in.in_layer.{weight,bias}, time_in.out_layer.{weight,bias}
  vector_in.in_layer/out_layer.{weight,bias}
  guidance_in.in_layer/out_layer.{weight,bias}
  final_layer.linear.weight/bias
  final_layer.adaLN_modulation.1.weight/bias

Our module tree uses the SAME names, except:
  - MLP layers are attributes `img_mlp_0`/`img_mlp_2` (underscore) instead of
    a Sequential `img_mlp.0`/`img_mlp.2` (dot) — direct rename.
  - `MLPEmbedder.in_layer`/`out_layer` for time_in/vector_in/guidance_in match
    the checkpoint exactly — no rename needed there.
  - `final_layer.adaLN_modulation` is an `nn.Sequential(SiLU, Linear)`; MLX's
    Sequential stores parameters as `adaLN_modulation.layers.{index}.*`
    (checkpoint uses flat `adaLN_modulation.1.*` — dot-index, no "layers").
  - QKNorm scale is stored as `*_norm.scale` in some checkpoint variants
    (RMSNorm convention) and must map to `.weight` (MLX's `nn.RMSNorm` param name).

This module handles exactly those three adjustments; everything else is a
straight prefix-strip pass-through.
"""

from __future__ import annotations

import mlx.core as mx


def normalize_flux_keys(
    state_dict: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Strip common ComfyUI/diffusion-model prefixes from checkpoint keys.

    Args:
        state_dict: Raw state dict from checkpoint file.

    Returns:
        State dict with cleaned keys.
    """
    normalized: dict[str, mx.array] = {}

    for key, value in state_dict.items():
        for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value

    return normalized


def detect_flux_in_channels(normalized: dict[str, mx.array]) -> int:
    """Read img_in's real input width from the checkpoint, mirroring
    comfy/model_base.py::Flux.concat_cond's own try/except on the live
    weight shape. 64 = plain FLUX.1 (16ch * 2x2 patch); 128 = depth/canny
    dev variants (32ch: 16 noise + 16 depth-or-canny latent).
    """
    weight = normalized.get("img_in.weight")
    if weight is None:
        return 64
    return int(weight.shape[1])


def map_flux_to_native(state_dict: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map a normalized FLUX.1 (BFL-native) state dict to our module naming.

    Args:
        state_dict: State dict from a FLUX.1 checkpoint, already prefix-stripped.

    Returns:
        State dict with keys mapped to our native module attribute paths.
    """
    result: dict[str, mx.array] = {}

    for key, value in state_dict.items():
        new_key = key

        # ── img_mlp.{0,2} / txt_mlp.{0,2} -> img_mlp_0/2, txt_mlp_0/2 ──────
        if ".img_mlp." in new_key:
            new_key = new_key.replace(".img_mlp.", ".img_mlp_")
        elif ".txt_mlp." in new_key:
            new_key = new_key.replace(".txt_mlp.", ".txt_mlp_")

        # ── QKNorm scale -> weight (RMSNorm convention some exporters use) ──
        elif new_key.endswith(("query_norm.scale", "key_norm.scale")):
            new_key = new_key[: -len("scale")] + "weight"

        # ── final_layer.adaLN_modulation.{1} -> insert MLX Sequential .layers ──
        elif new_key.startswith("final_layer.adaLN_modulation."):
            # "final_layer.adaLN_modulation.1.weight" -> "final_layer.adaLN_modulation.layers.1.weight"
            suffix = new_key[len("final_layer.adaLN_modulation."):]
            new_key = f"final_layer.adaLN_modulation.layers.{suffix}"

        result[new_key] = value

    return result
