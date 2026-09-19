"""
SDXL UNet architecture (native MLX).

Ported line-by-line from comfy's real UNetModel construction/forward loop
(`comfy/ldm/modules/diffusionmodules/openaimodel.py`) and its SpatialTransformer/
BasicTransformerBlock/CrossAttention (`comfy/ldm/modules/attention.py`), using
comfy's own SDXL config (`comfy/supported_models.py:200-267`,
`comfy/model_base.py:502-525` for the ADM/"y" vector).

Layout note: MLX's `nn.Conv2d` is channel-last (NHWC), unlike PyTorch's NCHW.
This UNet operates on `[B, H, W, C]` tensors throughout — the caller (bridge.py)
is responsible for transposing ComfyUI's NCHW latents in/out.

Every `nn.Sequential` here mirrors its PyTorch counterpart's child ordering
EXACTLY, including parameter-free placeholders (SiLU, Dropout) — MLX's
Sequential still allocates an index slot for them, so `weight_map.py` only
needs to insert `.layers.` after the attribute name, not renumber anything
(verified empirically: a 4-child Sequential with SiLU/Dropout at positions
1/2 keeps its Linear at `layers.3`, matching PyTorch's `Sequential.3`).
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .config import SDXLConfig


# ── Timestep / ADM embedding ────────────────────────────────────────────

def timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    """Sinusoidal timestep embedding, matching comfy's `timestep_embedding`.

    Unlike FLUX/Krea2's helper, SDXL timesteps are already raw discrete
    values in [0, 999] (no *1000 time_factor scaling) — comfy's own
    `openaimodel.py` calls `timestep_embedding(timesteps, model_channels,
    repeat_only=False)` with no extra scaling.
    """
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


def encode_adm(
    pooled_clip_g: mx.array,
    height: int,
    width: int,
    crop_h: int = 0,
    crop_w: int = 0,
    target_height: int | None = None,
    target_width: int | None = None,
) -> mx.array:
    """Build the SDXL ADM/"y" conditioning vector.

    Matches `comfy/model_base.py::SDXL.encode_adm` exactly: pooled CLIP-G
    (1280-dim) concatenated with 6 sinusoidal(256) embeddings of
    (height, width, crop_h, crop_w, target_height, target_width), IN THAT
    ORDER. Total: 1280 + 6*256 = 2816.

    Args:
        pooled_clip_g: [B, 1280] pooled CLIP-G output.
        height, width: generation resolution.
        crop_h, crop_w: crop offset (0 for standard generation).
        target_height, target_width: defaults to height/width if not given.

    Returns:
        [B, 2816] ADM vector.
    """
    if target_height is None:
        target_height = height
    if target_width is None:
        target_width = width

    batch = pooled_clip_g.shape[0]
    scalars = [height, width, crop_h, crop_w, target_height, target_width]
    parts = [pooled_clip_g]
    for s in scalars:
        t = mx.full((1,), float(s), dtype=mx.float32)
        parts.append(mx.broadcast_to(timestep_embedding(t, 256), (batch, 256)))
    return mx.concatenate(parts, axis=-1)


# ── Basic blocks ─────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """SDXL residual block: GroupNorm->SiLU->Conv, +timestep-emb, GroupNorm->SiLU->Conv.

    Matches `openaimodel.py::ResBlock` with `use_scale_shift_norm=False`
    (SDXL default): `h = h + emb_out` (post-norm additive), not scale+shift.
    """

    def __init__(self, channels: int, emb_channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels, eps=1e-5, pytorch_compatible=True),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels, eps=1e-5, pytorch_compatible=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if out_channels == channels:
            self.skip_connection = None
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        h = h + emb_out[:, None, None, :]
        h = self.out_layers(h)
        skip = x if self.skip_connection is None else self.skip_connection(x)
        return skip + h


class CrossAttention(nn.Module):
    """Standard multi-head attention, self- or cross- depending on `context`.

    No RoPE, no QK-norm — matches `attention.py::CrossAttention` exactly.
    """

    def __init__(self, query_dim: int, context_dim: int | None, heads: int, dim_head: int):
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(0.0))

    def __call__(self, x: mx.array, context: mx.array | None = None) -> mx.array:
        B, N, _ = x.shape
        context = context if context is not None else x
        M = context.shape[1]

        q = self.to_q(x).reshape(B, N, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = self.to_k(context).reshape(B, M, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, M, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.dim_head)
        attn = (q * scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head)
        return self.to_out(out)


class GEGLU(nn.Module):
    """GELU-gated linear unit: proj -> split -> x * gelu(gate)."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        return x * nn.gelu(gate)


class BasicTransformerBlock(nn.Module):
    """Pre-LN block: self-attn, cross-attn to `context`, GEGLU feed-forward.

    Matches `attention.py::BasicTransformerBlock`'s default path
    (`disable_self_attn=False`, `gated_ff=True`, no `ff_in`).
    """

    def __init__(self, dim: int, n_heads: int, d_head: int, context_dim: int):
        super().__init__()
        self.attn1 = CrossAttention(dim, None, n_heads, d_head)
        self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            GEGLU(dim, dim * 4),
            nn.Dropout(0.0),
            nn.Linear(dim * 4, dim),
        )

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """GroupNorm -> proj_in -> N x BasicTransformerBlock -> proj_out -> residual.

    `use_linear_in_transformer=True` for SDXL: proj_in/proj_out are Linear,
    operating on the flattened [B, H*W, C] sequence (no NHWC<->NCHW dance
    needed here since MLX is already channel-last).
    """

    def __init__(self, in_channels: int, n_heads: int, d_head: int, depth: int, context_dim: int):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, pytorch_compatible=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = [
            BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim) for _ in range(depth)
        ]
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        B, H, W, C = x.shape
        x_in = x
        x = self.norm(x)
        x = x.reshape(B, H * W, C)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context)
        x = self.proj_out(x)
        x = x.reshape(B, H, W, C)
        return x + x_in


class Downsample(nn.Module):
    """Stride-2 3x3 conv downsample. Attribute named `op` to match comfy's key."""

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.op(x)


class Upsample(nn.Module):
    """Nearest-neighbor 2x upsample + 3x3 conv. Attribute named `conv` to match comfy."""

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C)).reshape(B, H * 2, W * 2, C)
        return self.conv(x)


def _apply_block(module: nn.Module, h: mx.array, emb: mx.array, context: mx.array) -> mx.array:
    """Dispatch a single TimestepEmbedSequential child, matching
    `openaimodel.py::forward_timestep_embed`'s per-type routing."""
    if isinstance(module, ResBlock):
        return module(h, emb)
    if isinstance(module, SpatialTransformer):
        return module(h, context)
    return module(h)


# ── UNetModel ────────────────────────────────────────────────────────────

class UNetModel(nn.Module):
    """SDXL UNet: input_blocks (down) -> middle_block -> output_blocks (up).

    Block construction ported 1:1 from `openaimodel.py::UNetModel.__init__`
    (lines 528-836) to preserve the exact channel/attention-depth bookkeeping
    (`input_block_chans` skip-connection stack, `transformer_depth` consumed
    front-to-back on the way down / back-to-front on the way up).
    """

    def __init__(self, config: SDXLConfig | None = None):
        super().__init__()
        config = config or SDXLConfig()
        self.config = config
        self.dtype = config.mlx_dtype

        model_channels = config.model_channels
        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.label_emb = nn.Sequential(
            nn.Linear(config.adm_in_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        transformer_depth = list(config.transformer_depth)
        transformer_depth_output = list(config.transformer_depth_output)

        self.input_blocks: list[list[nn.Module]] = [
            [nn.Conv2d(config.in_channels, model_channels, kernel_size=3, padding=1)]
        ]
        input_block_chans = [model_channels]
        ch = model_channels

        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks):
                layers: list[nn.Module] = [ResBlock(ch, time_embed_dim, mult * model_channels)]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    num_heads = ch // config.num_head_channels
                    layers.append(SpatialTransformer(
                        ch, num_heads, config.num_head_channels, num_transformers, config.context_dim,
                    ))
                self.input_blocks.append(layers)
                input_block_chans.append(ch)
            if level != len(config.channel_mult) - 1:
                self.input_blocks.append([Downsample(ch, ch)])
                input_block_chans.append(ch)

        num_heads_mid = ch // config.num_head_channels
        self.middle_block: list[nn.Module] = [
            ResBlock(ch, time_embed_dim, ch),
            SpatialTransformer(
                ch, num_heads_mid, config.num_head_channels,
                config.transformer_depth_middle, config.context_dim,
            ),
            ResBlock(ch, time_embed_dim, ch),
        ]

        self.output_blocks: list[list[nn.Module]] = []
        for level, mult in list(enumerate(config.channel_mult))[::-1]:
            for i in range(config.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, model_channels * mult)]
                ch = model_channels * mult
                num_transformers = transformer_depth_output.pop()
                if num_transformers > 0:
                    num_heads = ch // config.num_head_channels
                    layers.append(SpatialTransformer(
                        ch, num_heads, config.num_head_channels, num_transformers, config.context_dim,
                    ))
                if level and i == config.num_res_blocks:
                    layers.append(Upsample(ch, ch))
                self.output_blocks.append(layers)

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch, eps=1e-5, pytorch_compatible=True),
            nn.SiLU(),
            nn.Conv2d(model_channels, config.out_channels, kernel_size=3, padding=1),
        )

    def __call__(self, x: mx.array, timesteps: mx.array, context: mx.array, y: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: [B, H, W, 4] latent (NHWC — caller transposes from ComfyUI's NCHW).
            timesteps: [B] raw discrete timestep values (see `timestep_embedding`).
            context: [B, T, 2048] concatenated CLIP-L+CLIP-G cross-attn context.
            y: [B, 2816] ADM vector (see `encode_adm`).

        Returns:
            [B, H, W, 4] predicted noise (NHWC).
        """
        t_emb = timestep_embedding(timesteps, self.config.model_channels).astype(self.dtype)
        emb = self.time_embed(t_emb)
        emb = emb + self.label_emb(y.astype(self.dtype))

        h = x.astype(self.dtype)
        hs = []
        for block_list in self.input_blocks:
            for module in block_list:
                h = _apply_block(module, h, emb, context)
            hs.append(h)

        for module in self.middle_block:
            h = _apply_block(module, h, emb, context)

        for block_list in self.output_blocks:
            hsp = hs.pop()
            h = mx.concatenate([h, hsp], axis=-1)
            for module in block_list:
                h = _apply_block(module, h, emb, context)

        return self.out(h)


# ── Loader ───────────────────────────────────────────────────────────────

def load_sdxl_unet(path, dtype: str = "float16") -> UNetModel:
    """Load an SDXL UNet checkpoint into a native MLX UNetModel.

    Same 6-step recipe as `load_transformer`/`load_krea2_transformer`:
    load safetensors -> normalize/map keys -> build model -> string-match
    `tree_flatten` keys directly against the checkpoint -> `tree_unflatten`
    -> `update()` -> `eval()`.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    from .. import _load_safetensors, _check_weight_match
    from .weight_map import normalize_sdxl_keys, map_sdxl_to_native

    state_dict = _load_safetensors(path)
    state_dict = normalize_sdxl_keys(state_dict)
    state_dict = map_sdxl_to_native(state_dict)

    config = SDXLConfig(dtype=dtype)
    unet = UNetModel(config)

    model_flat = tree_flatten(unet.parameters())
    new_flat = []
    matched = 0
    for flat_key, value in model_flat:
        if flat_key in state_dict:
            loaded = state_dict.pop(flat_key)
            if loaded.ndim == 4:
                # Every 4D tensor in this checkpoint is a Conv2d kernel:
                # PyTorch [out, in, kh, kw] -> MLX [out, kh, kw, in].
                loaded = loaded.transpose(0, 2, 3, 1)
            new_flat.append((flat_key, loaded.astype(config.mlx_dtype)))
            matched += 1
        else:
            new_flat.append((flat_key, value))
    new_nested = tree_unflatten(new_flat)
    unet.update(new_nested)
    mx.eval(unet.parameters())

    print(f"[ASDX] SDXL UNet: matched {matched}/{len(model_flat)} params from checkpoint")
    _check_weight_match(matched, len(model_flat), "SDXL UNet", path)
    return unet
