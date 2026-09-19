"""Qwen3-VL vision tower (ViT + DeepStack) in MLX, for MiniMax H3's
vision-grounded text conditioning (fl2va keyframes, ref2va references).

Ported from `comfy/text_encoders/qwen35.py::Qwen35VisionModel` and
`qwen3vl.py::Qwen3VLVisionModel`. Parameter names match the checkpoint's
`visual.*` keys (prefix stripped) so `vision_weight_map.py` assigns by name.
The one deliberate difference: the reference's `Conv3d` patch embed has
kernel == stride, so it is exactly a Linear over the flattened
`[C, T, P, P]` patch; `patch_embed.proj` is that Linear (weight `[hidden,
patch_dim]`) and the loader reshapes the 5-D / GGUF 4-D conv weight into it.

Everything runs in float32, like the reference (`image.to(float32)`).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    depth: int = 27
    num_heads: int = 16
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)
    out_hidden_size: int = 5120

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

    @property
    def merge_dim(self) -> int:
        return self.hidden_size * self.spatial_merge_size**2


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Split-half rotary embedding on `[N, heads, head_dim]`."""
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


class VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden, bias=True)
        self.linear_fc2 = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, spans: list[tuple[int, int]]) -> mx.array:
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        q = _apply_rope(qkv[:, 0], cos, sin)
        k = _apply_rope(qkv[:, 1], cos, sin)
        v = qkv[:, 2]
        outs = []
        for a, b in spans:  # attention never crosses a frame boundary
            qs, ks, vs = (t[a:b].transpose(1, 0, 2)[None] for t in (q, k, v))
            o = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=self.head_dim**-0.5)
            outs.append(o[0].transpose(1, 0, 2).reshape(b - a, -1))
        return self.proj(mx.concatenate(outs, axis=0))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config.hidden_size, config.num_heads)
        self.mlp = VisionMLP(config.hidden_size, config.intermediate_size)

    def __call__(self, x, cos, sin, spans):
        x = x + self.attn(self.norm1(x), cos, sin, spans)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.proj = nn.Linear(config.patch_dim, config.hidden_size, bias=True)


class PatchMerger(nn.Module):
    """Main merger: LayerNorm on the per-patch hidden size, THEN spatial merge."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.merge_dim = config.merge_dim
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(config.merge_dim, config.merge_dim, bias=True)
        self.linear_fc2 = nn.Linear(config.merge_dim, config.out_hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x).reshape(-1, self.merge_dim)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class DeepstackMerger(nn.Module):
    """DeepStack merger: spatial merge FIRST, then LayerNorm on the merged dim."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.merge_dim = config.merge_dim
        self.norm = nn.LayerNorm(config.merge_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(config.merge_dim, config.merge_dim, bias=True)
        self.linear_fc2 = nn.Linear(config.merge_dim, config.out_hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x.reshape(-1, self.merge_dim))
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class VisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config)
        self.deepstack_merger_list = [DeepstackMerger(config) for _ in config.deepstack_visual_indexes]

    # ---- host-side index/table construction (small, once per call) ----

    def _rotary_tables(self, grids: list[tuple[int, int, int]]) -> tuple[mx.array, mx.array]:
        cfg = self.config
        merge = cfg.spatial_merge_size
        dim = cfg.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        max_hw = max(max(h, w) for _, h, w in grids)
        table = np.outer(np.arange(max_hw, dtype=np.float32), inv_freq)  # [max_hw, dim/2]
        rows = []
        for t, h, w in grids:
            r = np.arange(h // merge)[:, None, None, None] * merge + np.arange(merge)[None, None, :, None]
            c = np.arange(w // merge)[None, :, None, None] * merge + np.arange(merge)[None, None, None, :]
            r = np.broadcast_to(r, (h // merge, w // merge, merge, merge)).reshape(-1)
            c = np.broadcast_to(c, (h // merge, w // merge, merge, merge)).reshape(-1)
            coords = np.stack([r, c], axis=-1)
            rows.append(np.tile(coords, (t, 1)) if t > 1 else coords)
        pos = np.concatenate(rows)  # [N, 2]
        freqs = table[pos].reshape(pos.shape[0], -1)  # [N, head_dim/2]
        emb = np.concatenate([freqs, freqs], axis=-1)  # [N, head_dim]
        return mx.array(np.cos(emb)[:, None, :]), mx.array(np.sin(emb)[:, None, :])

    def _pos_embeddings(self, grids: list[tuple[int, int, int]]) -> mx.array:
        cfg = self.config
        side = int(cfg.num_position_embeddings**0.5)
        merge = cfg.spatial_merge_size
        pieces = []
        for t, h, w in grids:
            h_idx = np.linspace(0, side - 1, h, dtype=np.float32)
            w_idx = np.linspace(0, side - 1, w, dtype=np.float32)
            h0, w0 = h_idx.astype(np.int64), w_idx.astype(np.int64)
            h1, w1 = np.minimum(h0 + 1, side - 1), np.minimum(w0 + 1, side - 1)
            dh, dw = (h_idx - h0)[:, None], (w_idx - w0)[None, :]
            idx = [(h0[:, None] * side + w0[None]), (h0[:, None] * side + w1[None]),
                   (h1[:, None] * side + w0[None]), (h1[:, None] * side + w1[None])]
            wts = [(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw]
            emb = sum(
                self.pos_embed(mx.array(i.reshape(-1))) * mx.array(wt.reshape(-1, 1).astype(np.float32))
                for i, wt in zip(idx, wts)
            )  # [h*w, C]
            emb = mx.tile(emb, (t, 1)) if t > 1 else emb
            emb = emb.reshape(t, h // merge, merge, w // merge, merge, -1)
            pieces.append(emb.transpose(0, 1, 3, 2, 4, 5).reshape(-1, emb.shape[-1]))
        return mx.concatenate(pieces, axis=0)

    # ---- forward ----

    def __call__(self, patches: mx.array, grids: list[tuple[int, int, int]]) -> tuple[mx.array, list[mx.array]]:
        """`patches`: `[N, patch_dim]` (see `vision_preprocess`), `grids`: one
        `(t, h, w)` per image. Returns `(merged, deepstack)`, each `[N/4,
        out_hidden]` (spatial_merge_size**2 patches per output row)."""
        x = self.patch_embed.proj(patches.astype(mx.float32))
        x = x + self._pos_embeddings(grids)
        cos, sin = self._rotary_tables(grids)
        spans, offset = [], 0
        for t, h, w in grids:
            for _ in range(t):  # one attention segment per frame
                spans.append((offset, offset + h * w))
                offset += h * w
        deepstack: list[mx.array] = []
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, spans)
            if i in self.config.deepstack_visual_indexes:
                deepstack.append(self.deepstack_merger_list[self.config.deepstack_visual_indexes.index(i)](x))
        return self.merger(x), deepstack
