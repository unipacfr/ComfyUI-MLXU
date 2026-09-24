"""
Native MLX Transformer for Flux2 (FLUX.2/Klein)
================================================
Matches the reference architecture in ComfyUI's `comfy/ldm/flux/{model,
layers,math}.py` for the `image_model == "flux2"` branch (same source file
as FLUX.1 — Flux2 is a parametrized variant of the same `Flux` class, not a
separate model file).

Delta vs the FLUX.1 port already in `native/__init__.py` (verified by
reading the real comfy source, not assumed from FLUX.1's structure):
  - Global modulation: 3 top-level `Modulation` layers
    (`double_stream_modulation_img`, `double_stream_modulation_txt`,
    `single_stream_modulation`) computed ONCE per forward pass and shared by
    every double/single block — FLUX.1 gives each block its own modulation.
  - MLP is a fused SiLU-gated GLU (`Linear(dim, 2*hidden, bias=False) ->
    silu(x1)*x2 -> Linear(hidden, dim, bias=False)`), not GELU-tanh.
  - Every Linear is bias-free (`ops_bias=False`), including qkv/proj.
  - 4-axis RoPE (vs FLUX.1's 3), theta=2000 (vs 10000). Text tokens get a
    sequential position on ONE dedicated axis (index 3) instead of sitting
    at position 0 on every axis.
  - 128 latent channels, patch_size=1 (no 2x2 token patchify).
  - No pooled/ADM vector conditioning (no `vector_in` for either real
    checkpoint on this machine). Guidance embedding is checkpoint-dependent:
    Klein has none (`guidance_in` stays `None`), the larger Flux2-D
    checkpoint has one — `Flux2Transformer` allocates and uses it only when
    `Flux2Config.guidance_embed` (detected from the checkpoint) is True.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import Flux2Config
from ..common import timestep_embedding


# ── RoPE (4-axis, paired-interleave convention) ─────────────────────────
# Duplicated from native/__init__.py rather than imported, matching this
# project's established per-family convention (native/krea2, native/zimage
# do the same) — avoids circular imports and keeps each family's math
# self-contained even though the underlying convention is identical.

def rope_freqs(pos: mx.array, dim: int, theta: float) -> mx.array:
    """[N, dim/2, 2, 2] rotation-matrix RoPE table for one axis."""
    assert dim % 2 == 0
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta ** scale)
    out = pos.astype(mx.float32)[:, None] * omega[None, :]
    cos, sin = mx.cos(out), mx.sin(out)
    return mx.stack([cos, -sin, sin, cos], axis=-1).reshape(*out.shape, 2, 2)


def embed_nd(ids: mx.array, axes_dim: tuple[int, ...], theta: float) -> mx.array:
    """N-axis RoPE embedding table (generic over axis count, unlike the name suggests)."""
    parts = [rope_freqs(ids[:, i], axes_dim[i], theta) for i in range(len(axes_dim))]
    return mx.concatenate(parts, axis=-3)


def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply paired-interleave RoPE rotation to Q or K. x: [B,H,N,D]."""
    B, H, N, D = x.shape
    x_pairs = x.reshape(B, H, N, D // 2, 1, 2)
    f = freqs[None, None]
    out = (f[..., 0] * x_pairs[..., 0]) + (f[..., 1] * x_pairs[..., 1])
    return out.reshape(B, H, N, D)


def joint_attention(q: mx.array, k: mx.array, v: mx.array, freqs: mx.array) -> mx.array:
    """Scaled dot-product attention with RoPE. q,k,v: [B,H,N,head_dim]."""
    B, H, N, Dh = q.shape
    q = apply_rope(q, freqs)
    k = apply_rope(k, freqs)
    scale = 1.0 / math.sqrt(Dh)
    attn = (q * scale) @ k.transpose(0, 1, 3, 2)
    attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
    out = attn @ v
    return out.transpose(0, 2, 1, 3).reshape(B, N, H * Dh)


class QKNorm(nn.Module):
    """Per-head RMSNorm applied to Q and K after the head split."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

    def __call__(self, q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
        return self.query_norm(q), self.key_norm(k)



class MLPEmbedder(nn.Module):
    """in_dim -> hidden_dim -> SiLU -> hidden_dim. Bias-free for Flux2 (ops_bias=False)."""

    def __init__(self, in_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.silu(self.in_layer(x)))


# ── Modulation (global, computed once per forward, shared across blocks) ─

class Modulation(nn.Module):
    """adaLN-style modulation: SiLU(vec) -> Linear -> chunk into (shift, scale, gate)(x N).

    double=True: returns (mod1, mod2), 6 chunks total.
    double=False: returns (mod1, None), 3 chunks total.
    Flux2 always constructs these with bias=False (comfy's Modulation(...,
    bias=False) for `global_modulation` layers specifically — FLUX.1's
    per-block Modulation layers keep the default bias=True, but those don't
    exist in this file at all).
    """

    def __init__(self, dim: int, double: bool, bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=bias)

    def __call__(self, vec: mx.array):
        out = self.lin(nn.silu(vec))
        parts = mx.split(out, self.multiplier, axis=-1)
        parts = [p[:, None, :] for p in parts]
        mod1 = (parts[0], parts[1], parts[2])
        mod2 = (parts[3], parts[4], parts[5]) if self.is_double else None
        return mod1, mod2


def apply_mod(x: mx.array, scale: mx.array, shift: mx.array | None) -> mx.array:
    """x * (1 + scale) [+ shift]. scale/shift are [B,1,D], x is [B,N,D]."""
    out = x * (1 + scale)
    if shift is not None:
        out = out + shift
    return out


# ── SiLU-gated GLU MLP (mlp_silu_act=True) ──────────────────────────────

def silu_glu(gate_up: mx.array) -> mx.array:
    """[B,N,2*H] fused gate+up projection -> silu(gate) * up -> [B,N,H]."""
    x1, x2 = mx.split(gate_up, 2, axis=-1)
    return nn.silu(x1) * x2


# ── Attention projection (qkv + QKNorm + output proj, bias-free) ────────

class SelfAttentionProj(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.norm = QKNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim, bias=bias)


# ── Double Block (joint img+txt self-attention, externally modulated) ───

class DoubleBlock(nn.Module):
    """Flux2 double-stream block.

    Unlike FLUX.1's DoubleBlock, this owns NO Modulation submodules of its
    own — `mod` (the precomputed global modulation output) is passed in
    fresh at every call, identical across all `num_double_blocks` instances
    within one forward pass (matches comfy: `global_modulation=True` implies
    `modulation=False` per-block, and the block's `vec` argument is already
    the `((img_mod1,img_mod2),(txt_mod1,txt_mod2))` tuple, not a raw vector).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.dim = dim
        self.img_norm1 = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.img_attn = SelfAttentionProj(dim, num_heads, bias=False)
        self.img_norm2 = nn.LayerNorm(dim, affine=False, eps=1e-6)

        self.txt_norm1 = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.txt_attn = SelfAttentionProj(dim, num_heads, bias=False)
        self.txt_norm2 = nn.LayerNorm(dim, affine=False, eps=1e-6)

        mlp_hidden = int(dim * mlp_ratio)
        self.img_mlp_0 = nn.Linear(dim, mlp_hidden * 2, bias=False)
        self.img_mlp_2 = nn.Linear(mlp_hidden, dim, bias=False)
        self.txt_mlp_0 = nn.Linear(dim, mlp_hidden * 2, bias=False)
        self.txt_mlp_2 = nn.Linear(mlp_hidden, dim, bias=False)

    def _qkv_heads(self, attn: SelfAttentionProj, x: mx.array):
        B, N, D = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = attn.norm(q, k)
        return q, k, v

    def __call__(self, img: mx.array, txt: mx.array, mod, freqs: mx.array):
        """
        Args:
            img: [B, N_img, D]
            txt: [B, N_txt, D]
            mod: ((img_mod1, img_mod2), (txt_mod1, txt_mod2)) precomputed globally.
            freqs: [N_txt+N_img, head_dim/2, 2, 2] RoPE table (txt positions first).
        """
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = mod
        img_shift1, img_scale1, img_gate1 = img_mod1
        img_shift2, img_scale2, img_gate2 = img_mod2
        txt_shift1, txt_scale1, txt_gate1 = txt_mod1
        txt_shift2, txt_scale2, txt_gate2 = txt_mod2

        img_modulated = apply_mod(self.img_norm1(img), img_scale1, img_shift1)
        img_q, img_k, img_v = self._qkv_heads(self.img_attn, img_modulated)

        txt_modulated = apply_mod(self.txt_norm1(txt), txt_scale1, txt_shift1)
        txt_q, txt_k, txt_v = self._qkv_heads(self.txt_attn, txt_modulated)

        q = mx.concatenate([txt_q, img_q], axis=2)
        k = mx.concatenate([txt_k, img_k], axis=2)
        v = mx.concatenate([txt_v, img_v], axis=2)

        attn = joint_attention(q, k, v, freqs)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        img = img + self.img_attn.proj(img_attn) * img_gate1
        txt = txt + self.txt_attn.proj(txt_attn) * txt_gate1

        img_mlp_in = apply_mod(self.img_norm2(img), img_scale2, img_shift2)
        img_mlp = self.img_mlp_2(silu_glu(self.img_mlp_0(img_mlp_in)))
        img = img + img_mlp * img_gate2

        txt_mlp_in = apply_mod(self.txt_norm2(txt), txt_scale2, txt_shift2)
        txt_mlp = self.txt_mlp_2(silu_glu(self.txt_mlp_0(txt_mlp_in)))
        txt = txt + txt_mlp * txt_gate2

        if txt.dtype == mx.float16:
            txt = mx.nan_to_num(txt, nan=0.0, posinf=65504.0, neginf=-65504.0)

        return img, txt


# ── Single Block (fused joint attention + MLP, externally modulated) ────

class SingleBlock(nn.Module):
    """Flux2 single-stream block. Owns no Modulation of its own (see DoubleBlock)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mlp_hidden = int(dim * mlp_ratio)

        self.pre_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.norm = QKNorm(self.head_dim)

        # qkv (3*dim) + fused gate_up mlp_in (2*mlp_hidden), all bias-free
        self.linear1 = nn.Linear(dim, dim * 3 + self.mlp_hidden * 2, bias=False)
        # attn_out (dim) + gated mlp_out (mlp_hidden) fused back to dim
        self.linear2 = nn.Linear(dim + self.mlp_hidden, dim, bias=False)

    def __call__(self, x: mx.array, mod, freqs: mx.array):
        """
        Args:
            x: [B, N, D] concatenated [txt, img] tokens.
            mod: (shift, scale, gate) precomputed globally, shared by every
                single block (matches comfy: `vec, _ =
                self.single_stream_modulation(vec_orig)` computed once).
            freqs: [N, head_dim/2, 2, 2] RoPE table.
        """
        shift, scale, gate = mod
        B, N, D = x.shape
        modulated = apply_mod(self.pre_norm(x), scale, shift)
        fused = self.linear1(modulated)
        qkv, mlp_in = mx.split(fused, [3 * self.dim], axis=-1)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.norm(q, k)

        attn = joint_attention(q, k, v, freqs)
        mlp_act = silu_glu(mlp_in)
        out = self.linear2(mx.concatenate([attn, mlp_act], axis=-1))
        x = x + out * gate

        if x.dtype == mx.float16:
            x = mx.nan_to_num(x, nan=0.0, posinf=65504.0, neginf=-65504.0)

        return x


# ── LastLayer ────────────────────────────────────────────────────────────

class LastLayer(nn.Module):
    """Final output layer: adaLN modulation (scale+shift only) -> LayerNorm -> Linear."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=False),
        )

    def __call__(self, x: mx.array, vec: mx.array) -> mx.array:
        shift, scale = mx.split(self.adaLN_modulation(vec), 2, axis=-1)
        x = apply_mod(self.norm_final(x), scale[:, None, :], shift[:, None, :])
        return self.linear(x)


# ── Main Transformer ──────────────────────────────────────────────────

class Flux2Transformer(nn.Module):
    """Complete Flux2/Klein transformer.

    Architecture:
      img_in / txt_in -> [double_stream_modulation_*, single_stream_modulation
      computed once] -> Nx DoubleBlock -> Mx SingleBlock -> final_layer
    """

    def __init__(self, config: Flux2Config | None = None):
        super().__init__()
        config = config or Flux2Config()
        self.config = config
        self.dtype = config.mlx_dtype

        self.img_in = nn.Linear(config.in_channels, config.hidden_size, bias=False)
        self.txt_in = nn.Linear(config.context_in_dim, config.hidden_size, bias=False)
        self.time_in = MLPEmbedder(256, config.hidden_size, bias=False)
        # Klein has no guidance_in.* keys (guidance_embed=False); the larger
        # Flux2-D checkpoint does — only allocate this branch when detected,
        # matching FLUX.1's `nn.Identity() if not guidance_embed` pattern
        # (here: simply None, checked in time_embed()).
        self.guidance_in = MLPEmbedder(256, config.hidden_size, bias=False) if config.guidance_embed else None

        self.double_blocks = [
            DoubleBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.num_double_blocks)
        ]
        self.single_blocks = [
            SingleBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.num_single_blocks)
        ]

        self.final_layer = LastLayer(config.hidden_size, config.in_channels)

        # Global modulation: computed once per forward, shared by every
        # block of its stream (double blocks reuse the same img/txt mod;
        # single blocks reuse the same mod) — the defining structural
        # difference from FLUX.1's per-block modulation.
        self.double_stream_modulation_img = Modulation(config.hidden_size, double=True, bias=False)
        self.double_stream_modulation_txt = Modulation(config.hidden_size, double=True, bias=False)
        self.single_stream_modulation = Modulation(config.hidden_size, double=False, bias=False)

    def get_rope(self, img_h: int, img_w: int, txt_len: int) -> mx.array:
        """Compute the 4-axis RoPE table for a [txt, img] sequence.

        Image tokens get (row, col) grid coordinates on axes 1/2 (axes 0 and
        3 stay zero for image — matches `process_img`, which only ever
        writes axes 0/1/2, leaving axis 3 untouched). Text tokens get a
        sequential index on `config.txt_ids_dim` (axis 3) ONLY — every other
        axis stays zero for text (matches `txt_ids_dims=[3]`, the inverse of
        FLUX.1's `txt_ids_dims=[]` where text sits at zero on every axis).

        Reference (Kontext-style) latents are out of scope for this v1 —
        `ref_index_scale`/`default_ref_method="index"` are recorded in the
        config for a future extension but not wired up here.
        """
        img_len = img_h * img_w
        total = txt_len + img_len
        n_axes = len(self.config.axes_dim)
        ids = mx.zeros((total, n_axes), dtype=mx.float32)

        if img_len > 0:
            rows = mx.arange(img_h, dtype=mx.float32)[:, None]
            cols = mx.arange(img_w, dtype=mx.float32)[None, :]
            rows = mx.broadcast_to(rows, (img_h, img_w)).reshape(-1)
            cols = mx.broadcast_to(cols, (img_h, img_w)).reshape(-1)
            ids[txt_len:txt_len + img_len, 1] = rows
            ids[txt_len:txt_len + img_len, 2] = cols

        if txt_len > 0:
            ids[:txt_len, self.config.txt_ids_dim] = mx.arange(txt_len, dtype=mx.float32)

        return embed_nd(ids, self.config.axes_dim, self.config.theta)

    def time_embed(self, t: mx.array, guidance: mx.array | None = None) -> mx.array:
        """[B, hidden_size] conditioning vector: timestep (+ guidance if the
        checkpoint has a guidance embedding — Flux2-D, unlike Klein). No
        pooled/ADM contribution either way (no `vector_in` in this family)."""
        vec = self.time_in(timestep_embedding(t, 256).astype(self.dtype))
        if guidance is not None and self.guidance_in is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).astype(self.dtype))
        return vec

    def __call__(
        self,
        img: mx.array,   # [B, N_img, in_channels] raw latent tokens (no patchify)
        txt: mx.array,   # [B, N_txt, context_in_dim] tapped-and-concatenated text embeddings
        t: mx.array,     # [B] timestep
        guidance: mx.array | None = None,  # [B] guidance scale (Flux2-D only; ignored if the checkpoint has no guidance_in)
        rope: mx.array | None = None,  # precomputed 4-axis RoPE table
    ) -> mx.array:
        """Forward pass. Returns [B, N_img, in_channels] noise prediction."""
        img_tokens = img.shape[1]

        img = self.img_in(img.astype(self.dtype))
        txt = self.txt_in(txt.astype(self.dtype))

        vec = self.time_embed(t, guidance=guidance)

        if rope is None:
            raise ValueError(
                "ASDX: Flux2Transformer.__call__ requires a precomputed `rope` table. "
                "Call get_rope(img_h, img_w, txt_len) with the real image token grid "
                "shape before calling predict()/__call__."
            )

        # Global modulation, computed once and shared by every block.
        double_mod = (self.double_stream_modulation_img(vec), self.double_stream_modulation_txt(vec))
        single_mod, _ = self.single_stream_modulation(vec)

        for block in self.double_blocks:
            img, txt = block(img, txt, double_mod, rope)

        if img.dtype == mx.float16:
            img = mx.nan_to_num(img, nan=0.0, posinf=65504.0, neginf=-65504.0)

        x = mx.concatenate([txt, img], axis=1)

        for block in self.single_blocks:
            x = block(x, single_mod, rope)

        img_out = x[:, txt.shape[1]:txt.shape[1] + img_tokens, :]

        return self.final_layer(img_out, vec)

    def predict(
        self,
        img: mx.array,
        txt: mx.array,
        timestep: float,
        guidance: float | None = None,
        rope: mx.array | None = None,
    ) -> mx.array:
        """Convenience method for one denoising step."""
        t = mx.array([timestep], dtype=mx.float32)
        g = mx.array([guidance], dtype=mx.float32) if guidance is not None else None
        return self(img, txt, t, guidance=g, rope=rope)


# ── Loader ────────────────────────────────────────────────────────────

def load_flux2_transformer(
    path: str | Path,
    dtype: str = "float16",
) -> Flux2Transformer:
    """Load a Flux2/Klein checkpoint into a Flux2Transformer.

    Follows the project's standard 6-step recipe (see native/krea2,
    native/sdxl, native/zimage): load raw safetensors -> normalize keys ->
    map to native naming -> build config+model -> match checkpoint keys
    against flattened model params BY STRING (never tuple — the Session 11
    bug) -> tree_unflatten -> model.update() -> mx.eval().

    Config is DETECTED from the checkpoint (`detect_flux2_config`), not a
    fixed default — this project has two real Flux2 checkpoints (Klein 9B
    and the larger Flux2-D) with different hidden_size/depth/guidance_embed.
    """
    from mlx.utils import tree_flatten, tree_unflatten
    from .config import detect_flux2_config
    from .weight_map import normalize_flux2_keys, map_flux2_to_native

    path = Path(path)
    from .. import _load_safetensors, _check_weight_match
    state = _load_safetensors(path)

    normalized = normalize_flux2_keys(state)
    normalized = map_flux2_to_native(normalized)

    config = detect_flux2_config(normalized, dtype=dtype)
    model = Flux2Transformer(config)

    model_flat = tree_flatten(model.parameters())
    new_flat = []
    matched = 0
    for flat_key, value in model_flat:
        if flat_key in normalized:
            new_flat.append((flat_key, normalized.pop(flat_key).astype(config.mlx_dtype)))
            matched += 1
        else:
            new_flat.append((flat_key, value))
    new_nested = tree_unflatten(new_flat)
    model.update(new_nested)
    mx.eval(model.parameters())

    print(f"[ASDX] Flux2 transformer: matched {matched}/{len(model_flat)} params from checkpoint")
    _check_weight_match(matched, len(model_flat), "Flux2 transformer", path)
    return model
