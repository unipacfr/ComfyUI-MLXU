"""
MiniMax H3 audio-video DiT -- MLX port (t2va, fl2va keyframe and ref2va reference conditioning rows).

Ports `comfy/ldm/minimax/model.py` block by block. Scope, per plan §5
Phase 6: reference conditioning (`MiniMaxH3ReferenceToVideo`) is built outside this module
(`minimax_h3_conditioning.py`); no extra guide frames (`MiniMaxH3AddGuide`), no VSA
sparse-attention gate (`gate_compress`), no PDD head bank (`FinalLayer`'s multi-head-per-timestep
variant -- not present on the real checkpoints this project targets, see
`config.py`'s `video_out.weight.shape[0] // out_features == 1` check in the
reference), batch size 1 only. Curve-form adaln only (`adaln_curve_grid` set)
-- the only variant seen on a real checkpoint so far; see config.py's module
docstring.

Weight names match the checkpoint directly (`blocks.N.attn.qkv_proj.weight`,
etc.) so `mlx.utils.tree_flatten`/`tree_unflatten` round-trips against the
real state dict without a translation table, the same convention every other
native/ family here uses.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .condition import AUDIO_COND_TIMESTEP, VISUAL_COND_TIMESTEP, PreparedCondition
from .config import MiniMaxH3Config
from .layout import PackedLayout
from .patchify import pack_audio, patchify_video, unpack_audio, unpatchify_video
from .rope import rms_norm as _rope_rms_norm
from .rope import rms_norm_rope_split_half, rope_cos_sin
from .rope import rope_freqs as _rope_freqs


class RMSNorm(nn.Module):
    """Plain RMSNorm (`x / sqrt(mean(x^2) + eps) * weight`) -- MiniMax H3's
    checkpoint weights center around 1.0 (e.g. a real `norm1.weight` sample:
    `[1.03, 1.19, 1.0, 1.01, 1.07]`), confirming the direct-multiply
    convention `torch.nn.functional.rms_norm` uses, NOT the "(1 + scale)"
    zero-centered convention some other families in this project (Krea2) use
    for a different checkpoint's training setup."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return _rope_rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    """Fused qkv self-attention with per-head RMSNorm and, when `cos`/`sin`
    are given, MiniMax H3's partial split-half RoPE (see `rope.py`).

    `cos`/`sin` are omitted entirely (plain per-head RMSNorm, no rotation)
    for `RefinerBlock`'s text-only pre-refinement -- matching
    `comfy/ldm/minimax/model.py::RefinerBlock.forward`, which calls
    `self.attn(...)` without `rope_freqs` (defaults to `None` there). Every
    main `DiTBlock`, by contrast, always passes rope info -- there is no real
    checkpoint or call path that skips rope for the main stack, so this port
    does not special-case that combination as `RefinerBlock` does not need it.
    """

    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float, rot_dim: int = 0):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.eps = eps
        self.rot_dim = rot_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(
        self, x: mx.array, cos: mx.array | None = None, sin: mx.array | None = None
    ) -> mx.array:
        """`x`: `[S, hidden]` (batch size 1, squeezed -- matches the packed
        single-sequence layout every caller here uses). `cos`/`sin`, when
        given: `[S, rot_dim // 2]`, broadcastable against the head axis."""
        s = x.shape[0]
        qkv = self.qkv_proj(x)
        inner = self.heads * self.head_dim
        q, k, v = qkv[:, :inner], qkv[:, inner : 2 * inner], qkv[:, 2 * inner :]
        q = q.reshape(s, self.heads, self.head_dim)
        k = k.reshape(s, self.heads, self.head_dim)
        v = v.reshape(s, self.heads, self.head_dim)

        if cos is not None:
            cos_b = cos[:, None, :]
            sin_b = sin[:, None, :]
            q = rms_norm_rope_split_half(q, self.q_norm.weight, self.eps, self.rot_dim, cos_b, sin_b)
            k = rms_norm_rope_split_half(k, self.k_norm.weight, self.eps, self.rot_dim, cos_b, sin_b)
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [S, heads, head_dim] -> [1, heads, S, head_dim] for mx.fast.sdpa
        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        scale = 1.0 / (self.head_dim**0.5)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out[0].transpose(1, 0, 2).reshape(s, inner)
        return self.out_proj(out)


class MLP(nn.Module):
    """SwiGLU MLP: `fc1` projects to `2*ffn` (gate + value, first half is the
    gate), `fc2` projects the gated `ffn`-wide result back to `hidden`.
    Matches `comfy.ops.linear_input_act(fc2, fc1(x), "swiglu")` via its
    `_swiglu_eager` implementation (`comfy/ops.py:947`):
    `gate, up = x.chunk(2, dim=-1); return silu(gate) * up`."""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.fc1(x)
        ffn = h.shape[-1] // 2
        gate, up = h[..., :ffn], h[..., ffn:]
        return self.fc2(nn.silu(gate) * up)


class AdalnProj(nn.Module):
    """Projects a per-unique-timestep embedding `t_emb` (`[M, t_dim]`) into
    `expand` modulation tensors, each `[M * modalities, hidden]` -- one row
    per (unique timestep, modality) pair. `modalities` is 3 for `DiTBlock`
    (video=0, text=1, audio=2 -- fixed tags every caller here uses) and 1 for
    `FinalLayer`. `apply_silu` is False for the curve-form adaln this project
    has a real checkpoint for (`curve["apply_silu"] = not use_adaln_curves`
    in `comfy/ldm/minimax/model.py::MiniMaxH3Model.__init__` -- the plain
    `TimeEmbedder` variant would set it True, out of scope here, see
    config.py's module docstring)."""

    def __init__(self, t_dim: int, hidden: int, expand: int, modalities: int, apply_silu: bool = False):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True)

    def __call__(self, t_emb: mx.array) -> tuple[mx.array, ...]:
        x = self.linear(nn.silu(t_emb) if self.apply_silu else t_emb)
        m = x.shape[0]
        x = x.reshape(m * self.modalities, self.expand * self.hidden)
        return tuple(mx.split(x, self.expand, axis=-1))


def mod_scale_shift(h: mx.array, shift: mx.array, scale: mx.array, segments: list[tuple[int, int, int]]) -> mx.array:
    """`h[a:b] = h[a:b] * (1 + scale[row]) + shift[row]` per segment, ported
    from `_mod_scale_shift`. MLX arrays are immutable, so this rebuilds `h`
    from per-segment slices instead of the reference's in-place `.mul_`/
    `.add_` -- same math, different (functional) execution style. `segments`
    covers `h` contiguously and in order (guaranteed by `PackedLayout`), so
    concatenating the rebuilt slices reproduces `h`'s original row order."""
    pieces = []
    for a, b, row in segments:
        pieces.append(h[a:b] * (1.0 + scale[row]) + shift[row])
    return mx.concatenate(pieces, axis=0)


def mod_gate(x: mx.array, gate: mx.array, other: mx.array, segments: list[tuple[int, int, int]]) -> mx.array:
    """`x[a:b] += other[a:b] * gate[row]` per segment, ported from
    `_mod_gate` (same functional-rebuild adaptation as `mod_scale_shift`)."""
    pieces = []
    for a, b, row in segments:
        pieces.append(x[a:b] + other[a:b] * gate[row])
    return mx.concatenate(pieces, axis=0)


class RefinerBlock(nn.Module):
    """Plain (unmodulated) residual attention + MLP block: pre-norm attention
    (no rope -- `Attention` called with `cos=sin=None`, see its docstring),
    pre-norm SwiGLU MLP. Used only by `TokenRefiner`, to refine text
    embeddings before they enter the packed multi-modal sequence."""

    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int, eps: float, qk_eps: float):
        super().__init__()
        self.norm1 = RMSNorm(hidden, eps=eps)
        self.norm2 = RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps)
        self.mlp = MLP(hidden, ffn)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TokenRefiner(nn.Module):
    """Stack of `RefinerBlock`s + a final RMSNorm -- 2 layers on the real
    checkpoint (`config.token_refiner_num_layers`)."""

    def __init__(self, num_layers: int, hidden: int, heads: int, head_dim: int, ffn: int, eps: float, qk_eps: float, final_eps: float):
        super().__init__()
        self.blocks = [RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps) for _ in range(num_layers)]
        self.final_norm = RMSNorm(hidden, eps=final_eps)

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    """One main-stack block: adaLN-modulated pre-norm attention (rope-carrying)
    + adaLN-modulated pre-norm SwiGLU MLP, both gated residual adds.
    `rot_dim` defaults to `head_dim` (attention_head_dim) since MiniMax H3's
    real checkpoint always rotates 96 of 128 head dims -- callers pass the
    config's actual value rather than relying on this default in practice."""

    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int, t_dim: int, eps: float, qk_eps: float, rot_dim: int | None = None):
        super().__init__()
        self.norm1 = RMSNorm(hidden, eps=eps)
        self.norm2 = RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, rot_dim=rot_dim if rot_dim is not None else head_dim)
        self.mlp = MLP(hidden, ffn)
        self.adaln_proj = AdalnProj(t_dim, hidden, expand=6, modalities=3, apply_silu=False)

    def __call__(
        self,
        x: mx.array,
        t_emb: mx.array,
        mod_segments: list[tuple[int, int, int]],
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = mod_gate(x, gate_msa, self.attn(h, cos=cos, sin=sin), mod_segments)
        h = mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


def curve_time_embedding(table: mx.array, t_vals: mx.array) -> mx.array:
    """Curve-form timestep embedding: linearly interpolate `adaln_t_table`
    (`[grid, time_embed_dim]`) at fractional row `t * (grid - 1)` for each
    `t` in `t_vals` (`[M]`, values in `[0, 1]`). Ported from
    `MiniMaxH3Model.forward`'s `use_adaln_curves` branch:

        pos = t.clamp(0, 1) * (grid - 1)
        i0 = pos.floor().clamp(max=grid - 2)   # keeps t=1.0 on the last interval
        t_emb = lerp(table[i0], table[i0 + 1], pos - i0)

    replaces the sinusoidal `TimeEmbedder` for this checkpoint variant (see
    config.py's module docstring) -- there is no separate embedder module to
    call, the table IS the embedder."""
    grid = table.shape[0]
    t_clamped = mx.clip(t_vals, 0.0, 1.0)
    pos = t_clamped * (grid - 1)
    i0 = mx.clip(mx.floor(pos), 0, grid - 2).astype(mx.int32)
    frac = (pos - i0.astype(pos.dtype))[:, None]
    row0 = table[i0]
    row1 = table[i0 + 1]
    return row0 + frac * (row1 - row0)


class FinalLayer(nn.Module):
    """adaLN-modulated final norm + separate video/audio output heads.

    The reference's `FinalLayer` also implements a "PDD head bank" (multiple
    output-head weight sets blended by how far the current denoising step
    spans the timeline) when `video_out.weight.shape[0] // video_out.
    out_features > 1`. Not implemented here: the real checkpoints this
    project targets have exactly one head (`n == 1`, see config.py), so that
    branch is dead code for them -- see module docstring's scope note."""

    def __init__(self, hidden: int, t_dim: int, video_dim: int, audio_dim: int, eps: float):
        super().__init__()
        self.norm = RMSNorm(hidden, eps=eps)
        self.adaln_proj = AdalnProj(t_dim, hidden, expand=2, modalities=1, apply_silu=False)
        self.video_out = nn.Linear(hidden, video_dim, bias=True)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        t_emb: mx.array,
        video_seg: tuple[int, int, int],
        audio_seg: tuple[int, int, int],
    ) -> tuple[mx.array, mx.array]:
        shift, scale = self.adaln_proj(t_emb)

        def mod(seg: tuple[int, int, int]) -> mx.array:
            a, b, row = seg
            return self.norm(x[a:b]) * (1.0 + scale[row]) + shift[row]

        return self.video_out(mod(video_seg)), self.audio_out(mod(audio_seg))


_SEG_TAG = {"video": 0, "text": 1, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}


def build_mod_segments(
    segments: list[tuple[int, int, str]],
    t_video: float,
    t_audio: float,
    *,
    vis_aug: float = 0.999,
    aud_aug: float = 1.0,
    text_tags: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, int]], list[float]]:
    """Assigns each packed-sequence segment a modulation row `t_row[t] * 3 +
    tag` and returns the sorted unique timestep values those rows index into
    (feed to `curve_time_embedding`/`AdalnProj` as `t_emb`'s `M` rows).

    Timestep per kind (`comfy/ldm/minimax/model.py::_forward`): text and
    video follow `t_video`, audio `t_audio`; keyframe/reference video rows
    are pinned at `max(t_video, vis_aug)` and their audio rows at
    `max(t_audio, aud_aug)`. Tags: video 0, text 1, audio 2 (condition
    rows carry their modality's tag). `text_tags` (`[text_len]`, 0 = vision
    row) splits the text segment into runs, vision rows taking tag 0.
    No denoise masks."""
    seg_t = {
        "text": t_video, "video": t_video, "audio": t_audio,
        "cond": max(t_video, vis_aug), "ref_img": max(t_video, vis_aug),
        "cond_audio": max(t_audio, aud_aug), "ref_audio": max(t_audio, aud_aug),
    }
    unique_t = sorted({t_video, t_audio} | {seg_t[k] for _, _, k in segments})
    t_row = {t: i for i, t in enumerate(unique_t)}
    mod_segments: list[tuple[int, int, int]] = []
    for a, b, kind in segments:
        base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = np.asarray(text_tags).reshape(-1).tolist()
            if len(tags) != b - a:
                raise ValueError(f"ASDX: text_token_tags has {len(tags)} entries, the text segment has {b - a} rows")
            run = 0
            for i in range(1, len(tags) + 1):
                if i == len(tags) or tags[i] != tags[run]:
                    mod_segments.append((a + run, a + i, base + int(tags[run])))
                    run = i
        else:
            mod_segments.append((a, b, base + _SEG_TAG[kind]))
    return mod_segments, unique_t


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """Converts a flow-matching sigma from one shift's grid to another's --
    ported verbatim from `comfy/ldm/minimax/model.py::time_shift_sigma`
    (inverts `sigma = s*b/(1+(s-1)*b)` to the base grid, re-applies the other
    shift). Used to derive the audio stream's own sigma from the video sigma
    the sampler drives (`sigma_shift_video=12.0` -> `sigma_shift_audio=3.0`
    on the real checkpoint)."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


class RopeBuffer(nn.Module):
    """Holds `rope.inv_freq` under its own submodule, matching the
    checkpoint's key path exactly (`rope.inv_freq`, not a bare top-level
    `inv_freq`) -- the reference does the same with a bare `nn.Module()`
    plus a registered buffer, for the same key-path reason."""

    def __init__(self, dim: int):
        super().__init__()
        self.inv_freq = mx.zeros(dim)


class MiniMaxH3Model(nn.Module):
    """MiniMax H3 audio-video DiT -- minimal t2va path (see module
    docstring for the full scope restriction list).

    Precondition on `video_latent`'s spatial/temporal dims: already a
    multiple of `patch_size` in every axis. The reference pads via
    `comfy.ldm.common_dit.pad_to_patch_size` before patchifying and crops
    the output back to the original size after -- omitted here because
    every real caller (the VAE's own downscale ratio combined with
    `EmptyMiniMaxH3LatentAV`'s `width`/`height` rounding, see
    `comfy_extras/nodes_minimax_h3.py`) already guarantees alignment, so
    the pad/crop is a no-op in practice; not silently dropped functionality,
    a documented precondition instead.

    `__call__`'s return is negated (`-video_out, -audio_out`), matching the
    reference's OUTER `forward()` (not just the `_forward` this class's
    `__call__` otherwise mirrors) -- that sign flip is part of the model's
    real output contract, not an implementation detail internal to
    `_forward`.
    """

    def __init__(self, config: MiniMaxH3Config):
        super().__init__()
        self.config = config
        hidden = config.hidden_size

        self.video_patch_proj = nn.Linear(config.video_patch_dim, hidden, bias=True)
        self.audio_patch_proj = nn.Linear(config.audio_latents_dim, hidden, bias=True)
        self.condition_proj = nn.Linear(config.text_dim, hidden, bias=True)
        if config.adaln_curve_grid is None:
            raise NotImplementedError(
                "ASDX: MiniMaxH3Model only implements the curve-form adaln variant "
                "(adaln_curve_grid set) -- see config.py's module docstring."
            )
        self.adaln_t_table = mx.zeros((config.adaln_curve_grid, config.time_embed_dim))
        self.rope = RopeBuffer(config.rope_inv_freq_len)
        self.token_refiner = TokenRefiner(
            config.token_refiner_num_layers, hidden, config.num_attention_heads,
            config.attention_head_dim, config.ffn_hidden_size, config.norm_eps,
            config.qk_norm_eps, config.final_norm_eps,
        )
        rot_dim = config.rope_inv_freq_len * 3 * 2  # 3 axes, cos/sin pair each -- see rope.py
        if rot_dim > config.attention_head_dim:
            raise ValueError(
                f"ASDX: MiniMaxH3Config's rope_inv_freq_len={config.rope_inv_freq_len} implies "
                f"rot_dim={rot_dim}, larger than attention_head_dim={config.attention_head_dim} "
                f"-- partial rope requires rot_dim <= head_dim."
            )
        self.blocks = [
            DiTBlock(
                hidden, config.num_attention_heads, config.attention_head_dim,
                config.ffn_hidden_size, config.time_embed_dim, config.norm_eps,
                config.qk_norm_eps, rot_dim=rot_dim,
            )
            for _ in range(config.num_layers)
        ]
        self.final_layer = FinalLayer(
            hidden, config.time_embed_dim, config.video_patch_dim, config.audio_latents_dim, config.final_norm_eps
        )

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        context: mx.array,
        sigma_v: float,
        cond: PreparedCondition | None = None,
    ) -> tuple[mx.array, mx.array]:
        """`video_latent`: `[1, latents_dim, T, H, W]`. `audio_latent`:
        `[1, audio_latents_dim, 2, T_audio]`. `context`: `[L, text_dim]`
        (Qwen3-VL text states, batch already squeezed -- this port's
        convention throughout, see `Attention`/`DiTBlock`). `sigma_v`: the
        video stream's flow-matching sigma in `[0, 1]` (this port's
        interface takes it directly rather than the reference's
        `timestep = sigma * 1000` ComfyUI convention, which is pure
        sampler-plumbing with no effect on the math below).
        `cond`: prepared keyframe/reference rows (`condition.prepare_condition`,
        computed once per sampling run); `None` = plain t2va. Returns the
        NEGATED velocities of the target streams (the reference's outer
        `forward()` sign flip)."""
        cfg = self.config
        shift_v, shift_a = cfg.sigma_shift_video, cfg.sigma_shift_audio
        t_v = 1.0 - sigma_v
        t_a = 1.0 - time_shift_sigma(sigma_v, shift_v, shift_a)

        payload = cond.payload if cond is not None else None
        text_len = context.shape[0]
        latent_t, lat_h, lat_w = video_latent.shape[2], video_latent.shape[3], video_latent.shape[4]
        audio_t = audio_latent.shape[-1]

        layout = PackedLayout(
            text_len, latent_t, lat_h, lat_w, audio_t,
            keyframes=payload.keyframes if payload else (), refs=payload.refs if payload else (),
        )
        mod_segments, unique_t = build_mod_segments(
            layout.segments, t_v, t_a,
            vis_aug=payload.visual_cond_noise_aug if payload else VISUAL_COND_TIMESTEP,
            aud_aug=payload.audio_cond_noise_aug if payload else AUDIO_COND_TIMESTEP,
            text_tags=payload.text_token_tags if payload else None,
        )
        t_emb = curve_time_embedding(self.adaln_t_table, mx.array(unique_t, dtype=mx.float32))

        video_embed = self.video_patch_proj(patchify_video(video_latent, cfg.patch_size))
        audio_embed = self.audio_patch_proj(pack_audio(audio_latent))
        cond_video = self.video_patch_proj(cond.video_rows) if cond is not None and cond.video_rows is not None else None
        cond_audio = self.audio_patch_proj(cond.audio_rows) if cond is not None and cond.audio_rows is not None else None

        text_states = context
        if text_states.shape[-1] != cfg.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states))

        # Assemble in the layout's segment order (condition rows interleaved with the target).
        pieces: list[mx.array] = []
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                pieces.append(text_states)
            elif kind in ("cond", "ref_img"):
                pieces.append(cond_video[voff:voff + n])
                voff += n
            elif kind in ("cond_audio", "ref_audio"):
                pieces.append(cond_audio[aoff:aoff + n])
                aoff += n
            elif kind == "audio":
                pieces.append(audio_embed)
            else:
                pieces.append(video_embed)
        h = mx.concatenate(pieces, axis=0)
        for name, used, rows in (("video", voff, cond.video_rows if cond else None), ("audio", aoff, cond.audio_rows if cond else None)):
            available = 0 if rows is None else rows.shape[0]
            if used != available:
                raise ValueError(
                    f"ASDX: layout consumed {used} {name} condition rows but the prepared condition has {available}"
                )

        angles = _rope_freqs(layout.position_ids, self.rope.inv_freq)
        cos, sin = rope_cos_sin(angles)

        for block in self.blocks:
            h = block(h, t_emb, mod_segments, cos, sin)

        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")
        v, a = self.final_layer(
            h, t_emb, (va, vb, unique_t.index(t_v)), (aa, ab, unique_t.index(t_a))
        )

        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, cfg.latents_dim, cfg.patch_size)
        audio_out = unpack_audio(a)
        return -video_out, -audio_out
