# Anima MLX Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Anima checkpoints (circlestone-labs Anima and its finetunes, bf16 and ComfyUI int8-convrot) through the ASDX MLX sampler, producing the same image as ComfyUI's own `UNETLoader` + `KSampler` for the same seed/prompt/steps.

**Architecture:** A new `apple_silicon_nodes/native/anima/` package ports ComfyUI's `comfy/ldm/anima/model.py` (LLM adapter) and `comfy/ldm/cosmos/predict2.py::MiniTrainDIT` (Cosmos-Predict2 DiT, image-only subset) to `mlx.core`/`mlx.nn`. The Qwen3-0.6B text encoder stays ComfyUI's own `AnimaTEModel` via `ASDX_CLIPLoader` (PyTorch-MPS, the canon's text-encoder exception); its conditioning dict already carries `t5xxl_ids`/`t5xxl_weights`, which the MLX LLM adapter consumes once per generation. VAE stays ComfyUI's `comfy.sd.VAE` (Wan21 latent format, 16 ch, 8x).

**Tech Stack:** Python 3.13, MLX (`mlx.core`, `mlx.nn`, `mx.fast`), ComfyUI V3 node API, pytest via `uv run pytest`.

**Spec:** this plan (user request 2026-09-26: "portage complet d'Anima en MLX/Metal/Apple Silicon ; regarder comment ComfyUI fait l'implementation, ainsi que SceneWorks, mlx-gen, mlx-rs et inference"). Research findings are recorded in "Reference facts" below.

## Global Constraints

- Ground truth for behavior: ComfyUI source at `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI` (`comfy/ldm/anima/model.py`, `comfy/ldm/cosmos/predict2.py`, `comfy/ldm/cosmos/position_embedding.py`, `comfy/model_detection.py:855-900`, `comfy/supported_models.py::Anima`, `comfy/model_base.py::Anima`). Second reference: `/Volumes/X10Pro/Images/Projet/inference/crates/media/mlx-gen/mlx-gen-anima/src` (newest copy, 2026-09-15; the `mlx-gen/` copy is older). Canon: "ComfyUI and the SceneWorks stack are the reference implementations".
- Where the two references disagree, ComfyUI wins (the user's workflows run ComfyUI). Known disagreement: q/k RMSNorm eps is `1e-6` in ComfyUI, `1e-5` in mlx-gen (diffusers). Use `1e-6`.
- Port to MLX, don't transcribe (canon "Porting from a reference means converting it to Apple Silicon"): `mx.fast.scaled_dot_product_attention`, `mx.fast.layer_norm`, `nn.RMSNorm`, no torch on the hot path, `mx.eval()` once per step.
- Every checkpoint load goes through `native/__init__.py::_load_safetensors` (CLAUDE.md rule 6) so int8-convrot is dequantized by the existing, already-verified `_dequantize_comfy_quant_int8` (marker `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256,"per_row":true}`, scale shape `[out,1]`, verified 2026-09-26 on `DaSiWa-ANIMA-ObsidianArchives-v2_int8_row-wise_convrot_runtime.safetensors`).
- Weight loading is strict: unexpected key, missing key, or shape mismatch raises. Never a partial-match "matched N/M" pass.
- No MLX VAE (CLAUDE.md rule 7). Decode goes through `ASDX_VAEDecode` -> `comfy.sd.VAE`.
- Python via `uv run`, never bare `python`/`pip`. Tests: `uv run pytest`.
- Stage explicit paths only; never `git add docs` (`docs/architecture/` stays untracked). Do not commit unless the user asks (Claudoscope rule); each "Commit" step below means "stage and propose the commit".
- Update `README.md` when the family ships (user feedback memory).

## Reference facts (from inspection, 2026-09-26)

- Checkpoint prefix: `model.diffusion_model.` (all local finetunes). mlx-gen notes the official base cut uses `net.`. Strip either.
- Bf16 checkpoint = 685 tensors = 567 DiT + 118 `llm_adapter`. Int8 file = 1581 tensors (448 `.weight_scale` F32 + 448 `.comfy_quant` U8 on top).
- DiT: `model_channels=2048`, 16 heads x 128, 28 blocks, `mlp_ratio=4` (GPT2FeedForward, no bias, exact GELU), `crossattn_emb_channels=1024`, `use_adaln_lora=True` with dim 256, `patch_spatial=2`, `patch_temporal=1`, `concat_padding_mask=True` (17-ch input -> `x_embedder.proj.1.weight` is `[2048, 68]`), `out_channels=16`, RoPE3D with `h/w extrapolation 4.0`, `t 1.0`, `extra_per_block_abs_pos_emb=False`.
- Detection in ComfyUI: `in_channels = x_embedder.proj.1.weight.shape[1] // 4 - 1`; `num_heads = 16 if model_channels == 2048 else 40 if 5120`.
- LLM adapter (hard-coded defaults): `Embedding(32128, 1024)`, 6 blocks each `[RMSNorm -> self-attn(RoPE) -> RMSNorm -> cross-attn into Qwen3 states (RoPE on q with target positions, on k with source positions) -> RMSNorm -> Linear(1024,4096,bias) GELU Linear(4096,1024,bias)]`, 16 heads x 64, RoPE theta 10000, then `norm(out_proj(x))`. Output multiplied by `t5xxl_weights` then right-padded with zeros to 512 tokens if shorter.
- Sampling: `ModelType.FLOW`, `shift=3.0`, `multiplier=1.0` => timestep fed to the DiT is sigma itself; `denoised = x - out * sigma`; schedule `time_snr_shift(3.0, t)` (same shape as Z-Image's). Base/aesthetic use CFG (~4.5); turbo runs at CFG 1.0.
- Latent format: `latent_formats.Wan21` (16 ch, per-channel mean/std in `native/config.py::WAN21_LATENTS_MEAN/STD`, helpers `process_wan21_latent_in/out`).
- Two layout gotchas: patchify is `b c (h m) (w n) -> b (h w) (c m n)` (channel-major); unpatchify is `B (H W) (p1 p2 C) -> B C (H p1) (W p2)` (channel-minor). They are NOT inverses of each other's axis order.
- DiT RoPE is applied in float32 (`comfy_kitchen.eager.rms_rope_split_half`: rms_norm then split-half rotation with `[[cos,-sin],[sin,cos]]`); the adapter RoPE casts cos/sin to the activation dtype first (`RotaryEmbedding.forward`). Only DiT self-attn gets RoPE; DiT cross-attn gets q/k norm only.
- fp16 compute: ComfyUI keeps the DiT residual stream in float32 when compute dtype is float16 (`predict2.py:907-912`), casts back to the context dtype before `final_layer`.
- Timestep embedding = `native/common.py::timestep_embedding(t, 2048, time_factor=1.0)` ([cos, sin], exponent / half_dim), cast to compute dtype. With adaLN-LoRA, `emb = t_embedding_norm(sincos)` and `adaln_lora = linear_2(silu(linear_1(sincos)))` (`[B, 3*2048]`). Each modulation: `(Linear2(Linear1(silu(emb))) + adaln_lora).split(3)` in order shift, scale, gate; final layer uses `adaln_lora[..., :2*D]` and order shift, scale.

## Review Focus

- Odd latent grid (width/height not a multiple of 16 px): expected a clear `require_divisible_dims` error before any compute, not a reshape crash. Pinned in Task 6.
- Base checkpoint with no negative conditioning while `guidance > 1`: expected a clear error telling the user to add `ASDX_ConditioningMerger`, not a silent CFG-free render. Pinned in Task 6.
- Conditioning from the wrong text encoder (no `t5xxl_ids` in the dict, or hidden size != 1024): expected an error naming `ASDX_CLIPLoader` + the Anima Qwen3-0.6B file, not a shape crash inside the adapter. Pinned in Task 5.
- Prompt longer than 512 T5 tokens: adapter output is NOT truncated or padded (ComfyUI pads only when shorter). Pinned in Task 2.
- A Wan "animate" checkpoint whose filename contains "anima": the filename hint must be overridden by key detection, not routed to the Anima loader. Pinned in Task 6.

---

### Task 1: `AnimaConfig` + RoPE tables

**Files:**
- Create: `apple_silicon_nodes/native/anima/__init__.py`
- Create: `apple_silicon_nodes/native/anima/config.py`
- Create: `apple_silicon_nodes/native/anima/rope.py`
- Create: `tests/support/anima_module_loader.py`
- Test: `tests/native/anima/__init__.py` (empty), `tests/native/anima/test_config.py`, `tests/native/anima/test_rope.py`

**Interfaces:**
- Produces: `AnimaConfig` (frozen dataclass, fields below, properties `head_dim`, `mlx_dtype`); `detect_anima_config(state: dict[str, mx.array], dtype: str) -> AnimaConfig`; `cosmos_rope_3d(head_dim, t, h, w, h_ratio, w_ratio, t_ratio) -> tuple[mx.array, mx.array]` (cos, sin, each `[t*h*w, head_dim//2]` float32); `adapter_rope(length, head_dim, theta=10000.0) -> tuple[mx.array, mx.array]` (each `[length, head_dim//2]` float32); `apply_rope_split_half(x, cos, sin) -> mx.array` (x `[B, H, L, D]`).

- [ ] **Step 1: Create the test loader** (same trick as `tests/support/qwen_image21_module_loader.py`, which bypasses `apple_silicon_nodes/__init__.py` and its `comfy_api` import)

```python
"""Load apple_silicon_nodes/native/anima/*.py standalone, bypassing
apple_silicon_nodes/__init__.py (which imports comfy_api)."""
from __future__ import annotations

from support.qwen_image21_module_loader import load_native_module

__all__ = ["load_native_module"]
```

- [ ] **Step 2: Write the failing tests**

`tests/native/anima/test_config.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.anima_module_loader import load_native_module

config_mod = load_native_module("anima.config")
AnimaConfig = config_mod.AnimaConfig
detect_anima_config = config_mod.detect_anima_config


def _state(model_channels=2048, in_ch_plus_mask=17, blocks=28):
    sd = {"x_embedder.proj.1.weight": mx.zeros((model_channels, in_ch_plus_mask * 4))}
    for i in range(blocks):
        sd[f"blocks.{i}.self_attn.q_proj.weight"] = mx.zeros((1, 1))
    return sd


def test_defaults_match_real_checkpoint():
    cfg = AnimaConfig()
    assert (cfg.model_channels, cfg.num_blocks, cfg.num_heads, cfg.head_dim) == (2048, 28, 16, 128)
    assert cfg.in_channels == 16 and cfg.out_channels == 16
    assert cfg.adaln_lora_dim == 256 and cfg.crossattn_emb_channels == 1024
    assert cfg.rope_ratios == (1.0, 4.0, 4.0)
    assert cfg.min_context_len == 512


def test_detect_reads_geometry():
    cfg = detect_anima_config(_state(), dtype="bfloat16")
    assert cfg.model_channels == 2048 and cfg.num_blocks == 28 and cfg.num_heads == 16
    assert cfg.mlx_dtype == mx.bfloat16


def test_detect_rejects_video_variant():
    # in_channels=17 (+mask=18) is Cosmos image-to-video, which uses other RoPE ratios.
    with pytest.raises(ValueError, match="in_channels"):
        detect_anima_config(_state(in_ch_plus_mask=18), dtype="bfloat16")


def test_detect_rejects_unknown_width():
    with pytest.raises(ValueError, match="model_channels"):
        detect_anima_config(_state(model_channels=1024), dtype="bfloat16")
```

`tests/native/anima/test_rope.py` (reference formulas copied from `position_embedding.py:80-163` and `anima/model.py:20-38`):

```python
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.anima_module_loader import load_native_module

rope = load_native_module("anima.rope")


def _ref_cosmos(head_dim, T, H, W, hr, wr, tr):
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h
    rs = np.arange(0, dim_h, 2)[: dim_h // 2].astype(np.float64) / dim_h
    rt = np.arange(0, dim_t, 2)[: dim_t // 2].astype(np.float64) / dim_t
    fh = 1.0 / ((10000.0 * hr ** (dim_h / (dim_h - 2))) ** rs)
    fw = 1.0 / ((10000.0 * wr ** (dim_h / (dim_h - 2))) ** rs)
    ft = 1.0 / ((10000.0 * tr ** (dim_t / (dim_t - 2))) ** rt)
    et, eh, ew = np.outer(np.arange(T), ft), np.outer(np.arange(H), fh), np.outer(np.arange(W), fw)
    ang = np.concatenate([
        np.broadcast_to(et[:, None, None, :], (T, H, W, et.shape[1])),
        np.broadcast_to(eh[None, :, None, :], (T, H, W, eh.shape[1])),
        np.broadcast_to(ew[None, None, :, :], (T, H, W, ew.shape[1])),
    ], axis=-1).reshape(T * H * W, -1)
    return np.cos(ang), np.sin(ang)


def test_cosmos_rope_matches_reference():
    cos, sin = rope.cosmos_rope_3d(128, 1, 6, 5, h_ratio=4.0, w_ratio=4.0, t_ratio=1.0)
    rc, rs = _ref_cosmos(128, 1, 6, 5, 4.0, 4.0, 1.0)
    assert cos.shape == (30, 64)
    np.testing.assert_allclose(np.array(cos), rc, atol=1e-5)
    np.testing.assert_allclose(np.array(sin), rs, atol=1e-5)


def test_split_half_rotation_matches_rotate_half():
    x = mx.random.normal((1, 2, 7, 8))
    cos, sin = rope.adapter_rope(7, 8)
    got = np.array(rope.apply_rope_split_half(x, cos, sin))
    xn = np.array(x)
    c = np.concatenate([np.array(cos)] * 2, -1)
    s = np.concatenate([np.array(sin)] * 2, -1)
    rot = np.concatenate([-xn[..., 4:], xn[..., :4]], -1)
    np.testing.assert_allclose(got, xn * c + rot * s, atol=1e-5)


def test_zero_position_is_identity():
    x = mx.random.normal((1, 1, 1, 128))
    cos, sin = rope.cosmos_rope_3d(128, 1, 1, 1, 4.0, 4.0, 1.0)
    np.testing.assert_allclose(np.array(rope.apply_rope_split_half(x, cos, sin)), np.array(x), atol=1e-6)
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/native/anima -q`
Expected: FAIL, `ModuleNotFoundError: apple_silicon_nodes.native.anima.config`.

- [ ] **Step 4: Implement**

`apple_silicon_nodes/native/anima/config.py`:

```python
"""Anima (Cosmos-Predict2 MiniTrainDIT + LLM adapter) config.

Values from `comfy/model_detection.py:855-900` (the `anima` branch) and
`comfy/ldm/anima/model.py::LLMAdapter` defaults, checked against the real
checkpoint headers (2048-wide, 28 blocks, 17-ch patch input)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import mlx.core as mx

from ..common import to_mlx_dtype

_HEADS_BY_WIDTH = {2048: 16, 5120: 40}


@dataclass(frozen=True)
class AnimaConfig:
    dtype: str = "bfloat16"
    in_channels: int = 16
    out_channels: int = 16
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    crossattn_emb_channels: int = 1024
    adaln_lora_dim: int = 256
    patch_spatial: int = 2
    # (t, h, w) RoPE NTK extrapolation ratios for in_channels == 16.
    rope_ratios: tuple[float, float, float] = (1.0, 4.0, 4.0)
    adapter_dim: int = 1024
    adapter_layers: int = 6
    adapter_heads: int = 16
    adapter_vocab: int = 32128
    min_context_len: int = 512

    @property
    def head_dim(self) -> int:
        return self.model_channels // self.num_heads

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return to_mlx_dtype(self.dtype, "Anima")


def detect_anima_config(state: dict[str, mx.array], dtype: str) -> AnimaConfig:
    """`state` keys are already prefix-stripped (`blocks.N...`, `x_embedder...`)."""
    w = state["x_embedder.proj.1.weight"]
    model_channels = int(w.shape[0])
    in_channels = int(w.shape[1]) // 4 - 1
    if in_channels != 16:
        raise ValueError(
            f"ASDX: Anima checkpoint has in_channels={in_channels}; only the 16-channel "
            "text-to-image variant is supported (17 is Cosmos image-to-video)."
        )
    if model_channels not in _HEADS_BY_WIDTH:
        raise ValueError(f"ASDX: unknown Anima model_channels={model_channels}.")
    num_blocks = 1 + max(int(m.group(1)) for k in state if (m := re.match(r"blocks\.(\d+)\.", k)))
    return AnimaConfig(
        dtype=dtype,
        model_channels=model_channels,
        num_blocks=num_blocks,
        num_heads=_HEADS_BY_WIDTH[model_channels],
    )
```

`apple_silicon_nodes/native/anima/rope.py`:

```python
"""RoPE for Anima: Cosmos 3-axis NTK RoPE (DiT self-attn) and the LLM
adapter's 1-D RoPE. Both rotate split halves (x[:D/2], x[D/2:]).

Ported from `comfy/ldm/cosmos/position_embedding.py::VideoRopePosition3DEmb`
and `comfy/ldm/anima/model.py::RotaryEmbedding`."""

from __future__ import annotations

import mlx.core as mx


def cosmos_rope_3d(
    head_dim: int, t: int, h: int, w: int, h_ratio: float, w_ratio: float, t_ratio: float
) -> tuple[mx.array, mx.array]:
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h
    spatial = mx.arange(0, dim_h, 2, dtype=mx.float32)[: dim_h // 2] / dim_h
    temporal = mx.arange(0, dim_t, 2, dtype=mx.float32)[: dim_t // 2] / dim_t
    h_freqs = 1.0 / ((10000.0 * h_ratio ** (dim_h / (dim_h - 2))) ** spatial)
    w_freqs = 1.0 / ((10000.0 * w_ratio ** (dim_h / (dim_h - 2))) ** spatial)
    t_freqs = 1.0 / ((10000.0 * t_ratio ** (dim_t / (dim_t - 2))) ** temporal)
    et = mx.arange(t, dtype=mx.float32)[:, None] * t_freqs[None, :]
    eh = mx.arange(h, dtype=mx.float32)[:, None] * h_freqs[None, :]
    ew = mx.arange(w, dtype=mx.float32)[:, None] * w_freqs[None, :]
    ang = mx.concatenate(
        [
            mx.broadcast_to(et[:, None, None, :], (t, h, w, et.shape[1])),
            mx.broadcast_to(eh[None, :, None, :], (t, h, w, eh.shape[1])),
            mx.broadcast_to(ew[None, None, :, :], (t, h, w, ew.shape[1])),
        ],
        axis=-1,
    ).reshape(t * h * w, -1)
    return mx.cos(ang), mx.sin(ang)


def adapter_rope(length: int, head_dim: int, theta: float = 10000.0) -> tuple[mx.array, mx.array]:
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    ang = mx.arange(length, dtype=mx.float32)[:, None] * inv_freq[None, :]
    return mx.cos(ang), mx.sin(ang)


def apply_rope_split_half(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: [B, H, L, D]; cos/sin: [L, D/2]. Computes in cos.dtype, returns x.dtype
    (the DiT passes float32 tables like comfy_kitchen; the adapter passes tables
    already cast to the activation dtype like RotaryEmbedding.forward)."""
    half = x.shape[-1] // 2
    t = x.astype(cos.dtype)
    x1, x2 = t[..., :half], t[..., half:]
    out = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return out.astype(x.dtype)
```

`apple_silicon_nodes/native/anima/__init__.py` (exports grow in later tasks):

```python
"""Anima native MLX implementation: Cosmos-Predict2 MiniTrainDIT + LLM adapter.

The Qwen3-0.6B text encoder is ComfyUI's own `AnimaTEModel` (via
`ASDX_CLIPLoader`); see `bridge.conditioning_anima_to_mlx`."""

from __future__ import annotations

from .config import AnimaConfig, detect_anima_config

__all__ = ["AnimaConfig", "detect_anima_config"]
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/native/anima -q`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add apple_silicon_nodes/native/anima tests/support/anima_module_loader.py tests/native/anima
git commit -m "feat(anima): config detection and RoPE tables"
```

---

### Task 2: LLM adapter, parity vs ComfyUI's `LLMAdapter`

**Files:**
- Create: `apple_silicon_nodes/native/anima/adapter.py`
- Modify: `tests/support/comfyui_reference_loader.py` (add `load_real_comfy_anima()`)
- Test: `tests/native/anima/test_adapter.py`

**Interfaces:**
- Consumes: `AnimaConfig`, `adapter_rope`, `apply_rope_split_half` (Task 1).
- Produces: `LLMAdapter(cfg: AnimaConfig)` with `__call__(source_hidden: mx.array [B,S,1024], target_ids: mx.array int32 [B,St]) -> mx.array [B,St,1024]`. Parameter tree names equal the checkpoint's `llm_adapter.*` suffixes: `embed.weight`, `blocks.N.{norm_self_attn,norm_cross_attn,norm_mlp}.weight`, `blocks.N.{self_attn,cross_attn}.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight`, `blocks.N.mlp.{0,2}.{weight,bias}`, `out_proj.{weight,bias}`, `norm.weight`.

- [ ] **Step 1: Add the reference loader** (append to `tests/support/comfyui_reference_loader.py`, same pattern as `load_real_comfy_minimax_model`)

```python
def load_real_comfy_anima():
    """Real `comfy.ldm.anima.model` + `comfy.ops`, for Anima parity tests."""
    if not _COMFYUI_ROOT.exists() or not _COMFYUI_VENV_SITE_PACKAGES.exists():
        pytest.skip("ComfyUI install not present on this machine")
    for name in list(sys.modules):
        if name == "comfy" or name.startswith("comfy."):
            del sys.modules[name]
    sys.path.insert(0, str(_COMFYUI_ROOT))
    sys.path.insert(0, str(_COMFYUI_VENV_SITE_PACKAGES))
    try:
        import comfy.ldm.anima.model as anima_model
        import comfy.ops as ops
    except ImportError as e:
        pytest.skip(f"comfy.ldm.anima.model not importable: {e}")
    return anima_model, ops
```

- [ ] **Step 2: Write the failing test** (random weights shared by both sides, float32, CPU torch)

```python
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.anima_module_loader import load_native_module
from support.comfyui_reference_loader import load_real_comfy_anima

adapter_mod = load_native_module("anima.adapter")
config_mod = load_native_module("anima.config")


def _copy_torch_to_mlx(torch_module, mlx_module):
    from mlx.utils import tree_flatten, tree_unflatten
    sd = {k: mx.array(v.detach().float().numpy()) for k, v in torch_module.state_dict().items()}
    names = [k for k, _ in tree_flatten(mlx_module.parameters())]
    assert sorted(names) == sorted(sd), set(names) ^ set(sd)
    mlx_module.update(tree_unflatten([(k, sd[k]) for k in names]))


@pytest.mark.parametrize("target_len", [5, 600])
def test_adapter_matches_comfy(target_len):
    import torch
    anima_model, ops = load_real_comfy_anima()
    torch.manual_seed(0)
    ref = anima_model.LLMAdapter(operations=ops.disable_weight_init, dtype=torch.float32)
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.02)
    ours = adapter_mod.LLMAdapter(config_mod.AnimaConfig(dtype="float32"))
    _copy_torch_to_mlx(ref, ours)

    src = torch.randn(1, 9, 1024)
    ids = torch.randint(0, 32128, (1, target_len))
    with torch.no_grad():
        want = ref(src, ids).numpy()
    got = np.array(ours(mx.array(src.numpy()), mx.array(ids.numpy().astype(np.int32))))
    assert got.shape == (1, target_len, 1024)  # no truncation for > 512 tokens
    np.testing.assert_allclose(got, want, atol=2e-4, rtol=2e-4)
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/native/anima/test_adapter.py -q`
Expected: FAIL, `ModuleNotFoundError: apple_silicon_nodes.native.anima.adapter`.

- [ ] **Step 4: Implement** `apple_silicon_nodes/native/anima/adapter.py`

```python
"""Anima LLM adapter: T5 token ids (learned query tokens) cross-attend into
Qwen3-0.6B hidden states. Port of `comfy/ldm/anima/model.py::LLMAdapter`
(inference path: no attention masks, `preprocess_text_embeds` passes none)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import AnimaConfig
from .rope import adapter_rope, apply_rope_split_half


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def __call__(self, x, context, rope_q, rope_k):
        b, lq, _ = x.shape
        lk = context.shape[1]
        q = self.q_norm(self.q_proj(x).reshape(b, lq, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(context).reshape(b, lk, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(context).reshape(b, lk, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        # RotaryEmbedding.forward casts cos/sin to the activation dtype before applying.
        q = apply_rope_split_half(q, rope_q[0].astype(x.dtype), rope_q[1].astype(x.dtype))
        k = apply_rope_split_half(k, rope_k[0].astype(x.dtype), rope_k[1].astype(x.dtype))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(b, lq, -1))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm_self_attn = nn.RMSNorm(dim, eps=1e-6)
        self.self_attn = _Attention(dim, heads)
        self.norm_cross_attn = nn.RMSNorm(dim, eps=1e-6)
        self.cross_attn = _Attention(dim, heads)
        self.norm_mlp = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = [nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)]

    def __call__(self, x, context, rope_x, rope_ctx):
        h = self.norm_self_attn(x)
        x = x + self.self_attn(h, h, rope_x, rope_x)
        x = x + self.cross_attn(self.norm_cross_attn(x), context, rope_x, rope_ctx)
        h = self.norm_mlp(x)
        for layer in self.mlp:
            h = layer(h)
        return x + h


class LLMAdapter(nn.Module):
    def __init__(self, cfg: AnimaConfig):
        super().__init__()
        d = cfg.adapter_dim
        self.head_dim = d // cfg.adapter_heads
        self.embed = nn.Embedding(cfg.adapter_vocab, d)
        self.blocks = [_Block(d, cfg.adapter_heads) for _ in range(cfg.adapter_layers)]
        self.out_proj = nn.Linear(d, d)
        self.norm = nn.RMSNorm(d, eps=1e-6)

    def __call__(self, source_hidden: mx.array, target_ids: mx.array) -> mx.array:
        x = self.embed(target_ids).astype(source_hidden.dtype)
        rope_x = adapter_rope(x.shape[1], self.head_dim)
        rope_ctx = adapter_rope(source_hidden.shape[1], self.head_dim)
        for block in self.blocks:
            x = block(x, source_hidden, rope_x, rope_ctx)
        return self.norm(self.out_proj(x))
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/native/anima/test_adapter.py -q`
Expected: 2 passed. If `atol` fails by a small margin only for the 600-token case, report the max abs diff; do not widen tolerance without that number.

- [ ] **Step 6: Commit**

```bash
git add apple_silicon_nodes/native/anima/adapter.py tests/native/anima/test_adapter.py tests/support/comfyui_reference_loader.py
git commit -m "feat(anima): MLX LLM adapter, parity with comfy LLMAdapter"
```

---

### Task 3: Anima DiT, full-forward parity vs ComfyUI's `Anima`

**Files:**
- Create: `apple_silicon_nodes/native/anima/model.py`
- Modify: `apple_silicon_nodes/native/anima/__init__.py` (export `AnimaTransformer`)
- Test: `tests/native/anima/test_model.py`

**Interfaces:**
- Consumes: Task 1 (`AnimaConfig`, `cosmos_rope_3d`, `apply_rope_split_half`), Task 2 (`LLMAdapter`), `native/common.py::timestep_embedding`.
- Produces: `AnimaTransformer(cfg)` with attribute `config: AnimaConfig`, and
  - `encode_context(qwen_hidden: mx.array [B,S,1024], t5_ids: mx.array int32 [B,St], t5_weights: mx.array | None [B,St]) -> mx.array [B, max(St,512), 1024]` (once per generation);
  - `__call__(x: mx.array [B,16,H,W], timestep: mx.array float32 [B], context: mx.array [B,L,1024]) -> mx.array [B,16,H,W]` (velocity, `denoised = x - out * sigma`).
  Parameter tree names equal the prefix-stripped checkpoint keys (`blocks.N.adaln_modulation_self_attn.1.weight`, `t_embedder.1.linear_1.weight`, `x_embedder.proj.1.weight`, `final_layer.adaln_modulation.2.weight`, `llm_adapter.*`, ...).

- [ ] **Step 1: Write the failing test** (tiny width, 2 blocks, same random weights both sides; adapter stays full-size because ComfyUI hard-codes it)

```python
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.anima_module_loader import load_native_module
from support.comfyui_reference_loader import load_real_comfy_anima

model_mod = load_native_module("anima.model")
config_mod = load_native_module("anima.config")

WIDTH, HEADS, BLOCKS = 256, 2, 2


def _ref_model(torch, anima_model, ops):
    return anima_model.Anima(
        max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=WIDTH, num_blocks=BLOCKS, num_heads=HEADS, crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=256, extra_per_block_abs_pos_emb=False,
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
        image_model="anima", operations=ops.disable_weight_init, dtype=torch.float32,
    )


def _ours():
    cfg = config_mod.AnimaConfig(dtype="float32", model_channels=WIDTH, num_blocks=BLOCKS, num_heads=HEADS)
    return model_mod.AnimaTransformer(cfg)


def _copy(ref, ours):
    from mlx.utils import tree_flatten, tree_unflatten
    sd = {k: mx.array(v.detach().float().numpy()) for k, v in ref.state_dict().items()}
    names = [k for k, _ in tree_flatten(ours.parameters())]
    assert sorted(names) == sorted(sd), set(names) ^ set(sd)
    ours.update(tree_unflatten([(k, sd[k]) for k in names]))


@pytest.mark.parametrize("hw", [(8, 6), (12, 10)])
def test_forward_matches_comfy(hw):
    import torch
    anima_model, ops = load_real_comfy_anima()
    torch.manual_seed(0)
    ref = _ref_model(torch, anima_model, ops)
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.02)
    ours = _ours()
    _copy(ref, ours)

    h, w = hw
    x = torch.randn(1, 16, 1, h, w)
    sigma = torch.tensor([0.7])
    qwen = torch.randn(1, 9, 1024)
    ids = torch.randint(0, 32128, (1, 6))
    weights = torch.rand(6) + 0.5
    with torch.no_grad():
        want = ref(x, sigma, qwen, t5xxl_ids=ids, t5xxl_weights=weights.view(1, -1, 1)).numpy()[:, :, 0]

    ctx = ours.encode_context(mx.array(qwen.numpy()), mx.array(ids.numpy().astype(np.int32)),
                              mx.array(weights.numpy()[None]))
    assert ctx.shape == (1, 512, 1024)
    got = np.array(ours(mx.array(x.numpy()[:, :, 0]), mx.array(sigma.numpy()), ctx))
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=5e-4, rtol=5e-4)


def test_rejects_odd_latent_grid():
    ours = _ours()
    with pytest.raises(ValueError, match="even"):
        ours(mx.zeros((1, 16, 7, 6)), mx.array([0.5]), mx.zeros((1, 512, 1024)))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/native/anima/test_model.py -q`
Expected: FAIL, `ModuleNotFoundError: apple_silicon_nodes.native.anima.model`.

- [ ] **Step 3: Implement** `apple_silicon_nodes/native/anima/model.py`

```python
"""Anima DiT: Cosmos-Predict2 `MiniTrainDIT` (image-only: T=1, no extra
per-block pos-emb, no padding-mask resize) + `LLMAdapter`.

Port of `comfy/ldm/cosmos/predict2.py` and `comfy/ldm/anima/model.py`.
List-valued attributes (`t_embedder`, `x_embedder.proj`, `adaln_modulation_*`)
keep the checkpoint's `nn.Sequential` indices so parameter names equal the
prefix-stripped checkpoint keys."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..common import timestep_embedding
from .adapter import LLMAdapter
from .config import AnimaConfig
from .rope import apply_rope_split_half, cosmos_rope_3d

_EPS = 1e-6


def _layer_norm(x: mx.array) -> mx.array:
    return mx.fast.layer_norm(x, None, None, _EPS)


def _run(seq: list, x: mx.array) -> mx.array:
    for layer in seq:
        x = layer(x)
    return x


class _Attention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, query_dim // heads
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.output_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=_EPS)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=_EPS)

    def __call__(self, x, context=None, rope=None):
        ctx = x if context is None else context
        b, lq, _ = x.shape
        lk = ctx.shape[1]
        q = self.q_norm(self.q_proj(x).reshape(b, lq, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(ctx).reshape(b, lk, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(ctx).reshape(b, lk, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        if rope is not None:  # self-attn only (predict2.py:184)
            q = apply_rope_split_half(q, *rope)
            k = apply_rope_split_half(k, *rope)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        return self.output_proj(o.transpose(0, 2, 1, 3).reshape(b, lq, -1))


def _adaln(d: int, lora: int, chunks: int) -> list:
    return [nn.SiLU(), nn.Linear(d, lora, bias=False), nn.Linear(lora, chunks * d, bias=False)]


class _Block(nn.Module):
    def __init__(self, cfg: AnimaConfig):
        super().__init__()
        d = cfg.model_channels
        self.self_attn = _Attention(d, d, cfg.num_heads)
        self.cross_attn = _Attention(d, cfg.crossattn_emb_channels, cfg.num_heads)
        self.mlp = _FeedForward(d, int(d * cfg.mlp_ratio))
        self.adaln_modulation_self_attn = _adaln(d, cfg.adaln_lora_dim, 3)
        self.adaln_modulation_cross_attn = _adaln(d, cfg.adaln_lora_dim, 3)
        self.adaln_modulation_mlp = _adaln(d, cfg.adaln_lora_dim, 3)

    def __call__(self, x, emb, context, rope, adaln_lora):
        res_dtype, compute_dtype = x.dtype, emb.dtype

        def mod(seq):
            return mx.split(_run(seq, emb) + adaln_lora, 3, axis=-1)  # shift, scale, gate

        shift, scale, gate = mod(self.adaln_modulation_self_attn)
        h = (_layer_norm(x) * (1 + scale) + shift).astype(compute_dtype)
        x = x + gate.astype(res_dtype) * self.self_attn(h, None, rope).astype(res_dtype)

        shift, scale, gate = mod(self.adaln_modulation_cross_attn)
        h = (_layer_norm(x) * (1 + scale) + shift).astype(compute_dtype)
        x = x + gate.astype(res_dtype) * self.cross_attn(h, context).astype(res_dtype)

        shift, scale, gate = mod(self.adaln_modulation_mlp)
        h = (_layer_norm(x) * (1 + scale) + shift).astype(compute_dtype)
        return x + gate.astype(res_dtype) * self.mlp(h).astype(res_dtype)


class _FeedForward(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.layer1 = nn.Linear(d, hidden, bias=False)
        self.layer2 = nn.Linear(hidden, d, bias=False)

    def __call__(self, x):
        return self.layer2(nn.gelu(self.layer1(x)))


class _TimestepEmbedding(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear_1 = nn.Linear(d, d, bias=False)
        self.linear_2 = nn.Linear(d, 3 * d, bias=False)

    def __call__(self, x):
        return self.linear_2(nn.silu(self.linear_1(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.proj = [nn.Identity(), nn.Linear(in_dim, d, bias=False)]

    def __call__(self, x):
        return self.proj[1](x)


class _FinalLayer(nn.Module):
    def __init__(self, cfg: AnimaConfig):
        super().__init__()
        d = cfg.model_channels
        self.linear = nn.Linear(d, cfg.patch_spatial ** 2 * cfg.out_channels, bias=False)
        self.adaln_modulation = _adaln(d, cfg.adaln_lora_dim, 2)

    def __call__(self, x, emb, adaln_lora):
        d = x.shape[-1]
        shift, scale = mx.split(_run(self.adaln_modulation, emb) + adaln_lora[..., : 2 * d], 2, axis=-1)
        return self.linear(_layer_norm(x) * (1 + scale) + shift)


class AnimaTransformer(nn.Module):
    def __init__(self, config: AnimaConfig):
        super().__init__()
        self.config = config
        d, p = config.model_channels, config.patch_spatial
        self.t_embedder = [nn.Identity(), _TimestepEmbedding(d)]
        self.t_embedding_norm = nn.RMSNorm(d, eps=_EPS)
        self.x_embedder = _PatchEmbed((config.in_channels + 1) * p * p, d)
        self.blocks = [_Block(config) for _ in range(config.num_blocks)]
        self.final_layer = _FinalLayer(config)
        self.llm_adapter = LLMAdapter(config)

    def encode_context(self, qwen_hidden, t5_ids, t5_weights=None):
        """`Anima.preprocess_text_embeds`: adapter, per-token weights, right-pad to 512."""
        dtype = self.config.mlx_dtype
        out = self.llm_adapter(qwen_hidden.astype(dtype), t5_ids)
        if t5_weights is not None:
            out = out * t5_weights[..., None].astype(dtype)
        pad = self.config.min_context_len - out.shape[1]
        if pad > 0:
            out = mx.pad(out, [(0, 0), (0, pad), (0, 0)])
        return out

    def __call__(self, x, timestep, context):
        cfg = self.config
        b, c, h, w = x.shape
        p = cfg.patch_spatial
        if h % p or w % p:
            raise ValueError(f"ASDX: Anima latent grid {h}x{w} must be even (patch {p}); use a width/height multiple of 16.")
        dtype = cfg.mlx_dtype
        gh, gw = h // p, w // p

        # concat_padding_mask (all zeros) -> patchify "b c (h m) (w n) -> b (h w) (c m n)".
        x = mx.concatenate([x.astype(dtype), mx.zeros((b, 1, h, w), dtype=dtype)], axis=1)
        x = x.reshape(b, c + 1, gh, p, gw, p).transpose(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, (c + 1) * p * p)
        x = self.x_embedder(x)

        rope = cosmos_rope_3d(cfg.head_dim, 1, gh, gw,
                              h_ratio=cfg.rope_ratios[1], w_ratio=cfg.rope_ratios[2], t_ratio=cfg.rope_ratios[0])

        sincos = timestep_embedding(timestep, cfg.model_channels, time_factor=1.0).astype(dtype)[:, None, :]
        adaln_lora = self.t_embedder[1](sincos)
        emb = self.t_embedding_norm(sincos)

        context = context.astype(dtype)
        if dtype == mx.float16:  # predict2.py:907-912: fp32 residual stream under fp16 compute
            x = x.astype(mx.float32)
        for block in self.blocks:
            x = block(x, emb, context, rope, adaln_lora)

        x = self.final_layer(x.astype(context.dtype), emb, adaln_lora)
        # unpatchify "B (H W) (p1 p2 C) -> B C (H p1) (W p2)" -- channel-minor, unlike patchify.
        x = x.reshape(b, gh, gw, p, p, cfg.out_channels).transpose(0, 5, 1, 3, 2, 4)
        return x.reshape(b, cfg.out_channels, h, w)
```

Add to `__init__.py`: `from .model import AnimaTransformer` and `"AnimaTransformer"` in `__all__`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/native/anima -q`
Expected: all pass. If parity fails, bisect by comparing intermediate outputs in this order before changing code: patch-embed output, `sincos`/`adaln_lora`, block 0 output, final layer. Report the first divergent stage and its max abs diff.

- [ ] **Step 5: Commit**

```bash
git add apple_silicon_nodes/native/anima/model.py apple_silicon_nodes/native/anima/__init__.py tests/native/anima/test_model.py
git commit -m "feat(anima): MLX Cosmos-Predict2 DiT, parity with comfy Anima"
```

---

### Task 4: Strict checkpoint loader on real files (bf16 + int8 convrot)

**Files:**
- Create: `apple_silicon_nodes/native/anima/weight_map.py`
- Modify: `apple_silicon_nodes/native/anima/__init__.py` (export `load_anima_checkpoint`)
- Test: `tests/native/anima/test_weight_map.py`

**Interfaces:**
- Consumes: `native/__init__.py::_load_safetensors(path) -> dict[str, mx.array]` (float32 for dequantized/upcast weights), `native/common.py::_check_weight_match`, Tasks 1-3.
- Produces: `load_anima_checkpoint(path: str | Path, dtype: str = "bfloat16") -> AnimaTransformer`; `strip_anima_prefix(key: str) -> str`.

- [ ] **Step 1: Write the failing test** (real files; skip if absent). `_load_safetensors` lives in `native/__init__.py`, which is importable standalone (it does not import `comfy_api`); if the module loader can't import it, load it via `load_native_module("__init__")`-equivalent used by `tests/native/krea2` tests and note it in the report.

```python
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.anima_module_loader import load_native_module

wm = load_native_module("anima.weight_map")

_DIR = Path("/Volumes/X10Pro/Images/models/diffusion_models/Anima")
_BF16 = _DIR / "anime/WaiHassakuAnima.safetensors"
_INT8 = _DIR / "anima/DaSiWa-ANIMA-ObsidianArchives-v2_int8_row-wise_convrot_runtime.safetensors"


def test_strip_prefix():
    assert wm.strip_anima_prefix("model.diffusion_model.blocks.0.mlp.layer1.weight") == "blocks.0.mlp.layer1.weight"
    assert wm.strip_anima_prefix("net.llm_adapter.embed.weight") == "llm_adapter.embed.weight"


@pytest.mark.parametrize("path", [_BF16, _INT8], ids=["bf16", "int8_convrot"])
def test_real_checkpoint_loads_strictly(path, capsys):
    if not path.exists():
        pytest.skip(f"no local {path.name}")
    model = wm.load_anima_checkpoint(path, dtype="bfloat16")
    assert "matched 685/685" in capsys.readouterr().out
    assert model.config.num_blocks == 28
    # Sanity: loaded weights are not random init (canon verify-checkpoint step 4).
    w = np.array(model.blocks[0].self_attn.q_proj.weight.astype(mx.float32))
    assert np.isfinite(w).all() and 1e-4 < w.std() < 1.0


def test_int8_dequant_close_to_bf16_sibling_shape():
    if not _INT8.exists():
        pytest.skip("no int8 file")
    model = wm.load_anima_checkpoint(_INT8, dtype="bfloat16")
    out = model(mx.random.normal((1, 16, 8, 8)), mx.array([0.5]), mx.random.normal((1, 512, 1024)))
    mx.eval(out)
    assert out.shape == (1, 16, 8, 8) and np.isfinite(np.array(out.astype(mx.float32))).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/native/anima/test_weight_map.py -q`
Expected: FAIL, `ModuleNotFoundError: ...anima.weight_map`.

- [ ] **Step 3: Implement** `apple_silicon_nodes/native/anima/weight_map.py`

```python
"""Strict Anima checkpoint loading. Goes through `_load_safetensors` so the
int8-convrot / fp8-scaled dequantization covers Anima like every family."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from ..common import _check_weight_match
from .config import detect_anima_config
from .model import AnimaTransformer

_PREFIXES = ("model.diffusion_model.", "net.")


def strip_anima_prefix(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_anima_checkpoint(path: str | Path, dtype: str = "bfloat16") -> AnimaTransformer:
    from .. import _load_safetensors

    path = Path(path)
    state = {strip_anima_prefix(k): v for k, v in _load_safetensors(path).items()}
    config = detect_anima_config(state, dtype=dtype)
    model = AnimaTransformer(config)

    expected = dict(tree_flatten(model.parameters()))
    unexpected = sorted(set(state) - set(expected))
    missing = sorted(set(expected) - set(state))
    if unexpected or missing:
        raise ValueError(
            f"ASDX: Anima checkpoint '{path.name}' does not match the architecture: "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}), {len(missing)} missing (e.g. {missing[:3]})."
        )
    weights = []
    for key, param in expected.items():
        tensor = state.pop(key)
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(f"ASDX: Anima key {key!r} has shape {tuple(tensor.shape)}, expected {tuple(param.shape)}.")
        weights.append((key, tensor.astype(config.mlx_dtype)))

    print(f"[ASDX] Anima DiT: matched {len(weights)}/{len(expected)} params from checkpoint")
    _check_weight_match(len(weights), len(expected), "Anima DiT", path)
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    mx.clear_cache()
    return model
```

Add the export to `__init__.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/native/anima -q`
Expected: all pass, the two real-file tests print `matched 685/685`.

- [ ] **Step 5: Run the project's checkpoint recipe**

Invoke the `verify-checkpoint` skill on `native/anima/` with `_BF16`, then dispatch the `weight-map-reviewer` agent on `native/anima/weight_map.py`. Record both verdicts in the task report.

- [ ] **Step 6: Commit**

```bash
git add apple_silicon_nodes/native/anima/weight_map.py apple_silicon_nodes/native/anima/__init__.py tests/native/anima/test_weight_map.py
git commit -m "feat(anima): strict checkpoint loader (bf16 + int8 convrot)"
```

---

### Task 5: Bridge (conditioning, noise, latent out) + Empty Latent format

**Files:**
- Modify: `apple_silicon_nodes/bridge.py` (add three functions next to the `qwen_image21` ones, around `bridge.py:458`, `:542`, `:764`)
- Modify: `apple_silicon_nodes/latent.py:55` (add `"anima": (16, 8)` to `_LATENT_FORMATS`)
- Test: `tests/test_bridge_anima.py`

**Interfaces:**
- Consumes: `native/config.py::process_wan21_latent_out`, `bridge._to_numpy`.
- Produces:
  - `conditioning_anima_to_mlx(conditioning, precision) -> tuple[mx.array, mx.array, mx.array]` = (qwen_hidden `[1,S,1024]` in `precision`, t5_ids `[1,St]` int32, t5_weights `[1,St]` float32);
  - `prepare_noise_from_latent_anima(latent: dict, seed: int, precision) -> tuple[mx.array, int, int, tuple[int,int]]` (noise `[B,16,H/8,W/8]`, height, width, (latent_h, latent_w));
  - `mlx_to_comfy_latent_anima(latents: mx.array, template: dict) -> dict` (applies `process_wan21_latent_out`, returns torch `samples` `[B,16,h,w]`).

- [ ] **Step 1: Inspect the real conditioning first** (CLAUDE.md "inspect the artifact"). In ComfyUI, run one `ASDX_CLIPLoader` (Anima TE file, the `type` the user's workflow uses) -> `ASDX_CLIPTextEncode`, and log `type(cond[0][0]), cond[0][0].shape, sorted(cond[0][1].keys()), cond[0][1]["t5xxl_ids"].shape, dtype`. Expected from source (`comfy/text_encoders/anima.py:48-52`, `comfy/sd.py:346`): hidden `[1,S,1024]`, dict with `t5xxl_ids` (1-D int) and `t5xxl_weights` (1-D float). If the real dict differs, stop and report before writing code.

- [ ] **Step 2: Write the failing test** (uses the comfy stub like `tests/test_bridge_qwen_image21.py`; copy its import header verbatim)

```python
import mlx.core as mx
import numpy as np
import pytest
import torch

# (import header copied from tests/test_bridge_qwen_image21.py: install_comfy_stubs(); bridge = load_node_module("bridge"))


def _cond(hidden=1024, with_ids=True):
    extra = {"pooled_output": None}
    if with_ids:
        extra["t5xxl_ids"] = torch.tensor([10, 20, 1], dtype=torch.int)
        extra["t5xxl_weights"] = torch.tensor([1.0, 1.2, 1.0])
    return [[torch.randn(1, 7, hidden), extra]]


def test_conditioning_extracts_all_three():
    h, ids, w = bridge.conditioning_anima_to_mlx(_cond(), mx.bfloat16)
    assert h.shape == (1, 7, 1024) and h.dtype == mx.bfloat16
    assert ids.shape == (1, 3) and ids.dtype == mx.int32
    np.testing.assert_allclose(np.array(w), [[1.0, 1.2, 1.0]], rtol=1e-6)


def test_conditioning_without_t5_ids_names_the_loader():
    with pytest.raises(RuntimeError, match="ASDX_CLIPLoader"):
        bridge.conditioning_anima_to_mlx(_cond(with_ids=False), mx.bfloat16)


def test_conditioning_wrong_width_names_the_loader():
    with pytest.raises(RuntimeError, match="ASDX_CLIPLoader"):
        bridge.conditioning_anima_to_mlx(_cond(hidden=4096), mx.bfloat16)


def test_latent_out_dewhitens_wan21():
    z = mx.zeros((1, 16, 4, 4))
    out = bridge.mlx_to_comfy_latent_anima(z, {"samples": z})["samples"]
    assert tuple(out.shape) == (1, 16, 4, 4)
    assert abs(float(out[0, 0, 0, 0]) - (-0.7571)) < 1e-4  # WAN21_LATENTS_MEAN[0]


def test_noise_rejects_non_16_channel_latent():
    with pytest.raises(RuntimeError, match="16-channel"):
        bridge.prepare_noise_from_latent_anima({"samples": torch.zeros(1, 64, 8, 8)}, 0, mx.bfloat16)
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_bridge_anima.py -q`
Expected: FAIL, `AttributeError: ... conditioning_anima_to_mlx`.

- [ ] **Step 4: Implement** (in `bridge.py`)

```python
ANIMA_LATENT_CHANNELS = 16
ANIMA_VAE_DOWNSCALE = 8


def conditioning_anima_to_mlx(conditioning: Any, precision: mx.Dtype) -> tuple[mx.array, mx.array, mx.array]:
    """Anima's text path: ComfyUI's `AnimaTEModel` (Qwen3-0.6B, via
    `ASDX_CLIPLoader`) gives `[1,S,1024]` hidden states plus `t5xxl_ids`/
    `t5xxl_weights` in the cond dict (`comfy/text_encoders/anima.py`). The MLX
    LLM adapter consumes all three (`AnimaTransformer.encode_context`)."""
    if isinstance(conditioning, dict):
        conditioning = conditioning.get("conditioning", conditioning)
    hidden_np = _to_numpy(conditioning[0][0])
    extra = conditioning[0][1] if len(conditioning[0]) > 1 else {}
    if hidden_np.ndim != 3 or hidden_np.shape[-1] != 1024 or "t5xxl_ids" not in extra:
        raise RuntimeError(
            "ASDX: Anima needs the Qwen3-0.6B Anima text encoder's conditioning ([1,S,1024] + "
            f"t5xxl_ids) from ASDX_CLIPLoader -> ASDX_CLIPTextEncode; got hidden {hidden_np.shape}, "
            f"keys {sorted(extra)}."
        )
    ids = mx.array(_to_numpy(extra["t5xxl_ids"]).astype(np.int32)[None])
    weights_src = extra.get("t5xxl_weights")
    weights = (mx.ones(ids.shape, dtype=mx.float32) if weights_src is None
               else mx.array(_to_numpy(weights_src).astype(np.float32)[None]))
    hidden = mx.array(hidden_np).astype(precision)
    mx.eval(hidden, ids, weights)
    return hidden, ids, weights


def mlx_to_comfy_latent_anima(latents: mx.array, template: dict[str, Any]) -> dict[str, Any]:
    """[B,16,h,w] model-space latent -> true VAE space (`latent_formats.Wan21.process_out`,
    applied once like `comfy/samplers.py`'s `process_latent_out`)."""
    from .native.config import process_wan21_latent_out

    samples = process_wan21_latent_out(latents.astype(mx.float32))
    out = dict(template)
    out["samples"] = torch.from_numpy(np.array(samples))
    return out


def prepare_noise_from_latent_anima(
    latent: dict[str, Any], seed: int, precision: mx.Dtype
) -> tuple[mx.array, int, int, tuple[int, int]]:
    """Unit-gaussian noise for Anima, NCHW (the DiT patchifies internally)."""
    if "samples" not in latent:
        raise RuntimeError("ASDX: latent must be a Comfy LATENT with 'samples'.")
    samples = latent["samples"]
    if samples.ndim != 4 or tuple(samples.shape)[1] != ANIMA_LATENT_CHANNELS:
        raise RuntimeError(
            f"ASDX: needs {ANIMA_LATENT_CHANNELS}-channel Anima latent (ASDX Empty Latent, "
            f"latent_format='anima'), got {tuple(samples.shape)}"
        )
    import comfy.sample
    noise = comfy.sample.prepare_noise(samples, int(seed), latent.get("batch_index"))
    noise_np = _to_numpy(noise)
    _, _, latent_h, latent_w = noise_np.shape
    noise_mlx = mx.array(noise_np).astype(precision)
    mx.eval(noise_mlx)
    return noise_mlx, latent_h * ANIMA_VAE_DOWNSCALE, latent_w * ANIMA_VAE_DOWNSCALE, (latent_h, latent_w)
```

Check how `mlx_to_comfy_latent_qwen_image21` builds its torch tensor and use the same conversion helper if one exists (do not add a second conversion idiom). In `latent.py`, add `"anima": (16, 8),` with a comment `# Wan21 16ch/8x VAE latent; the DiT patchifies 2x2 internally.`

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_bridge_anima.py tests/test_bridge_qwen_image21.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add apple_silicon_nodes/bridge.py apple_silicon_nodes/latent.py tests/test_bridge_anima.py
git commit -m "feat(anima): bridge conditioning/noise/latent + anima empty-latent format"
```

---

### Task 6: Sampler loop, schedule, loader dispatch, detection

**Files:**
- Modify: `apple_silicon_nodes/sampler/scheduling.py` (`generate_sigmas` Z-Image branch and `_flow_shift_fn`: add `"anima"`)
- Modify: `apple_silicon_nodes/sampler/core.py` (route `model_type == "anima"` next to the `qwen_image21` route at `core.py:164`; add `_run_anima`)
- Modify: `apple_silicon_nodes/sampler/__init__.py:263` (noise-prep dispatch)
- Modify: `apple_silicon_nodes/loader.py` (replace the Task-0 raise with `return "anima"`, add `_ANIMA_HINTS` to the key-verified hint chain, `_load_transformer_for_type` branch, `_MODEL_TYPE_CAPABILITY["anima"] = "anima_base"`)
- Modify: `apple_silicon_nodes/capability.py` (add `"anima_base"` profile after `qwen_image21_base`)
- Test: `tests/test_loader_anima.py` (rewrite), `tests/test_scheduling_anima.py`, `tests/test_sampler_anima.py`

**Interfaces:**
- Consumes: Tasks 3-5 (`load_anima_checkpoint`, `AnimaTransformer.encode_context/__call__`, bridge functions), `sampler/solvers.py::step`, `capability.require_divisible_dims`.
- Produces: `_SamplerCore._run_anima(steps, seed) -> dict` (Comfy LATENT with `sdmlx_model_type="anima"`).

- [ ] **Step 1: Write the failing tests**

`tests/test_loader_anima.py` (replaces the "raises" test):

```python
# (same header as today's file: install_comfy_stubs(); loader_mod = load_node_module("loader"))

@pytest.mark.parametrize("path", [_BF16_REAL, _INT8_REAL], ids=["bf16", "int8_convrot"])
def test_real_anima_checkpoint_detected(path):
    if not path.exists():
        pytest.skip(f"no local {path.name}")
    assert loader_mod._detect_model_type(path) == "anima"


def test_anima_hint_is_verified_by_keys(tmp_path):
    # A non-Anima file whose name contains "anima" (e.g. a Wan *animate* checkpoint)
    # must fall back to key detection, not route to the Anima loader.
    from safetensors.numpy import save_file
    import numpy as np
    p = tmp_path / "wan_animate_test.safetensors"
    save_file({"double_blocks.0.img_attn.qkv.weight": np.zeros((2, 2), np.float32)}, str(p))
    assert loader_mod._detect_model_type(p) == "dev"


def test_capability_entry():
    from apple_silicon_nodes.capability import CAPABILITY_PROFILES
    assert CAPABILITY_PROFILES[loader_mod._MODEL_TYPE_CAPABILITY["anima"]].latent_channels == 16
```

`tests/test_scheduling_anima.py` (Anima = `ModelSamplingDiscreteFlow(shift=3.0)`; the reference numbers come from running the real ComfyUI object once, recorded here):

```python
# header: load the scheduling module like tests/test_scheduling_qwen_image21.py does

def test_anima_normal_matches_comfy_reference():
    comfy_ms = load_real_comfy_model_sampling_flow(shift=3.0)  # see Step 3 helper
    import comfy.samplers
    want = comfy.samplers.calculate_sigmas(comfy_ms, "normal", 30).tolist()
    got = scheduling.calculate_sigmas("anima", "normal", 30)
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_anima_simple_matches_comfy_reference():
    comfy_ms = load_real_comfy_model_sampling_flow(shift=3.0)
    import comfy.samplers
    want = comfy.samplers.calculate_sigmas(comfy_ms, "simple", 20).tolist()
    np.testing.assert_allclose(scheduling.calculate_sigmas("anima", "simple", 20), want, atol=1e-5)
```

`tests/test_sampler_anima.py` (fake transformer, no weights; checks CFG wiring, guards, and output format):

```python
# header: install_comfy_stubs(); core_mod = load_node_module("sampler.core")

class _FakeAnima:
    """Velocity = 0 for the positive context, 1 for the negative, so CFG is observable."""
    def __init__(self):
        from types import SimpleNamespace
        self.config = SimpleNamespace(mlx_dtype=mx.float32, min_context_len=512)
        self.calls = []

    def encode_context(self, hidden, ids, weights):
        return mx.full((1, 512, 1024), float(hidden[0, 0, 0]))

    def __call__(self, x, t, ctx):
        self.calls.append(float(ctx[0, 0, 0]))
        return mx.zeros_like(x) if float(ctx[0, 0, 0]) > 0 else mx.ones_like(x)


def _cond(sign):
    return [[torch.full((1, 3, 1024), sign), {"t5xxl_ids": torch.tensor([1]), "t5xxl_weights": torch.tensor([1.0])}]]


def _core(guidance, negative=True, width=128, height=128):
    positive = {"conditioning": _cond(1.0)}
    if negative:
        positive["_negative"] = _cond(-1.0)
    # Build _SamplerCore exactly as sampler/__init__.py does; pass only the kwargs it accepts.
    ...  # implementer: copy the constructor call from sampler/__init__.py and fill with these values


def test_cfg_runs_two_passes_per_step():
    core = _core(guidance=4.5)
    core.run(steps=2, seed=0)
    assert len(core.transformer.calls) == 4


def test_cfg_one_is_single_pass_and_needs_no_negative():
    core = _core(guidance=1.0, negative=False)
    core.run(steps=2, seed=0)
    assert len(core.transformer.calls) == 2


def test_cfg_without_negative_raises():
    with pytest.raises(RuntimeError, match="ASDX_ConditioningMerger"):
        _core(guidance=4.5, negative=False).run(steps=1, seed=0)


def test_odd_size_raises_before_compute():
    with pytest.raises(Exception, match="16"):
        _core(guidance=1.0, width=120, height=128).run(steps=1, seed=0)
```

The `...` in `_core` is the one place the implementer must read `sampler/__init__.py` to copy the real `_SamplerCore(...)` call (its kwarg list is long and changes); do not invent kwargs.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_loader_anima.py tests/test_scheduling_anima.py tests/test_sampler_anima.py -q`
Expected: FAIL (detection raises; `calculate_sigmas("anima", ...)` falls to FLUX `mu` branch; no `_run_anima`).

- [ ] **Step 3: Implement**

`sampler/scheduling.py`: change `if model_type in ("zimage", "zimage_turbo"):` in both `generate_sigmas` and `_flow_shift_fn` to `if model_type in ("zimage", "zimage_turbo", "anima"):` and extend the comment: `# Anima: ModelSamplingDiscreteFlow shift=3.0, multiplier=1.0 (comfy/supported_models.py::Anima) -- same curve as Z-Image.` Add a test helper `load_real_comfy_model_sampling_flow(shift)` to `tests/support/comfyui_reference_loader.py` that imports real `comfy.model_sampling`, builds `ModelSamplingDiscreteFlow()` and calls `set_parameters(shift=shift, multiplier=1.0)`.

`loader.py`:

```python
_ANIMA_HINTS = ("anima",)
```

In `_detect_model_type`, after the `_FLUX2_HINTS` loop add:

```python
    if hint_type is None:
        for hint in _ANIMA_HINTS:
            if hint in name:
                hint_type = "anima"
                break
```

In `_detect_model_type_from_keys`, replace the Task-0 `raise` block with:

```python
    if any("llm_adapter." in k for k in keys):
        # Anima: Cosmos-Predict2 MiniTrainDIT + LLM adapter (comfy/ldm/anima).
        return "anima"
```

In `_load_transformer_for_type`, before the FLUX `else`:

```python
    elif model_type == "anima":
        from .native.anima import load_anima_checkpoint
        transformer = load_anima_checkpoint(path, dtype=dtype)
        return transformer, transformer.config
```

`_MODEL_TYPE_CAPABILITY["anima"] = "anima_base"`.

`capability.py`:

```python
    # ── Anima (Cosmos-Predict2 2B DiT + LLM adapter, 685/685 weights matched) ──
    "anima_base": CapabilityProfile(
        family="anima",
        name="Anima",
        generate_params={"cfg_scale": "float", "negative": "string", "width": "int", "height": "int", "steps": "int"},
        # Base/aesthetic need true CFG with a negative; turbo runs at cfg 1.0 without one,
        # so the negative is checked in _run_anima against the actual cfg, not required here.
        requires=frozenset(),
        hard_block=frozenset(),
        latent_channels=16,
    ),
```

`sampler/__init__.py`: add before the `else`:

```python
        elif model_type == "anima":
            noise, height, width, output_shape = bridge.prepare_noise_from_latent_anima(
                latent_image, int(seed), config.mlx_dtype
            )
```

`sampler/core.py`: route next to `qwen_image21` (`if model_type == "anima": return self._run_anima(steps, seed)`), and add:

```python
    def _run_anima(self, steps: int, seed: int) -> dict:
        """Anima sampling loop: ModelType.FLOW (shift 3.0, timestep = sigma),
        velocity output (denoised = x - v*sigma), true two-pass CFG on
        `self.guidance` when > 1 (base/aesthetic ~4.5), single pass at 1.0 (turbo).
        The LLM adapter runs once per prompt here, like ComfyUI's
        `Anima.extra_conds` -> `preprocess_text_embeds`, not per step."""
        cap_module.require_divisible_dims(self.width, self.height, 16, "anima")
        precision = self.config.mlx_dtype
        cfg_scale = float(self.guidance) if self.guidance and self.guidance > 0 else 1.0

        ctx_pos = self.transformer.encode_context(*bridge.conditioning_anima_to_mlx(self.positive, precision))
        ctx_neg = None
        if cfg_scale != 1.0:
            negative = self.positive.get("_negative") if isinstance(self.positive, dict) else None
            if negative is None:
                raise RuntimeError(
                    "ASDX: Anima with cfg > 1 needs a negative prompt. Merge the positive and "
                    "negative ASDX_CLIPTextEncode outputs with ASDX_ConditioningMerger, or set cfg to 1.0 (turbo)."
                )
            ctx_neg = self.transformer.encode_context(*bridge.conditioning_anima_to_mlx(negative, precision))
        mx.eval(ctx_pos, ctx_neg) if ctx_neg is not None else mx.eval(ctx_pos)

        def velocity(x_at, sigma_at):
            t = mx.array([sigma_at], dtype=mx.float32)
            v = self.transformer(x_at, t, ctx_pos)
            if ctx_neg is not None:
                v_neg = self.transformer(x_at, t, ctx_neg)
                v = v_neg + cfg_scale * (v - v_neg)
            return v

        sigmas = calculate_sigmas(self.model_type, self.scheduler_name, steps, self.width, self.height)
        solver_state: dict[str, Any] = {}
        mx.reset_peak_memory()
        step_times: list[float] = []
        t_start = time.perf_counter()
        for t in range(steps):
            step_start = time.perf_counter()
            sigma_t = sigmas[t]
            sigma_next = sigmas[t + 1] if t + 1 < len(sigmas) else 0.0
            if self.lora_schedule is not None:
                self.lora_schedule["step"] = t
                self.transformer = self._update_lora_schedule(self.transformer, self.config, self.lora_schedule, t, steps)

            v = velocity(self.noise, sigma_t)
            mx.eval(v)
            denoised = self.noise - v * sigma_t

            def _model_call(x_at, sigma_at):
                out = velocity(x_at, sigma_at)
                mx.eval(out)
                return x_at - out * sigma_at

            self.noise, solver_state = solvers.step(
                self.sampler_name, x=self.noise, sigma=sigma_t, sigma_next=sigma_next,
                denoised=denoised, state=solver_state, seed=seed, step_index=t,
                is_flow_matching=self._is_flow_matching, model_call=_model_call,
            )
            mx.eval(self.noise)
            step_times.append(time.perf_counter() - step_start)
            if (t + 1) % 5 == 0 or t == 0:
                print(f"[ASDX] Anima Step {t + 1}/{steps} - {step_times[-1]:.3f}s")

        if self.low_memory_mode:
            from ..loader import clear_model_cache
            clear_model_cache()
            self.transformer = None

        out_latent = bridge.mlx_to_comfy_latent_anima(self.noise, {"samples": self.noise})
        out_latent["sdmlx_model_type"] = self.model_type
        mem = bridge.collect_mlx_memory()
        avg = sum(step_times) / len(step_times) if step_times else 0
        print(f"[ASDX] Anima Sampling complete: {time.perf_counter() - t_start:.1f}s total, "
              f"{avg:.3f}s/step, {mem['peak_gb']:.1f}GB peak, cfg={cfg_scale:.1f}")
        if self.memory_shape is not None:
            record_observation(self.memory_shape, mx.get_peak_memory())
        bridge.clear_mlx_cache()
        return out_latent
```

Replace the one-line conditional `mx.eval` with a plain `if` block if the linter flags it. Check whether `self.guidance` is how `_run_sdxl` receives cfg (`core.py:1511`) and use the same attribute; if the sampler node exposes a separate cfg input for SDXL, use that instead and adjust the tests.

- [ ] **Step 4: Run to verify they pass, plus the full suite**

Run: `uv run pytest -q`
Expected: all pass (pre-existing skips unchanged). Report the before/after pass counts.

- [ ] **Step 5: Commit**

```bash
git add apple_silicon_nodes/sampler/scheduling.py apple_silicon_nodes/sampler/core.py apple_silicon_nodes/sampler/__init__.py apple_silicon_nodes/loader.py apple_silicon_nodes/capability.py tests/test_loader_anima.py tests/test_scheduling_anima.py tests/test_sampler_anima.py tests/support/comfyui_reference_loader.py
git commit -m "feat(anima): sampler loop with CFG, schedule, loader routing"
```

---

### Task 7: End-to-end parity against ComfyUI on real weights + docs

**Files:**
- Create: `scripts/anima_parity.py`
- Modify: `README.md` (supported families table / node reference), `apple_silicon_nodes/native/anima/__init__.py` docstring (final status)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: One-step DiT parity on the real checkpoint.** `scripts/anima_parity.py` loads `WaiHassakuAnima.safetensors` into (a) the real ComfyUI `Anima` (via the ComfyUI venv, `comfy.sd.load_diffusion_model`, float32 on CPU) and (b) `load_anima_checkpoint(..., dtype="float32")`, feeds both the same random latent `[1,16,1,64,64]`, sigma 0.8, random Qwen hidden `[1,20,1024]`, ids `[1,12]`, and prints max abs diff and cosine similarity of the velocity. Run with the ComfyUI interpreter: `/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/.venv/bin/python scripts/anima_parity.py` (this script needs both torch-comfy and mlx; if `mlx` is missing in that venv, run the MLX half under `uv run`, save `.npy`, and compare). Expected: cosine > 0.9999 in float32. Sanity-check the harness first: comparing ComfyUI against itself must give diff 0, cosine 1.0.

- [ ] **Step 2: Dispatch the `comfy-reference-diff` agent** on `apple_silicon_nodes/native/anima/` against `comfy/ldm/anima/model.py` + `comfy/ldm/cosmos/predict2.py`. Fix CONFIRMED findings in the owning task's file with a test first.

- [ ] **Step 3: User smoke test in ComfyUI** (the user runs this; report their result verbatim). Workflow: `ASDX_DiffusionLoader` (WaiHassakuAnima, bfloat16) -> `ASDX_CLIPLoader` (Anima Qwen3-0.6B TE) -> two `ASDX_CLIPTextEncode` -> `ASDX_ConditioningMerger` -> `ASDX_EmptyLatent` (1024x1024, `latent_format=anima`) -> `ASDX_MLXSampler` (30 steps, cfg 4.5, euler, normal) -> `ASDX_VAEDecode` (Anima/Qwen VAE). Then the same seed/prompt through ComfyUI `UNETLoader` + `KSampler` (euler/normal, cfg 4.5) for a side-by-side. Repeat once with the int8 DaSiWa file and once with `waiANIMA_v10_TurboAIO` at cfg 1.0, 10 steps.

- [ ] **Step 4: README + canon offer.** Add Anima to README's supported-model list and node notes (latent format `anima`, text encoder via `ASDX_CLIPLoader`, CFG behavior, int8 convrot supported, LoRA not yet supported). Offer the user a canon record: "Anima's LLM adapter runs once per prompt in MLX; the Qwen3 encoder stays in ComfyUI".

- [ ] **Step 5: Commit**

```bash
git add scripts/anima_parity.py README.md apple_silicon_nodes/native/anima/__init__.py
git commit -m "docs(anima): parity script and README"
```

---

## Out of scope (explicit follow-ups)

- Anima LoRA support (`MultiLoraLoader` key mapping for `blocks.N.*` / `llm_adapter.*`): separate plan once base generation is confirmed. mlx-gen's `adapters.rs` lists the 60 adapter targets the turbo LoRA uses.
- img2img / inpaint for Anima (needs `process_wan21_latent_in` on the source latent + noise blend): separate plan.
- ComfyUI's `pad_to_patch_size` for odd latent grids: replaced by a hard `/16` size requirement.
- Block streaming / memory ladder (mlx-gen `block_stream.rs`): the 2B DiT is ~4 GB in bf16 on a 64 GB machine.
