# ASDX — Apple Silicon Diffusion Nodes

Custom ComfyUI nodes optimized for **Apple Silicon** (M1/M2/M3/M4/M5) using **MLX native inference** with zero-copy Unified Memory semantics.

Inspired by [SDMLX](https://github.com/elef4/SDMLX), this project takes the core concepts and reimagines them with a cleaner architecture focused on:

1. **Pure MLX inference** — the diffusion loop runs entirely in MLX, no PyTorch overhead
2. **Strategic memory management** — `mx.eval()` at bridge points, `mx.clear_cache()` between phases
3. **Precision awareness** — float16/bfloat16 native on Apple Silicon
4. **Clean separation** — bridge, native core, and nodes are independently maintainable

## Supported checkpoints

| Family | Latent | Notes |
|--------|--------|-------|
| FLUX.1 (dev / kontext) | 16ch | Kontext KV cache for reference-image conditioning |
| Krea2 | 16ch | Identity Edit (source-token attention), txtfusion enhancer, image-grounded prompt encoding |
| Z-Image (+ turbo) | 16ch | Distinct patch-token axis order from FLUX |
| Flux2 / Klein | 128ch | 16x VAE downscale |
| SDXL (incl. Illustrious/Pony/NoobAI-style) | 4ch | Direct UNet latent grid, no 2x2 patchify |

Model family is auto-detected from the checkpoint filename/keys. Quantized checkpoints
(FP8_SCALED, INT8 ConvRot/tensorwise) are dequantized on load — the format is classified
from the safetensors header marker keys, never guessed from dtype alone.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ComfyUI Pipeline                                  │
└───────────────┬───────────────────────────┬─────────────────────────┘
                │                           │
     ┌──────────▼──────────┐    ┌──────────▼──────────┐
     │   Conditioning      │    │    Latent Input      │
     │   (CLIP/T5)         │    │  (4/16/128-channel)  │
     └──────────┬──────────┘    └──────────┬──────────┘
                │                          │
     ┌──────────▼──────────┐    ┌──────────▼──────────┐
     │  mlx_conditioning   │    │  torch.Tensor (MPS) │
     └──────────┬──────────┘    └──────────┬──────────┘
                │                          │
                └──────────┬───────────────┘
                           │
              ┌────────────▼────────────┐
              │   ASDX_MLXSampler       │
              │   (Pure MLX inference)  │
              │                         │
              │  • FLUX/Krea2/SDXL/     │
              │    Z-Image/Flux2 core   │
              │  • SeaCache / TeaCache  │
              │  • Kontext KV cache     │
              │  • LoRA injection       │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  mx.array (latent)      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   ASDX_VAEDecode        │
              │   (MLX VAE decode)      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  torch.Tensor (IMAGE)   │
              │  [B, H, W, C] [0,1]     │
              └─────────────────────────┘
```

## Nodes

### Loaders
| Node | Description |
|------|-------------|
| `🍏 ASDX Diffusion Loader` | Load a bare diffusion checkpoint (UNet/transformer only) into MLX |
| `🍏 ASDX Checkpoint Loader` | Load a full checkpoint (transformer + CLIP + VAE) into MLX |
| `🍏 ASDX Dual CLIP Loader` | Load CLIP-L + T5-XXL text encoders |
| `🍏 ASDX CLIP Loader` | Load a single CLIP text encoder |
| `🍏 ASDX VAE Loader` | Load a VAE checkpoint |

### Conditioning
| Node | Description |
|------|-------------|
| `🍏 ASDX CLIP Text Encode` | Encode prompts (T5+CLIP for FLUX-family, CLIP-only for SDXL) |
| `🍏 ASDX Conditioning Merger` | Merge conditioning inputs |
| `🍏 ASDX Krea2 Identity Edit` | Krea2-only: apply the Identity Edit LoRA and prepare the source (pixel-space fit + VAE encode + whiten + pack). Emits the model dict; the sampler consumes it in priority over its legacy `source_latent`/`source_image` inputs |
| `🍏 ASDX Krea2 Grounded Encode` | Krea2-only: encode a prompt alongside a source image through the CLIP's vision tower (image-grounded conditioning), matching the krea2_edit LoRA's training-time conditioning. Requires a Krea2 `mlx_clip` |
| `🍏 ASDX Depth Conditioning` | flux1_depth-only: validate and stage a depth map + VAE for depth-control generation. Emits the model dict; the sampler consumes it (`depth_cond`) in priority over its legacy `depth_image`/`depth_strength` inputs |
| `🍏 ASDX Kontext Reference` | Attach a Kontext reference latent + strength to the model dict. The sampler consumes it (`kontext`) in priority over its legacy `kontext`/`kontext_reference_latent`/`kontext_reference_strength` inputs |
| `🍏 ASDX Latent Noise Prep` | Validate and mode-tag img2img/inpaint/fill ingredients (`mode`/`image`/`mask`/`image_strength`/`mask_blur`/`mask_padding`). Emits the model dict; the sampler consumes it (`latent_prep`) in priority over its own same-named inputs (no schema removal — full backward compatibility) |

### Sampling
| Node | Description |
|------|-------------|
| `🍏 ASDX MLX Native Sampler` | Multi-sampler/scheduler MLX sampler with SeaCache/TeaCache |

`sampler_name`: `euler` (default), `euler_a`, `dpmpp_2m`, `dpmpp_2m_sde`,
`dpmpp_2s_ancestral`, `ddim`, `deis`.
`scheduler_name`: `normal` (default), `simple`, `karras`, `sgm_uniform`, `beta`.
Ported from ComfyUI's own sampler/scheduler algorithms (`comfy/k_diffusion/
sampling.py`, `comfy/samplers.py`) and verified numerically against them.
TeaCache/SeaCache require `sampler_name` in `euler`/`ddim` (their step-skip
heuristic assumes a single, stateless model call per step).

Krea2 Identity Edit: the recommended path is the dedicated
`🍏 ASDX Krea2 Identity Edit` node — wire your source **image** + the VAE into
it, and its `model` output into `ASDX_MLXSampler`. It applies the Identity Edit
LoRA and fits the source in **pixel space** (contain + /16 floor + bicubic, or a
minimal center-crop when the AR nearly matches), then VAE-encodes, whitens, and
packs it — the exact geometry the v1_2 LoRA was trained with and what
comfyui-krea2edit's `fit` mode does. The sampler reads the prepared source from
the model dict in priority over its legacy inputs.

The sampler's legacy `source_latent` / `source_image` + `vae` inputs still work
(unchanged) for saved workflows: `source_latent` is fitted onto the target's
latent grid (center-crop + resize), `source_image` + `vae` runs the pixel path.
If both the node and a legacy input are wired, the node wins (a log line says
so). `ref_boost` dials target→source attention (1.0 = off; the reference
workflow ships 4.0 for strong likeness) — set it on the node, or on the sampler
for the legacy path.

### Latent / VAE
| Node | Description |
|------|-------------|
| `🍏 ASDX Empty Latent` | Create an empty latent (`flux`/`flux2`/`sdxl` format) |
| `🍏 ASDX VAE Decode (MLX)` | Decode latents via MLX VAE |
| `🍏 ASDX VAE Encode (MLX)` | Encode images via MLX VAE |

Both VAE nodes retry automatically with a tiled encode/decode when MPS raises one
of the two failures tiling fixes: a plain out-of-memory `RuntimeError`, or
`MPSGraph does not support tensor dims larger than INT_MAX` (the VAE mid-block
attention matrix is `(H*W/64)**2` elements, so it trips past ~1723x1723 px).
comfy's own OOM fallback recognises neither on Apple Silicon.

### LoRA
| Node | Description |
|------|-------------|
| `🍏 ASDX LoRA Loader` | Load single LoRA (A/B, kohya, ComfyUI diff, diffusers/PEFT, or LyCORIS LoKr/LoHa) with strength scaling |
| `🍏 ASDX Multi LoRA Loader` | Stack up to 5 LoRAs simultaneously |
| `🍏 ASDX LoRA Schedule` | Per-step LoRA strength modulation (linear/cosine/ease-in-out) |

### Utilities
| Node | Description |
|------|-------------|
| `🍏 ASDX Memory Profiler` | Real-time MLX/MPS memory stats |
| `🍏 ASDX Cache Manager` | Clear model/CLIP/VAE caches between runs |
| `🍏 ASDX Depth Map` | Estimate a depth map from an image |
| `🍏 ASDX Live Preview` | Stream intermediate latents during sampling |

> ControlNet Union and IP-Adapter nodes exist in `disabled_nodes/` but are not currently
> registered — kept for a future revisit, not deleted.

## MiniMax H3 (Apple Silicon native)

Text-to-video, first/last-frame video and reference-to-video with audio, run on the native
MLX MiniMax H3 DiT and its Qwen3-VL text encoder.

| Node | Description |
|------|-------------|
| `🍏 ASDX MiniMax H3 Empty Latent (AV)` | Empty video + audio latents (length snapped up to the 17k+5 frame grid, 124 = ~5s at 24 fps) |
| `🍏 ASDX MiniMax H3 Sigma Shift` | Override the video/audio sigma shifts on the model dict |
| `🍏 ASDX MiniMax H3 Model Loader` | Load the DiT checkpoint, requantized to MLX 4-bit |
| `🍏 ASDX MiniMax H3 Text Encoder Loader` | Load the Qwen3-VL-32B text encoder. `load_vision` also loads the vision tower (about 2.4GB more); it is required whenever the prompt contains images: first/last frames, reference images and reference videos. Audio-only references do not need it |
| `🍏 ASDX MiniMax H3 Text Encode` | Text-only prompt to conditioning |
| `🍏 ASDX MiniMax H3 Image to Video` | Prompt plus optional `first_frame` / `last_frame` to conditioning and empty video/audio latents. With no frame it is plain text-to-video. Frames need the `vae` input |
| `🍏 ASDX MiniMax H3 Reference to Video` | Prompt plus reference images (up to 9), videos (up to 3, with optional soundtracks) and standalone audio (up to 3), to conditioning and empty latents. Refer to them in the prompt as `<Picture i>`, `<Video k>`, `<Audio j>` (1-based per type). `ref_image_size`: `match` (default) scales each reference down to the generation's pixel area, `max` uses a 2048px short edge |
| `🍏 ASDX MiniMax H3 Sampler` | Flow-matching Euler sampling of the video and audio latents |

Accepted checkpoint formats (DiT and text encoder): GGUF, and ComfyUI safetensors either dense
or INT8 ConvRot. The W4A8 variant is refused.

Loading the VAEs: the video VAE and the audio VAE are two separate files, each loaded by a
VAE loader node. `🍏 ASDX VAE Loader` works for both: it builds `comfy.sd.VAE`, which
recognizes the MiniMax H3 video VAE and the MiniMax H3 audio VAE from their state-dict keys
(`comfy/sd.py`), so use one loader node per file. ComfyUI's core `VAELoader` builds the same
class. The audio VAE is only needed for reference audio and for decoding the audio latent.

Minimal graphs. The `🍏 ASDX MiniMax H3 Sampler` takes `model` (from `Model Loader`),
`conditioning`, `video_latent` and `audio_latent` (all three emitted by the conditioning node
below), plus `steps` and `seed`. Text-to-video, first/last frame and reference graphs differ
only in the conditioning node:

- t2v: `Text Encoder Loader` -> `Image to Video` (no frame connected) -> `Sampler`.
- i2v / first-last frame: `Text Encoder Loader` (`load_vision` on) and a video VAE -> `Image to Video`
  with `first_frame` and/or `last_frame` -> `Sampler`.
- ref: `Text Encoder Loader` (`load_vision` on if any image or video reference is used), video VAE
  and audio VAE -> `Reference to Video` with the reference inputs (each `ref_*` input grows a new
  slot as you connect one) -> `Sampler`. Without a VAE the references only condition the text
  encoder.

After the sampler (the sampler emits two latents, not a video):

- `Sampler.video_latent` + video VAE -> `🍏 ASDX VAE Decode (MLX)` (accepts the 5D video latent)
  -> images.
- `Sampler.audio_latent` + audio VAE -> `🍏 ASDX VAE Decode Audio` -> AUDIO.
- images + audio -> core `CreateVideo` with `fps` set to 24 (its default is 30) -> core `SaveVideo`.

Known limits:

- No PDD (the reference implementation's head bank) and no AddGuide conditioning.
- The reference implementations select a distinct `ref2va` DiT checkpoint for the reference task. The
  loader here loads the fl2va and ref2va files unchanged (same tensor structure), so use the `ref2va`
  file for references.
- References lengthen the packed token sequence (a warning is logged past the row threshold);
  `ref_image_size = max` can be several times slower than `match`.

## Installation

1. Clone this repo into your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/unipacfr/ComfyUI-MLXU.git
   ```

2. Install dependencies:
   ```bash
   pip install -r ComfyUI-MLXU/requirements.txt
   ```

3. Restart ComfyUI

## Apple Silicon Optimizations

### Memory Management
- **MLX lazy evaluation**: Computations are not executed until `mx.eval()` is called
- **Strategic eval points**: Only at bridge boundaries (PyTorch ↔ MLX)
- **Cache lifecycle**: `mx.clear_cache()` after sampling completes
- **Peak memory tracking**: Built into every node for debugging OOM
- **Stale executor cache purge**: on checkpoint switch, drops ComfyUI's own node-output cache entries still holding a previous model (invisible to its RAM-pressure eviction, which doesn't recognize MLX-backed payloads) so `mx.clear_cache()` can actually reclaim them

### Precision
- **float16 default**: Native on Apple Silicon, best performance
- **bfloat16 support**: Better numerical stability for sensitive ops
- **No torch.float16 on MPS**: Avoids NaN issues seen in PyTorch MPS

### Unified Memory
- **Zero-copy between CPU/GPU**: MLX and PyTorch MPS share the same physical memory
- **NumPy as bridge**: `mx.array(np_array)` creates a view, not a copy, when possible
- **Device placement**: Latents created on MPS for zero-copy with sampler

### Inference Speed
- **SeaCache**: Skip computation when previous step output is similar enough
- **TeaCache**: Output-level caching with adaptive threshold interpolation
- **Precomputed rope**: Positional embeddings computed once, reused across steps
- **Precomputed text projection**: `txt_in(emb)` computed once, reused across steps

### Advanced Conditioning
- **LoRA Runtime Loading**: Standard A/B matrices, kohya-style, ComfyUI diff format (weights and `.diff_b` biases), and diffusers/PEFT-style keys for FLUX.1, Flux.2, Krea2 and Z-Image, with per-LoRA alpha scaling
- **LyCORIS adapters**: LoKr (Kronecker) applied as a structured forward-time residual that never materializes the full `[out, in]` delta, and LoHa (Hadamard); both full and low-rank factor forms
- **Multi-LoRA Stacking**: Up to 5 LoRAs applied simultaneously with strength scheduling
- **Kontext KV Cache**: Reference image tokens cached and injected into transformer attention layers
- **Quantized checkpoints**: FP8_SCALED and INT8 ConvRot/tensorwise dequantized on load
- **Krea2 image-grounded encoding** (`🍏 ASDX Krea2 Grounded Encode`): prompt is encoded alongside a source image through the CLIP's vision tower, matching the krea2_edit LoRA's training-time conditioning

## Comparison with SDMLX

| Feature | SDMLX | ASDX |
|---------|-------|------|
| FLUX native transformer | ✅ (C++ core) | ✅ (pure MLX) |
| SDXL support | ❌ | ✅ |
| LoRA runtime loading | ✅ | ✅ |
| Multi-LoRA stacking | ✅ | ✅ |
| TeaCache acceleration | ✅ | ✅ |
| SeaCache acceleration | ✅ | ✅ |
| ControlNet Union | ✅ | 🚧 disabled (WIP, `disabled_nodes/`) |
| IP-Adapter | ✅ | 🚧 disabled (WIP, `disabled_nodes/`) |
| Kontext reference | ✅ | ✅ |
| Code size | ~15K lines | ~3K lines |
| Learning curve | Steep | Gentle |
| Modularity | Tightly coupled | Cleanly separated |

ASDX is the **minimal viable set** — the core diffusion pipeline with advanced features (LoRA, quantized checkpoints, TeaCache/SeaCache) implemented in clean, readable MLX code. Use it as a starting point, learning reference, or production pipeline for Apple Silicon.

## License

MIT
