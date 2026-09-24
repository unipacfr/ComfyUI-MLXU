"""
VAE Decode / Encode nodes
=========================
MLX-native VAE decoding and encoding for FLUX.

Uses the MLX VAE implementation for zero-copy decoding on Apple Silicon,
bridging only at the input/output boundaries.
"""

from __future__ import annotations

import time
from typing import Any

from comfy_api.latest import io

import torch


# ── Globals ───────────────────────────────────────────────────────────

_VAE_CACHE: dict[str, Any] = {}


# ── MPS failures a tiled retry can fix ───────────────────────────────

def _needs_tiled_retry(e: Exception) -> bool:
    """True for the two plain `RuntimeError`s MPS raises that tiling resolves.

    1. **Out of memory.** `comfy.sd.VAE.decode()`/`.encode()` already retry via
       tiled decode when `model_management.is_oom(e)` recognises the exception —
       but `is_oom()` only matches `torch.cuda.OutOfMemoryError` or
       `torch.AcceleratorError` (error_code==2 / "out of memory" in message). A
       real MPS OOM on this machine (torch 2.13) raises a plain `RuntimeError`
       ("MPS backend out of memory..."), and `AcceleratorError` is a
       `RuntimeError` SUBCLASS, not the reverse — so `is_oom()` returns False,
       `raise_non_oom()` re-raises, and the node crashes instead of falling back
       to tiled decode. Verified directly against the installed torch:
       `isinstance(plain_mps_runtimeerror, AcceleratorError)` is False.

    2. **INT_MAX tensor limit.** MPSGraph refuses any tensor with more than
       INT_MAX (2**31-1) elements, regardless of free memory. The VAE mid-block
       self-attention materialises a full `(H*W/64)**2`-element matrix, so every
       image past ~1723x1723 px trips it — and `slice_attention()` only splits
       that matrix when it does NOT fit in free memory, so the more memory is
       free the more certainly it hits the limit (steps=1 on an idle 64 GB
       machine). Neither `is_oom()` nor `slice_attention()`'s own retry loop
       recognises it ("out of memory" is not in the message), so it also
       crashes. Reproduced on the Krea2 (Wan21) VAE at 2048x2048 for both encode
       and decode; tiling to 512px tiles brings the matrix back under the limit.
    """
    if not isinstance(e, RuntimeError):
        return False
    msg = str(e).lower()
    return "out of memory" in msg or "larger than int_max" in msg


# ── VAE Loader ───────────────────────────────────────────────────────

class ASDX_VAELoader(io.ComfyNode):
    """Load a standalone VAE checkpoint (e.g. ae.safetensors for FLUX.1,
    the Flux2/Krea2/Z-Image VAE, or a plain SDXL VAE file).

    Needed for any model loaded via ASDX_DiffusionLoader, which only
    returns the diffusion transformer — no VAE is embedded in a
    diffusion-only checkpoint. ASDX_CheckpointLoader already returns a
    real VAE for full checkpoints and does not need this node.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_VAELoader",
            display_name="🍏 ASDX VAE Loader",
            category="ASDX/Loaders",
            inputs=[
                io.Combo.Input("vae_name", options=cls._get_vae_names()),
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
            ],
        )

    @staticmethod
    def _get_vae_names() -> list[str]:
        try:
            import folder_paths
            return folder_paths.get_filename_list("vae")
        except Exception:
            return []

    @classmethod
    def execute(cls, vae_name: str) -> io.NodeOutput:
        if vae_name in _VAE_CACHE:
            return io.NodeOutput(_VAE_CACHE[vae_name])

        import comfy.sd
        import comfy.utils
        import folder_paths

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        vae.throw_exception_if_invalid()

        _VAE_CACHE[vae_name] = vae
        print(f"[ASDX] VAE loaded: {vae_name}")
        return io.NodeOutput(vae)


# ── VAE Decode ───────────────────────────────────────────────────────

class ASDX_VAEDecode(io.ComfyNode):
    """Decode FLUX latents to images using MLX VAE.

    The VAE decoder runs entirely in MLX, only the final image tensor
    is converted to PyTorch for ComfyUI downstream nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_VAEDecode",
            display_name="🍏 ASDX VAE Decode (MLX)",
            category="ASDX/Latent",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, samples: dict, vae: Any) -> io.NodeOutput:
        # Get latent samples
        if not isinstance(samples, dict) or "samples" not in samples:
            raise RuntimeError("ASDX VAE Decode: expected LATENT input.")

        latent = samples["samples"]

        # Always use the real ComfyUI/PyTorch VAE. The former MLX VAE path
        # (`mlx_vae.py`, now deleted) was an untrained placeholder that
        # silently produced noise for 16ch latents (FLUX.1/Z-Image).
        return io.NodeOutput(*cls._fallback_decode(latent, vae))

    @staticmethod
    def _fallback_decode(latent: torch.Tensor, vae: Any) -> tuple[torch.Tensor]:
        """Standard PyTorch VAE decode fallback.

        `latent` here is already the unwrapped tensor (`decode()` extracts
        `samples["samples"]` before calling this) — do not index it again.

        Some VAEs (e.g. Krea2's, which is architecturally a Wan 2.1 video
        VAE — `comfy.supported_models.Krea2.latent_format = latent_formats.
        Wan21`, confirmed against the real comfy source) declare
        `vae.latent_dim == 3` and require a 5D `[B,C,T,H,W]` tensor; comfy's
        own `VAE.decode()` only auto-squeezes 5D->4D for `latent_dim==2`
        VAEs, never the reverse (`comfy/sd.py` `VAE.decode`). Our own latent
        dicts are always plain 4D `[B,C,H,W]` (single image, no temporal
        axis), so add a size-1 temporal axis here for these VAEs and remove
        it again from the decoded image, mirroring stock `nodes.py::
        VAEDecode.decode()`'s own `if len(images.shape) == 5: reshape(...)`.
        """
        if getattr(vae, "latent_dim", 2) == 3 and latent.dim() == 4:
            latent = latent.unsqueeze(2)

        # comfy's own OOM->tiled retry never fires on MPS (see
        # `_needs_tiled_retry`), so do it here. Mirrors `comfy/sd.py::VAE.decode`'s
        # own structure: set a flag inside `except` and tile OUTSIDE it, because
        # the live exception keeps every tensor allocated at raise-time referenced
        # until the block exits — tiling inside it would run against that
        # still-held memory.
        do_tile = False
        try:
            image = vae.decode(latent)
        except Exception as e:
            if not _needs_tiled_retry(e):
                raise
            print(f"[ASDX] VAE Decode: MPS limit hit ({e}), retrying with tiled decode.")
            do_tile = True

        if do_tile:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
            if getattr(vae, "latent_dim", 2) == 3:
                # `VAE.decode_tiled()`'s dims==3 branch builds `overlap=(1, overlap,
                # overlap)` WITHOUT defaulting `overlap` first, so calling it with no
                # arguments passes `(1, None, None)` down to `tiled_scale_multidim`
                # and dies on `int - None`. Pass the tile/overlap `VAE.decode()`'s own
                # internal 3D fallback computes. (`encode_tiled()` does default its
                # overlap, so the encode side below needs no such argument.)
                tile = 256 // vae.spacial_compression_decode()
                image = vae.decode_tiled(latent, tile_x=tile, tile_y=tile, overlap=tile // 4)
            else:
                image = vae.decode_tiled(latent)

        if image.dim() == 5:
            image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        return (image,)


# ── VAE Encode ───────────────────────────────────────────────────────

class ASDX_VAEEncode(io.ComfyNode):
    """Encode images to FLUX latents using MLX VAE.

    The encoding runs in MLX for Apple Silicon acceleration.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_VAEEncode",
            display_name="🍏 ASDX VAE Encode (MLX)",
            category="ASDX/Latent",
            inputs=[
                io.Image.Input("pixels"),
                io.Vae.Input("vae"),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, pixels: torch.Tensor, vae: Any) -> io.NodeOutput:
        t0 = time.perf_counter()

        if pixels.ndim != 4:
            raise RuntimeError(f"ASDX VAE Encode: expected [B,H,W,C] image, got {pixels.shape}")

        # Always use the real ComfyUI/PyTorch VAE, mirroring ASDX_VAEDecode.execute()
        # above (the deleted MLX placeholder returned raw pixels mislabeled as a latent).
        latent_torch = cls._fallback_encode(pixels, vae)

        elapsed = time.perf_counter() - t0
        print(f"[ASDX] VAE Encode: {latent_torch.shape}, {elapsed:.2f}s")

        return io.NodeOutput({"samples": latent_torch})

    @staticmethod
    def _fallback_encode(pixels: torch.Tensor, vae: Any) -> torch.Tensor:
        """Standard PyTorch VAE encode fallback.

        Mirrors `ASDX_VAEDecode._fallback_decode`: comfy's own `VAE.encode()`
        already handles the `[B,H,W,C]` -> internal layout conversion and pixel
        normalization, and for `latent_dim == 3` VAEs (e.g. Krea2's Wan21-style
        VAE) inserts a size-1 temporal axis before encoding, returning a 5D
        `[B,C,T,H,W]` latent. Our own latent dicts are always plain 4D
        `[B,C,H,W]` (single image, no temporal axis) -- squeeze that axis back
        out here, the exact inverse of what `_fallback_decode` does before decode.
        """
        # Same MPS OOM/INT_MAX->tiled fallback as `_fallback_decode` above, same
        # flag-outside-except reasoning. `encode_tiled` returns the same layout
        # as `encode`, so the `latent_dim == 3` squeeze below covers both paths.
        do_tile = False
        try:
            latent = vae.encode(pixels)
        except Exception as e:
            if not _needs_tiled_retry(e):
                raise
            print(f"[ASDX] VAE Encode: MPS limit hit ({e}), retrying with tiled encode.")
            do_tile = True

        if do_tile:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
            latent = vae.encode_tiled(pixels)

        if getattr(vae, "latent_dim", 2) == 3 and latent.dim() == 5:
            latent = latent.squeeze(2)
        return latent


# ── VAE Decode Audio ─────────────────────────────────────────────────

class ASDX_VAEDecodeAudio(io.ComfyNode):
    """Decode an audio latent (e.g. MiniMax H3's audio stream) to a
    waveform. First AUDIO-type output in this project -- everything else so
    far has been IMAGE/video. Mirrors ComfyUI's own generic
    `comfy_extras.nodes_audio.VAEDecodeAudio`/`vae_decode_audio` exactly
    (not MiniMax-H3-specific: any audio VAE works here), including its
    post-decode std-normalization (avoids clipping without changing the
    audio VAE's own output convention) and the `is_nested`-latent unwrap
    for ComfyUI's own packed audio+video latents -- our own MiniMax H3
    pipeline emits separate video/audio latent dicts rather than a packed
    NestedTensor, so `is_nested` is always False for latents this project's
    own samplers produce, but real-world AUDIO latents saved by other tools
    could still arrive packed.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_VAEDecodeAudio",
            display_name="🍏 ASDX VAE Decode Audio",
            category="ASDX/Latent",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, samples: dict, vae: Any) -> io.NodeOutput:
        if not isinstance(samples, dict) or "samples" not in samples:
            raise RuntimeError("ASDX VAE Decode Audio: expected LATENT input.")
        return io.NodeOutput(cls._decode_audio(samples, vae))

    @staticmethod
    def _decode_audio(samples: dict, vae: Any) -> dict:
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[-1]

        do_tile = False
        try:
            audio = vae.decode(latent)
        except Exception as e:
            if not _needs_tiled_retry(e):
                raise
            print(f"[ASDX] VAE Decode Audio: MPS limit hit ({e}), retrying with tiled decode.")
            do_tile = True

        if do_tile:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
            audio = vae.decode_tiled(latent, tile_x=512, tile_y=512, overlap=64)

        audio = audio.movedim(-1, 1)  # vae's native [B, T, C] -> [B, C, T]

        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio = audio / std

        sample_rate = samples.get("sample_rate")
        if sample_rate is None:
            sample_rate = getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 44100))
        return {"waveform": audio, "sample_rate": sample_rate}


NODE_LIST = [
    ASDX_VAELoader,
    ASDX_VAEDecode,
    ASDX_VAEEncode,
    ASDX_VAEDecodeAudio,
]
