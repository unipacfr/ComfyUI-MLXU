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

import mlx.core as mx
import torch


# ── Globals ───────────────────────────────────────────────────────────

_VAE_CACHE: dict[str, Any] = {}


# ── MPS OOM detection ────────────────────────────────────────────────

def _is_mps_oom(e: Exception) -> bool:
    """True for the plain `RuntimeError` a real MPS OOM raises.

    `comfy.sd.VAE.decode()`/`.encode()` already retry via tiled decode when
    `model_management.is_oom(e)` recognises the exception — but `is_oom()` only
    matches `torch.cuda.OutOfMemoryError` or `torch.AcceleratorError`
    (error_code==2 / "out of memory" in message). A real MPS OOM on this machine
    (torch 2.13) raises a plain `RuntimeError` ("MPS backend out of memory..."),
    and `AcceleratorError` is a `RuntimeError` SUBCLASS, not the reverse — so
    `is_oom()` returns False, `raise_non_oom()` re-raises, and the node crashes
    instead of falling back to tiled decode. Verified directly against the
    installed torch: `isinstance(plain_mps_runtimeerror, AcceleratorError)` is
    False.
    """
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


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

        # Always use the real ComfyUI/PyTorch VAE. `_decode_with_mlx_vae()`
        # (below) has no real weight loading — it's an untrained placeholder
        # that silently ignores `vae` and produces noise, not an image, for
        # any latent that used to route through it (16ch: FLUX.1/Z-Image).
        # SDXL (4ch) and Flux2 (128ch) already always used this real path.
        # Left `_decode_with_mlx_vae`/`mlx_vae.py` in place, just unreferenced
        # from here — a native MLX VAE decoder remains a separate future task.
        return io.NodeOutput(*cls._fallback_decode(latent, vae))

    @staticmethod
    def _decode_with_mlx_vae(latent: mx.array) -> mx.array:
        """Decode using MLX VAE.

        In production, this would use the native MLX VAE from sdmlx.
        For now, we provide a placeholder that demonstrates the pattern.
        """
        # The real implementation would:
        # 1. Load the VAE weights into an MLX VAE module
        # 2. Call vae.decode(latent)
        # 3. Return the decoded output

        # Placeholder: for demonstration, just return the latent
        # A real implementation needs the VAE module from sdmlx/mlx_sd
        try:
            from .mlx_vae import get_vae_decoder
            vae = get_vae_decoder()
            if vae is not None:
                return vae.decode(latent)
        except Exception:
            pass

        # Fallback: just return the latent (will produce garbage without real VAE)
        # This should never be reached in a proper installation
        print("[ASDX] VAE Decode: no MLX VAE available, using fallback")
        return latent

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

        # comfy's own OOM->tiled retry never fires on MPS (see `_is_mps_oom`),
        # so do it here. Mirrors `comfy/sd.py::VAE.decode`'s own structure: set a
        # flag inside `except` and tile OUTSIDE it, because the live exception
        # keeps every tensor allocated at raise-time referenced until the block
        # exits — tiling inside it would run against that still-held memory.
        do_tile = False
        try:
            image = vae.decode(latent)
        except Exception as e:
            if not _is_mps_oom(e):
                raise
            print("[ASDX] VAE Decode: MPS out of memory, retrying with tiled decode.")
            do_tile = True

        if do_tile:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
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
        # above: `_encode_with_mlx_vae()` below has no real weight loading -- it's an
        # untrained placeholder that silently returns the raw pixels (mislabeled as a
        # "latent") whenever the MLX encoder is unavailable, which it always is today.
        latent_torch = cls._fallback_encode(pixels, vae)

        elapsed = time.perf_counter() - t0
        print(f"[ASDX] VAE Encode: {latent_torch.shape}, {elapsed:.2f}s")

        return io.NodeOutput({"samples": latent_torch})

    @staticmethod
    def _encode_with_mlx_vae(image: mx.array) -> mx.array:
        """Encode using MLX VAE encoder."""
        try:
            from .mlx_vae import get_vae_encoder
            vae = get_vae_encoder()
            if vae is not None:
                return vae.encode(image)
        except Exception:
            pass
        print("[ASDX] VAE Encode: no MLX VAE available, using fallback")
        return image

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
        # Same MPS OOM->tiled fallback as `_fallback_decode` above, same
        # flag-outside-except reasoning. `encode_tiled` returns the same layout
        # as `encode`, so the `latent_dim == 3` squeeze below covers both paths.
        do_tile = False
        try:
            latent = vae.encode(pixels)
        except Exception as e:
            if not _is_mps_oom(e):
                raise
            print("[ASDX] VAE Encode: MPS out of memory, retrying with tiled encode.")
            do_tile = True

        if do_tile:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
            latent = vae.encode_tiled(pixels)

        if getattr(vae, "latent_dim", 2) == 3 and latent.dim() == 5:
            latent = latent.squeeze(2)
        return latent


NODE_LIST = [
    ASDX_VAELoader,
    ASDX_VAEDecode,
    ASDX_VAEEncode,
]
