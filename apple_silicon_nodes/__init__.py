"""
Apple Silicon Diffusion Nodes
=============================
Custom ComfyUI nodes optimized for Apple Silicon (M1/M2/M3/M4/M5)
using MLX native inference with zero-copy Unified Memory semantics.

Key design principles:
  - MLX native for diffusion backbones (transformer/unet)
  - PyTorch MPS for ComfyUI compatibility where conversion is impractical
  - Strategic mx.eval() / mx.clear_cache() at memory hotspots
  - float16/bfloat16 native on Apple Silicon (avoid torch.float16 MPS NaN issues)

Advanced features:
  - LoRA runtime loading (standard A/B and ComfyUI diff format)
  - TeaCache acceleration (output-level step skipping)
  - SeaCache acceleration (residual-based step skipping)
  - Kontext KV cache (reference image conditioning)

Note: ControlNet Union and IP-Adapter (incl. CLIP Vision Encode) nodes are
temporarily moved out of the active package to `disabled_nodes/` at the repo
root pending a revisit -- not deleted, just not registered/importable here.
"""

from __future__ import annotations

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

__version__ = "0.2.1"

# ── Node display names ────────────────────────────────────────────────
# Only consulted on the dependency-missing fallback path below, where the
# real node modules (and their own display_name= in define_schema) can't be
# imported -- this is the sole remaining source of truth for those names.
_DISPLAY = {
    # Core
    "ASDX_DiffusionLoader": "🍏 ASDX Diffusion Loader",
    "ASDX_CheckpointLoader": "🍏 ASDX Checkpoint Loader",
    "ASDX_DualCLIPLoader": "🍏 ASDX Dual CLIP Loader",
    "ASDX_CLIPLoader": "🍏 ASDX CLIP Loader",
    "ASDX_CLIPTextEncode": "🍏 ASDX CLIP Text Encode",
    "ASDX_MLXSampler": "🍏 ASDX MLX Native Sampler",
    "ASDX_VAELoader": "🍏 ASDX VAE Loader",
    "ASDX_VAEDecode": "🍏 ASDX VAE Decode (MLX)",
    "ASDX_VAEEncode": "🍏 ASDX VAE Encode (MLX)",
    "ASDX_EmptyLatent": "🍏 ASDX Empty Latent",
    "ASDX_MemoryProfiler": "🍏 ASDX Memory Profiler",
    "ASDX_CacheManager": "🍏 ASDX Cache Manager",
    # Conditioning
    "ASDX_ConditioningMerger": "🍏 ASDX Conditioning Merger",
    "ASDX_Krea2Edit": "🍏 ASDX Krea2 Identity Edit",
    # LoRA
    "ASDX_LoraLoader": "🍏 ASDX LoRA Loader",
    "ASDX_MultiLoraLoader": "🍏 ASDX Multi LoRA Loader",
    "ASDX_LoraSchedule": "🍏 ASDX LoRA Schedule",
    # Image Chain
    "ASDX_ImageToLatent": "🍏 ASDX Image → Latent",
    "ASDX_MaskFromImage": "🍏 ASDX Mask From Image",
    "ASDX_MaskBlur": "🍏 ASDX Mask Blur",
    "ASDX_ImageCompositor": "🍏 ASDX Image Compositor",
    # Depth
    "ASDX_DepthMap": "🍏 ASDX Depth Map",
    # Utilities
    "ASDX_LivePreview": "🍏 ASDX Live Preview",
}

# ── Lazy import with graceful fallback ────────────────────────────────
_UNAVAILABLE_REASON: str | None = None
try:
    from .loader import NODE_LIST as _loader_nodes
    from .conditioning import NODE_LIST as _cond_nodes
    from .sampler import NODE_LIST as _sampler_nodes
    from .vae import NODE_LIST as _vae_nodes
    from .latent import NODE_LIST as _latent_nodes
    from .memory import NODE_LIST as _mem_nodes
    from .lora import NODE_LIST as _lora_nodes
    from .krea2_edit import NODE_LIST as _krea2_edit_nodes

    # Optional: image chain, depth map, live preview
    try:
        from .image_chain import NODE_LIST as _chain_nodes
    except Exception:
        _chain_nodes = []

    try:
        from .depth_map import NODE_LIST as _depth_nodes
    except Exception:
        _depth_nodes = []

    try:
        from .live_preview import NODE_LIST as _preview_nodes
    except Exception:
        _preview_nodes = []

    _ALL_NODES: list[type[io.ComfyNode]] = [
        *_loader_nodes,
        *_cond_nodes,
        *_sampler_nodes,
        *_vae_nodes,
        *_latent_nodes,
        *_mem_nodes,
        *_lora_nodes,
        *_krea2_edit_nodes,
        *_chain_nodes,
        *_depth_nodes,
        *_preview_nodes,
    ]
except ModuleNotFoundError as exc:
    # Allow import on non-macOS / non-MLX hosts (e.g. CI, Comfy Registry parser)
    _missing = {
        "mlx", "numpy", "torch", "PIL", "safetensors", "gguf",
        "huggingface_hub", "transformers", "comfy", "folder_paths",
    }
    if exc.name in _missing:
        _ALL_NODES = []
        _UNAVAILABLE_REASON = (
            "ASDX requires Apple Silicon MLX runtime dependencies. "
            "Install on macOS with: pip install mlx mlx-lm safetensors"
        )
    else:
        raise


def _make_unavailable_node(node_id: str, display_name: str) -> type[io.ComfyNode]:
    """Build a stub node that errors at runtime with a helpful message,
    used in place of the real nodes when required dependencies are missing.
    """
    class _Unavailable(io.ComfyNode):
        @classmethod
        def define_schema(cls) -> io.Schema:
            return io.Schema(
                node_id=node_id,
                display_name=display_name,
                category="ASDX/Utilities",
                inputs=[],
                outputs=[],
            )

        @classmethod
        def execute(cls) -> io.NodeOutput:
            raise RuntimeError(_UNAVAILABLE_REASON)

    _Unavailable.__name__ = node_id
    _Unavailable.__qualname__ = node_id
    return _Unavailable


class ASDXExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        if _UNAVAILABLE_REASON is not None:
            return [
                _make_unavailable_node(node_id, display_name)
                for node_id, display_name in _DISPLAY.items()
            ]
        return _ALL_NODES


async def comfy_entrypoint() -> ASDXExtension:
    return ASDXExtension()


__all__ = ["__version__"]
