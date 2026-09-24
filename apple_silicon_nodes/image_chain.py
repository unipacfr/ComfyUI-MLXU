"""MFLUX_IMAGE typed chain for multi-image workflows.

Mirrors the MfluxImage pattern from ComfyUI-mflux-AnyModel: a typed
container that passes between nodes for img2img, inpainting, depth
control, and multi-image edit/redux workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import torch
from comfy_api.latest import io


# ── MFLUX_IMAGE dataclass ─────────────────────────────────────────────

@dataclass
class MFLUX_IMAGE:
    """Typed image payload for chaining in multi-image workflows.

    Combines image, latent, depth map, and mask into a single typed
    container that can be passed between nodes without serialization.
    """

    image: torch.Tensor | None = None
    """[B, H, W, C] float32 [0, 1] — the original or generated image."""

    latent: torch.Tensor | None = None
    """[B, 16, H/8, W/8] — pre-encoded latent (FLUX format)."""

    depth_map: mx.array | None = None
    """[B, H/8, W/8, 1] — depth map from DepthPro (MLX array)."""

    mask: torch.Tensor | None = None
    """[B, H, W] float32 [0, 1] — inpainting mask."""

    source: str = "input"
    """Source identifier: 'input', 'generated', 'reference', 'depth'."""

    metadata: dict = field(default_factory=dict)
    """Arbitrary metadata (strength, resolution, etc.)."""

    @property
    def has_image(self) -> bool:
        return self.image is not None and self.image.numel() > 0

    @property
    def has_latent(self) -> bool:
        return self.latent is not None and self.latent.numel() > 0

    @property
    def has_mask(self) -> bool:
        return self.mask is not None and self.mask.numel() > 0

    @property
    def has_depth(self) -> bool:
        return self.depth_map is not None and self.depth_map.size > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for ComfyUI bridge (type hints only, no tensor data)."""
        return {
            "type": "mflux_image",
            "has_image": self.has_image,
            "has_latent": self.has_latent,
            "has_mask": self.has_mask,
            "has_depth": self.has_depth,
            "source": self.source,
            "metadata": self.metadata,
        }


# ── Node: ImageToLatent ───────────────────────────────────────────────

class ASDX_ImageToLatent(io.ComfyNode):
    """Wrap an image in an image-only MFLUX_IMAGE payload (no latent)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_ImageToLatent",
            display_name="🍏 ASDX Image → Latent",
            category="ASDX/ImageChain",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Custom("mflux_image").Output(display_name="image_latent"),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor) -> io.NodeOutput:
        # Image-only payload: the former MLX VAE encode path imported a
        # nonexistent `MLXVAE` and always fell through to this. Real encoding
        # belongs to ASDX_VAEEncode (comfy.sd.VAE).
        return io.NodeOutput(MFLUX_IMAGE(image=image, source="input"))


# ── Node: MaskFromImage ───────────────────────────────────────────────

class ASDX_MaskFromImage(io.ComfyNode):
    """Generate a binary mask from an image using threshold.

    Converts a grayscale or RGB image to a binary [0, 1] mask.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MaskFromImage",
            display_name="🍏 ASDX Mask From Image",
            category="ASDX/ImageChain",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.Custom("mflux_image").Output(display_name="image_with_mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        threshold: float,
        invert: bool,
    ) -> io.NodeOutput:
        # Convert to grayscale if needed
        if image.ndim == 4 and image.shape[-1] > 1:
            mask = image.mean(dim=-1, keepdim=True).squeeze(-1)
        else:
            mask = image.squeeze(-1) if image.ndim == 4 else image

        # Threshold
        mask = (mask > threshold).float()
        if invert:
            mask = 1.0 - mask

        # Expand to [B, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        payload = MFLUX_IMAGE(
            image=image,
            mask=mask,
            source="mask",
            metadata={"threshold": threshold, "invert": invert},
        )
        return io.NodeOutput(mask, payload)


# ── Node: MaskBlur ────────────────────────────────────────────────────

class ASDX_MaskBlur(io.ComfyNode):
    """Apply Gaussian blur to a mask for soft edges."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MaskBlur",
            display_name="🍏 ASDX Mask Blur",
            category="ASDX/ImageChain",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("blur_radius", default=4, min=0, max=64),
            ],
            outputs=[
                io.Mask.Output(display_name="blurred_mask"),
                io.Custom("mflux_image").Output(display_name="image_with_mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        blur_radius: int,
    ) -> io.NodeOutput:
        if blur_radius <= 0:
            return io.NodeOutput(mask, MFLUX_IMAGE(mask=mask, source="mask"))

        try:
            import torch.nn.functional as F

            # Pad mask for edge convolution
            pad = blur_radius // 2
            if pad > 0:
                padded = F.pad(mask, (pad, pad, pad, pad), mode="reflect")
                # 2D Gaussian blur via separable convolution
                k_size = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                kernel = cls._gaussian_kernel(k_size, blur_radius / 3.0)
                blurred = F.conv2d(
                    padded.unsqueeze(1), kernel, padding=k_size // 2
                ).squeeze(1)
                # Crop to original size
                if pad > 0:
                    blurred = blurred[:, pad:-pad, pad:-pad]
            else:
                blurred = mask
        except Exception:
            blurred = mask

        if blurred.ndim == 3 and blurred.shape[0] > 1:
            pass  # batch already correct
        elif blurred.ndim == 2:
            blurred = blurred.unsqueeze(0)

        return io.NodeOutput(blurred, MFLUX_IMAGE(mask=blurred, source="mask_blur"))

    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a 1D Gaussian kernel, then make it 2D via outer product."""
        import math
        coords = torch.arange(kernel_size, dtype=torch.float32)
        center = kernel_size // 2
        diff = (coords - center).float() / sigma
        kernel_1d = torch.exp(-0.5 * diff * diff)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.unsqueeze(0).unsqueeze(0)


# ── Node: ImageCompositor ─────────────────────────────────────────────

class ASDX_ImageCompositor(io.ComfyNode):
    """Composite a generated image over an original using a mask.

    Implements mask-preserve compositing: where mask=1, keep the original;
    where mask=0, use the generated image. Blended proportionally.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_ImageCompositor",
            display_name="🍏 ASDX Image Compositor",
            category="ASDX/ImageChain",
            inputs=[
                io.Image.Input("original_image"),
                io.Image.Input("generated_image"),
                io.Mask.Input("mask"),
            ],
            outputs=[
                io.Image.Output(display_name="composited"),
                io.Custom("mflux_image").Output(display_name="image_chain"),
            ],
        )

    @classmethod
    def execute(
        cls,
        original_image: torch.Tensor,
        generated_image: torch.Tensor,
        mask: torch.Tensor,
    ) -> io.NodeOutput:
        # Expand mask to match image dimensions
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, original_image.shape[-1])
        elif mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, original_image.shape[-1])

        # Ensure same size
        h, w = original_image.shape[1], original_image.shape[2]
        if generated_image.shape[1] != h or generated_image.shape[2] != w:
            generated_image = torch.nn.functional.interpolate(
                generated_image.permute(0, 3, 1, 2),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        # Clamp mask to [0, 1]
        mask = mask.clamp(0.0, 1.0)

        # Composite: original * mask + generated * (1 - mask)
        composited = original_image * mask + generated_image * (1.0 - mask)
        composited = composited.clamp(0.0, 1.0)

        payload = MFLUX_IMAGE(
            image=composited,
            source="composited",
            metadata={"mask_used": True},
        )
        return io.NodeOutput(composited, payload)


NODE_LIST = [
    ASDX_ImageToLatent,
    ASDX_MaskFromImage,
    ASDX_MaskBlur,
    ASDX_ImageCompositor,
]
