"""Structural test for FLUX depth conditioning (channel-concat img_in path).

No live checkpoint needed: a reduced random-weight FluxTransformer with
``in_channels=128`` (the depth/canny-dev shape: 16 noise + 16 depth-latent
channels, 2x2 patched) is fed a synthetic wide image-token tensor through
``predict()``, mirroring this project's "random-weight forward pass" step of
its own checkpoint-verification recipe. Confirms only that the wider img_in
plumbing (Task C1 + C2) shapes and runs correctly -- not that a real
flux1-depth-dev checkpoint produces a depth-following image (no such
checkpoint is available in this environment; see Task C4 in
docs/superpowers/plans/2026-09-13-node-extraction-and-depth-control.md).
"""

from __future__ import annotations

import mlx.core as mx

from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
native = load_node_module("native")
FluxConfig = native.FluxConfig
FluxTransformer = native.FluxTransformer


def _reduced_depth_model() -> FluxTransformer:
    config = FluxConfig(
        num_double_blocks=1, num_single_blocks=1, dtype="float16", in_channels=128,
    )
    return FluxTransformer(config)


def test_depth_concat_forward_pass():
    model = _reduced_depth_model()

    img_h, img_w = 8, 8
    n_img = img_h * img_w
    txt_len = 4

    img = mx.random.normal((1, n_img, 128)).astype(mx.float16)
    txt = mx.random.normal((1, txt_len, 4096)).astype(mx.float16)
    pooled = mx.random.normal((1, 768)).astype(mx.float16)
    rope = model.get_rope(img_h, img_w, txt_len)

    out = model.predict(img=img, txt=txt, timestep=0.5, guidance=3.5, pooled=pooled, rope=rope)
    mx.eval(out)

    assert out.shape == (1, n_img, 64)
    assert bool(mx.all(mx.isfinite(out)).item())
