"""Loads the `visual.*` weights of MiniMax H3's Qwen3-VL-32B text-encoder
checkpoint (safetensors or GGUF, both dense BF16 here) into a `VisionTower`.

Format facts (inspected on the real files, 351 `visual.*` tensors each, same
names in both): the Conv3d `patch_embed.proj.weight` is `(1152, 3, 2, 16, 16)`
in safetensors but `(3456, 2, 16, 16)` in GGUF (the RGB axis folded into the
first dimension); both flatten row-major to `(1152, 1536)`, which is the
Linear the tower uses. Only the Qwen3-VL-32B geometry exists, so shapes are
verified against `VisionConfig()` rather than auto-detected.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .checkpoint_source import open_checkpoint
from .vision_tower import VisionConfig, VisionTower

_PREFIX = "visual."


def verify_vision_shapes(shapes: dict[str, tuple[int, ...]], config: VisionConfig) -> None:
    """Raise `ValueError` unless `shapes` (names without the `visual.` prefix)
    describe exactly `config`'s geometry."""
    depth = len([k for k in shapes if re.fullmatch(r"blocks\.\d+\.norm1\.weight", k)])
    n_ds = len([k for k in shapes if re.fullmatch(r"deepstack_merger_list\.\d+\.norm\.weight", k)])
    expected = {
        "pos_embed.weight": (config.num_position_embeddings, config.hidden_size),
        "merger.linear_fc2.weight": (config.out_hidden_size, config.merge_dim),
        "blocks.0.mlp.linear_fc1.weight": (config.intermediate_size, config.hidden_size),
    }
    problems = []
    if depth != config.depth:
        problems.append(f"blocks: found {depth}, expected {config.depth}")
    if n_ds != len(config.deepstack_visual_indexes):
        problems.append(f"deepstack mergers: found {n_ds}, expected {len(config.deepstack_visual_indexes)}")
    for name, shape in expected.items():
        if tuple(shapes.get(name, ())) != shape:
            problems.append(f"{name}: found {shapes.get(name)}, expected {shape}")
    patch = shapes.get("patch_embed.proj.weight")
    if patch is None or math.prod(patch) != config.hidden_size * config.patch_dim:
        problems.append(f"patch_embed.proj.weight: found {patch}, expected {config.hidden_size * config.patch_dim} elements")
    if problems:
        raise ValueError("ASDX: visual tower checkpoint does not match Qwen3-VL-32B: " + "; ".join(problems))


def assign_vision_weights(model: VisionTower, tensors: dict[str, mx.array]) -> int:
    """Assign `tensors` (names without the `visual.` prefix) to `model` by
    name, converting to float32. Any missing name or shape mismatch raises; only
    `patch_embed.proj.weight` may differ in rank (element count must match)."""
    flat = dict(tree_flatten(model.parameters()))
    missing = sorted(set(flat) - set(tensors))
    if missing:
        raise KeyError(f"ASDX: vision tower weights missing from checkpoint: {missing[:5]} ({len(missing)} total)")
    new: dict[str, mx.array] = {}
    for name, current in flat.items():
        tensor = tensors[name]
        if name == "patch_embed.proj.weight":  # Conv3d weight: 5-D or GGUF 4-D, only the element count must match
            fits = tensor.size == current.size
        else:
            fits = tuple(tensor.shape) == tuple(current.shape)
        if not fits:
            raise ValueError(f"ASDX: vision weight '{name}': checkpoint shape {tuple(tensor.shape)} does not fit {tuple(current.shape)}")
        new[name] = tensor.reshape(current.shape).astype(mx.float32)
    model.update(tree_unflatten(list(new.items())))
    mx.eval(model.parameters())
    mx.clear_cache()  # return the freed load-time buffers MLX keeps cached (up to ~6 GB after the text encoder)
    return len(new)


def load_vision_tower(path: str | Path) -> VisionTower:
    """Build the Qwen3-VL-32B `VisionTower` and load its `visual.*` weights."""
    source = open_checkpoint(path)
    shapes = {n[len(_PREFIX):]: s for n, s in source.shapes().items() if n.startswith(_PREFIX)}
    config = VisionConfig()
    verify_vision_shapes(shapes, config)
    tensors = {name: source.get(_PREFIX + name) for name in shapes}
    model = VisionTower(config)
    assigned = assign_vision_weights(model, tensors)
    del tensors  # the raw checkpoint buffers are only freed once this dict drops them; then return them to the system
    mx.clear_cache()
    print(f"[ASDX] MiniMax H3 vision tower ({Path(path).suffix.lstrip('.').lower()}): assigned {assigned} params")
    return model
