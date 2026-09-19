"""Unfolds MiniMax H3's tokenizer presentation into what the vision-grounded
text encoder consumes: the same construction as
`comfy/sd1_clip.py::SDClipModel.process_tokens` (each vision entry becomes
`size` merged rows spliced between its `<|vision_start|>`/`<|vision_end|>`
ids) plus `comfy/text_encoders/minimax.py::token_tags_from_embeds_info`.
"""

from __future__ import annotations

import numbers

import mlx.core as mx
import numpy as np

from .text_encoder import Qwen3TextEncoder, VisionInputs
from .vision_preprocess import preprocess_image, preprocess_video_block
from .vision_rope import mrope_position_ids
from .vision_tower import VisionTower

# Any id works: rows at vision positions are replaced by the tower's output.
PLACEHOLDER_ID = 151643


def token_tags(seq_len: int, infos: list[dict]) -> np.ndarray:
    """adaLN token tags: 1 for text, 0 (video modality) for every vision block,
    including the flanking `<|vision_start|>`/`<|vision_end|>` tokens."""
    tags = np.ones(seq_len, dtype=np.int64)
    for info in infos:
        tags[max(0, info["index"] - 1): info["index"] + info["size"] + 1] = 0
    return tags


def _to_numpy(data) -> np.ndarray:
    if hasattr(data, "detach"):  # torch tensor (ComfyUI IMAGE)
        data = data.detach().cpu().float().numpy()
    return np.asarray(data, dtype=np.float32)


def build_vision_conditioning(
    entries: list, tower: VisionTower | None, *, rope_dims: tuple[int, int, int] = (24, 20, 20)
) -> tuple[np.ndarray, VisionInputs | None, np.ndarray]:
    """`entries`: `MiniMaxH3Tokenizer.tokenize_with_weights(...)["qwen3vl_32b"][0]`.
    Returns `(input_ids int32 [S], vision or None, token_tags [S])`."""
    ids: list[int] = []
    rows: list[mx.array] = []
    deepstack_parts: list[list[mx.array]] = []
    infos: list[dict] = []
    for item, _weight in entries:
        if isinstance(item, numbers.Integral):
            ids.append(int(item))
            continue
        if tower is None:
            raise ValueError("ASDX: the prompt contains images but no vision tower is loaded (enable load_vision on the text encoder loader)")
        data = _to_numpy(item["data"])
        if item.get("minimax_video_block", False):
            patches, grid = preprocess_video_block(data)
        else:
            patches, grid = preprocess_image(data)
        merged, deepstack = tower(mx.array(patches), [grid])
        mx.eval(merged, *deepstack)
        size = merged.shape[0]
        infos.append({"type": "image", "index": len(ids), "size": size, "grid": grid})
        ids.extend([PLACEHOLDER_ID] * size)
        rows.append(merged)
        deepstack_parts.append(deepstack)

    input_ids = np.asarray(ids, dtype=np.int32)
    tags = token_tags(len(ids), infos)
    if not infos:
        return input_ids, None, tags
    vision = VisionInputs(
        rows=mx.concatenate(rows, axis=0),
        row_indices=np.concatenate([np.arange(i["index"], i["index"] + i["size"]) for i in infos]),
        deepstack=[mx.concatenate([part[k] for part in deepstack_parts], axis=0) for k in range(len(deepstack_parts[0]))],
        position_ids=mrope_position_ids(infos, len(ids)),
        rope_dims=rope_dims,
    )
    return input_ids, vision, tags


def encode_with_vision(
    encoder: Qwen3TextEncoder,
    tower: VisionTower | None,
    entries: list,
    *,
    rope_dims: tuple[int, int, int] = (24, 20, 20),
) -> tuple[mx.array, np.ndarray]:
    """Run the presentation through the encoder. Returns `(hidden [S, hidden], token_tags [S])`."""
    input_ids, vision, tags = build_vision_conditioning(entries, tower, rope_dims=rope_dims)
    hidden = encoder(mx.array(input_ids), vision=vision)
    mx.eval(hidden)
    return hidden, tags
