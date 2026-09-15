"""Standard (non-multimodal) RoPE for the Qwen3 text-only path.

For text-only input (no image tokens), `comfy/text_encoders/llama.py`'s
`precompute_freqs_cis` builds `position_ids` as `[1, S]` (a single sequential
axis; see `Llama2_.forward`'s `torch.arange(...).unsqueeze(0)` when
`position_ids is None`), which fails its own `position_ids.shape[0] > 1`
gate for Qwen3-VL's interleaved M-RoPE -- so it always takes the plain
single-axis branch:

    inv_freq[i] = 1 / theta^(2i/head_dim), i in [0, head_dim/2)
    angle[s, i] = s * inv_freq[i]                    (s = sequence position)
    cos/sin = cat(angle, angle).cos()/.sin()          (duplicated, not tiled per-pair)

and `apply_rope`'s rotation is:

    out[:half] = x[:half] * cos - x[half:] * sin
    out[half:] = x[half:] * cos + x[:half] * sin

which is bit-for-bit the same "split in half, not interleaved pairs"
rotation `rope.py::apply_rope_split_half` already implements for MiniMax
H3's DiT (that math is generic RoPE, not something MiniMax H3-specific) --
this module only supplies Qwen3's angle computation (single axis, its own
`rope_theta`, full head_dim rotated -- no partial-rope `rot_dim` cutoff).
`rope.rms_norm_rope_split_half(x, weight, eps, head_dim, cos, sin)` (passing
the FULL head_dim as `rot_dim`) is reused as-is for the fused per-head
RMSNorm + rotation, verified in `test_text_encoder_rope.py` against the real
`comfy.text_encoders.llama.precompute_freqs_cis`/`apply_rope`.
"""

from __future__ import annotations

import mlx.core as mx


def qwen3_rope_cos_sin(seq_len: int, head_dim: int, theta: float) -> tuple[mx.array, mx.array]:
    """`[S, head_dim // 2]` cos/sin for sequential positions `0..seq_len-1`,
    single axis, full `head_dim` rotated -- matches
    `precompute_freqs_cis(head_dim, torch.arange(seq_len).unsqueeze(0), theta)`'s
    plain (non-interleaved-MRoPE) branch."""
    half = head_dim // 2
    exponent = mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim
    inv_freq = 1.0 / (theta**exponent)  # [half]
    positions = mx.arange(seq_len, dtype=mx.float32)  # [S]
    angles = positions[:, None] * inv_freq[None, :]  # [S, half]
    return mx.cos(angles), mx.sin(angles)
