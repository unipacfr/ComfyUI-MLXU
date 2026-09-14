"""
MiniMax H3 native MLX implementation (in progress).

So far:
- MiniMaxH3Config / detect_minimax_h3_config: architecture configuration,
  shape-derived from a checkpoint the same way `comfy/model_detection.py`
  does (see config.py's module docstring for the real checkpoints this was
  verified against).

Not yet implemented: the DiT itself (`model.py`), weight mapping
(`weight_map.py`), text encoder, VAE wiring. See
`docs/plan-multi-modeles-apple-silicon.md` §5 Phase 6.
"""

from __future__ import annotations

from .config import MiniMaxH3Config, detect_minimax_h3_config

__all__ = [
    "MiniMaxH3Config",
    "detect_minimax_h3_config",
]
