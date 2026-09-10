"""DCT v3.11 Per-Slot Interpretable Transport.

Replaces the direction loss (DCR ≈ 0.526 ≈ random) with per-slot survival
supervision and a diversity constraint, making slot representations directly
interpretable without an OT bottleneck in the gradient path.

Status: candidate (not yet validated on real datasets).
"""

from .model import DCTV311SlotInterpretable

__all__ = ["DCTV311SlotInterpretable"]
