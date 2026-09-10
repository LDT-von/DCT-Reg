"""DCT v3.10 with fixed pre-computed anchors [ARCHIVED - ABLATION].

LEGACY: This method is for ablation studies only.
NOT for paper use. Use DCTV310DirectionalRegularizedTransport instead.
"""

from .model import DCTV310FixedAnchors

__all__ = ["DCTV310FixedAnchors"]
