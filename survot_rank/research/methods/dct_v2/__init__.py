"""Independent DCT v2; own encoders and survival head, six objectives."""
from .config import DCTV2Config
from .model import DCTV2

__all__ = ["DCTV2", "DCTV2Config"]
