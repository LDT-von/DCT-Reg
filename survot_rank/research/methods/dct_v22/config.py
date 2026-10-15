"""Version-local capacity controls; inherited v2.1 objectives are unchanged."""
from dataclasses import dataclass
import math

from survot_rank.research.methods.dct_v21.config import DCTV21Config


@dataclass(frozen=True)
class DCTV22Config(DCTV21Config):
    dim: int = 128
    dropout: float = .30
    wsi_hidden_dim: int = 256
    wsi_dropout: float = .30
    relation_gradient_interval: int = 0

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.wsi_hidden_dim, bool) or not isinstance(self.wsi_hidden_dim, int) or self.wsi_hidden_dim < 1:
            raise ValueError("wsi_hidden_dim must be a positive integer")
        value = self.wsi_dropout
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("wsi_dropout must be finite in [0,1)")
        value = self.relation_gradient_interval
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("relation_gradient_interval must be a nonnegative integer")

    @classmethod
    def from_args(cls, args):
        values = DCTV21Config.from_args(args).to_dict()
        # The parent's defaults describe v2.1. Honor explicit common fields,
        # but apply this version's defaults when no value was supplied.
        for name, key, default in (("dim", "wsi_projection_dim", 128),
                                   ("dropout", "otehv2_dropout", .30)):
            if getattr(args, key, None) is None:
                values[name] = default
        # Euclidean scale is derived from dim unless explicitly configured.
        if getattr(args, "dct_v2_euclidean_scale", None) is None:
            values["euclidean_scale"] = None
        for name in ("wsi_hidden_dim", "wsi_dropout", "relation_gradient_interval"):
            value = getattr(args, "dct_v22_" + name, None)
            if value is not None:
                values[name] = value
        return cls(**values)
