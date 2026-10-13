"""Version-local readout and objective choices; all six v2 terms remain."""
from dataclasses import dataclass
import math

from survot_rank.research.methods.dct_v2.config import DCTV2Config


@dataclass(frozen=True)
class DCTV21Config(DCTV2Config):
    local_readout: str = "centered_pair"
    relation_objective: str = "contrast_probe"
    relation_temperature: float = .05
    relation_probe_gain: float = .10
    diversity_objective: str = "collision_distance"

    def __post_init__(self):
        super().__post_init__()
        if self.local_readout not in {"centered_pair", "legacy_mean", "none"}:
            raise ValueError("invalid v2.1 local readout")
        if self.relation_objective not in {"contrast_probe", "legacy_hinge"}:
            raise ValueError("invalid v2.1 relation objective")
        if self.diversity_objective not in {"collision_distance", "legacy_squared"}:
            raise ValueError("invalid v2.1 diversity objective")
        for name in ("relation_temperature", "relation_probe_gain"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.relation_probe_gain > 1:
            raise ValueError("relation_probe_gain must be <=1")
        if self.local_readout == "centered_pair" and not self.local_gating:
            raise ValueError("centered_pair already uses actual local margins; use local_readout=none to ablate it")
        if self.relation_objective == "contrast_probe" and self.relation_tolerance != 0:
            raise ValueError("v2.1 contrast_probe uses zero margin; relation_tolerance is for legacy_hinge only")

    @classmethod
    def from_args(cls, args):
        values = DCTV2Config.from_args(args).to_dict()
        for name in ("local_readout", "relation_objective", "relation_temperature",
                     "relation_probe_gain", "diversity_objective"):
            value = getattr(args, "dct_v21_" + name, None)
            if value is not None:
                values[name] = value
        return cls(**values)
