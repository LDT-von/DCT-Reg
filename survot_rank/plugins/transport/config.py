"""Explicit configuration for the extracted DCT transport mechanism."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TransportPluginConfig:
    wsi_dim: int
    omics_dim: int
    dim: int = 256
    wsi_slots: int = 8
    omics_slots: int = 8
    stages: int = 4
    heads: int = 4
    event_layers: int = 2
    slot_iters: int = 3
    slot_heads: int = 8
    slot_dim_head: int = 64
    slot_hidden_dim: int = 128
    dropout: float = 0.15
    slot_init_mode: str = "deterministic"
    slot_eval_seed: int = 1729
    geometries: tuple[str, ...] = ("cosine", "euclidean", "positive_dot")
    plan_mode: str = "learned"
    coordinate_temperature: float = 0.30
    stage_cost_weight: float = 0.20
    evidence_cost_weight: float = 0.0
    evidence_mass_floor: float = 0.05
    marginal_strength: float = 1.0
    eps: float = 0.05
    eps_start: float = 0.10
    eps_anneal_epochs: int = 12
    sinkhorn_iters: int = 50
    projection_iters: int = 10000
    projection_tol: float = 1e-4
    residual_init: float = 0.0

    def __post_init__(self):
        for name in ("wsi_dim", "omics_dim", "dim", "wsi_slots", "omics_slots",
                     "stages", "heads", "event_layers", "slot_iters", "slot_heads",
                     "slot_dim_head", "slot_hidden_dim", "eps_anneal_epochs",
                     "sinkhorn_iters", "projection_iters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.slot_init_mode not in {"deterministic", "learned", "gaussian"}:
            raise ValueError("unsupported slot_init_mode")
        if self.plan_mode not in {"learned", "independent", "uniform"}:
            raise ValueError("plan_mode must be learned, independent, or uniform")
        geometries = tuple(self.geometries)
        if not geometries or len(set(geometries)) != len(geometries):
            raise ValueError("geometries must be nonempty and unique")
        if not set(geometries) <= {"cosine", "euclidean", "positive_dot"}:
            raise ValueError("unsupported transport geometry")
        object.__setattr__(self, "geometries", geometries)
        for name in ("coordinate_temperature", "eps", "eps_start", "projection_tol",
                     "evidence_mass_floor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("stage_cost_weight", "evidence_cost_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.marginal_strength <= 1:
            raise ValueError("marginal_strength must be in [0, 1]")
        if not math.isfinite(self.residual_init):
            raise ValueError("residual_init must be finite")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "TransportPluginConfig":
        return cls(**values)
