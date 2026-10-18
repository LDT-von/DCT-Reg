"""Feature-only DCT coupling for a host that retains its own objective.

Only the DCT cost networks and balanced/KL-relaxed solvers are reused. There
are no DCT slots, event encoders, predictors, decoders, or auxiliary losses.
Native short slot/prototype sequences are coupled directly. Long sequences
use soft prototype pooling solely to bound the size of the transport matrix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .solver import DCTTransportSolver


@dataclass(frozen=True)
class TransportOnlyConfig:
    """A coupling recipe in the feature space already owned by the host.

    ``legacy`` is the normalized balanced v3.13 cost. ``bounded`` retains
    absolute cost offsets and is required by the optional UOT solver. A
    missing cost mode selects the appropriate recipe automatically.
    """

    host_dim: int = 256
    slot_budget: int = 8
    stages: int = 4
    geometries: tuple[str, ...] = ("cosine", "euclidean", "positive_dot")
    transport_mode: str = "balanced"
    cost_mode: str | None = None
    plan_mode: str = "learned"
    coordinate_temperature: float = .30
    stage_cost_weight: float = .20
    evidence_cost_weight: float = 0.
    evidence_mass_floor: float = .05
    marginal_strength: float = 1.
    eps: float = .05
    eps_start: float = .10
    eps_anneal_epochs: int = 12
    sinkhorn_iters: int = 50
    projection_iters: int = 10000
    projection_tol: float = 1e-4
    tau_wsi: float = .5
    tau_omics: float = .5
    cost_scale: float = 1.
    euclidean_scale: float | None = None
    uot_iters: int = 200
    uot_tolerance: float = 1e-7
    residual_init: float = .05

    def __post_init__(self):
        for name in ("host_dim", "slot_budget", "stages", "eps_anneal_epochs",
                     "sinkhorn_iters", "projection_iters", "uot_iters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.transport_mode not in {"balanced", "uot"}:
            raise ValueError("transport_mode must be balanced or uot")
        cost_mode = self.cost_mode
        if cost_mode is None:
            cost_mode = "legacy" if self.transport_mode == "balanced" else "bounded"
        if cost_mode == "absolute":
            cost_mode = "bounded"
        if cost_mode not in {"legacy", "bounded"}:
            raise ValueError("cost_mode must be legacy or bounded (absolute)")
        if self.transport_mode == "uot" and cost_mode != "bounded":
            raise ValueError("UOT requires absolute bounded costs")
        object.__setattr__(self, "cost_mode", cost_mode)
        if self.plan_mode not in {"learned", "independent"}:
            raise ValueError("plan_mode must be learned or independent")
        geometries = tuple(self.geometries)
        if not geometries or len(geometries) != len(set(geometries)) or not set(
            geometries
        ) <= {"cosine", "euclidean", "positive_dot"}:
            raise ValueError("geometries must be a nonempty unique selection of supported costs")
        object.__setattr__(self, "geometries", geometries)
        for name in ("coordinate_temperature", "evidence_mass_floor", "eps", "eps_start",
                     "projection_tol", "tau_wsi", "tau_omics", "cost_scale", "uot_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("stage_cost_weight", "evidence_cost_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(self.marginal_strength, bool) or not 0 <= self.marginal_strength <= 1:
            raise ValueError("marginal_strength must be in [0, 1]")
        if isinstance(self.residual_init, bool) or not math.isfinite(self.residual_init):
            raise ValueError("residual_init must be finite")
        scale = math.sqrt(self.host_dim) if self.euclidean_scale is None else self.euclidean_scale
        if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("euclidean_scale must be finite and positive")
        object.__setattr__(self, "euclidean_scale", float(scale))

    @property
    def dim(self):
        return self.host_dim

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


class DCTTransportMechanism(DCTTransportSolver):
    """Return two feature tensors; the host alone computes its task loss.

    Subclassing permits exact reuse of the original solver's ``super`` calls.
    Its constructor is deliberately bypassed: no old slot/event/readout module
    is instantiated. ``last_plans`` contains detached, zero-padded matrices;
    its last two axes index the valid tokens or pooled prototypes per patient.
    """

    def __init__(self, config: TransportOnlyConfig):
        nn.Module.__init__(self)
        if not isinstance(config, TransportOnlyConfig):
            raise TypeError("transport-only mechanism requires TransportOnlyConfig")
        self.config = config
        self.enabled = True
        self.current_epoch = 0
        self.plan_mode = config.plan_mode
        dim = config.host_dim
        self.stage_pair_cost = nn.Sequential(
            nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(),
            nn.Linear(dim, config.stages),
        )
        self.evidence_gate = nn.Sequential(
            nn.LayerNorm(dim * 5), nn.Linear(dim * 5, dim), nn.GELU(), nn.Linear(dim, 1),
        )
        self.register_buffer("stage_embedding", torch.linspace(-1., 1., config.stages)
                             .unsqueeze(1).repeat(1, dim))
        self.wsi_prototypes = nn.Parameter(torch.randn(config.slot_budget, dim) * .02)
        self.omics_prototypes = nn.Parameter(torch.randn(config.slot_budget, dim) * .02)
        self.wsi_gain = nn.Parameter(torch.tensor(float(config.residual_init)))
        self.omics_gain = nn.Parameter(torch.tensor(float(config.residual_init)))
        self.last_diagnostics = {}
        self.last_solver_diagnostics = {}
        self.last_plans = None
        self.last_reference_rows = None
        self.last_reference_cols = None

    def set_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self.enabled = enabled
        return self

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.current_epoch = epoch
        return self

    def set_plan_mode(self, mode):
        if mode not in {"learned", "independent"}:
            raise ValueError("plan_mode must be learned or independent")
        self.plan_mode = mode
        return self

    def get_extra_state(self):
        return {"schema": "dct_transport_only_v1", "config": self.config.to_dict(),
                "enabled": self.enabled, "epoch": self.current_epoch, "plan_mode": self.plan_mode}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("schema") != "dct_transport_only_v1":
            raise ValueError("transport-only checkpoint metadata is required")
        saved = TransportOnlyConfig.from_dict(state["config"])
        if saved.to_dict() != self.config.to_dict():
            raise ValueError("construct transport-only module from the exact checkpoint configuration")
        self.set_enabled(state["enabled"])
        self.set_epoch(state["epoch"])
        self.set_plan_mode(state["plan_mode"])

    @staticmethod
    def _mask(tokens, mask, name):
        if mask is None:
            return torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool or mask.device != tokens.device:
            raise ValueError(f"{name} must be a boolean [B,N] tensor on the feature device")
        return mask

    def _pool(self, tokens, prototypes):
        if tokens.size(0) <= self.config.slot_budget:
            return tokens, None
        # A single soft assignment is sufficient for bounded transport. It is
        # not an iterative slot-attention block or a second encoder.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            values = tokens.to(prototypes.dtype)
            scores = F.normalize(values, dim=-1) @ F.normalize(prototypes, dim=-1).T
            assignment = torch.softmax(scores / self.config.coordinate_temperature, dim=-1)
            weights = assignment.T / assignment.sum(0).unsqueeze(-1).clamp_min(1e-8)
            return weights @ values, assignment

    def forward(self, wsi, omics, *, epoch=None, enabled=None, wsi_mask=None, omics_mask=None):
        if wsi.ndim != 3 or omics.ndim != 3 or wsi.size(0) != omics.size(0) or wsi.size(0) < 1:
            raise ValueError("expected paired [B,N,D] and [B,P,D] feature tensors")
        if wsi.size(-1) != self.config.host_dim or omics.size(-1) != self.config.host_dim:
            raise ValueError("both host feature dimensions must equal host_dim")
        if wsi.device != omics.device or wsi.dtype != omics.dtype or not wsi.is_floating_point():
            raise ValueError("paired features require the same floating dtype and device")
        active = self.enabled if enabled is None else enabled
        if not isinstance(active, bool):
            raise TypeError("enabled must be bool")
        epoch = self.current_epoch if epoch is None else epoch
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.last_plans = self.last_reference_rows = self.last_reference_cols = None
        self.last_solver_diagnostics = {}
        if not active:
            self.last_diagnostics = {"schema": "transport_only_diagnostics_v1", "enabled": False}
            return wsi, omics
        if wsi.device != self.stage_embedding.device:
            raise ValueError("features and transport mechanism must be on the same device")
        wm, om = self._mask(wsi, wsi_mask, "wsi_mask"), self._mask(omics, omics_mask, "omics_mask")
        if not bool(torch.isfinite(wsi[wm]).all()) or not bool(torch.isfinite(omics[om]).all()):
            raise ValueError("valid host features must be finite")
        batch, cfg = wsi.size(0), self.config
        width, height = min(wsi.size(1), cfg.slot_budget), min(omics.size(1), cfg.slot_budget)
        dtype = next(self.stage_pair_cost.parameters()).dtype
        plans = torch.zeros(batch, cfg.stages, len(cfg.geometries), width, height,
                            device=wsi.device, dtype=dtype)
        rows = plans.new_zeros(batch, cfg.stages, width)
        cols = plans.new_zeros(batch, cfg.stages, height)
        available = wm.any(-1) & om.any(-1)
        w_outputs, o_outputs, solver_reports, shapes = [], [], [], []
        for patient in range(batch):
            if not bool(available[patient]):
                w_outputs.append(wsi[patient])
                o_outputs.append(omics[patient])
                shapes.append((0, 0))
                continue
            wi, oi = wm[patient].nonzero(as_tuple=True)[0], om[patient].nonzero(as_tuple=True)[0]
            w, wa = self._pool(wsi[patient, wi], self.wsi_prototypes)
            o, oa = self._pool(omics[patient, oi], self.omics_prototypes)
            costs, a, b = self.costs_and_marginals(w.unsqueeze(0), o.unsqueeze(0))
            p, a, b = self.plans_from_costs(costs, a, b, epoch)
            solver_reports.append(dict(self.last_solver_diagnostics))
            nw, no = w.size(0), o.size(0)
            # Geometry and stage averages introduce no trainable readout.
            # Dividing by REFERENCE marginals retains UOT mass strength;
            # normalizing by realized marginals would erase that signal.
            with torch.autocast(device_type=wsi.device.type, enabled=False):
                consensus = p[0].mean(1)
                w_update = (torch.einsum("swo,od->swd", consensus, o.to(p.dtype))
                            / a[0, ..., None].clamp_min(1e-8)).mean(0)
                o_update = (torch.einsum("swo,wd->sod", consensus, w.to(p.dtype))
                            / b[0, ..., None].clamp_min(1e-8)).mean(0)
                if wa is not None:
                    w_update = wa.to(p.dtype) @ w_update
                if oa is not None:
                    o_update = oa.to(p.dtype) @ o_update
            w_outputs.append(wsi[patient].index_add(
                0, wi, (torch.tanh(self.wsi_gain) * w_update).to(wsi.dtype)))
            o_outputs.append(omics[patient].index_add(
                0, oi, (torch.tanh(self.omics_gain) * o_update).to(omics.dtype)))
            plans[patient, ..., :nw, :no] = p[0].detach()
            rows[patient, :, :nw], cols[patient, :, :no] = a[0].detach(), b[0].detach()
            shapes.append((nw, no))
        self.last_plans = plans.detach()
        self.last_reference_rows, self.last_reference_cols = rows.detach(), cols.detach()
        with torch.no_grad():
            mass = plans.sum((-2, -1))
            paired_mass = mass[available]
            self.last_diagnostics = {
                "schema": "transport_only_diagnostics_v1", "enabled": True,
                "transport_mode": cfg.transport_mode, "cost_mode": cfg.cost_mode,
                "plan_mode": self.plan_mode, "epoch": epoch, "epsilon": self.epsilon(epoch),
                "paired_patients": int(available.sum()), "transport_shapes": shapes,
                "wsi_compressed": [int(mask.sum()) > cfg.slot_budget for mask in wm],
                "omics_compressed": [int(mask.sum()) > cfg.slot_budget for mask in om],
                "total_mass_mean": float(paired_mass.mean()) if paired_mass.numel() else 0.,
                "wsi_residual_gain": float(torch.tanh(self.wsi_gain)),
                "omics_residual_gain": float(torch.tanh(self.omics_gain)),
                "solver": solver_reports,
            }
        return torch.stack(w_outputs), torch.stack(o_outputs)
