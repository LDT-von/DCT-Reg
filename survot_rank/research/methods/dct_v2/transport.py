"""Version-owned DCT feature computations and UOT/projection solvers.

Feature equations descend from DCT's slots, prototype coordinates and event
readout. Numerical UOT/projection code is copied from the locally audited
transport candidate; this independent module imports no external host or
plugin package. Source hashes are recorded in delivery provenance.
"""
from dataclasses import dataclass, fields
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.func import functional_call
from survot_rank.research.components.slot_attention import MultiHeadSlotAttention
from .config import DCTV2Config

def frozen_call(module, *args):
    params = {k: v.detach() for k, v in module.named_parameters()}
    buffers = {k: v.detach().clone() for k, v in module.named_buffers()}
    return functional_call(module, (params, buffers), args, strict=True)

@dataclass
class TransportState:
    wsi_tokens: torch.Tensor
    omics_tokens: torch.Tensor
    wsi_mask: torch.Tensor
    omics_mask: torch.Tensor
    wsi_slots: torch.Tensor
    omics_slots: torch.Tensor
    plans: torch.Tensor
    row_marginals: torch.Tensor
    col_marginals: torch.Tensor
    event_tokens: torch.Tensor
    stage_weights: torch.Tensor
    transported_wsi: torch.Tensor
    transported_omics: torch.Tensor
    pair_available: torch.Tensor


def pair_tokens(wsi, omics):
    w = wsi[:, :, None, :].expand(-1, -1, omics.size(1), -1)
    o = omics[:, None, :, :].expand(-1, wsi.size(1), -1, -1)
    return torch.cat((w, o, w * o, (w - o).abs()), dim=-1)


def normalize_cost(cost):
    cost = cost - cost.amin(dim=(-2, -1), keepdim=True)
    return cost / cost.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def semantic_slots(tokens, prototypes, temperature):
    scores = torch.einsum("bnd,kd->bkn", F.normalize(tokens, dim=-1),
                          F.normalize(prototypes, dim=-1))
    assignments = torch.softmax(scores / temperature, dim=1)
    weights = assignments / assignments.sum(-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bkn,bnd->bkd", weights, tokens)


class DCTEventReadout(nn.Module):
    """DCT MultiScaleOTFusion without a host hazard predictor.

Three plan embedding channels remain fixed for matched geometry controls. If a
geometry is omitted, its channel receives the first retained plan explicitly.
"""
    def __init__(self, dim, stages, heads, dropout):
        super().__init__()
        self.cost_convs = nn.ModuleDict({
            "cosine": nn.Linear(1, dim),
            "euclidean": nn.Sequential(nn.Linear(1, dim // 2), nn.GELU(),
                                       nn.Linear(dim // 2, dim)),
            "dot": nn.Linear(1, dim),
        })
        self.proj = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)
        self.event_queries = nn.Parameter(torch.randn(stages, dim) * 0.02)
        self.cross_attn = nn.TransformerEncoderLayer(
            dim, heads, dim * 2, dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.refine = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, wsi, omics, plan_cos, plan_euc, plan_dot):
        dim = wsi.size(-1)
        contents = pair_tokens(wsi, omics)
        mass_features = torch.cat((
            self.cost_convs["cosine"](plan_cos.unsqueeze(-1).to(wsi.dtype)),
            self.cost_convs["euclidean"](plan_euc.unsqueeze(-1).to(wsi.dtype)),
            self.cost_convs["dot"](plan_dot.unsqueeze(-1).to(wsi.dtype)),
        ), dim=-1)
        tokens = self.proj(mass_features + contents[..., :dim * 3])
        tokens = tokens.reshape(wsi.size(0), -1, dim)
        mass = plan_cos.reshape(wsi.size(0), -1).clamp_min(1e-8).log().unsqueeze(-1)
        scores = torch.einsum("kd,bpd->bpk", F.normalize(self.event_queries, dim=-1),
                              F.normalize(tokens, dim=-1)) + mass.to(tokens.dtype)
        assignments = torch.softmax(scores.transpose(1, 2), dim=-1)
        events = torch.bmm(assignments, tokens)
        events = self.norm(self.cross_attn(events))
        return events + self.refine(events), assignments


class DCTFeatureCore(nn.Module):
    def __init__(self, config: DCTV2Config):
        super().__init__()
        self.config = config
        self.plan_mode = config.plan_mode
        dim = config.dim
        slot_kwargs = dict(dim=dim, heads=config.slot_heads, dim_head=config.slot_dim_head,
                           iters=config.slot_iters, hidden_dim=config.slot_hidden_dim,
                           init_mode=config.slot_init_mode)
        self.slot_attention_wsi = MultiHeadSlotAttention(config.wsi_slots, **slot_kwargs,
                                                        eval_seed=config.slot_eval_seed)
        self.slot_attention_omic = MultiHeadSlotAttention(config.omics_slots, **slot_kwargs,
                                                         eval_seed=config.slot_eval_seed + 1)
        self.shared_wsi_prototypes = nn.Parameter(torch.randn(config.wsi_slots, dim) * .02)
        self.shared_omic_prototypes = nn.Parameter(torch.randn(config.omics_slots, dim) * .02)
        self.stage_pair_cost = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim),
                                            nn.GELU(), nn.Linear(dim, config.stages))
        self.evidence_gate = nn.Sequential(nn.LayerNorm(dim * 5), nn.Linear(dim * 5, dim),
                                          nn.GELU(), nn.Linear(dim, 1))
        self.register_buffer("stage_embedding", torch.linspace(-1., 1., config.stages)
                             .unsqueeze(1).repeat(1, dim))
        self.fusion = DCTEventReadout(dim, config.stages, config.heads, config.dropout)
        encoder = nn.TransformerEncoderLayer(dim, config.heads, dim * 2, config.dropout,
                                             activation="gelu", batch_first=True, norm_first=True)
        self.event_encoder = nn.TransformerEncoder(encoder, config.event_layers)
        self.event_norm = nn.LayerNorm(dim)
        self.event_gate = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(),
                                        nn.Dropout(config.dropout), nn.Linear(dim // 2, 1))


    def epsilon(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        fraction = min(1., float(epoch) / self.config.eps_anneal_epochs)
        return self.config.eps_start + fraction * (self.config.eps - self.config.eps_start)

    def costs_and_marginals(self, wsi, omics):
        # Costs and both marginal normalizations share an FP32/FP64 path.
        # Half-precision marginals can have unequal totals after rounding,
        # making any balanced projection mathematically infeasible.
        dtype = next(self.stage_pair_cost.parameters()).dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("keep transport parameters FP32/FP64 and use autocast for mixed precision")
        with torch.autocast(device_type=wsi.device.type, enabled=False):
            return self._costs_and_marginals(wsi.to(dtype), omics.to(dtype))

    def _costs_and_marginals(self, wsi, omics):
        cfg = self.config
        contents = pair_tokens(wsi, omics)
        stage_cost = F.softplus(self.stage_pair_cost(contents)).permute(0, 3, 1, 2)
        base = {
            "cosine": 1. - torch.bmm(F.normalize(wsi, dim=-1), F.normalize(omics, dim=-1).transpose(1, 2)),
            # cdist defines a finite zero-distance gradient for coincident slots.
            "euclidean": torch.cdist(wsi, omics, p=2),
            "positive_dot": F.softplus(-torch.bmm(wsi, omics.transpose(1, 2))),
        }
        base = [normalize_cost(base[name]) for name in cfg.geometries]
        costs, rows, cols = [], [], []
        for stage in range(cfg.stages):
            code = self.stage_embedding[stage].view(1, 1, 1, -1).expand(*contents.shape[:-1], cfg.dim)
            gate = torch.sigmoid(self.evidence_gate(torch.cat((contents, code), -1)).squeeze(-1))
            evidence_cost = normalize_cost(-torch.log(gate.clamp_min(1e-6)))
            prognosis_cost = normalize_cost(stage_cost[:, stage])
            costs.append(torch.stack([c + cfg.stage_cost_weight * prognosis_cost
                                      + cfg.evidence_cost_weight * evidence_cost for c in base], 1))
            rows.append(gate.mean(-1).clamp_min(cfg.evidence_mass_floor))
            cols.append(gate.mean(-2).clamp_min(cfg.evidence_mass_floor))
        rows, cols = torch.stack(rows, 1), torch.stack(cols, 1)
        rows, cols = rows / rows.sum(-1, keepdim=True), cols / cols.sum(-1, keepdim=True)
        strength = cfg.marginal_strength
        rows = (1. - strength) / rows.size(-1) + strength * rows
        cols = (1. - strength) / cols.size(-1) + strength * cols
        return torch.stack(costs, 1), rows, cols

    def plans_from_costs(self, costs, rows, cols, epoch):
        cfg = self.config
        if self.plan_mode not in {"learned", "independent", "uniform"}:
            raise ValueError("unsupported transport plan_mode")
        dtype = torch.float32 if costs.dtype in (torch.float16, torch.bfloat16) else costs.dtype
        costs, rows, cols = costs.to(dtype), rows.to(dtype), cols.to(dtype)
        rows = rows / rows.sum(-1, keepdim=True)
        cols = cols / cols.sum(-1, keepdim=True)
        if self.plan_mode == "uniform":
            rows = torch.full_like(rows, 1. / rows.size(-1))
            cols = torch.full_like(cols, 1. / cols.size(-1))
        if self.plan_mode != "learned":
            plans = (rows[..., :, None] * cols[..., None, :]).unsqueeze(2)
            return plans.expand(-1, -1, len(cfg.geometries), -1, -1), rows, cols
        stage_plans = []
        for stage in range(cfg.stages):
            geometries = []
            for geometry in range(len(cfg.geometries)):
                cost = costs[:, stage, geometry]
                kernel = (-cost / self.epsilon(epoch)).clamp(-60., 60.)
                a, b = rows[:, stage], cols[:, stage]
                log_a, log_b = a.clamp_min(1e-8).log(), b.clamp_min(1e-8).log()
                u, v = torch.zeros_like(a), torch.zeros_like(b)
                for _ in range(cfg.sinkhorn_iters):
                    u = log_a - torch.logsumexp(kernel + v[:, None, :], 2)
                    v = log_b - torch.logsumexp(kernel + u[:, :, None], 1)
                plan = (kernel + u[:, :, None] + v[:, None, :]).exp().clamp_min(1e-30)
                for _ in range(cfg.projection_iters):
                    plan = plan * (a[:, :, None] / plan.sum(-1, keepdim=True).clamp_min(1e-8))
                    plan = plan * (b[:, None, :] / plan.sum(-2, keepdim=True).clamp_min(1e-8))
                    error = torch.maximum((plan.sum(-1) - a).abs().amax(),
                                          (plan.sum(-2) - b).abs().amax())
                    if bool(error.detach() <= cfg.projection_tol):
                        break
                if not bool(torch.isfinite(plan).all()) or bool(error.detach() > cfg.projection_tol):
                    raise RuntimeError("transport coupling failed marginal tolerance; increase projection_iters")
                geometries.append(plan)
            stage_plans.append(torch.stack(geometries, 1))
        return torch.stack(stage_plans, 1), rows, cols

    def events_from_plans(self, wsi, omics, plans, *, frozen=False):
        call = frozen_call if frozen else lambda module, *args: module(*args)
        selected = []
        names = self.config.geometries
        for stage in range(self.config.stages):
            p = plans[:, stage]
            channels = [p[:, names.index(name) if name in names else 0]
                        for name in ("cosine", "euclidean", "positive_dot")]
            events, _ = call(self.fusion, wsi, omics, *channels)
            selected.append(events[:, stage:stage + 1])
        tokens = torch.cat(selected, 1) + self.stage_embedding.unsqueeze(0)
        tokens = call(self.event_norm, call(self.event_encoder, tokens))
        return tokens, torch.softmax(call(self.event_gate, tokens).squeeze(-1), 1)

    def _slots(self, tokens, mask, attention, prototypes):
        available = mask.any(-1)
        result = tokens.new_zeros(tokens.size(0), prototypes.size(0), tokens.size(-1))
        if bool(available.any()):
            indices = available.nonzero(as_tuple=True)[0]
            local = attention(tokens[indices], token_mask=mask[indices])
            slots = semantic_slots(local, prototypes, self.config.coordinate_temperature)
            result = result.index_copy(0, indices, slots.to(result.dtype))
        return result

    def forward(self, wsi, omics, *, wsi_mask, omics_mask, epoch=0):
        cfg = self.config
        sw = self._slots(wsi, wsi_mask, self.slot_attention_wsi, self.shared_wsi_prototypes)
        so = self._slots(omics, omics_mask, self.slot_attention_omic, self.shared_omic_prototypes)
        available = wsi_mask.any(-1) & omics_mask.any(-1)
        batch = wsi.size(0)
        plans = wsi.new_zeros(batch, cfg.stages, len(cfg.geometries), cfg.wsi_slots, cfg.omics_slots,
                             dtype=torch.float32 if wsi.dtype in (torch.float16, torch.bfloat16) else wsi.dtype)
        rows = plans.new_zeros(batch, cfg.stages, cfg.wsi_slots)
        cols = plans.new_zeros(batch, cfg.stages, cfg.omics_slots)
        events = wsi.new_zeros(batch, cfg.stages, cfg.dim)
        weights = wsi.new_zeros(batch, cfg.stages)
        if bool(available.any()):
            indices = available.nonzero(as_tuple=True)[0]
            costs, a, b = self.costs_and_marginals(sw[indices], so[indices])
            p, a, b = self.plans_from_costs(costs, a, b, epoch)
            e, g = self.events_from_plans(sw[indices], so[indices], p)
            plans = plans.index_copy(0, indices, p.to(plans.dtype))
            rows, cols = rows.index_copy(0, indices, a), cols.index_copy(0, indices, b)
            events = events.index_copy(0, indices, e.to(events.dtype))
            weights = weights.index_copy(0, indices, g.to(weights.dtype))
        # FP32 barycentric normalization prevents 0/0 on absent patients under
        # autocast; the epsilon cannot be represented as an FP16 denominator.
        with torch.autocast(device_type=wsi.device.type, enabled=False):
            consensus = plans.mean(2)
            moved_wsi = torch.einsum("bswo,bwd->bsod", consensus, sw.to(plans.dtype))
            moved_wsi = moved_wsi / consensus.sum(-2).unsqueeze(-1).clamp_min(1e-8)
            moved_omics = torch.einsum("bswo,bod->bswd", consensus, so.to(plans.dtype))
            moved_omics = moved_omics / consensus.sum(-1).unsqueeze(-1).clamp_min(1e-8)
            transported_wsi = torch.einsum("bs,bsod->bod", weights.detach().to(plans.dtype), moved_wsi)
            transported_omics = torch.einsum("bs,bswd->bwd", weights.detach().to(plans.dtype), moved_omics)
        return TransportState(wsi, omics, wsi_mask, omics_mask, sw, so, plans, rows, cols,
                              events, weights,
                              transported_wsi.to(wsi.dtype), transported_omics.to(omics.dtype), available)


def realized_independent_plan(plan):
    """Remove topology without changing positive realized mass or marginals."""
    if plan.ndim < 2 or not plan.is_floating_point():
        raise ValueError("plan must be a floating transport matrix")
    if not bool(torch.isfinite(plan).all()) or bool((plan < 0).any()):
        raise ValueError("plan must be finite and nonnegative")
    rows, cols = plan.sum(-1), plan.sum(-2)
    mass = rows.sum(-1)
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    # Normalized factors avoid underflow in r*c for very small positive mass.
    return mass[..., None, None] * (rows / safe_mass[..., None])[..., :, None] * (
        cols / safe_mass[..., None])[..., None, :]


@dataclass
class TransportStateV2(TransportState):
    transport_mass: torch.Tensor
    wsi_confidence: torch.Tensor
    omics_confidence: torch.Tensor
    writeback_confidence: torch.Tensor


class DCTV2Transport(DCTFeatureCore):
    def __init__(self, config: DCTV2Config):
        if not isinstance(config, DCTV2Config):
            raise TypeError("DCTV2Transport requires its versioned configuration")
        super().__init__(config)
        self.last_solver_diagnostics = {}


    def _costs_and_marginals(self, wsi, omics):
        cfg = self.config
        if cfg.cost_mode == "legacy":
            return super()._costs_and_marginals(wsi, omics)
        contents = pair_tokens(wsi, omics)
        stage_cost = torch.sigmoid(self.stage_pair_cost(contents)).permute(0, 3, 1, 2)
        cosine = torch.bmm(F.normalize(wsi, dim=-1), F.normalize(omics, dim=-1).transpose(1, 2))
        distance = torch.cdist(wsi, omics, p=2)
        base = {
            "cosine": (1. - cosine.clamp(-1., 1.)) / 2.,
            # Raw norms remain informative; this is not normalized squared L2,
            # which would be algebraically identical to the cosine branch.
            "euclidean": distance / (distance + cfg.euclidean_scale),
            "positive_dot": torch.sigmoid(-torch.bmm(wsi, omics.transpose(1, 2)) / math.sqrt(cfg.dim)),
        }
        costs, rows, cols = [], [], []
        for stage in range(cfg.stages):
            code = self.stage_embedding[stage].view(1, 1, 1, -1).expand(*contents.shape[:-1], cfg.dim)
            gate = torch.sigmoid(self.evidence_gate(torch.cat((contents, code), -1)).squeeze(-1))
            costs.append(cfg.cost_scale * torch.stack([
                base[name] + cfg.stage_cost_weight * stage_cost[:, stage]
                + cfg.evidence_cost_weight * (1. - gate) for name in cfg.geometries], 1))
            rows.append(gate.mean(-1).clamp_min(cfg.evidence_mass_floor))
            cols.append(gate.mean(-2).clamp_min(cfg.evidence_mass_floor))
        rows, cols = torch.stack(rows, 1), torch.stack(cols, 1)
        rows, cols = rows / rows.sum(-1, keepdim=True), cols / cols.sum(-1, keepdim=True)
        strength = cfg.marginal_strength
        return torch.stack(costs, 1), ((1.-strength) / rows.size(-1) + strength * rows), (
            (1.-strength) / cols.size(-1) + strength * cols)

    def plans_from_costs(self, costs, rows, cols, epoch):
        cfg = self.config
        if cfg.transport_mode == "balanced":
            self.last_solver_diagnostics = {"mode": "balanced", "actual_marginal_control": False}
            return super().plans_from_costs(costs, rows, cols, epoch)
        if self.plan_mode not in {"learned", "independent"}:
            raise ValueError("v2 plan_mode must be learned or independent")
        if costs.ndim != 5 or rows.shape != (costs.size(0), costs.size(1), costs.size(3)) or (
            cols.shape != (costs.size(0), costs.size(1), costs.size(4))):
            raise ValueError("expected costs [B,S,G,W,O], rows [B,S,W], cols [B,S,O]")
        if not bool(torch.isfinite(costs).all()) or not bool(torch.isfinite(rows).all()) or not bool(
            torch.isfinite(cols).all()) or bool((rows <= 0).any()) or bool((cols <= 0).any()):
            raise ValueError("costs must be finite and reference marginals finite and strictly positive")
        dtype = torch.float64 if costs.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=costs.device.type, enabled=False):
            c, a, b = costs.to(dtype), rows.to(dtype), cols.to(dtype)
            a, b = a / a.sum(-1, keepdim=True), b / b.sum(-1, keepdim=True)
            log_a, log_b = a.log().unsqueeze(2), b.log().unsqueeze(2)
            eps = self.epsilon(epoch)
            log_kernel = log_a[..., :, None] + log_b[..., None, :] - c / eps
            theta_h, theta_g = cfg.tau_wsi / (cfg.tau_wsi + eps), cfg.tau_omics / (cfg.tau_omics + eps)
            u, v = torch.zeros_like(log_kernel[..., 0]), torch.zeros_like(log_kernel[..., 0, :])
            effective_tolerance = max(cfg.uot_tolerance, 8. * torch.finfo(dtype).eps)
            potential_scale = 1.
            change = float("inf")
            for iteration in range(cfg.uot_iters):
                old_u, old_v = u, v
                u = theta_h * (log_a - torch.logsumexp(log_kernel + v[..., None, :], -1))
                v = theta_g * (log_b - torch.logsumexp(log_kernel + u[..., :, None], -2))
                if (iteration + 1) % 10 == 0 or iteration + 1 == cfg.uot_iters:
                    potential_scale = float(torch.maximum(u.abs().amax(), v.abs().amax()).detach().clamp_min(1.))
                    effective_tolerance = max(cfg.uot_tolerance, 8. * torch.finfo(dtype).eps * potential_scale)
                    change = torch.maximum((u-old_u).abs().amax(), (v-old_v).abs().amax()).detach()
                    if bool(change <= effective_tolerance):
                        break
            if not bool(change <= effective_tolerance):
                raise RuntimeError("UOT did not converge within uot_iters; increase the budget or recalibrate tau/eps")
            plan = (log_kernel + u[..., :, None] + v[..., None, :]).exp()
            if not bool(torch.isfinite(plan).all()):
                raise RuntimeError("UOT produced a nonfinite plan; check cost/tau/eps scales")
            self.last_solver_diagnostics = {"mode": "uot", "iterations": iteration + 1,
                "max_log_update": float(change), "effective_tolerance": effective_tolerance,
                "requested_tolerance": cfg.uot_tolerance, "max_log_potential": potential_scale,
                "converged": True,
                "actual_marginal_control": self.plan_mode == "independent"}
            if self.plan_mode == "independent":
                plan = realized_independent_plan(plan)
            return plan, a, b

    def forward(self, wsi, omics, *, wsi_mask, omics_mask, epoch=0):
        # v1 dispatches to this version's cost and solver methods. It still owns
        # slot extraction and event networks, so their parameter layout is fixed.
        state = super().forward(wsi, omics, wsi_mask=wsi_mask, omics_mask=omics_mask, epoch=epoch)
        consensus = state.plans.mean(2)
        actual_rows, actual_cols = consensus.sum(-1), consensus.sum(-2)
        mass = actual_rows.sum(-1)
        a, b = state.row_marginals, state.col_marginals
        safe_a = torch.where(a > 0, a, torch.ones_like(a))
        safe_b = torch.where(b > 0, b, torch.ones_like(b))
        row_gate, col_gate = (actual_rows / safe_a).clamp(0., 1.), (actual_cols / safe_b).clamp(0., 1.)
        weights = state.stage_weights.to(mass.dtype)
        wsi_confidence = torch.einsum("bs,bsw->bw", weights.detach(), row_gate)
        omics_confidence = torch.einsum("bs,bso->bo", weights.detach(), col_gate)
        confidence = (weights * mass.clamp(0., 1.)).sum(-1)
        values = {field.name: getattr(state, field.name) for field in fields(TransportState)}
        if self.config.mass_aware:
            # Gate BEFORE averaging stages: a well-matched stage must not
            # lend its strength to the normalized feature of a weak stage.
            with torch.autocast(device_type=wsi.device.type, enabled=False):
                moved_wsi = torch.einsum("bswo,bwd->bsod", consensus, state.wsi_slots.to(consensus.dtype))
                moved_omics = torch.einsum("bswo,bod->bswd", consensus, state.omics_slots.to(consensus.dtype))
                moved_wsi = moved_wsi / actual_cols[..., None].clamp_min(1e-8)
                moved_omics = moved_omics / actual_rows[..., None].clamp_min(1e-8)
                values["transported_wsi"] = torch.einsum(
                    "bs,bsod->bod", weights.detach(), moved_wsi * col_gate[..., None]).to(wsi.dtype)
                values["transported_omics"] = torch.einsum(
                    "bs,bswd->bwd", weights.detach(), moved_omics * row_gate[..., None]).to(omics.dtype)
        return TransportStateV2(**values, transport_mass=mass, wsi_confidence=wsi_confidence,
                                omics_confidence=omics_confidence, writeback_confidence=confidence)


def fixed_realized_marginal_plan(costs: Tensor, rows: Tensor, cols: Tensor, *,
                                 epsilon: float, iterations: int = 1000,
                                 tolerance: float = 1e-6, min_mass: float = 1e-8) -> Tensor:
    """Differentiable entropy projection with detached actual margins.

    Row/column factors in the KL reference can be absorbed by dual scalings at
    fixed margins, so the kernel exp(-C/epsilon) has the same minimizer. Tiny
    positive masses below min_mass retain an independent feasible plan and have
    zero cost gradient; zero rows/columns remain exactly zero. A failed active
    solve raises instead of silently substituting another matching mechanism.
    """
    if (costs.ndim != 5 or rows.shape != costs.shape[:3] + costs.shape[3:4]
            or cols.shape != costs.shape[:3] + costs.shape[4:5]):
        raise ValueError("expected costs [B,S,G,W,O], actual rows [B,S,G,W], cols [B,S,G,O]")
    for name, value, positive in (("epsilon", epsilon, True), ("tolerance", tolerance, True), ("min_mass", min_mass, False)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"invalid {name}")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be positive")
    if not all(value.is_floating_point() and value.device == costs.device for value in (costs, rows, cols)):
        raise ValueError("costs and margins require floating tensors on the same device")
    if not all(bool(torch.isfinite(value).all()) for value in (costs, rows, cols)) or bool((rows < 0).any()) or bool((cols < 0).any()):
        raise ValueError("costs must be finite and actual margins finite/nonnegative")
    dtype = torch.float64 if costs.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=costs.device.type, enabled=False):
        c = costs.to(dtype).reshape(-1, costs.size(-2), costs.size(-1))
        r = rows.detach().to(dtype).reshape(-1, costs.size(-2))
        o = cols.detach().to(dtype).reshape(-1, costs.size(-1))
        mass_r, mass_o = r.sum(-1), o.sum(-1)
        if not bool(torch.isfinite(mass_r).all()) or not bool(torch.isfinite(mass_o).all()):
            raise ValueError("actual total mass must be finite")
        mass_scale = torch.maximum(mass_r, mass_o)
        safe_scale = torch.where(mass_scale > 0, mass_scale, torch.ones_like(mass_scale))
        relative_mass_error = (mass_r-mass_o).abs() / safe_scale
        if bool(((mass_r > 0) != (mass_o > 0)).any()) or bool((relative_mass_error > 1e-5).any()):
            raise ValueError("actual row and column mass must agree")
        safe_r = torch.where(mass_r > 0, mass_r, torch.ones_like(mass_r))
        safe_o = torch.where(mass_o > 0, mass_o, torch.ones_like(mass_o))
        rn, on = r / safe_r[:, None], o / safe_o[:, None]
        result = (mass_r[:, None, None] * rn[:, :, None] * on[:, None, :]).detach() + c * 0.
        active = (mass_r > min_mass).nonzero(as_tuple=True)[0]
        if active.numel():
            rr, oo = rn[active], on[active]
            kernel = -c[active] / epsilon
            # A constant offset is valid HERE because actual total mass is fixed.
            kernel = kernel - kernel.amax((-2, -1), keepdim=True).detach()
            tiny = torch.finfo(dtype).tiny
            log_r, log_o = rr.clamp_min(tiny).log(), oo.clamp_min(tiny).log()
            u = torch.where(rr > 0, torch.zeros_like(rr), torch.full_like(rr, -torch.inf))
            v = torch.where(oo > 0, torch.zeros_like(oo), torch.full_like(oo, -torch.inf))
            effective = max(tolerance, 8 * torch.finfo(dtype).eps)
            error = math.inf
            for iteration in range(iterations):
                next_u = log_r - torch.logsumexp(kernel + v[:, None, :], -1)
                u = torch.where(rr > 0, next_u, torch.full_like(next_u, -torch.inf))
                next_v = log_o - torch.logsumexp(kernel + u[:, :, None], -2)
                v = torch.where(oo > 0, next_v, torch.full_like(next_v, -torch.inf))
                if (iteration + 1) % 10 == 0 or iteration + 1 == iterations:
                    q = (kernel + u[:, :, None] + v[:, None, :]).exp()
                    error = torch.maximum((q.sum(-1) - rr).abs().amax(), (q.sum(-2) - oo).abs().amax()).detach()
                    if bool(error <= effective):
                        break
            if not bool(error <= effective):
                raise RuntimeError("fixed-realized-marginal projection did not converge")
            plan = q * mass_r[active, None, None]
            if not bool(torch.isfinite(plan).all()):
                raise RuntimeError("fixed-realized-marginal projection produced nonfinite values")
            result = result.index_copy(0, active, plan)
        return result.reshape_as(costs)
