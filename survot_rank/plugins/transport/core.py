"""Feature-only extraction of DCT's prototype, cost, plan and event computations.

The equations follow the prediction path at commit 530f87b. No survival head,
training reference, intervention anchor, patient file or reconstruction decoder
is owned by this module. The host-level residual bridge is a new integration.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from survot_rank.research.components.slot_attention import MultiHeadSlotAttention
from .config import TransportPluginConfig


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


class TransportEventReadout(nn.Module):
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


class TransportCore(nn.Module):
    def __init__(self, config: TransportPluginConfig):
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
        self.fusion = TransportEventReadout(dim, config.stages, config.heads, config.dropout)
        encoder = nn.TransformerEncoderLayer(dim, config.heads, dim * 2, config.dropout,
                                             activation="gelu", batch_first=True, norm_first=True)
        self.event_encoder = nn.TransformerEncoder(encoder, config.event_layers)
        self.event_norm = nn.LayerNorm(dim)
        self.event_gate = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(),
                                        nn.Dropout(config.dropout), nn.Linear(dim // 2, 1))

    @classmethod
    def from_dct(cls, model):
        """Copy only feature computations from an existing DCT model.

        Inputs must already be encoded by the original WSI/omics encoders.
        The caller still owns the source model and its prediction heads.
        """
        if float(getattr(model, "dct_geometry_reliability_strength", 0.0)) != 0.0:
            raise ValueError("geometry reliability is not part of the extracted plugin recipe")
        args = model.args
        dim = int(model.wsi_projection_dim)
        def slot_recipe(slot):
            heads = int(slot.split_heads.axes_lengths["h"])
            return (heads, slot.to_q.out_features // heads, slot.mlp[0].out_features,
                    slot.iters, slot.init_mode)
        slot_heads, slot_dim_head, slot_hidden_dim, _, _ = slot_recipe(model.slot_attention_wsi)
        if slot_recipe(model.slot_attention_wsi) != slot_recipe(model.slot_attention_omic):
            raise ValueError("exported WSI and omics slots require the same configuration")
        config = TransportPluginConfig(
            wsi_dim=dim, omics_dim=dim, dim=dim,
            wsi_slots=model.shared_wsi_prototypes.size(0),
            omics_slots=model.shared_omic_prototypes.size(0), stages=model.spt_num_stages,
            heads=int(getattr(args, "otehv2_heads", 4)),
            event_layers=len(model.event_encoder.layers),
            slot_iters=model.slot_attention_wsi.iters,
            slot_heads=slot_heads, slot_dim_head=slot_dim_head, slot_hidden_dim=slot_hidden_dim,
            dropout=float(getattr(args, "otehv2_dropout", .15)),
            slot_init_mode=model.slot_attention_wsi.init_mode,
            slot_eval_seed=int(getattr(args, "dct_slot_eval_seed", 1729)),
            coordinate_temperature=model.dct_coordinate_temperature,
            stage_cost_weight=model.spt_prog_cost_weight,
            evidence_cost_weight=model.dct_evidence_cost_weight,
            evidence_mass_floor=model.dct_evidence_mass_floor,
            marginal_strength=model.dct_evidence_marginal_strength,
            eps=float(model.ot_eps), eps_start=float(getattr(args, "rg_eps_start", model.ot_eps * 2)),
            eps_anneal_epochs=int(getattr(args, "rg_eps_anneal", 12)),
            sinkhorn_iters=model.ot_iter,
            projection_iters=model.dct_coupling_projection_iters,
            projection_tol=model.dct_coupling_projection_tol,
            plan_mode=str(getattr(model, "_plan_mode", "learned")),
        )
        # Construction of the exporter must not perturb the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            result = cls(config)
        for name in ("slot_attention_wsi", "slot_attention_omic", "stage_pair_cost",
                     "evidence_gate", "fusion", "event_encoder", "event_norm", "event_gate"):
            setattr(result, name, copy.deepcopy(getattr(model, name)))
        result.shared_wsi_prototypes = nn.Parameter(model.shared_wsi_prototypes.detach().clone(),
                                                    requires_grad=model.shared_wsi_prototypes.requires_grad)
        result.shared_omic_prototypes = nn.Parameter(model.shared_omic_prototypes.detach().clone(),
                                                     requires_grad=model.shared_omic_prototypes.requires_grad)
        result.stage_embedding = model.stage_embedding.detach().clone()
        result.train(model.training)
        return result

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

    def events_from_plans(self, wsi, omics, plans):
        selected = []
        names = self.config.geometries
        for stage in range(self.config.stages):
            p = plans[:, stage]
            channels = [p[:, names.index(name) if name in names else 0]
                        for name in ("cosine", "euclidean", "positive_dot")]
            events, _ = self.fusion(wsi, omics, *channels)
            selected.append(events[:, stage:stage + 1])
        tokens = torch.cat(selected, 1) + self.stage_embedding.unsqueeze(0)
        tokens = self.event_norm(self.event_encoder(tokens))
        return tokens, torch.softmax(self.event_gate(tokens).squeeze(-1), 1)

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
