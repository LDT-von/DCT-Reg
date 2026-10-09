"""Version-local DCT transport machinery.

Transport/reader equations are adapted from the repository's DCT implementation
(8180e17). No runtime imports from any older method package. Historical loss
heads and intervention-training losses are intentionally not instantiated.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from survot_rank.research.components.omics_encoder import SNN_Block, WSI_Mlp


def cosine_cost(x, y):
    return 1.0 - F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).transpose(-1, -2)


def euclidean_cost(x, y):
    # vector_norm has a defined zero subgradient at identical vectors.
    return torch.linalg.vector_norm(x.unsqueeze(2) - y.unsqueeze(1), dim=-1)


class PrototypeSlotAttention(nn.Module):
    """Persistent prototype queries; outputs pool actual patient tokens.

    Identity affects routing only. It is never added to output values, so
    distinct prototype vectors cannot alone satisfy a content-diversity loss.
    """
    def __init__(self, dim, num_slots, iters):
        super().__init__()
        self.num_slots, self.iters = num_slots, iters
        if num_slots < 1 or iters < 1:
            raise ValueError("slot count and iterations must be positive")
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.last_pooling_attention = None

    def forward(self, inputs, prototypes):
        if inputs.ndim != 3 or inputs.size(1) < 1:
            raise ValueError("slot input must be a nonempty [B, N, D] tensor")
        identity = F.normalize(prototypes, dim=-1).unsqueeze(0)
        states = inputs.mean(1, keepdim=True).expand(-1, self.num_slots, -1)
        keys = F.normalize(self.to_k(self.norm_input(inputs)), dim=-1)
        for _ in range(self.iters):
            # A recurrent content query refines, but cannot erase, slot identity.
            content_query = F.normalize(self.to_q(self.norm_slots(states)), dim=-1)
            query = F.normalize(content_query + 2.0 * identity, dim=-1)
            # Unit-vector dot products shrink as 1/sqrt(D). A fixed cosine
            # temperature made D=256 routing nearly uniform even when D=32
            # smoke tests looked healthy. Keep logit scale dimension-stable.
            scores = torch.einsum("bkd,bnd->bkn", query, keys) * math.sqrt(inputs.size(-1)) / 0.5
            assignment = scores.softmax(dim=1)
            pooling = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-8)
            content = pooling @ inputs
            states = self.gru(content.reshape(-1, inputs.size(-1)),
                              states.reshape(-1, inputs.size(-1))).view_as(states)
        self.last_pooling_attention = pooling.detach()
        return content, pooling


class MultiScaleOTFusion(nn.Module):
    """Three cost matrices → three OT plans → concat → cross-attention fusion."""

    def __init__(self, dim, num_events=16, nhead=4, dropout=0.1):
        super().__init__()
        self.num_events = num_events
        self.cost_convs = nn.ModuleDict({
            "cosine": nn.Linear(1, dim),
            "euclidean": nn.Sequential(
                nn.Linear(1, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, dim),
            ),
            "dot": nn.Linear(1, dim),
        })
        self.proj = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)

        self.event_queries = nn.Parameter(torch.randn(num_events, dim) * 0.02)

        self.cross_attn = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )

        self.refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _build_pair_tokens(self, slots_wsi, slots_omic):
        bsz, sw, so, dim = slots_wsi.shape[0], slots_wsi.shape[1], slots_omic.shape[1], slots_wsi.shape[-1]
        w = slots_wsi.unsqueeze(2).expand(bsz, sw, so, dim)
        o = slots_omic.unsqueeze(1).expand(bsz, sw, so, dim)
        return torch.cat([w, o, w * o, (w - o).abs()], dim=-1)

    def forward(self, slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot):
        bsz, sw, dim = slots_wsi.shape
        so = slots_omic.shape[1]

        pair_tokens = self._build_pair_tokens(slots_wsi, slots_omic)  # [B, sw, so, dim*4]

        # Project each cost type
        c_cos = self.cost_convs["cosine"](plan_cos.unsqueeze(-1))
        c_euc = self.cost_convs["euclidean"](plan_euc.unsqueeze(-1))
        c_dot = self.cost_convs["dot"](plan_dot.unsqueeze(-1))
        cost_concat = torch.cat([c_cos, c_euc, c_dot], dim=-1)  # [B, sw*so, dim*3]
        pair_context = pair_tokens[..., : dim * 3]
        pair_tokens = self.proj(cost_concat + pair_context)

        # Aggregate into events via attention
        pair_tokens = pair_tokens.reshape(bsz, sw * so, dim)
        pair_mass = plan_cos.reshape(bsz, sw * so).clamp_min(1e-8).log().unsqueeze(-1)
        q = F.normalize(self.event_queries, dim=-1)
        t = F.normalize(pair_tokens, dim=-1)
        scores = torch.einsum("kd,bpd->bpk", q, t)
        scores = scores + pair_mass
        assign = torch.softmax(scores.transpose(1, 2), dim=-1)
        events = torch.bmm(assign, pair_tokens)

        # Cross-attention refinement (WSI ↔ Omic bidirectional)
        events = self.norm(self.cross_attn(events))

        return events + self.refine(events), assign


class TransportBackbone(nn.Module):
    _LOW_RISK, _HIGH_RISK = 0, 1

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__()
        self.args = args
        self.wsi_embedding_dim = int(args.encoding_dim)
        self.wsi_projection_dim = dim = int(args.wsi_projection_dim)
        self.num_classes = int(args.n_classes)
        self.omic_sizes = list(args.omic_sizes or [])
        self.num_pathways = len(self.omic_sizes)
        if not self.num_pathways or any(int(n) < 1 for n in self.omic_sizes):
            raise ValueError("v3.14 requires nonempty, positive omic_sizes")
        heads = int(getattr(args, "otehv2_heads", 4))
        if dim < 2 or heads < 1 or dim % heads:
            raise ValueError("projection dimension must be >= 2 and divisible by otehv2_heads")
        self.spt_num_stages = self.num_events = int(getattr(args, "dct_num_stages", 4))
        self.ot_iter = int(getattr(args, "otehv2_iter", 50))
        self.dct_coupling_projection_iters = int(getattr(args, "dct_coupling_projection_iters", 1000))
        self.dct_coupling_projection_tol = float(getattr(args, "dct_coupling_projection_tol", 1e-4))
        if min(self.spt_num_stages, self.ot_iter, self.dct_coupling_projection_iters) < 1:
            raise ValueError("transport stages and iteration counts must be positive")
        positive_values = (float(getattr(args, "otehv2_eps", .05)), float(getattr(args, "rg_eps_start", .1)),
                           self.dct_coupling_projection_tol)
        if any(not math.isfinite(value) or value <= 0 for value in positive_values):
            raise ValueError("transport epsilon and tolerance must be positive")
        self.spt_prog_cost_weight = float(getattr(args, "spt_prog_cost", .2))
        if not math.isfinite(self.spt_prog_cost_weight) or self.spt_prog_cost_weight < 0:
            raise ValueError("prognostic cost weight must be finite and nonnegative")
        if self.wsi_embedding_dim < 1 or self.num_classes < 1:
            raise ValueError("input dimension and number of time bins must be positive")
        if int(getattr(args, "otehv2_layers", 2)) < 1:
            raise ValueError("event encoder must have at least one layer")
        self.dct_ipcw_rank_margin = .02
        self.dct_ipcw_rank_temperature = .5
        self.dct_ipcw_max_weight = 10.
        self.dct_ipcw_rank_memory_size = 64
        self.dct_anchor_momentum = .9
        self.dct_evidence_cost_weight = 0.
        self.dct_evidence_mass_floor = .05
        self.dct_evidence_marginal_strength = 1.
        self.dct_geometry_reliability_strength = 0.
        self.dct_fixed_coupling = False
        self.dct_random_anchors = False
        self.dct_perm_labels_seed = 0
        self.dct_stage_jitter_fraction = 0.
        self.dct_mix_ratio = 1.
        self._factual_plan_cache = None
        self._last_transport_reliability = None
        self._rank_memory_epoch = None
        self._rank_memory_risk = self._rank_memory_times = self._rank_memory_censorship = None

        self.sig_networks = nn.ModuleList([
            nn.Sequential(SNN_Block(n, dim), SNN_Block(dim, dim)) for n in self.omic_sizes
        ])
        self.wsi_mlp = WSI_Mlp(self.wsi_embedding_dim, dim)
        sw, so = int(args.slot_num_wsi), int(args.slot_num_omics)
        self.shared_wsi_prototypes = nn.Parameter(torch.randn(sw, dim) * .02)
        self.shared_omic_prototypes = nn.Parameter(torch.randn(so, dim) * .02)
        self.slot_attention_wsi = PrototypeSlotAttention(dim, sw, int(args.slot_iters))
        self.slot_attention_omic = PrototypeSlotAttention(dim, so, int(args.slot_iters))
        dropout = float(getattr(args, "otehv2_dropout", .15))
        self.fusion = MultiScaleOTFusion(dim, self.spt_num_stages, heads, dropout)
        layer = nn.TransformerEncoderLayer(dim, heads, dim*2, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.event_encoder = nn.TransformerEncoder(layer, int(getattr(args, "otehv2_layers", 2)))
        self.event_norm = nn.LayerNorm(dim)
        self.event_hazard = nn.Linear(dim, self.num_classes)
        self.event_gate = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(dim//2, 1))
        self.stage_pair_cost = nn.Sequential(nn.LayerNorm(dim*4), nn.Linear(dim*4, dim),
                                            nn.GELU(), nn.Linear(dim, self.spt_num_stages))
        self.evidence_gate = nn.Sequential(nn.LayerNorm(dim*5), nn.Linear(dim*5, dim),
                                          nn.GELU(), nn.Linear(dim, 1))
        # Nonconstant feature directions survive LayerNorm; scalar offsets do not.
        positions = torch.arange(self.spt_num_stages).float().unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.)/dim))
        stage_embedding = torch.zeros(self.spt_num_stages, dim)
        stage_embedding[:, 0::2] = torch.sin(positions * frequencies)
        stage_embedding[:, 1::2] = torch.cos(positions * frequencies[:dim//2])
        self.register_buffer("stage_embedding", stage_embedding)
        self.register_buffer("risk_anchor_costs", torch.zeros(self.spt_num_stages, 2, 3, sw, so))
        self.register_buffer("risk_anchor_seen", torch.zeros(self.spt_num_stages, 2, dtype=torch.bool))
        self.register_buffer("dct_stage_edges", torch.empty(0))
        self.register_buffer("dct_censor_times", torch.empty(0))
        self.register_buffer("dct_censor_survival", torch.empty(0))
        # Sinkhorn epsilon is annealed by epoch. Preserve the trained epoch so
        # a freshly constructed inference model does not silently use eps(0).
        self.register_buffer("transport_epoch", torch.tensor(int(getattr(args, "cur_epoch", 0)), dtype=torch.long))
        self.last_training_losses = {}
        self.last_explanations = None

    def _encode_omics(self, kwargs):
        return torch.stack([encoder(kwargs[f"x_omic{i+1}"])
                            for i, encoder in enumerate(self.sig_networks)], dim=1)

    def _encode_transport_slots(self, x_wsi_proj, x_omics, kwargs):
        w, wa = self.slot_attention_wsi(x_wsi_proj, self.shared_wsi_prototypes)
        o, oa = self.slot_attention_omic(x_omics, self.shared_omic_prototypes)
        return w, o, wa, oa

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ("dct_stage_edges", "dct_censor_times", "dct_censor_survival"):
            key = prefix + name
            if key in state_dict:
                self._buffers[name] = torch.empty_like(state_dict[key], device=self.risk_anchor_costs.device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        # The shared batch adapter explicitly forwards args.cur_epoch even in
        # inference; synchronize it as well as the buffer after checkpoint load.
        self.args.cur_epoch = int(self.transport_epoch.item())
        self._rank_memory_epoch = None
        self._rank_memory_risk = self._rank_memory_times = self._rank_memory_censorship = None

    @staticmethod
    def _risk(logits):
        hazards = torch.sigmoid(logits)
        return -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)

    @property
    def has_train_reference(self):
        return self.dct_stage_edges.numel() == self.spt_num_stages + 1

    @torch.no_grad()
    def configure_train_reference(self, event_times, censorship):
        """Fit time stages and IPCW censoring survival from one fold's train set.

        ``c == 0`` denotes an observed event and ``c == 1`` denotes censoring.
        Stage upper edges are event-time quantiles.  The final edge is finite so
        patients followed beyond it can contribute to the final low-risk anchor.

        Ablation ``dct_stage_jitter_fraction`` permutes the upper edges within a
        bounded range while preserving monotonicity, proving that the actual
        edge placement—not the existence of stages—carries the IPCW signal.
        Ablation ``dct_perm_labels_seed`` permutes ``event_times`` before edge
        fitting so that the null-calibration model never sees consistent
        censoring.
        """
        device = self.risk_anchor_costs.device
        times = torch.as_tensor(event_times, dtype=torch.float32, device=device).flatten()
        cens = torch.as_tensor(censorship, dtype=torch.float32, device=device).flatten()
        if times.numel() == 0 or times.shape != cens.shape or not torch.isfinite(times).all():
            raise ValueError("training reference needs matching, nonempty finite times/censorship")
        if not torch.all((cens == 0) | (cens == 1)) or (times < 0).any():
            raise ValueError("training reference requires nonnegative times and binary censorship")
        if self.dct_perm_labels_seed > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.dct_perm_labels_seed)
            times = times[torch.randperm(times.numel(), generator=generator, device=device)]
        observed = times[cens < 0.5]
        if observed.numel() < self.spt_num_stages:
            raise ValueError(
                "DCT needs at least dct_num_stages observed training events to fit stage anchors."
            )
        quantiles = torch.linspace(
            1.0 / self.spt_num_stages, 1.0, self.spt_num_stages, device=device
        )
        upper = torch.quantile(observed, quantiles)
        if self.dct_stage_jitter_fraction > 0.0:
            spread = (upper[-1] - upper[0]).clamp_min(1e-3)
            jitter = torch.empty_like(upper).uniform_(
                -self.dct_stage_jitter_fraction,
                self.dct_stage_jitter_fraction,
                device=device,
            )
            upper = upper + jitter * spread
        # Strictly increasing edges make stage membership deterministic even with ties.
        upper = torch.maximum(upper, torch.cummax(upper, dim=0).values)
        for idx in range(1, upper.numel()):
            upper[idx] = torch.maximum(upper[idx], torch.nextafter(upper[idx - 1], upper.new_tensor(float("inf"))))
        self.dct_stage_edges = torch.cat([upper.new_tensor([-float("inf")]), upper])

        unique_times = torch.unique(times, sorted=True)
        censor_survival = torch.ones_like(unique_times)
        value = torch.ones((), dtype=times.dtype, device=device)
        for idx, time in enumerate(unique_times):
            # Reverse Kaplan-Meier: observed events leave the risk set before
            # censoring at tied times, matching sksurv's censoring estimator.
            tied_observed = ((times == time) & (cens < 0.5)).sum()
            at_risk = ((times >= time).sum() - tied_observed).to(times.dtype).clamp_min(1.0)
            censor_events = ((times == time) & (cens >= 0.5)).sum().to(times.dtype)
            value = value * (1.0 - censor_events / at_risk)
            censor_survival[idx] = value
        self.dct_censor_times = unique_times
        self.dct_censor_survival = censor_survival.clamp_min(0.05)
        self.risk_anchor_costs.zero_()
        self.risk_anchor_seen.zero_()
        self._rank_memory_epoch = None
        self._rank_memory_risk = self._rank_memory_times = self._rank_memory_censorship = None

    def _ipcw(self, query_times, *, before=False):
        query_times = torch.as_tensor(query_times, device=self.dct_censor_times.device, dtype=torch.float32)
        if self.dct_censor_times.numel() == 0:
            return torch.ones_like(query_times)
        indices = torch.searchsorted(self.dct_censor_times, query_times.contiguous(), right=not before) - 1
        # G(t)=1 before the first observed follow-up time. Clamping -1 to zero
        # would incorrectly reuse the first post-time KM value.
        values = torch.ones_like(query_times)
        valid = indices >= 0
        values[valid] = self.dct_censor_survival[indices[valid]]
        return values.clamp_min(0.05).reciprocal()

    def _sinkhorn_eps(self, epoch):
        end = float(getattr(self.args, "otehv2_eps", 0.05))
        start = float(getattr(self.args, "rg_eps_start", end * 2.0))
        anneal = max(1, int(getattr(self.args, "rg_eps_anneal", 12)))
        return start + min(1.0, epoch / anneal) * (end - start)

    def _cost_tensor(self, slots_wsi, slots_omic):
        # Keep distances and Sinkhorn inputs in FP32 under AMP.
        with torch.autocast(device_type=slots_wsi.device.type, enabled=False):
            return self._cost_tensor_float(slots_wsi.float(), slots_omic.float())

    def _cost_tensor_float(self, slots_wsi, slots_omic):
        """Return stage costs and evidence-conditioned OT marginals."""
        pair_tokens = self._pair_tokens(slots_wsi, slots_omic)
        bsz, sw, so, dim4 = pair_tokens.shape
        dim = dim4 // 4
        stage_cost = F.softplus(self.stage_pair_cost(pair_tokens)).permute(0, 3, 1, 2)
        base_costs = (
            self._normalize_cost(cosine_cost(slots_wsi, slots_omic)),
            self._normalize_cost(euclidean_cost(slots_wsi, slots_omic)),
            self._normalize_cost(self._positive_dot_cost(slots_wsi, slots_omic)),
        )

        all_stage_costs, row_marginals, col_marginals, gates = [], [], [], []
        reliabilities = []
        for stage_idx in range(self.spt_num_stages):
            stage_code = self.stage_embedding[stage_idx].view(1, 1, 1, dim)
            stage_code = stage_code.expand(bsz, sw, so, dim)
            gate = torch.sigmoid(
                self.evidence_gate(torch.cat([pair_tokens, stage_code], dim=-1)).squeeze(-1)
            )
            evidence_cost = self._normalize_stage_cost(-torch.log(gate.clamp_min(1e-6)))
            prognostic_cost = self._normalize_stage_cost(stage_cost[:, stage_idx])
            current_stage_costs = torch.stack([
                base_cost
                + self.spt_prog_cost_weight * prognostic_cost
                + self.dct_evidence_cost_weight * evidence_cost
                for base_cost in base_costs
            ], dim=1)
            all_stage_costs.append(current_stage_costs)
            if self.dct_geometry_reliability_strength > 0.0:
                reliabilities.append(self._geometry_reliability(current_stage_costs))
            # Gate affects both the energy and how much each semantic slot is
            # allowed to transport.  This avoids forcing weak evidence to carry
            # uniform mass merely because standard balanced OT requires it.
            row_marginals.append(gate.mean(dim=-1).clamp_min(self.dct_evidence_mass_floor))
            col_marginals.append(gate.mean(dim=-2).clamp_min(self.dct_evidence_mass_floor))
            gates.append(gate)
        rows = torch.stack(row_marginals, dim=1)
        cols = torch.stack(col_marginals, dim=1)
        rows = rows / rows.sum(dim=-1, keepdim=True)
        cols = cols / cols.sum(dim=-1, keepdim=True)
        strength = self.dct_evidence_marginal_strength
        reliability_strength = self.dct_geometry_reliability_strength
        if reliability_strength > 0.0:
            reliability = torch.stack(reliabilities, dim=1)
            effective_strength = strength * (
                (1.0 - reliability_strength) + reliability_strength * reliability
            )
            uniform_rows = torch.full_like(rows, 1.0 / rows.size(-1))
            uniform_cols = torch.full_like(cols, 1.0 / cols.size(-1))
            rows = uniform_rows + effective_strength.unsqueeze(-1) * (rows - uniform_rows)
            cols = uniform_cols + effective_strength.unsqueeze(-1) * (cols - uniform_cols)
            self._last_transport_reliability = reliability.detach()
        else:
            if strength < 1.0:
                uniform_rows = torch.full_like(rows, 1.0 / rows.size(-1))
                uniform_cols = torch.full_like(cols, 1.0 / cols.size(-1))
                rows = (1.0 - strength) * uniform_rows + strength * rows
                cols = (1.0 - strength) * uniform_cols + strength * cols
            self._last_transport_reliability = None
        return torch.stack(all_stage_costs, dim=1), rows, cols, torch.stack(gates, dim=1)

    @staticmethod
    def _log_sinkhorn(cost, rows, cols, eps, max_iter):
        # Sharp learned costs can otherwise contaminate every logsumexp with
        # NaN/Inf and make the whole fold look invalid.
        if not torch.isfinite(cost).all():
            raise FloatingPointError("nonfinite v3.14 transport cost")
        finite_cost = cost
        finite_cost = finite_cost.clamp(min=-1e4, max=1e4)
        eps = max(float(eps), float(torch.finfo(cost.dtype).eps))
        kernel = (-finite_cost / eps).clamp(min=-60.0, max=60.0)
        log_rows = rows.clamp_min(1e-8).log()
        log_cols = cols.clamp_min(1e-8).log()
        log_u = torch.zeros_like(log_rows)
        log_v = torch.zeros_like(log_cols)
        for _ in range(max_iter):
            log_u = log_rows - torch.logsumexp(kernel + log_v.unsqueeze(1), dim=2)
            log_v = log_cols - torch.logsumexp(kernel + log_u.unsqueeze(2), dim=1)
            log_u = torch.nan_to_num(log_u, nan=0.0, posinf=60.0, neginf=-60.0)
            log_v = torch.nan_to_num(log_v, nan=0.0, posinf=60.0, neginf=-60.0)
        plan = (kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)).exp()
        return torch.nan_to_num(plan, nan=0.0, posinf=1.0, neginf=0.0)

    def _project_coupling(self, plan, rows, cols):
        """Numerically project a positive plan to its evidence-conditioned marginals."""
        for _ in range(self.dct_coupling_projection_iters):
            plan = plan * (rows.unsqueeze(-1) / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8))
            plan = plan * (cols.unsqueeze(1) / plan.sum(dim=-2, keepdim=True).clamp_min(1e-8))
            row_error = (plan.sum(dim=-1) - rows).abs().amax()
            col_error = (plan.sum(dim=-2) - cols).abs().amax()
            if bool(torch.maximum(row_error, col_error).detach() <= self.dct_coupling_projection_tol):
                break
        return plan

    def _plans_from_cost_tensor(self, costs, rows, cols, epoch, *, replay_fixed=False):
        eps = self._sinkhorn_eps(epoch)
        plans, distances = [], []
        for stage_idx in range(self.spt_num_stages):
            stage_plans, stage_distances = [], []
            for cost_idx in range(costs.size(2)):
                if replay_fixed and bool(getattr(self, "dct_fixed_coupling", False)):
                    plan = self._replay_cached_plan(stage_idx, cost_idx, costs, rows, cols)
                else:
                    plan = self._log_sinkhorn(
                        costs[:, stage_idx, cost_idx], rows[:, stage_idx], cols[:, stage_idx],
                        eps=eps, max_iter=self.ot_iter,
                    )
                    plan = self._project_coupling(plan, rows[:, stage_idx], cols[:, stage_idx])
                stage_plans.append(plan)
                stage_distances.append((plan * costs[:, stage_idx, cost_idx]).sum(dim=(1, 2)))
            plans.append(tuple(stage_plans))
            distances.append(torch.stack(stage_distances).mean())
        return plans, torch.stack(distances).mean()

    @torch.no_grad()
    def _replay_cached_plan(self, stage_idx, cost_idx, costs, rows, cols):
        """Project a stale factual plan to a new intervention's marginals.

        Ablation ``fixed_coupling`` proves that re-solving Sinkhorn under each
        intervention actually changes the plan; without re-solving, the audit
        chain reduces to a fixed-coupling marginal projection.
        """
        cached = self._factual_plan_cache
        if cached is None:
            return self._log_sinkhorn(
                costs[:, stage_idx, cost_idx],
                rows[:, stage_idx], cols[:, stage_idx],
                eps=float(getattr(self.args, "otehv2_eps", 0.05)),
                max_iter=self.ot_iter,
            )
        return self._project_coupling(cached[stage_idx][cost_idx], rows[:, stage_idx], cols[:, stage_idx])

    def _stage_membership_weights(self, event_time, censorship):
        """IPCW weights for event-in-stage (high) and survived-past-stage (low)."""
        bsz = event_time.size(0)
        high = event_time.new_zeros(bsz, self.spt_num_stages)
        low = event_time.new_zeros(bsz, self.spt_num_stages)
        if not self.has_train_reference:
            return low, high
        observed = censorship < 0.5
        for stage_idx in range(self.spt_num_stages):
            lower = self.dct_stage_edges[stage_idx]
            upper = self.dct_stage_edges[stage_idx + 1]
            in_stage = observed & (event_time > lower) & (event_time <= upper)
            # Event after the upper boundary or censoring after it proves survival
            # through this stage.  Censoring before the boundary contributes zero.
            survived_stage = (event_time > upper) | ((censorship >= 0.5) & (event_time >= upper))
            high[:, stage_idx] = in_stage.to(event_time.dtype) * self._ipcw(event_time, before=True)
            low[:, stage_idx] = survived_stage.to(event_time.dtype) * self._ipcw(
                torch.full_like(event_time, upper)
            )
        return low, high

    def _ipcw_pairwise_ranking_loss(self, factual_logits, event_time, censorship):
        """Smooth Uno-style ranking over the exact validation risk score.

        A pair ``(i, j)`` is comparable when patient ``i`` has an observed
        event before patient ``j``.  The event case receives inverse squared
        censoring-survival weight, matching IPCW concordance weighting.  The
        softplus temperature gives a smooth approximation to a margin hinge
        while retaining gradients for near-correct pairs.
        """
        risk = self._risk(factual_logits)
        times = event_time.float().view(-1)
        censoring = censorship.float().view(-1)
        observed = censoring < 0.5

        # Sparse-event survival cohorts often provide too few comparable pairs
        # in a small batch.  The optional within-epoch memory supplies detached
        # reference risks from previous batches; current risks remain attached,
        # so gradients still update only the current batch.  It stays opt-in so
        # historical score-first runs remain exactly reproducible.
        memory_count = 0
        if self.dct_ipcw_rank_memory_size > 0 and self._rank_memory_risk is not None:
            memory_count = int(self._rank_memory_risk.numel())
            risk = torch.cat([risk, self._rank_memory_risk.to(risk.device)], dim=0)
            times = torch.cat([times, self._rank_memory_times.to(times.device)], dim=0)
            censoring = torch.cat(
                [censoring, self._rank_memory_censorship.to(censoring.device)], dim=0
            )
            observed = censoring < 0.5

        current_mask = torch.ones(risk.shape[0], dtype=torch.bool, device=risk.device)
        if memory_count:
            current_mask[-memory_count:] = False
        comparable = observed[:, None] & (times[:, None] < times[None, :])
        comparable &= current_mask[:, None] | current_mask[None, :]
        self.last_ipcw_pair_count = comparable.sum().detach()
        if not bool(comparable.any()):
            return risk.sum() * 0.0

        differences = risk[:, None] - risk[None, :]
        temperature = self.dct_ipcw_rank_temperature
        pair_losses = temperature * F.softplus(
            (self.dct_ipcw_rank_margin - differences) / temperature
        )

        # Uno's concordance uses delta_i / G(T_i)^2. Stabilised clipping avoids
        # one late, heavily censored event dominating a small batch.
        event_weights = self._ipcw(times, before=True).square().clamp_max(self.dct_ipcw_max_weight)
        pair_weights = event_weights[:, None].expand_as(pair_losses)[comparable]
        return (
            pair_losses[comparable] * pair_weights
        ).sum() / pair_weights.sum().clamp_min(1e-6)

    def _remember_ipcw_batch(self, risk, times, censorship):
        if self.dct_ipcw_rank_memory_size <= 0:
            return
        risks = [risk.detach()]
        all_times = [times.detach()]
        all_censorship = [censorship.detach()]
        if self._rank_memory_risk is not None:
            risks.insert(0, self._rank_memory_risk.to(risk.device))
            all_times.insert(0, self._rank_memory_times.to(times.device))
            all_censorship.insert(0, self._rank_memory_censorship.to(censorship.device))
        keep = self.dct_ipcw_rank_memory_size
        self._rank_memory_risk = torch.cat(risks, dim=0)[-keep:].clone()
        self._rank_memory_times = torch.cat(all_times, dim=0)[-keep:].clone()
        self._rank_memory_censorship = torch.cat(all_censorship, dim=0)[-keep:].clone()

    def _reset_ipcw_memory_for_epoch(self, epoch):
        if self._rank_memory_epoch != int(epoch):
            self._rank_memory_epoch = int(epoch)
            self._rank_memory_risk = None
            self._rank_memory_times = None
            self._rank_memory_censorship = None

    @torch.no_grad()
    def _update_risk_anchors(self, costs, low_weights, high_weights):
        if self.dct_random_anchors:
            # Ablation: keep anchor buffer populated with non-zero random tensors
            # so downstream code sees a valid (but meaningless) anchor. This
            # isolates whether the IPCW anchor itself carries the prognostic
            # signal or whether it is leaked by some other module.
            for stage_idx in range(self.spt_num_stages):
                for risk_idx in (self._LOW_RISK, self._HIGH_RISK):
                    if bool(self.risk_anchor_seen[stage_idx, risk_idx]):
                        continue
                    sampled = costs[:, stage_idx].mean(dim=0)
                    self.risk_anchor_costs[stage_idx, risk_idx].copy_(
                        sampled + 0.05 * torch.randn_like(sampled)
                    )
                    self.risk_anchor_seen[stage_idx, risk_idx] = True
            return
        for stage_idx in range(self.spt_num_stages):
            for risk_idx, weights in (
                (self._LOW_RISK, low_weights[:, stage_idx]),
                (self._HIGH_RISK, high_weights[:, stage_idx]),
            ):
                if not bool((weights > 0).any()):
                    continue
                weighted = costs[:, stage_idx] * weights.view(-1, 1, 1, 1)
                current = weighted.sum(dim=0) / weights.sum().clamp_min(1e-6)
                if bool(self.risk_anchor_seen[stage_idx, risk_idx]):
                    self.risk_anchor_costs[stage_idx, risk_idx].lerp_(
                        current.to(dtype=self.risk_anchor_costs.dtype),
                        1.0 - self.dct_anchor_momentum,
                    )
                else:
                    self.risk_anchor_costs[stage_idx, risk_idx].copy_(
                        current.to(dtype=self.risk_anchor_costs.dtype)
                    )
                    self.risk_anchor_seen[stage_idx, risk_idx] = True

    def _counterfactual_costs(self, factual_costs):
        alpha = min(1.0, max(0.0, self.dct_mix_ratio))
        bsz = factual_costs.size(0)
        anchors = self.risk_anchor_costs.to(device=factual_costs.device, dtype=factual_costs.dtype)
        low_anchor = anchors[:, self._LOW_RISK].unsqueeze(0).expand(bsz, -1, -1, -1, -1)
        high_anchor = anchors[:, self._HIGH_RISK].unsqueeze(0).expand(bsz, -1, -1, -1, -1)
        low_seen = self.risk_anchor_seen[:, self._LOW_RISK].view(1, -1, 1, 1, 1)
        high_seen = self.risk_anchor_seen[:, self._HIGH_RISK].view(1, -1, 1, 1, 1)
        low_anchor = torch.where(low_seen, low_anchor, factual_costs)
        high_anchor = torch.where(high_seen, high_anchor, factual_costs)
        return (
            (1.0 - alpha) * factual_costs + alpha * low_anchor,
            (1.0 - alpha) * factual_costs + alpha * high_anchor,
        )

    @staticmethod
    def _marginal_error(plans, rows, cols):
        errors = []
        for stage_idx, stage_plans in enumerate(plans):
            for plan in stage_plans:
                row_error = (plan.sum(dim=-1) - rows[:, stage_idx]).abs().amax(dim=-1)
                col_error = (plan.sum(dim=-2) - cols[:, stage_idx]).abs().amax(dim=-1)
                errors.append(torch.maximum(row_error, col_error))
        return torch.stack(errors, dim=1).amax(dim=1)

    def _encode_logits_from_plans(self, slots_wsi, slots_omic, plans):
        tokens = self._selected_stage_events(slots_wsi, slots_omic, plans)
        tokens = tokens + self.stage_embedding.unsqueeze(0)
        tokens = self.event_norm(self.event_encoder(tokens))
        event_logits = self.event_hazard(tokens)
        gate = torch.softmax(self.event_gate(tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)
        return logits, gate

    @staticmethod
    def _positive_dot_cost(x, y):
        similarity = torch.bmm(x, y.transpose(1, 2))
        return F.softplus(-similarity)

    @staticmethod
    def _normalize_cost(cost):
        cost = cost - cost.amin(dim=(1, 2), keepdim=True)
        return cost / cost.mean(dim=(1, 2), keepdim=True).clamp_min(1e-2)

    @staticmethod
    def _pair_tokens(slots_wsi, slots_omic):
        bsz, sw, dim = slots_wsi.shape
        so = slots_omic.shape[1]
        w = slots_wsi.unsqueeze(2).expand(bsz, sw, so, dim)
        o = slots_omic.unsqueeze(1).expand(bsz, sw, so, dim)
        return torch.cat([w, o, w * o, (w - o).abs()], dim=-1)

    @staticmethod
    def _normalize_stage_cost(cost):
        cost = cost - cost.amin(dim=(1, 2), keepdim=True)
        return cost / cost.mean(dim=(1, 2), keepdim=True).clamp_min(1e-2)

    def _selected_stage_events(self, slots_wsi, slots_omic, plans):
        selected = []
        for stage_idx, (plan_cos, plan_euc, plan_dot) in enumerate(plans):
            all_events, _ = self.fusion(
                slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot
            )
            selected.append(all_events[:, stage_idx:stage_idx + 1])
        return torch.cat(selected, dim=1)

    def explain_last_batch(self):
        """Return detached stage/slot evidence from the most recent forward pass."""
        if self.last_explanations is None:
            raise RuntimeError("Run a forward pass before requesting explanations")
        return self.last_explanations
