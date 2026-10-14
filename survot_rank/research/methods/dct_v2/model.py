"""Independent DCT v2: own encoders/head, partial transport and six losses."""
from contextlib import contextmanager
from dataclasses import fields
import torch
from torch import nn

from survot_rank.research.components.omics_encoder import SNN_Block, WSI_Mlp
from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.diversity import repaired_slot_diversity_loss
from survot_rank.research.methods.dct_v2.config import DCTV2Config
from survot_rank.research.methods.dct_v2.transport import DCTV2Transport, frozen_call, fixed_realized_marginal_plan, realized_independent_plan
from survot_rank.research.methods.dct_v2.losses import censor_km, inverse_censor_weight, ipcw_pairwise_ranking_loss, logit_nll, risk_from_logits


@contextmanager
def deterministic_readout(model):
    flags = [(module, module.training) for module in model.modules()]
    try:
        for module, _ in flags:
            module.training = False
        yield
    finally:
        for module, mode in flags:
            module.training = mode


class DCTV2(nn.Module):
    """Full survival model, not an adapter to another survival backbone.

    ``forward`` returns raw logits plus five weighted auxiliary terms. The
    shared trainer supplies main NLL. ``forward_with_losses`` also returns the
    complete six-term objective for focused validation/custom training loops.
    """
    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__()
        self.args = args
        self.config = args if isinstance(args, DCTV2Config) else DCTV2Config.from_args(args)
        cfg = self.config
        self.num_classes, self.num_pathways = cfg.num_classes, len(cfg.omic_sizes)
        self.wsi_projection_dim = cfg.dim
        self.wsi_mlp = WSI_Mlp(cfg.wsi_input_dim, cfg.dim)
        self.sig_networks = nn.ModuleList([
            nn.Sequential(SNN_Block(width, cfg.dim, cfg.dropout), SNN_Block(cfg.dim, cfg.dim, cfg.dropout))
            for width in cfg.omic_sizes
        ])
        self.core = DCTV2Transport(cfg)
        # One shared DCT head reads native and updated features; no external
        # host head and no new head just for the native auxiliary objective.
        self.event_hazard = nn.Linear(cfg.dim, cfg.num_classes)
        self.per_slot_hazard_wsi = nn.Linear(cfg.dim, cfg.num_classes)
        self.per_slot_hazard_omic = nn.Linear(cfg.dim, cfg.num_classes)
        self.residual_gain = nn.Parameter(torch.tensor(float(cfg.residual_init)))
        self.register_buffer("censor_times", torch.empty(0))
        self.register_buffer("censor_survival", torch.empty(0))
        self._rank_memory = None
        self._rank_epoch = None
        self.last_state = None
        self.last_training_losses = {}
        self.last_diagnostics = {}
        self.last_relation_diagnostics = {}
        self.last_native_hazards = None

    def get_extra_state(self):
        return {"schema": "dct_v2_standalone_v1", "config": self.config.to_dict(),
                "output": "raw_logits", "losses": ["global_nll", "ipcw_rank", "per_slot_nll",
                "slot_diversity", "relation", "native"], "reconstruction": False}

    def set_extra_state(self, value):
        if not isinstance(value, dict) or value.get("schema") != "dct_v2_standalone_v1":
            raise ValueError("DCT v2 requires its standalone checkpoint schema")
        if DCTV2Config.from_dict(value["config"]).to_dict() != self.config.to_dict():
            raise ValueError("construct DCT v2 with the exact saved recipe before loading")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.set_extra_state(state_dict.get("_extra_state"))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        times, surv = state_dict.get(prefix+"censor_times"), state_dict.get(prefix+"censor_survival")
        if times is not None and surv is not None:
            if times.ndim != 1 or times.shape != surv.shape or not bool(torch.isfinite(times).all()) or not bool(torch.isfinite(surv).all()):
                raise ValueError("invalid saved censor reference")
            if bool((times < 0).any()) or bool((times[1:] <= times[:-1]).any()) or bool(((surv < 0) | (surv > 1)).any()) or bool((surv[1:] > surv[:-1]).any()):
                raise ValueError("saved censor reference must be valid KM statistics")
            for name, value in (("censor_times", times), ("censor_survival", surv)):
                current = getattr(self, name)
                setattr(self, name, current.new_empty(value.shape))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @property
    def has_train_reference(self):
        return self.censor_times.numel() > 0

    @torch.no_grad()
    def set_training_reference(self, event_times, censorship):
        times, surv = censor_km(event_times, censorship)
        self.censor_times = times.to(self.event_hazard.weight.device)
        self.censor_survival = surv.to(self.event_hazard.weight.device)
        self._rank_memory, self._rank_epoch = None, None

    configure_train_reference = set_training_reference

    def objective_weights_effective(self, epoch=0):
        cfg = self.config
        return dict(global_nll=1., ipcw_rank=cfg.ipcw_weight, per_slot_nll=cfg.slot_weight,
                    slot_diversity=cfg.diversity_weight, relation=cfg.relation_effective_weight(epoch), native=cfg.native_weight)

    def _encode(self, kwargs):
        cfg = self.config
        x = kwargs.get("x_wsi")
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.size(0) < 1 or x.size(-1) != cfg.wsi_input_dim or x.size(1) < 1 or not x.is_floating_point():
            raise ValueError("x_wsi must be floating [B,N,wsi_input_dim]")
        mask = kwargs.get("wsi_mask", kwargs.get("wsi_valid_mask"))
        mask = x.ne(0).any(-1) if mask is None else mask
        omask = kwargs.get("omics_mask")
        omask = torch.ones(x.size(0), self.num_pathways, dtype=torch.bool, device=x.device) if omask is None else omask
        for name, value, shape in (("wsi_mask", mask, x.shape[:2]), ("omics_mask", omask, (x.size(0), self.num_pathways))):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.device != x.device or value.shape != shape:
                raise ValueError(name + " must be a boolean mask with the input token shape/device")
        for key in ("wsi_missing", "omic_missing"):
            if not isinstance(kwargs.get(key, False), bool):
                raise ValueError(key + " must be boolean")
        if kwargs.get("wsi_missing", False):
            mask = torch.zeros_like(mask)
        if kwargs.get("omic_missing", False):
            omask = torch.zeros_like(omask)
        if bool((~(mask.any(-1) | omask.any(-1))).any()):
            raise ValueError("each patient needs at least one available modality")
        safe = x.masked_fill(~mask[..., None], 0.)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("valid WSI features must be finite")
        wsi = self.wsi_mlp(safe).masked_fill(~mask[..., None], 0.)
        tokens = []
        for i, (width, encoder) in enumerate(zip(cfg.omic_sizes, self.sig_networks), 1):
            value = kwargs.get(f"x_omic{i}")
            if not isinstance(value, torch.Tensor) or value.shape != (x.size(0), width) or value.device != x.device or not value.is_floating_point():
                raise ValueError(f"x_omic{i} must be floating [B,{width}] on the WSI device")
            safe_value = value.masked_fill(~omask[:, i-1:i], 0.)
            if not bool(torch.isfinite(safe_value).all()):
                raise ValueError("valid omics inputs must be finite")
            tokens.append(encoder(safe_value).masked_fill(~omask[:, i-1:i], 0.))
        return wsi, torch.stack(tokens, 1), mask, omask

    @staticmethod
    def _native_feature(state):
        wa, oa = state.wsi_mask.any(-1), state.omics_mask.any(-1)
        return (state.wsi_slots.mean(1)*wa[:, None] + state.omics_slots.mean(1)*oa[:, None]) / (wa.float()+oa.float()).clamp_min(1)[:, None]

    @staticmethod
    def _stage_weights(state):
        return torch.where(state.pair_available[:, None], state.stage_weights,
                           torch.full_like(state.stage_weights, 1/state.stage_weights.size(1)))

    def _quality(self, state):
        p = state.plans.mean(2)
        rows, cols = p.sum(-1), p.sum(-2)
        mass = rows.sum(-1)
        a, b = state.row_marginals, state.col_marginals
        qrow = (rows/torch.where(a > 0, a, torch.ones_like(a))).clamp(0, 1)
        qcol = (cols/torch.where(b > 0, b, torch.ones_like(b))).clamp(0, 1)
        return mass, qrow, qcol

    def _predict(self, state, plans=None, *, frozen=False, quality=None):
        cfg = self.config
        plans = state.plans if plans is None else plans
        if plans is state.plans and not frozen:
            events = state.event_tokens
        else:
            events, _ = self.core.events_from_plans(state.wsi_slots, state.omics_slots, plans, frozen=frozen)
        mass, qr, qc = self._quality(state) if quality is None else quality
        with torch.autocast(device_type=plans.device.type, enabled=False):
            consensus = plans.mean(2)
            sw, so = state.wsi_slots.to(consensus.dtype), state.omics_slots.to(consensus.dtype)
            rows, cols = consensus.sum(-1), consensus.sum(-2)
            moved_w = torch.einsum("bswo,bwd->bsod", consensus, sw)/cols[..., None].clamp_min(1e-8)
            moved_o = torch.einsum("bswo,bod->bswd", consensus, so)/rows[..., None].clamp_min(1e-8)
            if not cfg.local_gating:
                qr, qc = torch.ones_like(qr), torch.ones_like(qc)
            local = .5*((qc[..., None]*(moved_w-so[:, None])).mean(2) +
                        (qr[..., None]*(moved_o-sw[:, None])).mean(2))
            strength = mass.clamp(0, 1) if cfg.mass_aware else (mass > 0).to(mass.dtype)
            update = strength[..., None]*(events.to(consensus.dtype)+local)
        native = self._native_feature(state)
        gain = self.residual_gain.detach() if frozen else self.residual_gain
        features = native[:, None] + torch.tanh(gain).to(native.dtype)*update.to(native.dtype)
        call = frozen_call if frozen else lambda module, *args: module(*args)
        per_stage_logits = call(self.event_hazard, features)
        weights = self._stage_weights(state).to(per_stage_logits.dtype)
        logits = torch.einsum("bs,bsc->bc", weights, per_stage_logits)
        native_logits = call(self.event_hazard, native)
        # Exact bypass at zero realized exchange, including any event biases.
        logits = torch.where((mass.sum(-1) == 0)[:, None], native_logits, logits)
        return logits, native_logits, update

    def reference_deficit(self, state):
        r, c = state.plans.detach().sum(-1), state.plans.detach().sum(-2)
        a, b = state.row_marginals.detach().unsqueeze(2), state.col_marginals.detach().unsqueeze(2)
        deficits = .5*((a-r).clamp_min(0).sum(-1)+(b-c).clamp_min(0).sum(-1)).mean(2)
        return (self._stage_weights(state).detach()*deficits).sum(-1).clamp(0, 1).detach()

    def relation_loss(self, state, y, c, epoch=0):
        """Cost-only hinge conditional on fixed detached realized budgets."""
        cfg = self.config
        if self.core.plan_mode == "independent":
            # The matched candidate is itself independent in this control;
            # do not secretly train a hypothetical learned topology.
            self.last_relation_diagnostics = {}
            return state.plans.sum()*0.
        dtype = self.event_hazard.weight.dtype
        values = {f.name: (getattr(state, f.name).detach().to(dtype) if getattr(state, f.name).is_floating_point()
                          else getattr(state, f.name).detach()) for f in fields(state)}
        detached = type(state)(**values)
        with deterministic_readout(self), torch.autocast(device_type=state.plans.device.type, enabled=False):
            costs, _, _ = self.core.costs_and_marginals(detached.wsi_slots, detached.omics_slots)
            actual = detached.plans
            projected = fixed_realized_marginal_plan(costs, actual.sum(-1), actual.sum(-2),
                epsilon=self.core.epsilon(epoch), iterations=cfg.projection_iters,
                tolerance=cfg.projection_tolerance, min_mass=cfg.min_plan_mass)
            quality = tuple(v.detach() for v in self._quality(detached))
            learned_logits, _, _ = self._predict(detached, projected, frozen=True, quality=quality)
            with torch.no_grad():
                independent_logits, _, _ = self._predict(detached, realized_independent_plan(actual), frozen=True, quality=quality)
                reference = logit_nll(independent_logits, y, c, cfg.nll_alpha, "none")
            factual = logit_nll(learned_logits, y, c, cfg.nll_alpha, "none")
            loss = torch.relu(factual-reference.detach()-cfg.relation_tolerance).mean()
        self.last_relation_diagnostics = {"factual_nll": factual.detach().mean(), "independent_nll": reference.detach().mean(),
            "row_error": (projected.sum(-1)-actual.sum(-1)).detach().abs().max(),
            "col_error": (projected.sum(-2)-actual.sum(-2)).detach().abs().max()}
        return loss

    def _slot_objectives(self, state, y, c, ipcw):
        cfg = self.config
        slots = (state.wsi_slots, state.omics_slots)
        logits = (self.per_slot_hazard_wsi(slots[0]), self.per_slot_hazard_omic(slots[1]))
        available = (state.wsi_mask.any(-1), state.omics_mask.any(-1))
        count = (available[0].float()+available[1].float()).clamp_min(1)
        slot_nll = sum(logit_nll(z, y, c, cfg.nll_alpha, "none").mean(1)*a for z, a in zip(logits, available))
        slot_nll = (slot_nll/count*ipcw).mean()
        diversity = slots[0].sum()*0.
        denominator = sum(a.sum() for a in available).clamp_min(1)
        for s, z, valid in zip(slots, logits, available):
            if bool(valid.any()):
                h = torch.sigmoid(z[valid])
                loss, _ = repaired_slot_diversity_loss(s[valid], s[valid], h, h, variance_min=cfg.variance_min,
                    variance_max=cfg.variance_max, target_pair_dist=cfg.target_pair_dist, pair_weight=cfg.pair_weight)
                diversity = diversity + loss*valid.sum()/denominator
        return slot_nll, diversity

    def forward(self, **kwargs):
        cfg = self.config
        epoch = kwargs.get("epoch", kwargs.get("cur_epoch", getattr(self.args, "cur_epoch", 0)))
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        wsi, omics, wm, om = self._encode(kwargs)
        state = self.core(wsi, omics, wsi_mask=wm, omics_mask=om, epoch=epoch)
        logits, native_logits, update = self._predict(state)
        self.last_state = state
        self.last_native_hazards = torch.sigmoid(native_logits)
        self.last_relation_diagnostics = {}
        self.last_training_losses = {}
        zero = logits.sum()*0.
        auxiliary = zero
        weights = self.objective_weights_effective(epoch)
        if self.training and kwargs.get("compute_aux", True):
            if any(kwargs.get(k) is None for k in ("y", "event_time", "c")):
                raise ValueError("training objectives require y, event_time and c")
            y, c, t = kwargs["y"], kwargs["c"], kwargs["event_time"]
            logit_nll(logits, y, c, cfg.nll_alpha)  # validate labels before side effects
            t = torch.as_tensor(t, device=logits.device, dtype=torch.float32).flatten()
            if t.shape != (logits.size(0),) or not bool(torch.isfinite(t).all()) or bool((t < 0).any()):
                raise ValueError("event_time must be finite nonnegative [B]")
            c = torch.as_tensor(c, device=logits.device, dtype=torch.float32).flatten()
            ipcw = inverse_censor_weight(t, self.censor_times, self.censor_survival, max_weight=cfg.ipcw_max_weight) if cfg.ipcw_weight > 0 or cfg.slot_weight > 0 else torch.ones_like(t)
            if self._rank_epoch != epoch:
                self._rank_memory, self._rank_epoch = None, epoch
            ranking, pairs = zero, t.new_zeros(())
            rank_weights = inverse_censor_weight(t, self.censor_times, self.censor_survival, power=2., max_weight=cfg.ipcw_max_weight) if cfg.ipcw_weight > 0 else torch.ones_like(t)
            if cfg.ipcw_weight > 0:
                ranking, pairs = ipcw_pairwise_ranking_loss(logits, t, c, rank_weights, margin=cfg.ipcw_rank_margin,
                    temperature=cfg.ipcw_rank_temperature, memory=self._rank_memory)
            slot, div = self._slot_objectives(state, y, c, ipcw) if cfg.slot_weight > 0 or cfg.diversity_weight > 0 else (zero, zero)
            relation = self.relation_loss(state, y, c, epoch) if weights["relation"] > 0 else zero
            deficit = self.reference_deficit(state)
            native = (deficit*logit_nll(native_logits, y, c, cfg.nll_alpha, "none")).mean() if cfg.native_weight > 0 else zero
            raw = dict(ipcw_rank=ranking, per_slot_nll=slot, slot_diversity=div, relation=relation, native=native)
            auxiliary = sum(weights[k]*v for k, v in raw.items())
            self.last_training_losses = {**{k:v.detach() for k,v in raw.items()},
                **{k+"_weighted": (weights[k]*v).detach() for k,v in raw.items()},
                **{k+"_effective_weight": logits.new_tensor(weights[k]) for k in raw},
                "reference_deficit": deficit.detach().mean(), "ipcw_pair_count": pairs,
                **{"relation_"+k:v for k,v in self.last_relation_diagnostics.items()}}
            if cfg.ipcw_weight > 0 and cfg.ipcw_memory_size > 0:
                current = (risk_from_logits(logits).detach(), t.detach(), c.detach(), rank_weights.detach())
                memory = current if self._rank_memory is None else tuple(torch.cat((old.to(new.device), new)) for old,new in zip(self._rank_memory,current))
                self._rank_memory = tuple(v[-cfg.ipcw_memory_size:].clone() for v in memory)
        self.last_diagnostics = {"schema":"dct_v2_standalone_diagnostics_v1", "transport_mass":state.transport_mass.detach(),
            "native_hazards":self.last_native_hazards.detach(), "update_norm":update.detach().norm(dim=-1),
            "residual_gain":torch.tanh(self.residual_gain).detach(), "solver":dict(self.core.last_solver_diagnostics)}
        return logits, auxiliary

    def forward_with_losses(self, **kwargs):
        logits, auxiliary = self(**kwargs)
        primary = logit_nll(logits, kwargs["y"], kwargs["c"], self.config.nll_alpha)
        return {"logits":logits, "hazards":torch.sigmoid(logits), "auxiliary":auxiliary, "loss":primary+auxiliary,
                "terms":{"global_nll":primary.detach(), **self.last_training_losses}, "state":self.last_state}
