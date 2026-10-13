"""Independent DCT v2.1, preserving v2 modules and exactly six objectives."""
from dataclasses import fields

import torch
from torch.nn import functional as F

from survot_rank.research.methods.dct_v2.model import DCTV2, deterministic_readout
from survot_rank.research.methods.dct_v2.losses import logit_nll
from survot_rank.research.methods.dct_v2.transport import (
    frozen_call, fixed_realized_marginal_plan, realized_independent_plan,
)
from survot_rank.research.methods.dct_v21.config import DCTV21Config
from survot_rank.research.methods.dct_v21.objectives import centered_pair_coactivation, collision_safe_separation


class DCTV21(DCTV2):
    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        config = args if isinstance(args, DCTV21Config) else DCTV21Config.from_args(args)
        super().__init__(config, omic_input_dim, omic_names, pathway_names)
        self.args = args
        self.last_local_readout = None
        self.last_diversity_diagnostics = {}

    def get_extra_state(self):
        return {"schema": "dct_v21_standalone_v1", "config": self.config.to_dict(),
                "output": "raw_logits", "losses": ["global_nll", "ipcw_rank", "per_slot_nll",
                "slot_diversity", "relation", "native"], "reconstruction": False}

    def set_extra_state(self, value):
        if not isinstance(value, dict) or value.get("schema") != "dct_v21_standalone_v1":
            raise ValueError("DCT v2.1 requires its own checkpoint schema")
        if DCTV21Config.from_dict(value["config"]).to_dict() != self.config.to_dict():
            raise ValueError("construct DCT v2.1 with the exact saved recipe")

    def _predict(self, state, plans=None, *, frozen=False, quality=None, gain_override=None):
        cfg = self.config
        if cfg.local_readout == "legacy_mean" and gain_override is None:
            return super()._predict(state, plans, frozen=frozen, quality=quality)
        plans = state.plans if plans is None else plans
        events = (state.event_tokens if plans is state.plans and not frozen else
                  self.core.events_from_plans(state.wsi_slots, state.omics_slots, plans, frozen=frozen)[0])
        mass, qr, qc = self._quality(state) if quality is None else quality
        strength = mass.clamp(0, 1) if cfg.mass_aware else (mass > 0).to(mass.dtype)
        call = frozen_call if frozen else lambda module, *args: module(*args)
        gain = self.residual_gain.detach() if frozen else self.residual_gain
        effective_gain = torch.tanh(gain) if gain_override is None else gain.new_tensor(gain_override)
        native = self._native_feature(state)
        update = strength[..., None]*events.to(strength.dtype)
        local = plans.new_zeros(plans.size(0), plans.size(1), cfg.num_classes)
        if cfg.local_readout == "centered_pair":
            local = centered_pair_coactivation(
                plans, call(self.per_slot_hazard_wsi, state.wsi_slots),
                call(self.per_slot_hazard_omic, state.omics_slots), mass_aware=cfg.mass_aware)
        elif cfg.local_readout == "legacy_mean":
            # Probe-only override retains the old formula for an explicit ablation.
            with torch.autocast(device_type=plans.device.type, enabled=False):
                consensus = plans.mean(2)
                sw, so = state.wsi_slots.to(consensus.dtype), state.omics_slots.to(consensus.dtype)
                rows, cols = consensus.sum(-1), consensus.sum(-2)
                moved_w = torch.einsum("bswo,bwd->bsod", consensus, sw)/cols[..., None].clamp_min(1e-8)
                moved_o = torch.einsum("bswo,bod->bswd", consensus, so)/rows[..., None].clamp_min(1e-8)
                if not cfg.local_gating:
                    qr, qc = torch.ones_like(qr), torch.ones_like(qc)
                delta = .5*((qc[..., None]*(moved_w-so[:, None])).mean(2)
                            +(qr[..., None]*(moved_o-sw[:, None])).mean(2))
                update = strength[..., None]*(events.to(consensus.dtype)+delta)
        features = native[:, None]+effective_gain.to(native.dtype)*update.to(native.dtype)
        per_stage = call(self.event_hazard, features)
        per_stage = per_stage+effective_gain.to(per_stage.dtype)*local.to(per_stage.dtype)
        logits = torch.einsum("bs,bsc->bc", self._stage_weights(state).to(per_stage.dtype), per_stage)
        native_logits = call(self.event_hazard, native)
        logits = torch.where((mass.sum(-1) == 0)[:, None], native_logits, logits)
        if not frozen:
            self.last_local_readout = local.detach()
        return logits, native_logits, update

    def relation_loss(self, state, y, c, epoch=0):
        cfg = self.config
        if cfg.relation_objective == "legacy_hinge":
            return super().relation_loss(state, y, c, epoch)
        if self.core.plan_mode == "independent":
            self.last_relation_diagnostics = {}
            return state.plans.sum()*0.
        dtype = self.event_hazard.weight.dtype
        values = {f.name: (getattr(state, f.name).detach().to(dtype) if getattr(state, f.name).is_floating_point()
                          else getattr(state, f.name).detach()) for f in fields(state)}
        detached = type(state)(**values)
        with deterministic_readout(self), torch.autocast(device_type=state.plans.device.type, enabled=False):
            actual = detached.plans
            rows, cols = actual.sum(-1), actual.sum(-2)
            freedom = ((rows > cfg.min_plan_mass).sum(-1) >= 2)&((cols > cfg.min_plan_mass).sum(-1) >= 2)
            mass_weight = (self._stage_weights(detached)*(actual.sum((-2, -1))*freedom).mean(2)).sum(-1).clamp(0, 1).detach()
            if not bool((mass_weight > 0).any()):
                self.last_relation_diagnostics = {"probe_gain":actual.new_tensor(cfg.relation_probe_gain),
                                                  "effective_mass":mass_weight.mean()}
                return state.plans.sum()*0.
            costs, _, _ = self.core.costs_and_marginals(detached.wsi_slots, detached.omics_slots)
            projected = fixed_realized_marginal_plan(
                costs, rows, cols, epsilon=self.core.epsilon(epoch), iterations=cfg.projection_iters,
                tolerance=cfg.projection_tolerance, min_mass=cfg.min_plan_mass)
            quality = tuple(v.detach() for v in self._quality(detached))
            learned, _, _ = self._predict(detached, projected, frozen=True, quality=quality,
                                          gain_override=cfg.relation_probe_gain)
            with torch.no_grad():
                independent, _, _ = self._predict(detached, realized_independent_plan(actual), frozen=True,
                                                  quality=quality, gain_override=cfg.relation_probe_gain)
                reference = logit_nll(independent, y, c, cfg.nll_alpha, "none")
            factual = logit_nll(learned, y, c, cfg.nll_alpha, "none")
            gap = factual-reference.detach()
            loss = (mass_weight*cfg.relation_temperature*F.softplus(gap/cfg.relation_temperature)).mean()
        self.last_relation_diagnostics = {
            "probe_factual_nll":factual.detach().mean(), "probe_independent_nll":reference.mean(),
            "probe_nll_gap":gap.detach().mean(), "probe_gain":gap.new_tensor(cfg.relation_probe_gain),
            "effective_mass":mass_weight.mean(),
            "row_error":(projected.sum(-1)-rows).detach().abs().max(),
            "col_error":(projected.sum(-2)-cols).detach().abs().max(),
        }
        return loss

    def _slot_objectives(self, state, y, c, ipcw):
        slot, legacy = super()._slot_objectives(state, y, c, ipcw)
        if self.config.diversity_objective == "legacy_squared":
            return slot, legacy
        cfg = self.config
        available = (state.wsi_mask.any(-1), state.omics_mask.any(-1))
        denominator = sum(a.sum() for a in available).clamp_min(1)
        diversity = state.wsi_slots.sum()*0.
        for name, s, head, valid in zip(("wsi", "omics"), (state.wsi_slots, state.omics_slots),
                                      (self.per_slot_hazard_wsi, self.per_slot_hazard_omic), available):
            if bool(valid.any()):
                sv = s[valid]
                h = torch.sigmoid(head(sv))
                variance = (h-h.mean(1, keepdim=True)).square().mean((1, 2))
                band = (F.relu(cfg.variance_min-variance)+F.relu(variance-cfg.variance_max)).mean()
                separation = collision_safe_separation(sv, cfg.target_pair_dist)
                term = band+cfg.pair_weight*separation
                diversity = diversity+term*valid.sum()/denominator
                self.last_diversity_diagnostics[name+"_separation"] = separation.detach()
        return slot, diversity

    def forward(self, **kwargs):
        self.last_local_readout = None
        self.last_diversity_diagnostics = {}
        logits, auxiliary = super().forward(**kwargs)
        self.last_diagnostics.update({"schema":"dct_v21_standalone_diagnostics_v1",
            "local_readout":self.config.local_readout, "relation_objective":self.config.relation_objective,
            "relation_probe_gain":self.config.relation_probe_gain,
            "diversity_objective":self.config.diversity_objective,
            "local_readout_norm":None if self.last_local_readout is None else self.last_local_readout.norm(dim=-1)})
        return logits, auxiliary
