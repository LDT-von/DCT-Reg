"""DCT v3.15: one-shot slots with a residual OT interaction.

No imports or inheritance from prior model versions. Attention pooling, cosine
OT, and centered bilinear moments are established building blocks; this is a
project-specific candidate, not a claim of first invention or clinical efficacy.
The historical NLL-only recipe remains available with a zero rank weight.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class OneShotSlots(nn.Module):
    """Queries select actual token values; no GRU or identity-valued output."""
    def __init__(self, dim, slots):
        super().__init__()
        if not 1 <= slots <= dim:
            raise ValueError("slot count must be between 1 and latent dimension")
        self.queries = nn.Parameter(torch.empty(slots, dim))
        nn.init.orthogonal_(self.queries)

    def forward(self, tokens):
        # Only routing is centered: patient-common content stays in the values.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            values = tokens.float()
            centered = values - values.mean(1, keepdim=True)
            keys = F.normalize(centered, dim=-1, eps=1e-6)
            query = F.normalize(self.queries.float(), dim=-1, eps=1e-6)
            scores = torch.einsum("kd,bnd->bkn", query, keys) * math.sqrt(tokens.size(-1)) / .5
            weights = scores.softmax(-1)
            slots = weights @ values
        return slots, weights


def uniform_sinkhorn(cost, epsilon=.2, iterations=40):
    """Positive balanced coupling; final rounding enforces uniform marginals.

    Differentiable row/column downscaling followed by a rank-one deficit
    correction avoids a long, data-dependent projection loop.
    """
    if not math.isfinite(epsilon) or epsilon <= 0 or iterations < 1:
        raise ValueError("Sinkhorn epsilon and iterations must be positive")
    if cost.ndim != 3 or min(cost.shape) < 1 or not torch.isfinite(cost).all():
        raise ValueError("cost must be finite nonempty [B, K, L]")
    with torch.autocast(device_type=cost.device.type, enabled=False):
        cost = cost.float() if cost.dtype in (torch.float16, torch.bfloat16) else cost
        batch, kw, ko = cost.shape
        kernel = -cost / epsilon
        loga, logb = -math.log(kw), -math.log(ko)
        u, v = cost.new_zeros(batch, kw), cost.new_zeros(batch, ko)
        for _ in range(iterations):
            u = loga - torch.logsumexp(kernel + v[:, None, :], -1)
            v = logb - torch.logsumexp(kernel + u[:, :, None], -2)
        plan = (kernel + u[:, :, None] + v[:, None, :]).exp()
        plan = plan * ((1./kw) / plan.sum(-1, keepdim=True).clamp_min(1e-12)).clamp_max(1.)
        plan = plan * ((1./ko) / plan.sum(-2, keepdim=True).clamp_min(1e-12)).clamp_max(1.)
        row_deficit = (1./kw - plan.sum(-1)).clamp_min(0.)
        col_deficit = (1./ko - plan.sum(-2)).clamp_min(0.)
        correction = row_deficit[:, :, None] * col_deficit[:, None, :] / row_deficit.sum(-1)[:, None, None].clamp_min(1e-12)
        return plan + correction


class ResidualTransportInteraction(nn.Module):
    """Read only cross-modal interactions beyond independent slot pairing."""
    def __init__(self, dim, classes, epsilon=.2, iterations=40, mode="ot"):
        super().__init__()
        if mode not in {"ot", "independent"}:
            raise ValueError("transport mode must be ot or independent")
        if not math.isfinite(epsilon) or epsilon <= 0 or iterations < 1:
            raise ValueError("invalid transport settings")
        self.epsilon, self.iterations, self.mode = epsilon, iterations, mode
        self.readout = nn.Linear(dim, classes, bias=False)

    @staticmethod
    def residuals(slots):
        centered = slots - slots.mean(1, keepdim=True)
        # A variance floor bounds amplification of nearly identical slots.
        return centered / (centered.square().mean(1, keepdim=True) + 1e-4).sqrt()

    @staticmethod
    def interaction(w, o, plan):
        independent = torch.full_like(plan, 1./(w.size(1)*o.size(1)))
        return torch.einsum("bkl,bkd,bld->bd", plan-independent, w, o)

    def forward(self, wsi, omics, paired):
        with torch.autocast(device_type=wsi.device.type, enabled=False):
            w, o = self.residuals(wsi.float()), self.residuals(omics.float())
            cost = 1. - F.normalize(w, dim=-1, eps=1e-4) @ F.normalize(o, dim=-1, eps=1e-4).transpose(1, 2)
            if self.mode == "ot":
                plan = uniform_sinkhorn(cost, self.epsilon, self.iterations)
            else:
                plan = cost.new_full(cost.shape, 1./(w.size(1)*o.size(1)))
            moment = self.interaction(w, o, plan) * paired[:, None]
            delta = self.readout(moment)
        return delta, moment, plan, w, o


class DCTV315ResidualTransport(nn.Module):
    """Standalone survival baseline; labels never enter forward prediction."""
    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__()
        self.args = args
        if getattr(args, "bag_loss", "nll_surv") != "nll_surv" or getattr(args, "rna_format", "Pathways") != "Pathways":
            raise ValueError("v3.15 requires nll_surv and Pathways")
        self.encoding_dim = int(args.encoding_dim)
        dim = self.dim = int(args.wsi_projection_dim)
        classes = self.num_classes = int(args.n_classes)
        self.omic_sizes = tuple(int(n) for n in args.omic_sizes)
        kw, ko = int(args.slot_num_wsi), int(args.slot_num_omics)
        dropout = float(getattr(args, "dct_v315_dropout", .1))
        self.alpha_surv = float(getattr(args, "alpha_surv", 0.))
        self.rti_rank_weight = float(getattr(args, "dct_v315_lambda_rti_rank", 0.))
        self.rti_rank_margin = float(getattr(args, "dct_v315_rti_rank_margin", .02))
        self.rti_rank_temperature = float(getattr(args, "dct_v315_rti_rank_temperature", .5))
        self.ipcw_max_weight = float(getattr(args, "dct_v315_ipcw_max_weight", 10.))
        if min(self.encoding_dim, dim, classes) < 1 or not self.omic_sizes or min(self.omic_sizes) < 1:
            raise ValueError("positive dimensions and nonempty omic_sizes required")
        if not 0 <= dropout < 1 or not 0 <= self.alpha_surv <= 1:
            raise ValueError("invalid dropout or alpha_surv")
        if (not math.isfinite(self.rti_rank_weight) or self.rti_rank_weight < 0 or
                not math.isfinite(self.rti_rank_margin) or self.rti_rank_margin < 0 or
                not math.isfinite(self.rti_rank_temperature) or self.rti_rank_temperature <= 0 or
                not math.isfinite(self.ipcw_max_weight) or self.ipcw_max_weight <= 0):
            raise ValueError("invalid v3.15 RTI rank settings")
        # One projection per modality token; no million-parameter WSI pre-MLP.
        def encoder(width):
            return nn.Sequential(nn.Linear(width, dim), nn.LayerNorm(dim), nn.GELU())
        self.wsi_encoder = encoder(self.encoding_dim)
        self.omic_encoders = nn.ModuleList([encoder(width) for width in self.omic_sizes])
        self.wsi_slots, self.omic_slots = OneShotSlots(dim, kw), OneShotSlots(dim, ko)
        self.global_head = nn.Sequential(nn.LayerNorm(dim*2+2), nn.Linear(dim*2+2, dim),
                                         nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, classes))
        self.transport = ResidualTransportInteraction(
            dim, classes, float(getattr(args, "dct_v315_ot_epsilon", .2)),
            int(getattr(args, "dct_v315_ot_iters", 40)),
            str(getattr(args, "dct_v315_transport_mode", "ot")))
        self.register_buffer("v315_censor_times", torch.empty(0), persistent=False)
        self.register_buffer("v315_censor_survival", torch.empty(0), persistent=False)
        self.last_training_losses, self.last_explanations = {}, None

    def objective_weights(self):
        rank_weight = self.rti_rank_weight if self.transport.mode == "ot" else 0.
        return {"nll": 1., "rti_ipcw_rank": rank_weight, "per_slot_nll": 0.,
                "slot_diversity": 0., "reconstruction": 0.}

    def get_extra_state(self):
        state = {"schema": 1, "architecture": "one_shot_rti", "alpha_surv": self.alpha_surv,
                "mode": self.transport.mode, "epsilon": self.transport.epsilon,
                "iterations": self.transport.iterations, "dropout": self.global_head[3].p,
                "omic_sizes": self.omic_sizes}
        if self.rti_rank_weight > 0:
            state.update(schema=2, lambda_rti_rank=self.rti_rank_weight,
                         rti_rank_margin=self.rti_rank_margin,
                         rti_rank_temperature=self.rti_rank_temperature,
                         ipcw_max_weight=self.ipcw_max_weight)
        return state

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("v3.15 checkpoint recipe differs from constructed model")

    @property
    def has_train_reference(self):
        return self.v315_censor_times.numel() > 0

    @torch.no_grad()
    def configure_train_reference(self, event_times, censorship):
        """Fit fold-local censoring KM for left-limit IPCW weights."""
        device = self.transport.readout.weight.device
        times = torch.as_tensor(event_times, dtype=torch.float32, device=device).flatten()
        cens = torch.as_tensor(censorship, dtype=torch.float32, device=device).flatten()
        if times.numel() < 1 or times.numel() != cens.numel() or not torch.isfinite(times).all():
            raise ValueError("finite nonempty train times and matching censorship required")
        if not torch.all((cens == 0) | (cens == 1)):
            raise ValueError("train censorship must be binary")
        unique_times = torch.unique(times, sorted=True)
        survival = torch.ones_like(unique_times)
        value = times.new_ones(())
        for index, time in enumerate(unique_times):
            at_risk = (times >= time).sum().to(times.dtype).clamp_min(1.)
            censor_events = ((times == time) & (cens >= .5)).sum().to(times.dtype)
            value = value * (1. - censor_events / at_risk)
            survival[index] = value
        self.v315_censor_times = unique_times
        self.v315_censor_survival = survival.clamp_min(.05)

    def _ipcw(self, query_times):
        if not self.has_train_reference:
            raise RuntimeError("configure_train_reference must run before RTI-IPCW ranking")
        # Uno-style comparable pairs use G(T_i-), excluding censoring events
        # tied at the event time itself.
        indices = torch.searchsorted(self.v315_censor_times, query_times, right=False) - 1
        values = torch.ones_like(query_times)
        valid = indices >= 0
        values[valid] = self.v315_censor_survival[indices[valid]]
        return values.clamp_min(.05).reciprocal()

    @staticmethod
    def _risk(logits):
        hazards = torch.sigmoid(logits)
        return -torch.cumprod(1. - hazards, dim=1).sum(dim=1)

    def _rti_directed_ipcw_rank(self, base, delta, event_time, censorship, paired):
        """Train RTI only on comparable pairs the detached base does not rank."""
        times = torch.as_tensor(event_time, dtype=torch.float32, device=delta.device).flatten()
        cens = torch.as_tensor(censorship, dtype=torch.float32, device=delta.device).flatten()
        if times.numel() != len(delta) or cens.numel() != len(delta) or not torch.isfinite(times).all():
            raise ValueError("event_time and censorship must match the batch")
        if not torch.all((cens == 0) | (cens == 1)):
            raise ValueError("censorship must be binary")
        if self.transport.mode != "ot":
            zero = delta.sum() * 0.
            return zero, delta.new_zeros(()), delta.new_zeros(())
        if not self.has_train_reference:
            raise RuntimeError("configure_train_reference must run before RTI-IPCW ranking")

        paired = paired.bool().flatten()
        if paired.numel() != len(delta):
            raise ValueError("paired availability must match the batch")
        comparable = ((cens < .5)[:, None] & (times[:, None] < times[None, :]) &
                      paired[:, None] & paired[None, :])
        base_logits = base.detach().float()
        base_risk = self._risk(base_logits)
        base_gap = base_risk[:, None] - base_risk[None, :]
        hard = comparable & (base_gap < self.rti_rank_margin)
        pair_count, hard_count = comparable.sum().detach(), hard.sum().detach()
        if not bool(hard.any()):
            return delta.sum() * 0., pair_count, hard_count

        corrected_risk = self._risk(base_logits + delta.float())
        corrected_gap = corrected_risk[:, None] - corrected_risk[None, :]
        pair_losses = self.rti_rank_temperature * F.softplus(
            (self.rti_rank_margin - corrected_gap) / self.rti_rank_temperature)
        event_weights = self._ipcw(times).square().clamp_max(self.ipcw_max_weight)
        weights = event_weights[:, None].expand_as(pair_losses)[hard]
        loss = (pair_losses[hard] * weights).sum() / weights.sum().clamp_min(1e-6)
        return loss, pair_count, hard_count

    @staticmethod
    def _mask(value, batch, device, default):
        mask = torch.as_tensor(default if value is None else value, device=device)
        if not torch.all((mask == 0) | (mask == 1)) or mask.numel() not in (1, batch):
            raise ValueError("availability must be boolean scalar or [B]")
        return mask.bool().reshape(-1).expand(batch)

    def _inputs(self, kwargs):
        x = kwargs.get("x_wsi")
        if not torch.is_tensor(x) or x.ndim != 3 or min(x.shape[:2]) < 1 or x.size(-1) != self.encoding_dim or not x.is_floating_point():
            raise ValueError("x_wsi must be nonempty floating [B, N, encoding_dim]")
        b, device = len(x), x.device
        wa = self._mask(kwargs.get("wsi_available"), b, device, True) & ~self._mask(kwargs.get("wsi_missing"), b, device, False)
        oa = self._mask(kwargs.get("omic_available", kwargs.get("omics_available")), b, device, True) & ~self._mask(kwargs.get("omic_missing"), b, device, False)
        if not (wa | oa).all():
            raise ValueError("each patient needs at least one modality")
        def clean(value, available):
            if not torch.isfinite(value[available]).all():
                raise ValueError("nonfinite available inputs")
            return torch.where(available.view([b]+[1]*(value.ndim-1)), value, 0.)
        omics = []
        for i, width in enumerate(self.omic_sizes, 1):
            value = kwargs.get(f"x_omic{i}")
            if not torch.is_tensor(value) or value.shape != (b, width) or value.device != device or not value.is_floating_point():
                raise ValueError(f"x_omic{i} must be floating [B,{width}] on WSI device")
            omics.append(clean(value, oa))
        return clean(x, wa), omics, wa, oa

    @staticmethod
    def _slot_metrics(slots, attention, available):
        if not bool(available.any()) or slots.size(1) < 2:
            return slots.new_zeros(()), slots.new_zeros(()), slots.new_zeros(())
        s, a = slots[available].detach().float(), attention[available].detach().float()
        off = ~torch.eye(s.size(1), device=s.device, dtype=torch.bool)
        norm, attn = F.normalize(s, dim=-1), F.normalize(a, dim=-1)
        return ((norm @ norm.transpose(1, 2))[:, off].mean(), s.std(1, unbiased=False).mean(),
                (attn @ attn.transpose(1, 2))[:, off].mean())

    def forward(self, **kwargs):
        x, omics, wa, oa = self._inputs(kwargs)
        w = self.wsi_encoder(x) * wa[:, None, None]
        o = torch.stack([net(value) for net, value in zip(self.omic_encoders, omics)], 1) * oa[:, None, None]
        sw, aw = self.wsi_slots(w)
        so, ao = self.omic_slots(o)
        global_input = torch.cat([sw.mean(1), so.mean(1), wa[:, None], oa[:, None]], -1)
        base = self.global_head(global_input)
        delta, moment, plan, rw, ro = self.transport(sw, so, wa & oa)
        logits = base.float() + delta
        rank_loss = logits.new_zeros(())
        pair_count = logits.new_zeros(())
        hard_pair_count = logits.new_zeros(())
        if self.training and self.rti_rank_weight > 0:
            if kwargs.get("event_time") is None or kwargs.get("c") is None:
                raise ValueError("event_time and censorship are required for RTI-IPCW ranking")
            rank_loss, pair_count, hard_pair_count = self._rti_directed_ipcw_rank(
                base, delta, kwargs["event_time"], kwargs["c"], wa & oa)
        auxiliary = self.rti_rank_weight * rank_loss
        error = torch.maximum((plan.sum(-1)-1./sw.size(1)).abs().amax(),
                              (plan.sum(-2)-1./so.size(1)).abs().amax())
        self.last_training_losses = {"v315_auxiliary": auxiliary.detach(),
            "v315_rti_ipcw_rank": rank_loss.detach(), "v315_rank_pairs": pair_count,
            "v315_hard_rank_pairs": hard_pair_count,
            "v315_marginal_error": error.detach(), "v315_interaction_rms": moment.detach().square().mean().sqrt(),
            "v315_delta_logit_rms": delta.detach().square().mean().sqrt()}
        for name, s, a, available in (("wsi", sw, aw, wa), ("omic", so, ao, oa)):
            cosine, std, overlap = self._slot_metrics(s, a, available)
            self.last_training_losses.update({f"v315_{name}_cosine": cosine, f"v315_{name}_slot_std": std,
                                               f"v315_{name}_attention_overlap": overlap})
        # Detached caches only; explaining does not retain the training graph.
        self.last_explanations = {"global_logits": base.detach().float(), "interaction_logits": delta.detach(),
            "interaction": moment.detach(), "transport_plan": plan.detach(), "wsi_slots": sw.detach(),
            "omic_slots": so.detach(), "wsi_assignment": aw.detach(), "omic_assignment": ao.detach(),
            "wsi_residuals": rw.detach(), "omic_residuals": ro.detach(), "paired": (wa & oa).detach(),
            "interaction_weight": self.transport.readout.weight.detach().float().clone(),
            "factual_risk": (-torch.cumprod(torch.sigmoid(-logits), -1).sum(-1)).detach()}
        return logits, auxiliary

    @torch.no_grad()
    def explain_last_batch(self):
        if self.last_explanations is None:
            raise RuntimeError("run forward before explaining")
        result = dict(self.last_explanations)
        plan, w, o = result["transport_plan"], result["wsi_residuals"], result["omic_residuals"]
        excess = plan - 1./(w.size(1)*o.size(1))
        result["pair_logit_contributions"] = torch.einsum("bkl,bkd,bld,cd->bklc", excess, w, o,
            result["interaction_weight"]) * result["paired"][:, None, None, None]
        return result
