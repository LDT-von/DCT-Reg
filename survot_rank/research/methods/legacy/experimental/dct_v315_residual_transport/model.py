"""DCT v3.15: one-shot slots, one residual OT interaction, one NLL.

No imports or inheritance from prior model versions. Attention pooling, cosine
OT, and centered bilinear moments are established building blocks; this is a
project-specific baseline, not a claim of first invention or clinical efficacy.
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
        if min(self.encoding_dim, dim, classes) < 1 or not self.omic_sizes or min(self.omic_sizes) < 1:
            raise ValueError("positive dimensions and nonempty omic_sizes required")
        if not 0 <= dropout < 1 or not 0 <= self.alpha_surv <= 1:
            raise ValueError("invalid dropout or alpha_surv")
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
        self.last_training_losses, self.last_explanations = {}, None

    @staticmethod
    def objective_weights():
        return {"nll": 1., "ipcw_rank": 0., "per_slot_nll": 0.,
                "slot_diversity": 0., "reconstruction": 0.}

    def get_extra_state(self):
        return {"schema": 1, "architecture": "one_shot_rti", "alpha_surv": self.alpha_surv,
                "mode": self.transport.mode, "epsilon": self.transport.epsilon,
                "iterations": self.transport.iterations, "dropout": self.global_head[3].p,
                "omic_sizes": self.omic_sizes}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("v3.15 checkpoint recipe differs from constructed model")

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
        error = torch.maximum((plan.sum(-1)-1./sw.size(1)).abs().amax(),
                              (plan.sum(-2)-1./so.size(1)).abs().amax())
        self.last_training_losses = {"v315_auxiliary": logits.new_zeros(()),
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
        return logits, logits.new_zeros(())

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
