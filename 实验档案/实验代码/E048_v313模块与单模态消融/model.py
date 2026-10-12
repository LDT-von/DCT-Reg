"""Isolated, retrainable DCT v3.13 controls; historical models stay untouched.

Unimodal arms never encode or consume the other modality. Structural controls
are applied in both train and eval, including every consumer of the OT plans.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import (
    DCTV313TransportReconstruction,
)

ARM_NAMES = (
    "full", "wsi_only", "omics_only", "concat", "no_self", "no_cross",
    "no_reconstruction", "direct", "independent", "no_prototypes",
    "shared_stage_plan", "single_geometry", "no_event_encoder",
    "pooled_transport", "no_rank", "no_slot_nll", "no_diversity",
)


def configure_arm(args, arm):
    """Resolve effective coefficients before any inherited constructor runs."""
    if arm not in ARM_NAMES:
        raise ValueError(f"Unknown E048 arm {arm!r}; choose from {ARM_NAMES}")
    settings = {
        "dct_v313_recon_weighting": "per_branch",
        "dct_v313_lambda_reconstruction_scale": 1.0,
        "dct_v313_lambda_wsi_reconstruction": 0.0,
        "dct_v313_disable_self_reconstruction": arm in {
            "no_self", "no_reconstruction", "wsi_only"},
        "dct_v313_disable_cross_reconstruction": arm in {
            "no_cross", "no_reconstruction", "wsi_only", "omics_only"},
        "dct_v313_cross_mode": "direct" if arm in {"direct", "concat"} else "transport",
        "dct_v313_plan_mode": "independent" if arm == "independent" else "learned",
        "dct_lambda_ipcw_rank": 0.0 if arm == "no_rank" else 0.10,
        "dct_v311_lambda_slot_nll": 0.0 if arm == "no_slot_nll" else 0.05,
        "dct_v311_lambda_slot_diversity": 0.0 if arm == "no_diversity" else 0.10,
        "dct_v311_target_pair_dist": 1.0,
        "dct_v311_lambda_pair_dist": 0.5,
        "dct_v38_lambda_direction": 0.0,
        "dct_v38_lambda_dose": 0.0,
        "dct_v38_lambda_reconfiguration": 0.0,
    }
    for key, value in settings.items():
        setattr(args, key, value)
    args._dct_user_overrides = set(getattr(args, "_dct_user_overrides", ())) | set(settings)
    return args


class E048Control(DCTV313TransportReconstruction):
    def __init__(self, args, arm, omic_input_dim=None, omic_names=None, pathway_names=None):
        if getattr(args, "dct_v313_revision", "legacy") != "legacy":
            raise ValueError("E048 v1 is a legacy two-reconstruction recipe; use the repaired three-branch ablation plan")
        configure_arm(args, arm)
        super().__init__(args, omic_input_dim, omic_names, pathway_names)
        self.arm = arm
        if arm in {"wsi_only", "omics_only", "concat", "pooled_transport"}:
            width = self.wsi_projection_dim * (2 if arm in {"concat", "pooled_transport"} else 1)
            # Additional heads must not change the inherited RNG continuation.
            with torch.random.fork_rng(devices=[]):
                self.pooled_head = nn.Sequential(
                    nn.LayerNorm(width), nn.Linear(width, self.wsi_projection_dim),
                    nn.GELU(), nn.Linear(self.wsi_projection_dim, self.num_classes),
                )
        self._freeze_bypassed_parameters()

    def _freeze_bypassed_parameters(self):
        if self.arm in {"wsi_only", "omics_only"}:
            active = ({"wsi_mlp", "slot_attention_wsi", "shared_wsi_prototypes", "per_slot_hazard_wsi"}
                      if self.arm == "wsi_only" else
                      {"sig_networks", "slot_attention_omic", "shared_omic_prototypes",
                       "per_slot_hazard_omic", "pathway_reconstruction_decoder"}) | {"pooled_head"}
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.split(".")[0] in active)
        else:
            bypassed = set()
            if self.arm == "concat":
                bypassed |= {"fusion", "stage_pair_cost", "evidence_gate", "stage_embedding",
                             "event_encoder", "event_norm", "event_hazard", "event_gate", "stage_score"}
            if self.arm == "pooled_transport":
                bypassed |= {"fusion", "event_encoder", "event_norm", "event_hazard", "event_gate"}
            if self.arm == "no_event_encoder":
                bypassed.add("event_encoder")
            if self.arm == "no_prototypes":
                bypassed |= {"shared_wsi_prototypes", "shared_omic_prototypes"}
            if self.arm == "no_reconstruction":
                bypassed.add("pathway_reconstruction_decoder")
            for name, parameter in self.named_parameters():
                if name.split(".")[0] in bypassed:
                    parameter.requires_grad_(False)

    def audit(self):
        return {
            "arm": self.arm, "modality": self.arm if self.arm.endswith("_only") else "wsi+omics",
            "rank": self.dct_lambda_ipcw_rank,
            "slot_nll": self.dct_v313_lambda_slot_nll_effective,
            "diversity": self.dct_v313_lambda_slot_diversity_effective,
            "self": self.dct_v313_reconstruction_self_coef,
            "cross": self.dct_v313_reconstruction_cross_coef,
            "wsi_reconstruction": 0.0,
            "reconstruction_target": "detached_encoded_pathway_tokens",
            "cross_mode": self._cross_mode, "plan_mode": self._plan_mode,
            "stages": self.spt_num_stages, "output_time_bins": self.num_classes,
            "parameters_total": sum(p.numel() for p in self.parameters()),
            "parameters_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def _encode_transport_slots(self, x_wsi_proj, x_omics, kwargs):
        if self.arm == "no_prototypes":
            sw = self.slot_attention_wsi(x_wsi_proj)
            so = self.slot_attention_omic(x_omics)
            # The inherited eval diagnostics require an assignment tensor.
            # Local slots retain their own coordinates when prototypes are absent.
            aw = torch.eye(sw.size(1), device=sw.device, dtype=sw.dtype).expand(sw.size(0), -1, -1)
            ao = torch.eye(so.size(1), device=so.device, dtype=so.dtype).expand(so.size(0), -1, -1)
            return sw, so, aw, ao
        return super()._encode_transport_slots(x_wsi_proj, x_omics, kwargs)

    def _cost_tensor(self, slots_wsi, slots_omic):
        costs, rows, cols, gates = super()._cost_tensor(slots_wsi, slots_omic)
        if self.arm == "shared_stage_plan":
            # Tie stage-specific cost AND marginals, keeping the four-event reader.
            costs = costs[:, :1].expand_as(costs)
            rows = rows[:, :1].expand_as(rows)
            cols = cols[:, :1].expand_as(cols)
            gates = gates[:, :1].expand_as(gates)
        if self.arm == "single_geometry":
            # Cosine only, replicated into the existing three-channel interface.
            # This is a geometry-information control, not a speed benchmark.
            costs = costs[:, :, :1].expand_as(costs)
        return costs, rows, cols, gates

    def _encode_logits_from_plans(self, slots_wsi, slots_omic, plans):
        if self.arm == "pooled_transport":
            gate = slots_wsi.new_full((slots_wsi.size(0), len(plans)), 1.0 / len(plans))
            transported, _ = self._transport_wsi_to_omic(slots_wsi, plans, gate)
            logits = self.pooled_head(torch.cat([transported.mean(1), slots_omic.mean(1)], dim=-1))
            return logits, gate
        if self.arm == "no_event_encoder":
            tokens = self.event_norm(self._selected_stage_events(slots_wsi, slots_omic, plans)
                                     + self.stage_embedding.unsqueeze(0))
            gate = torch.softmax(self.event_gate(tokens).squeeze(-1), dim=1)
            return torch.einsum("be,bec->bc", gate, self.event_hazard(tokens)), gate
        return super()._encode_logits_from_plans(slots_wsi, slots_omic, plans)

    def _rank_loss(self, logits, kwargs):
        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        self._reset_ipcw_memory_for_epoch(epoch)
        if self.dct_lambda_ipcw_rank == 0.0:
            return logits.new_zeros(())
        rank = self._ipcw_pairwise_ranking_loss(logits, kwargs["event_time"], kwargs["c"])
        self._remember_ipcw_batch(self._risk(logits), kwargs["event_time"].float().view(-1),
                                 kwargs["c"].float().view(-1))
        return rank

    def _single_losses(self, slots, modality, kwargs):
        head = self.per_slot_hazard_wsi if modality == "wsi" else self.per_slot_hazard_omic
        hazard = torch.sigmoid(head(slots))
        y = kwargs["y"]
        onehot = F.one_hot(y.long(), self.num_classes).to(hazard.dtype) if y.ndim == 1 else y.to(hazard.dtype)
        censor = kwargs["c"].float().view(-1, 1)
        ipcw = self._ipcw(kwargs["event_time"].float()).view(-1, 1).clamp_min(1e-3)
        nll = self._nll_surv_per_slot(hazard, onehot, censor < .5, censor >= .5, ipcw).mean()
        variance = ((hazard - hazard.mean(1, keepdim=True)) ** 2).mean((1, 2))
        lower = float(self.args.dct_v311_variance_min)
        upper = float(self.args.dct_v311_variance_max)
        diversity = (F.relu(lower - variance) + F.relu(variance - upper)).mean()
        if slots.size(1) > 1:
            pair_sq = (slots.unsqueeze(2) - slots.unsqueeze(1)).square().sum(-1)
            mask = 1 - torch.eye(slots.size(1), device=slots.device).unsqueeze(0)
            # Preserve the inherited recipe's pair statistic for this matched control.
            divisor = max(float(mask.sum()) / slots.size(0), 1.0)
            distance = (pair_sq * mask).sum((1, 2)) / divisor
            diversity = diversity + .5 * F.relu(1.0 - distance).mean()
        return nll, diversity

    def _forward_unimodal(self, kwargs):
        is_wsi = self.arm == "wsi_only"
        if is_wsi:
            tokens = self.wsi_mlp(kwargs["x_wsi"])
            local = self.slot_attention_wsi(tokens)
            slots, _ = self._semantic_slots(local, self.shared_wsi_prototypes)
        else:
            tokens = self._encode_omics(kwargs)
            local = self.slot_attention_omic(tokens)
            slots, _ = self._semantic_slots(local, self.shared_omic_prototypes)
        logits = self.pooled_head(slots.mean(1))
        if not self.training:
            return logits, 0.0
        rank = self._rank_loss(logits, kwargs)
        nll, diversity = self._single_losses(slots, "wsi" if is_wsi else "omics", kwargs)
        reconstruction = logits.new_zeros(())
        if not is_wsi:
            available = self._omics_available_mask(kwargs, tokens)
            reconstruction = self._reconstruction_distance(
                self.pathway_reconstruction_decoder(slots), tokens.detach(), available)
        ramp = self._reconstruction_ramp(int(getattr(self.args, "cur_epoch", 0)))
        aux = (.10 * rank + .05 * nll + .10 * diversity
               + ramp * self.dct_v313_reconstruction_self_coef * reconstruction)
        self.last_training_losses = {"ipcw_rank": rank.detach(), "v311_per_slot_nll": nll.detach(),
                                     "v311_slot_diversity": diversity.detach(),
                                     "v313_reconstruction_self": reconstruction.detach(),
                                     "v313_reconstruction_cross": logits.new_zeros(())}
        return logits, aux

    def _forward_concat(self, kwargs):
        wsi = self.wsi_mlp(kwargs["x_wsi"])
        omics = self._encode_omics(kwargs)
        sw, so, _, _ = self._encode_transport_slots(wsi, omics, kwargs)
        logits = self.pooled_head(torch.cat([sw.mean(1), so.mean(1)], -1))
        if not self.training:
            return logits, 0.0
        rank = self._rank_loss(logits, kwargs)
        nll = self.per_slot_nll_loss(sw, so, kwargs["y"], kwargs["event_time"], kwargs["c"])
        diversity = self.slot_diversity_loss(sw, so)
        available = self._omics_available_mask(kwargs, omics)
        self_recon = self._reconstruction_distance(self.pathway_reconstruction_decoder(so), omics, available)
        cross_recon = self._reconstruction_distance(self.pathway_reconstruction_decoder(sw), omics, available)
        ramp = self._reconstruction_ramp(int(getattr(self.args, "cur_epoch", 0)))
        aux = (.10 * rank + .05 * nll + .10 * diversity + ramp * (.05 * self_recon + .05 * cross_recon))
        self.last_training_losses = {"ipcw_rank": rank.detach(), "v311_per_slot_nll": nll.detach(),
                                     "v311_slot_diversity": diversity.detach(),
                                     "v313_reconstruction_self": self_recon.detach(),
                                     "v313_reconstruction_cross": cross_recon.detach()}
        return logits, aux

    def forward(self, **kwargs):
        if self.arm in {"wsi_only", "omics_only"}:
            return self._forward_unimodal(kwargs)
        if self.arm == "concat":
            return self._forward_concat(kwargs)
        return super().forward(**kwargs)


def build_model(args, arm, omic_input_dim=None, omic_names=None, pathway_names=None):
    return E048Control(args, arm, omic_input_dim, omic_names, pathway_names)
