"""DCT v3.11: Per-Slot Interpretability via Direct Survival Supervision.

This module replaces the direction loss (DCR ≈ 0.526 ≈ random) with a
mechanism that directly supervises slot-level representations:

1. **Per-Slot NLL**: Each WSI slot and each Omics slot independently predicts
   survival outcomes through its own hazard head. Gradients flow directly back
   to the slot attention mechanism — no Sinkhorn bottleneck in the gradient path.

2. **Slot Diversity Loss**: Forces slot predictions to have non-trivial variance,
   preventing the model from collapsing all slots to the same representation.
   The diversity target is a bounded range [σ²_min, σ²_max], enforced by a
   two-sided hinge that cannot be bypassed by the encoder or OT plan.

Key difference from v3.10:
  - v3.10: direction loss gradient is attenuated by Sinkhorn (epsilon=0.05)
  - v3.11: per-slot NLL gradient reaches slot attention directly; diversity
    constraint lives purely in representation space

This class keeps all v3.10 frozen invariants except:
  - dct_v38_lambda_direction = 0.0  (direction loss disabled)
  - New: per_slot_nll and slot_diversity objectives
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.dct_v310_directional_regularized_transport.model import (
    DCTV310DirectionalRegularizedTransport,
)


class DCTV311SlotInterpretable(DCTV310DirectionalRegularizedTransport):
    """DCT v3.11: per-slot survival supervision + diversity constraint.

    Replaces the direction loss with a mechanism that makes slot representations
    directly interpretable via independent hazard predictions.  The key invariant:

    - Every WSI slot and every Omics slot has its own hazard head.
    - Each head is supervised by IPCW-weighted NLL.
    - Slot predictions must vary (σ² > 0) so the model cannot collapse them.

    The OT plan and event encoder remain intact; they still contribute to the
    main survival prediction.  The new losses add *additional* supervision that
    the gradient analysis proved the direction loss could not provide.
    """

    NLL_WEIGHT = 1.0
    IPCW_RANK_WEIGHT = 0.10
    # Direction loss is disabled because DCR ≈ 0.526 ≈ random (audit results 2026).
    DIRECTION_WEIGHT = 0.0

    # New v3.11 objective weights (frozen, paper-facing).
    PER_SLOT_NLL_WEIGHT = 0.05
    SLOT_DIVERSITY_WEIGHT = 0.02

    # Diversity target range (σ² per sample, per batch).
    VARIANCE_MIN = 0.005
    VARIANCE_MAX = 0.050

    FROZEN_ARGUMENTS = dict(DCTV310DirectionalRegularizedTransport.FROZEN_ARGUMENTS)
    FROZEN_ARGUMENTS.update({
        # Disable direction loss explicitly.
        "dct_v38_lambda_direction": 0.0,
        # Disable dose and reconfiguration (they depend on direction loss).
        "dct_v38_lambda_dose": 0.0,
        "dct_v38_lambda_reconfiguration": 0.0,
        # Per-slot NLL weight.
        "dct_v311_lambda_slot_nll": PER_SLOT_NLL_WEIGHT,
        # Slot diversity weight.
        "dct_v311_lambda_slot_diversity": SLOT_DIVERSITY_WEIGHT,
        # Diversity variance bounds.
        "dct_v311_variance_min": VARIANCE_MIN,
        "dct_v311_variance_max": VARIANCE_MAX,
        # Zero out all v3.8 structural losses (dose/reconfig) for a clean baseline.
        "dct_v38_warmup_epochs": 0,
        "dct_v38_ramp_epochs": 0,
    })

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        # Enforce frozen recipe before any parent construction.
        for name, value in self.FROZEN_ARGUMENTS.items():
            setattr(args, name, value)
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        # ---- Per-Slot Hazard Heads ----
        # Gradient path: per_slot_nll → slot_hazard_head → slots → slot_attention
        # No Sinkhorn, no OT plan, no event_encoder in this gradient chain.
        dim = self.wsi_projection_dim
        num_classes = self.num_classes
        num_wsi_slots = int(getattr(args, "slot_num_wsi", 8))
        num_omic_slots = int(getattr(args, "slot_num_omics", 4))

        self.per_slot_hazard_wsi = nn.Linear(dim, num_classes)
        self.per_slot_hazard_omic = nn.Linear(dim, num_classes)
        self.num_wsi_slots = num_wsi_slots
        self.num_omic_slots = num_omic_slots

        # ---- Model attribute sync (for diagnostics) ----
        self.dct_lambda_ipcw_rank = self.IPCW_RANK_WEIGHT
        self.dct_v38_lambda_direction = 0.0
        self.dct_v38_lambda_dose = 0.0
        self.dct_v38_lambda_reconfiguration = 0.0

        # ---- Diagnostics (updated each forward pass) ----
        self._last_per_slot_nll = 0.0
        self._last_slot_diversity = 0.0
        self._last_slot_variance = 0.0

    @classmethod
    def objective_weights(cls) -> dict[str, float]:
        """Return the immutable paper objective for manifests and tests."""
        return {
            "nll": cls.NLL_WEIGHT,
            "ipcw_rank": cls.IPCW_RANK_WEIGHT,
            "per_slot_nll": cls.PER_SLOT_NLL_WEIGHT,
            "slot_diversity": cls.SLOT_DIVERSITY_WEIGHT,
        }

    @classmethod
    def key_contributions(cls) -> list[str]:
        """Return DCT v3.11 contribution claims in priority order."""
        return [
            "Per-slot survival supervision: each WSI/Omics slot independently predicts risk",
            "Diversity constraint: prevents slot collapse, enables multi-subtype representation",
            "IPCW-aware ranking for reliable survival curves",
            "Interpretable: slot hazard predictions are directly readable (no OT bottleneck)",
        ]

    # ============================================================
    #  Per-Slot NLL Loss
    # ============================================================

    def _nll_surv_per_slot(self, hazard, y_onehot, event_mask, censor_mask, ipcw):
        """Compute IPCW-weighted discrete-time NLL for a [B, K, C] hazard tensor.

        Discrete-time NLL:
            L = -Σ_t [ I(T=t, E=1) * log H(t) + I(T>=t) * log(1-H(t)) ]
              = event_NLL + survival_NLL

        Args:
            hazard: [B, K, C] sigmoid hazard per slot
            y_onehot: [B, C] one-hot time bin
            event_mask: [B, 1] True where event is observed
            censor_mask: [B, 1] True where patient is censored
            ipcw: [B, 1] IPCW weight per patient

        Returns:
            [B, K] NLL loss per slot, already IPCW-weighted.
        """
        eps = 1e-7
        log_h = (hazard + eps).clamp_max(1.0 - eps).log()
        log_s = (1.0 - hazard + eps).clamp_max(1.0 - eps).log()

        # Event NLL: -log H(t*) per slot.
        event_nll = -(y_onehot.unsqueeze(1) * log_h).sum(dim=-1)            # [B, K]

        # Survival NLL: -Σ_{t'} log(1-H(t')) up to event time, per slot.
        survival_nll = -(y_onehot.unsqueeze(1) * torch.cumsum(log_s, dim=-1)).sum(dim=-1)  # [B, K]

        nll = event_nll * event_mask + survival_nll * censor_mask         # [B, K]
        return nll * ipcw                                                    # [B, K]

    def per_slot_nll_loss(self, slots_wsi, slots_omic, y, event_time, c):
        """IPCW-weighted NLL per slot, averaged across all slots.

        Args:
            slots_wsi: [B, K_w, D] WSI slot representations
            slots_omic: [B, K_o, D] Omics slot representations
            y: [B] discrete time-bin labels (long) OR [B, C] one-hot floats
            event_time: [B] raw event times (not used directly, for IPCW)
            c: [B] censorship flag (0=event, 1=censored)

        Returns:
            Scalar loss = mean NLL across WSI slots and Omics slots.

        Gradient path (critical difference from direction loss):
            L_slot → per_slot_hazard_* → slots_wsi/omic → slot_attention
            No Sinkhorn, no OT plan, no event_encoder.
        """
        bsz = slots_wsi.size(0)

        # Per-slot hazard predictions.  Per-slot heads broadcast over the
        # time-bin dimension; their out_features must equal the number of
        # discrete bins (``n_classes`` in the trainer).
        hazard_wsi = torch.sigmoid(self.per_slot_hazard_wsi(slots_wsi))    # [B, K_w, C]
        hazard_omic = torch.sigmoid(self.per_slot_hazard_omic(slots_omic)) # [B, K_o, C]
        num_classes = hazard_wsi.size(-1)

        # Normalize y → one-hot of shape [B, num_classes].
        if y.dim() == 1:
            y_idx = y.long().view(bsz)
            if y_idx.max().item() >= num_classes:
                raise ValueError(
                    f"per_slot_nll_loss: label index {int(y_idx.max())} "
                    f">= num_classes {num_classes}"
                )
            y_onehot = torch.zeros(bsz, num_classes, device=y.device, dtype=hazard_wsi.dtype)
            y_onehot.scatter_(1, y_idx.unsqueeze(1), 1.0)
        elif y.dim() == 2:
            y_onehot = y.float()
        else:
            raise ValueError(
                f"per_slot_nll_loss: y must be [B] or [B, C], got {tuple(y.shape)}"
            )

        event_mask = (c.float() < 0.5).view(bsz, 1)      # [B, 1]
        censor_mask = (c.float() >= 0.5).view(bsz, 1)     # [B, 1]

        # IPCW weights.
        ipcw = self._ipcw(event_time.float()).view(bsz, 1).clamp_min(1e-3)  # [B, 1]

        # WSI slot NLL.
        nll_wsi = self._nll_surv_per_slot(hazard_wsi, y_onehot, event_mask, censor_mask, ipcw)  # [B, K_w]
        loss_wsi = nll_wsi.mean()

        # Omics slot NLL.
        nll_omic = self._nll_surv_per_slot(hazard_omic, y_onehot, event_mask, censor_mask, ipcw)  # [B, K_o]
        loss_omic = nll_omic.mean()

        self._last_per_slot_nll = (loss_wsi.item() + loss_omic.item()) * 0.5
        return 0.5 * (loss_wsi + loss_omic)

    # ============================================================
    #  Slot Diversity Loss
    # ============================================================

    def slot_diversity_loss(self, slots_wsi, slots_omic):
        """Two-sided hinge on variance of per-slot hazard predictions.

        Prevents:
          (a) Collapse: all slots → same hazard → no interpretability
          (b) Noise: all slots → independent random hazard → no structure

        The loss lives entirely in representation space with NO gradient
        path through the OT plan, event encoder, or Sinkhorn.

        Args:
            slots_wsi: [B, K_w, D] WSI slot representations
            slots_omic: [B, K_o, D] Omics slot representations

        Returns:
            Scalar diversity loss.
        """
        hazard_wsi = torch.sigmoid(self.per_slot_hazard_wsi(slots_wsi))   # [B, K_w, C]
        hazard_omic = torch.sigmoid(self.per_slot_hazard_omic(slots_omic))  # [B, K_o, C]

        # Concatenate all slot predictions per sample.
        all_preds = torch.cat([hazard_wsi, hazard_omic], dim=1)   # [B, K_w+K_o, C]

        # Mean hazard across slots per sample.
        mean_pred = all_preds.mean(dim=1, keepdim=True)             # [B, 1, C]

        # Per-sample variance across slots, averaged over hazard bins.
        variance = ((all_preds - mean_pred) ** 2).mean(dim=(1, 2))   # [B]

        # Batch mean variance.
        batch_variance = variance.mean()
        self._last_slot_variance = batch_variance.item()

        # Two-sided hinge.
        margin_min = float(getattr(self.args, "dct_v311_variance_min", self.VARIANCE_MIN))
        margin_max = float(getattr(self.args, "dct_v311_variance_max", self.VARIANCE_MAX))

        loss = F.relu(margin_min - batch_variance) + F.relu(batch_variance - margin_max)
        self._last_slot_diversity = loss.item()
        return loss

    # ============================================================
    #  Full Forward Pass
    # ============================================================

    def forward(self, **kwargs):
        """Forward pass with per-slot survival supervision.

        This re-implements the DCT base forward chain with per-slot NLL
        and diversity injected into the auxiliary loss.  The parent class's
        `_combine_auxiliary_objectives` is NOT called; we replace it here
        to keep the chain clean and avoid accidentally mixing in the
        direction loss from parent classes.
        """
        # ---- Encoding ----
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)

        (
            slots_wsi,
            slots_omic,
            wsi_coord_assign,
            omic_coord_assign,
        ) = self._encode_transport_slots(x_wsi_proj, x_omics, kwargs)

        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        if self.training:
            self._reset_ipcw_memory_for_epoch(epoch)

        # ---- Transport plan ----
        factual_costs, rows, cols, evidence_gate = self._cost_tensor(slots_wsi, slots_omic)
        factual_plans, ot_distance = self._plans_from_cost_tensor(
            factual_costs, rows, cols, epoch, replay_fixed=False
        )
        if self.dct_fixed_coupling:
            self._factual_plan_cache = [
                [plan.detach() for plan in stage_plans]
                for stage_plans in factual_plans
            ]

        self._last_factual_costs = factual_costs.detach()
        self._last_factual_rows = rows.detach()
        self._last_factual_cols = cols.detach()
        self._last_slots_wsi = slots_wsi.detach()
        self._last_slots_omic = slots_omic.detach()

        # ---- Event encoding and risk prediction ----
        factual_logits, factual_gate = self._encode_logits_from_plans(
            slots_wsi, slots_omic, factual_plans
        )
        factual_risk = self._risk(factual_logits)

        # ---- IPCW ranking loss ----
        low_weights = factual_costs.new_zeros(
            factual_costs.size(0), self.spt_num_stages
        )
        high_weights = torch.zeros_like(low_weights)
        ipcw_rank_loss = factual_costs.new_zeros(())

        if kwargs.get("event_time") is not None and kwargs.get("c") is not None:
            low_weights, high_weights = self._stage_membership_weights(
                kwargs["event_time"], kwargs["c"]
            )
            if self.training and self.dct_lambda_ipcw_rank != 0.0:
                ipcw_rank_loss = self._ipcw_pairwise_ranking_loss(
                    factual_logits, kwargs["event_time"], kwargs["c"]
                )
                self._remember_ipcw_batch(
                    factual_risk,
                    kwargs["event_time"].float().view(-1),
                    kwargs["c"].float().view(-1),
                )
            if self.training:
                self._update_risk_anchors(factual_costs.detach(), low_weights, high_weights)

        # ---- Per-slot losses (only during training with labels) ----
        y = kwargs.get("y")
        event_time = kwargs.get("event_time")
        c = kwargs.get("c")

        per_slot_nll = factual_costs.new_zeros(())
        slot_diversity = factual_costs.new_zeros(())

        if self.training and y is not None and event_time is not None and c is not None:
            per_slot_nll = self.per_slot_nll_loss(
                slots_wsi, slots_omic, y, event_time, c
            )
            slot_diversity = self.slot_diversity_loss(slots_wsi, slots_omic)

        # ---- Combine auxiliary objectives (v3.11 recipe) ----
        lambda_slot_nll = float(
            getattr(self.args, "dct_v311_lambda_slot_nll", self.PER_SLOT_NLL_WEIGHT)
        )
        lambda_diversity = float(
            getattr(self.args, "dct_v311_lambda_slot_diversity", self.SLOT_DIVERSITY_WEIGHT)
        )

        aux_loss = (
            self.IPCW_RANK_WEIGHT * ipcw_rank_loss
            + lambda_slot_nll * per_slot_nll
            + lambda_diversity * slot_diversity
        )

        # ---- Training diagnostics ----
        if self.training:
            active_stage_fraction = (
                ((low_weights > 0).any(dim=0) & (high_weights > 0).any(dim=0))
                .to(factual_costs.dtype)
                .mean()
            )
            import math
            row_entropy = -(
                rows.clamp_min(1e-8) * rows.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(max(2, rows.size(-1)))
            col_entropy = -(
                cols.clamp_min(1e-8) * cols.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(max(2, cols.size(-1)))

            self.last_training_losses = {
                "ot": ot_distance.detach(),
                "ipcw_rank": ipcw_rank_loss.detach(),
                "v311_per_slot_nll": (
                    per_slot_nll.detach()
                    if torch.is_tensor(per_slot_nll)
                    else factual_costs.new_tensor(float(per_slot_nll))
                ),
                "v311_slot_diversity": (
                    slot_diversity.detach()
                    if torch.is_tensor(slot_diversity)
                    else factual_costs.new_tensor(float(slot_diversity))
                ),
                "v311_slot_variance": factual_costs.new_tensor(self._last_slot_variance),
                "v311_per_slot_nll_lambda": factual_costs.new_tensor(lambda_slot_nll),
                "v311_slot_diversity_lambda": factual_costs.new_tensor(lambda_diversity),
                "active_stage_fraction": active_stage_fraction.detach(),
                "anchor_coverage": self.risk_anchor_seen.to(factual_costs.dtype).mean().detach(),
                "evidence_marginal_entropy": torch.cat(
                    [row_entropy.flatten(), col_entropy.flatten()]
                ).mean().detach(),
            }

        # ---- Evaluation mode: counterfactual audit ----
        if not self.training:
            low_costs, high_costs = self._counterfactual_costs(factual_costs)
            low_plans, _ = self._plans_from_cost_tensor(
                low_costs, rows, cols, epoch, replay_fixed=True
            )
            high_plans, _ = self._plans_from_cost_tensor(
                high_costs, rows, cols, epoch, replay_fixed=True
            )
            low_logits, _ = self._encode_logits_from_plans(slots_wsi, slots_omic, low_plans)
            high_logits, _ = self._encode_logits_from_plans(slots_wsi, slots_omic, high_plans)
            low_risk = self._risk(low_logits)
            high_risk = self._risk(high_logits)

            self.last_explanations = {
                "stage_slot_pair_evidence": torch.stack(
                    [item[0] for item in factual_plans], dim=1
                ).detach(),
                "evidence_gate": evidence_gate.detach(),
                "wsi_coordinate_assignment": wsi_coord_assign.detach(),
                "omic_coordinate_assignment": omic_coord_assign.detach(),
                "factual_risk": factual_risk.detach(),
                "low_risk_counterfactual": low_risk.detach(),
                "high_risk_counterfactual": high_risk.detach(),
                "counterfactual_risk_delta_low": (low_risk - factual_risk).detach(),
                "counterfactual_risk_delta_high": (high_risk - factual_risk).detach(),
                "risk_anchor_costs": self.risk_anchor_costs.detach(),
                "risk_anchor_seen": self.risk_anchor_seen.detach(),
                "stage_edges": self.dct_stage_edges.detach(),
                "event_gate": factual_gate.detach(),
                # Per-slot hazard predictions (new interpretability output).
                "per_slot_hazard_wsi": torch.sigmoid(
                    self.per_slot_hazard_wsi(slots_wsi)
                ).detach(),
                "per_slot_hazard_omic": torch.sigmoid(
                    self.per_slot_hazard_omic(slots_omic)
                ).detach(),
            }

        return factual_logits, aux_loss
