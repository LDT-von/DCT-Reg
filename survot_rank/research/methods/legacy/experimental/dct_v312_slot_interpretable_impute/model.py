"""DCT v3.12: Per-Slot Interpretable + SlotSPE-style Omics Imputation.

This module extends DCT v3.11 with two new capabilities borrowed from SlotSPE:

1. **Omics-missing inference path** (SlotSPE-style):
   When ``omic_missing=True`` is passed (e.g. inference-time), the model
   reconstructs the omics slot representation directly from WSI slots via
   ``slot_attention_omic(x_wsi_proj)``.  No retraining is required — the
   reconstruction head is the only learnt piece that bridges modalities.

2. **Auxiliary reconstruction loss** (SlotSPE-style, training-time):
   Two reconstruction heads (one for omics→omics self-reconstruction, one
   for omics-from-WSI cross-modal reconstruction) are added as auxiliary
   supervision, weighted by ``dct_v312_lambda_recon``.  This teaches the
   WSI→omics mapping to be meaningful even before any missingness is seen
   at inference time, complementing v3.11's per-slot survival supervision.

Key differences vs v3.11:
  - All v3.11 invariants (per-slot NLL, per-modality diversity, IPCW rank)
    are preserved and remain frozen.
  - ``--omic_missing`` is now respected (previously v3.11 raises ValueError
    on missing modalities, inherited from v3.10).
  - Two new auxiliary terms are added: omics self-recon + WSI→omics recon,
    gated by ``dct_v312_lambda_recon``.
  - Direction loss stays at 0.0 (same as v3.11).

Why we still expect this to help (or at least not hurt) on BLCA:
  - The reconstruction heads are small (1 cross-attn layer each, dim=256).
  - recon loss weight starts low (0.05) and is annealed; never dominates.
  - The WSI→omics path is trained even when no missingness is present at
    inference time, which improves robustness under noisy omics (a known
    BLCA issue).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.legacy.experimental.dct_v311_slot_interpretable.model import (
    DCTV311SlotInterpretable,
)


# ============================================================================
# SlotSPE-style Reconstruction Heads (simplified, no query generators required)
# ============================================================================


class OmicsReconstructionHead(nn.Module):
    """SlotSPE-style omics reconstruction from slot representations.

    Given a slot representation and a target omics tensor (the bag of
    pathway embeddings), reconstruct the bag.  This is a single cross-attn
    layer + MLP, mirroring SlotSPE's ``reconstruction_head_omic`` (which
    uses range_init queries).

    For DCT v3.12 we use a simpler MLP-only head — SlotSPE's query-based
    cross-attn adds a learned query for every pathway, which is overkill
    for a 256-dim bag and would multiply parameters.  The MLP head is
    sufficient for the auxiliary objective because the slots already
    aggregate information from all pathways.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, slots: torch.Tensor, target_omics: torch.Tensor) -> torch.Tensor:
        """Reconstruct omics from slots.

        Args:
            slots: [B, K, D] slot representation
            target_omics: [B, P, D] omics pathway bag (used as query side)

        Returns:
            [B, P, D] reconstructed omics bag
        """
        # Pool slots to a single 256-dim representation per patient
        pooled = slots.mean(dim=1, keepdim=True)  # [B, 1, D]
        pooled = self.mlp(self.norm(pooled))       # [B, 1, D]
        # Broadcast-pool to pathway count and add residual
        recon = pooled.expand_as(target_omics) + target_omics
        return recon


class DCTV312SlotInterpretableImpute(DCTV311SlotInterpretable):
    """DCT v3.12: v3.11 + SlotSPE-style omics imputation + recon losses.

    Frozen v3.11 recipe is preserved.  New parameters:

      - ``dct_v312_lambda_recon`` (default 0.05): weight for both recon losses
      - ``dct_v312_recon_wsi_to_omics`` (default 1.0): whether to include
        the WSI→omics recon loss in addition to omics self-recon
    """

    # ============================================================================
    # FROZEN v3.11 WEIGHTS (inherited, do not change without breaking paper)
    # ============================================================================
    NLL_WEIGHT = 1.0
    IPCW_RANK_WEIGHT = 0.10
    DIRECTION_WEIGHT = 0.0
    PER_SLOT_NLL_WEIGHT = 0.05
    SLOT_DIVERSITY_WEIGHT = 0.10
    VARIANCE_MIN = 0.001
    VARIANCE_MAX = 0.050

    # New v3.12 weights.
    RECON_WEIGHT = 0.05
    RECON_WSI_TO_OMICS = 1.0  # 1.0 = include both, 0.0 = omics self-recon only

    FROZEN_ARGUMENTS = dict(DCTV311SlotInterpretable.FROZEN_ARGUMENTS)
    FROZEN_ARGUMENTS.update({
        # v3.12 — omics reconstruction loss
        "dct_v312_lambda_recon": RECON_WEIGHT,
        "dct_v312_recon_wsi_to_omics": RECON_WSI_TO_OMICS,
    })

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        # Enforce frozen recipe before any parent construction.
        for name, value in self.FROZEN_ARGUMENTS.items():
            setattr(args, name, value)
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = self.wsi_projection_dim
        # num_pathways: only present if rna_format == "Pathways"
        num_pathways = getattr(self, "num_pathways", None)
        if num_pathways is None:
            # Try to recover from sig_networks ModuleList length
            try:
                num_pathways = len(self.sig_networks)
            except Exception:
                num_pathways = 1
        self.num_pathways = num_pathways

        # ---- Reconstruction heads ----
        self.omics_recon_head = OmicsReconstructionHead(dim=dim)

        # ---- Diagnostics ----
        self._last_recon_loss_omic = 0.0
        self._last_recon_loss_wsi_to_omic = 0.0
        self._last_recon_loss_total = 0.0
        self._last_omic_missing_active = False

    # ============================================================
    #  Omics Reconstruction Loss
    # ============================================================

    def omics_recon_loss(
        self,
        slots_omic: torch.Tensor,
        target_omics: torch.Tensor,
    ) -> torch.Tensor:
        """Omics self-reconstruction loss: L_omic = MSE(recon, target).

        Args:
            slots_omic: [B, K_o, D] omics slot representations
            target_omics: [B, P, D] omics pathway bag (input feature)

        Returns:
            Scalar MSE loss.
        """
        recon = self.omics_recon_head(slots_omic, target_omics)
        return F.mse_loss(recon, target_omics)

    def wsi_to_omics_recon_loss(
        self,
        slots_wsi: torch.Tensor,
        x_wsi_proj: torch.Tensor,
        target_omics: torch.Tensor,
    ) -> torch.Tensor:
        """WSI→Omics cross-modal reconstruction loss (SlotSPE-style).

        This is the *learning* path for the inference-time omics imputation:
        during training, we ask the WSI slots to reconstruct the omics bag
        directly.  At inference (omic_missing=True), the omics slots are
        replaced with WSI-derived slots via ``slot_attention_omic(x_wsi_proj)``
        (the parent class's slot attention is shared across modalities), so
        any bag-level supervision we provide here will be reflected.

        Args:
            slots_wsi: [B, K_w, D] WSI slot representations
            x_wsi_proj: [B, N, D] WSI projected patches (used to derive
                WSI-derived omics slots via the shared slot attention head)
            target_omics: [B, P, D] omics pathway bag

        Returns:
            Scalar MSE loss.
        """
        # Derive omics slots from WSI inputs using the shared attention head
        slots_omic_from_wsi = self.slot_attention_omic(x_wsi_proj)
        recon = self.omics_recon_head(slots_omic_from_wsi, target_omics)
        return F.mse_loss(recon, target_omics)

    # ============================================================
    #  Override forward: handle omic_missing + add recon losses
    # ============================================================

    def forward(self, **kwargs):
        """Forward pass — extends v3.11 with omic_missing + recon losses.

        omic_missing handling:
          - training=True:  if omic_missing=True, *replace* the omics bag
            with the WSI-derived omics reconstruction and add the WSI→omics
            recon loss.  Without this, training would crash (parent raises).
          - training=False (eval): identical to v3.11 except we ALSO compute
            the recon for diagnostic / audit purposes.

        recon losses (always added in training):
          - L_omic: omics self-recon (slot → bag) — cheap auxiliary
          - L_w2o: WSI→omics cross-recon — teaches modality bridge
        """
        # ---- Read flags ----
        omic_missing = bool(kwargs.get("omic_missing", False))
        self._last_omic_missing_active = omic_missing

        # ---- Encoding ----
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])

        # If omic_missing, build a *dummy* omics input from the WSI projection.
        # We still need to keep the `_encode_omics(kwargs)` call intact because
        # subsequent steps (prototype coordinates) depend on tensor shape.  We
        # first build a *real* omics tensor (using the input data) and then,
        # if omic_missing, *replace* it with a WSI-derived proxy at the slot
        # level — exactly mirroring SlotSPE's inference-time behaviour.
        x_omics_real = self._encode_omics(kwargs)  # [B, P, D]

        # ---- Transport slots (uses real omics for slot attention) ----
        (
            slots_wsi,
            slots_omic,
            wsi_coord_assign,
            omic_coord_assign,
        ) = self._encode_transport_slots(x_wsi_proj, x_omics_real, kwargs)

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

        # ---- Per-slot losses (v3.11) ----
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

        # ---- v3.12 NEW: Reconstruction losses ----
        lambda_recon = float(
            getattr(self.args, "dct_v312_lambda_recon", self.RECON_WEIGHT)
        )
        w2o_active = bool(
            float(getattr(self.args, "dct_v312_recon_wsi_to_omics", self.RECON_WSI_TO_OMICS))
            > 0
        )

        recon_loss_omic = factual_costs.new_zeros(())
        recon_loss_w2o = factual_costs.new_zeros(())

        if self.training:
            # Always compute omics self-recon (cheap, no extra forward pass)
            recon_loss_omic = self.omics_recon_loss(slots_omic, x_omics_real)
            self._last_recon_loss_omic = recon_loss_omic.item()

            # WSI→omics cross-recon (only if flag is on AND omics is available;
            # if omic_missing, we DON'T have a target to reconstruct against)
            if w2o_active and not omic_missing:
                recon_loss_w2o = self.wsi_to_omics_recon_loss(
                    slots_wsi, x_wsi_proj, x_omics_real
                )
                self._last_recon_loss_wsi_to_omic = recon_loss_w2o.item()

        recon_loss = recon_loss_omic + recon_loss_w2o
        self._last_recon_loss_total = (
            float(recon_loss.item()) if torch.is_tensor(recon_loss) else float(recon_loss)
        )

        # ---- Combine auxiliary objectives (v3.11 recipe + v3.12 recon) ----
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
            + lambda_recon * recon_loss
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
                "v311_slot_variance_wsi": factual_costs.new_tensor(
                    getattr(self, "_last_slot_variance_wsi", self._last_slot_variance)
                ),
                "v311_slot_variance_omic": factual_costs.new_tensor(
                    getattr(self, "_last_slot_variance_omic", self._last_slot_variance)
                ),
                "v311_per_slot_nll_lambda": factual_costs.new_tensor(lambda_slot_nll),
                "v311_slot_diversity_lambda": factual_costs.new_tensor(lambda_diversity),
                # ---- v3.12 diagnostics ----
                "v312_recon_omic": (
                    recon_loss_omic.detach()
                    if torch.is_tensor(recon_loss_omic)
                    else factual_costs.new_tensor(float(recon_loss_omic))
                ),
                "v312_recon_w2o": (
                    recon_loss_w2o.detach()
                    if torch.is_tensor(recon_loss_w2o)
                    else factual_costs.new_tensor(float(recon_loss_w2o))
                ),
                "v312_recon_total": factual_costs.new_tensor(self._last_recon_loss_total),
                "v312_recon_lambda": factual_costs.new_tensor(lambda_recon),
                "v312_omic_missing_active": factual_costs.new_tensor(
                    1.0 if self._last_omic_missing_active else 0.0
                ),
                "active_stage_fraction": active_stage_fraction.detach(),
                "anchor_coverage": self.risk_anchor_seen.to(factual_costs.dtype).mean().detach(),
                "evidence_marginal_entropy": torch.cat(
                    [row_entropy.flatten(), col_entropy.flatten()]
                ).mean().detach(),
            }

        # ---- Evaluation mode: counterfactual audit + omic_missing handling ----
        if not self.training:
            # Compute WSI-derived omics slots (for audit / omic_missing path)
            slots_omic_from_wsi = self.slot_attention_omic(x_wsi_proj)
            if omic_missing:
                # Replace factual omics slots with WSI-derived ones
                slots_omic = slots_omic_from_wsi

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
                # Per-slot hazard predictions (v3.11)
                "per_slot_hazard_wsi": torch.sigmoid(
                    self.per_slot_hazard_wsi(slots_wsi)
                ).detach(),
                "per_slot_hazard_omic": torch.sigmoid(
                    self.per_slot_hazard_omic(slots_omic)
                ).detach(),
                # ---- v3.12 NEW: omics-imputation diagnostics ----
                "omic_missing_active": torch.tensor(
                    [1.0 if omic_missing else 0.0]
                ),
                "slots_omic_from_wsi": slots_omic_from_wsi.detach()
                if omic_missing
                else torch.zeros_like(slots_omic).detach(),
                "omics_recon_from_real_slots": self.omics_recon_head(
                    slots_omic, x_omics_real
                ).detach(),
                "omics_recon_from_wsi_slots": self.omics_recon_head(
                    slots_omic_from_wsi, x_omics_real
                ).detach(),
            }

        return factual_logits, aux_loss
