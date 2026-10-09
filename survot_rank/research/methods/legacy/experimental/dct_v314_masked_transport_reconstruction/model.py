"""DCT v3.14: masked transport reconstruction.

This module intentionally does not inherit from any older version's model.
All version-specific per-slot supervision, diversity,
masking, reconstruction, diagnostics, and loss composition are local so the
v3.14 implementation can be isolated without carrying older experiment code.

The factual survival path consumes complete paired modalities.  A separate
auxiliary view replaces a fixed fraction of encoded pathway tokens with a
learned mask token, re-encodes the omics slots, and re-solves Sinkhorn against
the factual WSI slots.  Reconstruction is scored only at masked positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.legacy.experimental.dct_v314_masked_transport_reconstruction.backbone import (
    TransportBackbone,
)


class OmicsTokenReconstructionDecoder(nn.Module):
    """Decode a fixed set of pathway tokens from semantic slot memory."""

    def __init__(self, dim: int, num_pathways: int, num_heads: int) -> None:
        super().__init__()
        self.pathway_queries = nn.Parameter(torch.empty(1, num_pathways, dim))
        nn.init.trunc_normal_(self.pathway_queries, std=0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError(f"slot memory must be [B, K, D], got {tuple(memory.shape)}")
        queries = self.pathway_queries.expand(memory.size(0), -1, -1)
        normalized_memory = self.memory_norm(memory)
        decoded, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        decoded = queries + decoded
        return decoded + self.output_mlp(self.output_norm(decoded))


class DCTV314MaskedTransportReconstruction(
    TransportBackbone
):
    """Version-local v3.14 objective with masked self/OT reconstruction."""

    NLL_WEIGHT = 1.0
    IPCW_RANK_WEIGHT = 0.10
    DIRECTION_WEIGHT = 0.0
    PER_SLOT_NLL_WEIGHT = 0.05
    SLOT_DIVERSITY_WEIGHT = 0.02

    MTR_WEIGHT = 0.10
    MTR_SELF_FRACTION = 0.50
    MTR_CROSS_FRACTION = 0.50
    MTR_MASK_RATIO = 0.20
    MTR_RAMP_START = 2
    MTR_RAMP_EPOCHS = 5

    FROZEN_ARGUMENTS = {
            "dct_lambda_ipcw_rank": IPCW_RANK_WEIGHT,
            "dct_v38_lambda_direction": 0.0,
            "dct_v38_lambda_dose": 0.0,
            "dct_v38_lambda_reconfiguration": 0.0,
            "dct_v38_warmup_epochs": 0,
            "dct_v38_ramp_epochs": 0,
            "dct_lambda_etar": 0.0,
            "dct_lambda_listwise": 0.0,
            "dct_ipcw_rank_margin": 0.02,
            "dct_ipcw_rank_temperature": 0.50,
            "dct_ipcw_max_weight": 10.0,
            "dct_ipcw_rank_memory_size": 64,
            "dct_anchor_momentum": 0.90,
            "dct_evidence_cost_weight": 0.0,
            "dct_evidence_mass_floor": 0.05,
            "dct_evidence_marginal_strength": 1.0,
            "dct_geometry_reliability_strength": 0.0,
            "dct_fixed_coupling": False,
            "dct_random_anchors": False,
            "dct_perm_labels_seed": 0,
            "dct_stage_jitter_fraction": 0.0,
            "dct_mix_ratio": 1.0,
            "spt_lambda_ot": 0.0,
            "spt_lambda_rank": 0.0,
            "spt_lambda_stage": 0.0,
            "dct_v382_lambda_mgptr": 0.0,
            "dct_v382_adaptive_aux_weights": False,
    }

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        if str(getattr(args, "bag_loss", "nll_surv")) != "nll_surv":
            raise ValueError("DCT v3.14 requires bag_loss='nll_surv'")
        if str(getattr(args, "rna_format", "Pathways")) != "Pathways":
            raise ValueError(
                "DCT v3.14 reconstructs masked pathway tokens and requires "
                "rna_format='Pathways'"
            )
        for name, value in self.FROZEN_ARGUMENTS.items():
            setattr(args, name, value)
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = int(self.wsi_projection_dim)
        num_classes = int(self.num_classes)
        self.per_slot_hazard_wsi = nn.Linear(dim, num_classes)
        self.per_slot_hazard_omic = nn.Linear(dim, num_classes)
        self.num_wsi_slots = int(getattr(args, "slot_num_wsi", 8))
        self.num_omic_slots = int(getattr(args, "slot_num_omics", 4))

        heads = self._compatible_num_heads(dim, int(getattr(args, "otehv2_heads", 4)))
        self.pathway_reconstruction_decoder = OmicsTokenReconstructionDecoder(
            dim=dim,
            num_pathways=int(self.num_pathways),
            num_heads=heads,
        )
        self.pathway_mask_token = nn.Parameter(torch.empty(1, self.num_pathways, dim))
        nn.init.trunc_normal_(self.pathway_mask_token, std=0.02)

        self.dct_lambda_ipcw_rank = self.IPCW_RANK_WEIGHT
        self.dct_v38_lambda_direction = 0.0
        self.dct_v38_lambda_dose = 0.0
        self.dct_v38_lambda_reconfiguration = 0.0
        self.dct_v314_lambda_slot_nll = float(getattr(args, "dct_v314_lambda_slot_nll", self.PER_SLOT_NLL_WEIGHT))
        self.dct_v314_lambda_slot_diversity = float(getattr(args, "dct_v314_lambda_slot_diversity", self.SLOT_DIVERSITY_WEIGHT))
        self.dct_v314_lambda_mtr = float(getattr(args, "dct_v314_lambda_mtr", self.MTR_WEIGHT))
        self.dct_v314_mask_ratio = float(getattr(args, "dct_v314_mask_ratio", self.MTR_MASK_RATIO))
        self.reconstruction_mode = str(getattr(args, "dct_v314_reconstruction_mode", "masked"))
        self.alpha_surv = float(getattr(args, "alpha_surv", 0.0))
        for value in (self.dct_v314_lambda_slot_nll, self.dct_v314_lambda_slot_diversity, self.dct_v314_lambda_mtr):
            if not math.isfinite(value) or value < 0:
                raise ValueError("v3.14 loss weights must be finite and nonnegative")
        if not 0 < self.dct_v314_mask_ratio < 1:
            raise ValueError("v3.14 mask ratio must be strictly between zero and one")
        if self.reconstruction_mode not in {"masked", "full", "hybrid", "off"}:
            raise ValueError("invalid v3.14 reconstruction mode")
        if not 0 <= self.alpha_surv <= 1:
            raise ValueError("alpha_surv must be in [0, 1]")
        self.dct_v314_mtr_self_fraction = self.MTR_SELF_FRACTION
        self.dct_v314_mtr_cross_fraction = self.MTR_CROSS_FRACTION
        self.dct_v314_mtr_ramp_start = self.MTR_RAMP_START
        self.dct_v314_mtr_ramp_epochs = self.MTR_RAMP_EPOCHS

        self._last_per_slot_nll = 0.0
        self._last_slot_diversity = 0.0
        self._last_slot_variance = 0.0
        self._last_slot_variance_wsi = 0.0
        self._last_slot_variance_omic = 0.0
        self._last_pathway_mask = None

    @staticmethod
    def _compatible_num_heads(dim: int, preferred: int) -> int:
        heads = max(1, min(int(preferred), int(dim)))
        while dim % heads != 0:
            heads -= 1
        return heads

    def objective_weights(self) -> dict[str, float]:
        return {
            "nll": self.NLL_WEIGHT,
            "ipcw_rank": self.IPCW_RANK_WEIGHT,
            "per_slot_nll": self.dct_v314_lambda_slot_nll,
            "slot_diversity": self.dct_v314_lambda_slot_diversity,
            "masked_transport_reconstruction": self.dct_v314_lambda_mtr if self.reconstruction_mode != "off" else 0.0,
        }

    def get_extra_state(self):
        return {"schema": 3, "weights": self.objective_weights(),
                "reconstruction_mode": self.reconstruction_mode,
                "mask_ratio": self.dct_v314_mask_ratio, "alpha_surv": self.alpha_surv,
                "routing": "prototype_scaled_cosine_v1", "diversity": "unit_content_distance_hinge_v1"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("v3.14 checkpoint recipe differs from constructed model")

    @classmethod
    def key_contributions(cls) -> list[str]:
        return [
            "Masked pathway reconstruction without masked-target input leakage",
            "Auxiliary Sinkhorn re-solve from masked omics slots",
            "Per-slot survival supervision with a weak anti-collapse constraint",
            "IPCW-aware patient-risk ranking",
        ]

    def _nll_surv_per_slot(self, hazard, y_onehot, event_mask, censor_mask, ipcw):
        """Cumulative discrete-time likelihood for every semantic slot."""
        eps = 1e-7
        hazard = hazard.clamp(min=eps, max=1.0 - eps)
        log_hazard = hazard.log()
        log_survival = torch.log1p(-hazard).cumsum(dim=-1)
        log_survival_padded = torch.cat(
            [torch.zeros_like(log_survival[..., :1]), log_survival], dim=-1
        )

        y_index = y_onehot.argmax(dim=-1, keepdim=True)
        y_index = y_index.unsqueeze(1).expand(-1, hazard.size(1), -1)
        survival_before = torch.gather(
            log_survival_padded, dim=-1, index=y_index
        ).squeeze(-1)
        hazard_at_event = torch.gather(log_hazard, dim=-1, index=y_index).squeeze(-1)
        survival_at_censor = torch.gather(
            log_survival_padded,
            dim=-1,
            index=(y_index + 1).clamp(max=log_survival_padded.size(-1) - 1),
        ).squeeze(-1)

        event_nll = -(survival_before + hazard_at_event) * event_mask
        censor_nll = -survival_at_censor * censor_mask
        return (event_nll + censor_nll) * ipcw

    def per_slot_nll_loss(self, slots_wsi, slots_omic, y, event_time, c, wsi_available=None, omic_available=None):
        from survot_rank.research.methods.legacy.experimental.dct_v314_masked_transport_reconstruction.losses import discrete_nll
        losses = []
        for slots, head, available in (
            (slots_wsi, self.per_slot_hazard_wsi, wsi_available),
            (slots_omic, self.per_slot_hazard_omic, omic_available),
        ):
            if available is None:
                available = torch.ones(slots.size(0), device=slots.device, dtype=torch.bool)
            if bool(available.any()):
                # Censoring is already modeled by the likelihood. No second
                # 1/G(T) factor on censored patients (which previously reached 20x).
                losses.append(discrete_nll(head(slots[available]), y[available],
                                           c[available], self.alpha_surv).mean())
        loss = torch.stack(losses).mean() if losses else slots_wsi.reshape(-1)[:0].sum()
        self._last_per_slot_nll = float(loss.detach())
        return loss

    def slot_diversity_loss(self, slots_wsi, slots_omic, wsi_available=None, omic_available=None):
        """Bound excessive content cosine similarity; do not force hazards apart.

        Per-slot NLL gives slots the same patient outcome. Requiring different
        hazards conflicts with that supervision. Here diversity measures actual
        pooled patient content, with a dimensionless [0, 1] penalty.
        """
        losses = []
        for name, slots, head, available in (
            ("wsi", slots_wsi, self.per_slot_hazard_wsi, wsi_available),
            ("omic", slots_omic, self.per_slot_hazard_omic, omic_available),
        ):
            if available is None:
                available = torch.ones(slots.size(0), device=slots.device, dtype=torch.bool)
            selected = slots[available]
            value = slots.new_zeros(())
            cosine = slots.new_zeros(())
            if len(selected) and selected.size(1) > 1:
                content = F.layer_norm(selected.float(), (selected.size(-1),))
                content = F.normalize(content, dim=-1)
                similarity = content @ content.transpose(-1, -2)
                off_diagonal = ~torch.eye(selected.size(1), device=slots.device, dtype=torch.bool)
                pairs = similarity[:, off_diagonal]
                # Unit-vector distance >= sqrt(.2) corresponds to cosine <= .9.
                # A cosine hinge has a vanishing gradient near parallel slots;
                # vector_norm retains a useful subgradient for small NONZERO
                # differences. Exact symmetry still needs distinct routing.
                distance = torch.linalg.vector_norm(
                    content.unsqueeze(2) - content.unsqueeze(1), dim=-1)
                losses.append(F.relu(1. - distance[:, off_diagonal] / math.sqrt(.2)).square().mean())
                cosine = pairs.detach().mean()
                value = head(selected).sigmoid().var(1, unbiased=False).detach().mean()
            setattr(self, "_last_slot_variance_" + name, float(value))
            setattr(self, "_last_slot_cosine_" + name, float(cosine))
        self._last_slot_variance = .5 * (self._last_slot_variance_wsi + self._last_slot_variance_omic)
        result = torch.stack(losses).mean() if losses else slots_wsi.reshape(-1)[:0].sum()
        self._last_slot_diversity = float(result.detach())
        return result

    def _mtr_ramp(self, epoch: int) -> float:
        start = int(self.dct_v314_mtr_ramp_start)
        duration = int(self.dct_v314_mtr_ramp_epochs)
        if duration <= 0:
            return 1.0
        return min(1.0, max(0.0, (float(epoch) - float(start)) / float(duration)))

    @staticmethod
    def _batch_bool_mask(value, *, batch: int, device, default: bool) -> torch.Tensor:
        if value is None:
            return torch.full((batch,), default, dtype=torch.bool, device=device)
        mask = torch.as_tensor(value, device=device)
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("availability mask values must be boolean")
        mask = mask.bool().reshape(-1)
        if mask.numel() == 1:
            return mask.expand(batch)
        if mask.numel() != batch:
            raise ValueError(f"availability mask must contain 1 or {batch} values, got {mask.numel()}")
        return mask

    def _omics_available_mask(self, kwargs, target: torch.Tensor) -> torch.Tensor:
        batch = target.size(0)
        explicit = kwargs.get("omic_available", kwargs.get("omics_available"))
        available = self._batch_bool_mask(
            explicit, batch=batch, device=target.device, default=True
        )
        missing = self._batch_bool_mask(
            kwargs.get("omic_missing"),
            batch=batch,
            device=target.device,
            default=False,
        )
        return available & ~missing

    def _sample_pathway_mask(
        self, target: torch.Tensor, available: torch.Tensor
    ) -> torch.Tensor:
        if target.ndim != 3:
            raise ValueError(f"pathway tokens must be [B, P, D], got {tuple(target.shape)}")
        pathways = target.size(1)
        if pathways < 1:
            raise ValueError("masked reconstruction requires at least one pathway token")
        requested = int(round(pathways * float(self.dct_v314_mask_ratio)))
        masked_count = max(1, requested)
        if pathways > 1:
            masked_count = min(pathways - 1, masked_count)
        else:
            masked_count = 1
        scores = torch.rand(
            target.size(0), pathways, device=target.device, dtype=torch.float32
        )
        indices = scores.topk(masked_count, dim=1, largest=False).indices
        mask = torch.zeros(
            target.size(0), pathways, dtype=torch.bool, device=target.device
        )
        mask.scatter_(1, indices, True)
        return mask & available.view(-1, 1)

    def _apply_pathway_mask(
        self, x_omics: torch.Tensor, pathway_mask: torch.Tensor
    ) -> torch.Tensor:
        if pathway_mask.shape != x_omics.shape[:2]:
            raise ValueError(
                f"pathway mask must be {tuple(x_omics.shape[:2])}, "
                f"got {tuple(pathway_mask.shape)}"
            )
        token = self.pathway_mask_token.to(dtype=x_omics.dtype)
        return torch.where(pathway_mask.unsqueeze(-1), token, x_omics)

    @staticmethod
    def _transport_wsi_to_omic(
        slots_wsi: torch.Tensor,
        plans,
        stage_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(plans) != stage_gate.size(1):
            raise ValueError("transport stages must match the factual event-gate width")
        transported_stages = []
        for stage_plans in plans:
            if len(stage_plans) < 1:
                raise ValueError("every transport stage needs at least one geometry")
            consensus = torch.stack(tuple(stage_plans), dim=0).mean(dim=0)
            column_mass = consensus.sum(dim=1).clamp_min(1e-8)
            transported = torch.einsum("bwo,bwd->bod", consensus, slots_wsi)
            transported_stages.append(transported / column_mass.unsqueeze(-1))
        staged = torch.stack(transported_stages, dim=1)
        weights = stage_gate.detach()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bs,bsod->bod", weights, staged), staged

    @staticmethod
    def _masked_reconstruction_distance(prediction, target, pathway_mask, available):
        if prediction.shape != target.shape:
            raise ValueError("reconstruction shape does not match target")
        if pathway_mask.shape != target.shape[:2]:
            raise ValueError("pathway mask shape does not match reconstruction target")
        valid = pathway_mask & available.view(-1, 1)
        if not bool(valid.any()):
            return prediction.reshape(-1)[:0].sum()
        # Select BEFORE normalization: unavailable NaN targets never participate.
        prediction = prediction[valid].float()
        target = target.detach()[valid].float()
        prediction = F.layer_norm(prediction, (prediction.size(-1),))
        target = F.layer_norm(target, (target.size(-1),))
        cosine = (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()
        return .5 * (cosine + F.smooth_l1_loss(prediction, target))

    def masked_transport_reconstruction_losses(
        self, *, x_omics, slots_wsi, factual_gate, available, epoch,
        pathway_mask=None, target=None, wsi_available=None,
    ):
        # A detached full-omics gate still leaks information. Auxiliary stages
        # are averaged uniformly; no full-omics prediction is consulted here.
        del factual_gate
        target = x_omics.detach() if target is None else target.detach()
        if wsi_available is None:
            wsi_available = torch.ones_like(available)
        if pathway_mask is None:
            pathway_mask = self._sample_pathway_mask(target, available)
        else:
            pathway_mask = torch.as_tensor(pathway_mask, device=target.device)
            if pathway_mask.dtype != torch.bool or pathway_mask.shape != target.shape[:2]:
                raise ValueError("explicit pathway mask must be bool [B, P]")
            pathway_mask = pathway_mask & available[:, None]

        zero = x_omics.reshape(-1)[:0].sum()
        if not available.any() or self.reconstruction_mode == "off":
            self._last_pathway_mask = torch.zeros_like(pathway_mask)
            return zero, zero, zero, slots_wsi.new_zeros(len(slots_wsi), self.num_omic_slots, slots_wsi.size(-1)), self._last_pathway_mask, slots_wsi.new_zeros(len(slots_wsi), self.num_omic_slots, slots_wsi.size(-1))

        def reconstruct(mask_input, loss_mask):
            view = self._apply_pathway_mask(x_omics, mask_input)
            omic_slots, _ = self.slot_attention_omic(view, self.shared_omic_prototypes)
            self_pred = self.pathway_reconstruction_decoder(omic_slots)
            self_loss = self._masked_reconstruction_distance(self_pred, target, loss_mask, available)
            # Both modalities are needed for the cross branch.
            cross_available = available & wsi_available
            costs, rows, cols, _ = self._cost_tensor(slots_wsi, omic_slots)
            plans, _ = self._plans_from_cost_tensor(costs, rows, cols, epoch, replay_fixed=False)
            gate = slots_wsi.new_ones(len(slots_wsi), self.spt_num_stages)
            transported, _ = self._transport_wsi_to_omic(slots_wsi, plans, gate)
            cross_pred = self.pathway_reconstruction_decoder(transported)
            cross_loss = self._masked_reconstruction_distance(cross_pred, target, loss_mask, cross_available)
            return self_loss, cross_loss, transported, omic_slots

        if self.reconstruction_mode == "full":
            loss_mask = available[:, None].expand_as(pathway_mask)
            self_loss, cross_loss, transported, omic_slots = reconstruct(torch.zeros_like(pathway_mask), loss_mask)
            recorded_mask = torch.zeros_like(pathway_mask)
        else:
            self_loss, cross_loss, transported, omic_slots = reconstruct(pathway_mask, pathway_mask)
            recorded_mask = pathway_mask
            if self.reconstruction_mode == "hybrid":
                full = reconstruct(torch.zeros_like(pathway_mask), available[:, None].expand_as(pathway_mask))
                self_loss, cross_loss = .5*(self_loss+full[0]), .5*(cross_loss+full[1])
        total = self.dct_v314_mtr_self_fraction*self_loss + self.dct_v314_mtr_cross_fraction*cross_loss
        self._last_pathway_mask = recorded_mask.detach()
        return self_loss, cross_loss, total, transported, recorded_mask, omic_slots

    def forward(self, **kwargs):
        kwargs, wsi_available, omic_available = self._validated_inputs(kwargs)
        if self.training:
            epoch = int(kwargs.get("cur_epoch", getattr(self.args, "cur_epoch", 0)))
            self.transport_epoch.fill_(epoch)
        else:
            epoch = int(kwargs.get("cur_epoch", self.transport_epoch.item()))
        self.last_explanations = None
        self.last_training_losses = {}
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        x_wsi_proj = torch.where(wsi_available[:, None, None], x_wsi_proj, 0.)
        x_omics = torch.where(omic_available[:, None, None], x_omics, 0.)
        slots_wsi, slots_omic, wa, oa = self._encode_transport_slots(x_wsi_proj, x_omics, kwargs)
        factual_costs, rows, cols, evidence_gate = self._cost_tensor(slots_wsi, slots_omic)
        plans, ot_distance = self._plans_from_cost_tensor(factual_costs, rows, cols, epoch, replay_fixed=False)
        for name, value in (("factual_costs", factual_costs), ("factual_rows", rows),
                            ("factual_cols", cols), ("slots_wsi", slots_wsi), ("slots_omic", slots_omic)):
            setattr(self, "_last_" + name, value.detach())
        logits, factual_gate = self._encode_logits_from_plans(slots_wsi, slots_omic, plans)
        if not self.training:
            self._last_pathway_mask = None
            low_costs, high_costs = self._counterfactual_costs(factual_costs)
            low_plans, _ = self._plans_from_cost_tensor(low_costs, rows, cols, epoch)
            high_plans, _ = self._plans_from_cost_tensor(high_costs, rows, cols, epoch)
            low_logits, _ = self._encode_logits_from_plans(slots_wsi, slots_omic, low_plans)
            high_logits, _ = self._encode_logits_from_plans(slots_wsi, slots_omic, high_plans)
            risk, low, high = self._risk(logits), self._risk(low_logits), self._risk(high_logits)
            self.last_explanations = {
                "stage_slot_pair_evidence": torch.stack([p[0] for p in plans], 1).detach(),
                "evidence_gate": evidence_gate.detach(), "event_gate": factual_gate.detach(),
                "wsi_coordinate_assignment": wa.detach(), "omic_coordinate_assignment": oa.detach(),
                "factual_risk": risk.detach(), "low_risk_counterfactual": low.detach(),
                "high_risk_counterfactual": high.detach(), "counterfactual_risk_delta_low": (low-risk).detach(),
                "counterfactual_risk_delta_high": (high-risk).detach(),
                "risk_anchor_costs": self.risk_anchor_costs.detach(), "risk_anchor_seen": self.risk_anchor_seen.detach(),
                "stage_edges": self.dct_stage_edges.detach(),
                "factual_coupling_marginal_error": self._marginal_error(plans, rows, cols).detach(),
                "low_coupling_marginal_error": self._marginal_error(low_plans, rows, cols).detach(),
                "high_coupling_marginal_error": self._marginal_error(high_plans, rows, cols).detach(),
                "per_slot_hazard_wsi": self.per_slot_hazard_wsi(slots_wsi).sigmoid().detach(),
                "per_slot_hazard_omic": self.per_slot_hazard_omic(slots_omic).sigmoid().detach(),
            }
            return logits, logits.new_zeros(())

        self._reset_ipcw_memory_for_epoch(epoch)
        zero = logits.reshape(-1)[:0].sum()
        ipcw, slot_nll = zero, zero
        event_time, c, y = kwargs.get("event_time"), kwargs.get("c"), kwargs.get("y")
        low_weights = high_weights = logits.new_zeros(len(logits), self.spt_num_stages)
        if event_time is not None and c is not None:
            low_weights, high_weights = self._stage_membership_weights(event_time, c)
            ipcw = self._ipcw_pairwise_ranking_loss(logits, event_time, c)
            self._remember_ipcw_batch(self._risk(logits), event_time, c)
            self._update_risk_anchors(factual_costs.detach(), low_weights, high_weights)
            if y is not None:
                slot_nll = self.per_slot_nll_loss(slots_wsi, slots_omic, y, event_time, c, wsi_available, omic_available)
        diversity = self.slot_diversity_loss(slots_wsi, slots_omic, wsi_available, omic_available)
        ramp = self._mtr_ramp(epoch)
        weight = self.dct_v314_lambda_mtr * ramp if self.reconstruction_mode != "off" else 0.
        rec_self = rec_cross = rec = zero
        mask = torch.zeros(x_omics.shape[:2], device=x_omics.device, dtype=torch.bool)
        if weight > 0 and bool(omic_available.any()):
            target = self._clean_reconstruction_target(kwargs)
            rec_self, rec_cross, rec, _, mask, _ = self.masked_transport_reconstruction_losses(
                x_omics=x_omics, target=target, slots_wsi=slots_wsi, factual_gate=None,
                available=omic_available, wsi_available=wsi_available, epoch=epoch)
        else:
            self._last_pathway_mask = mask
        auxiliary = self.IPCW_RANK_WEIGHT*ipcw + self.dct_v314_lambda_slot_nll*slot_nll + self.dct_v314_lambda_slot_diversity*diversity + weight*rec
        def number(value):
            return logits.new_tensor(value)
        self.last_training_losses = {
            "ot": ot_distance.detach(), "ipcw_rank": ipcw.detach(),
            "v314_per_slot_nll": slot_nll.detach(), "v314_slot_diversity": diversity.detach(),
            "v314_slot_variance": number(self._last_slot_variance),
            "v314_slot_variance_wsi": number(self._last_slot_variance_wsi),
            "v314_slot_variance_omic": number(self._last_slot_variance_omic),
            "v314_slot_cosine_wsi": number(self._last_slot_cosine_wsi),
            "v314_slot_cosine_omic": number(self._last_slot_cosine_omic),
            "v314_mtr_self": rec_self.detach(), "v314_mtr_cross": rec_cross.detach(),
            "v314_mtr_total": rec.detach(), "v314_mtr_ramp": number(ramp), "v314_mtr_weight": number(weight),
            "v314_masked_pathway_fraction": mask.sum().float()/(omic_available.sum().clamp_min(1)*mask.size(1)),
            "v314_masked_pathway_count": mask.sum(1).float().mean(),
            "v314_omics_available_fraction": omic_available.float().mean(),
            "v314_nll_weighted": (self.dct_v314_lambda_slot_nll*slot_nll).detach(),
            "v314_diversity_weighted": (self.dct_v314_lambda_slot_diversity*diversity).detach(),
            "v314_reconstruction_weighted": (weight*rec).detach(),
            "v314_factual_marginal_error": self._marginal_error(plans, rows, cols).mean().detach(),
            "anchor_coverage": self.risk_anchor_seen.float().mean().detach(),
        }
        return logits, auxiliary

    def _validated_inputs(self, kwargs):
        kwargs = dict(kwargs)
        x = kwargs.get("x_wsi")
        if not torch.is_tensor(x) or x.ndim != 3 or min(x.shape[:2]) < 1 or x.size(-1) != self.wsi_embedding_dim:
            raise ValueError("x_wsi must be nonempty [B, N, encoding_dim]")
        batch, device = len(x), x.device
        masks = []
        for modality in ("wsi", "omic"):
            explicit = kwargs.get(modality + "_available")
            if modality == "omic" and explicit is None:
                explicit = kwargs.get("omics_available")
            available = self._batch_bool_mask(explicit, batch=batch, device=device, default=True)
            missing = self._batch_bool_mask(kwargs.get(modality + "_missing"), batch=batch, device=device, default=False)
            masks.append(available & ~missing)
        wsi_available, omic_available = masks
        if not (wsi_available | omic_available).all():
            raise ValueError("each patient must have at least one available modality")
        inputs = [("x_wsi", x, wsi_available)]
        for i, width in enumerate(self.omic_sizes, 1):
            value = kwargs.get(f"x_omic{i}")
            if not torch.is_tensor(value) or value.shape != (batch, width) or value.device != device:
                raise ValueError(f"x_omic{i} must be [B, {width}] on the WSI device")
            inputs.append((f"x_omic{i}", value, omic_available))
        for name, value, available in inputs:
            if not torch.isfinite(value[available]).all():
                raise ValueError(f"nonfinite values in available {name}")
            shape = [batch] + [1]*(value.ndim-1)
            kwargs[name] = torch.where(available.view(shape), value, 0.)
        for name in ("event_time", "c"):
            if kwargs.get(name) is not None:
                value = torch.as_tensor(kwargs[name], device=device, dtype=torch.float32).reshape(-1)
                if value.numel() != batch or not torch.isfinite(value).all():
                    raise ValueError(f"{name} must contain B finite values")
                if name == "c" and not torch.all((value == 0) | (value == 1)):
                    raise ValueError("c must be binary: 0=event, 1=censored")
                if name == "event_time" and (value < 0).any():
                    raise ValueError("event_time must be nonnegative")
                kwargs[name] = value
        if kwargs.get("y") is not None:
            kwargs["y"] = torch.as_tensor(kwargs["y"], device=device)
        return kwargs, wsi_available, omic_available

    @torch.no_grad()
    def _clean_reconstruction_target(self, kwargs):
        # The previous target included random AlphaDropout corruption. Use
        # deterministic, detached current encoder outputs; restore all modes.
        modes = [(module, module.training) for module in self.sig_networks.modules()]
        try:
            self.sig_networks.eval()
            return self._encode_omics(kwargs).detach()
        finally:
            for module, training in modes:
                module.training = training
