"""DCT v3.30 candidate: confidence-aware closed-loop prognostic transport.

The prediction path is deliberately closed:

    encoded tokens -> initial slots -> cross-modal OT -> original-token re-read
    -> reaggregated slots -> stage transport events -> survival risk.

No raw or pre-feedback modality representation is concatenated into the risk
head.  Survival labels enter only the optional training-time representation
contrast; they never determine a patient's feedback gate or transport plan.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from survot_rank.research.components.confidence_aware_transport_reaggregation import (
    ConfidenceAwareTransportReaggregation,
)
from survot_rank.research.methods.dct_v310_directional_regularized_transport.model import (
    DCTV310DirectionalRegularizedTransport,
)
from survot_rank.research.methods.distributional_counterfactual_transport.model import (
    DistributionalCounterfactualTransport,
)


class DCTV330ClosedLoopPrognosticTransport(DistributionalCounterfactualTransport):
    """Closed-loop OT feedback with a transport-only survival path."""

    VARIANTS = ConfidenceAwareTransportReaggregation.VARIANTS

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        if str(getattr(args, "bag_loss", "nll_surv")) != "nll_surv":
            raise ValueError("DCT v3.30 requires bag_loss='nll_surv'")
        variant = str(getattr(args, "dct_v330_variant", "prognostic_rank"))
        if variant not in self.VARIANTS:
            raise ValueError(f"dct_v330_variant must be one of {self.VARIANTS}")

        # Keep the final prediction architecture matched to DCT-Reg while
        # removing its anchor/direction objectives.  v3.30 supervises only the
        # post-reaggregation transport representation in its final arm.
        values = dict(DCTV310DirectionalRegularizedTransport.FROZEN_ARGUMENTS)
        values.update(
            dct_lambda_ipcw_rank=0.0,
            dct_lambda_etar=0.0,
            dct_v38_lambda_direction=0.0,
            dct_v38_lambda_dose=0.0,
            dct_v38_lambda_reconfiguration=0.0,
            fet_lambda_sparse=0.0,
            fet_lambda_faith=0.0,
            spt_lambda_ot=0.0,
            spt_lambda_rank=0.0,
            spt_lambda_stage=0.0,
        )
        for name, value in values.items():
            setattr(args, name, value)
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        self.dct_v330_variant = variant
        self.dct_v330_repr_rank_weight = float(
            getattr(args, "dct_v330_repr_rank_weight", 0.10)
        )
        self.dct_v330_repr_temperature = float(
            getattr(args, "dct_v330_repr_temperature", 0.10)
        )
        self.dct_v330_high_quantile = float(
            getattr(args, "dct_v330_high_quantile", 0.40)
        )
        self.dct_v330_low_quantile = float(
            getattr(args, "dct_v330_low_quantile", 0.60)
        )
        self.dct_v330_final_eps_start = float(
            getattr(args, "dct_v330_final_eps_start", 0.50)
        )
        self.dct_v330_final_eps_end = float(
            getattr(args, "dct_v330_final_eps_end", 0.10)
        )
        self.dct_v330_final_eps_anneal_epochs = int(
            getattr(args, "dct_v330_final_eps_anneal_epochs", 12)
        )
        if self.dct_v330_repr_rank_weight < 0.0:
            raise ValueError("dct_v330_repr_rank_weight must be non-negative")
        if self.dct_v330_repr_temperature <= 0.0:
            raise ValueError("dct_v330_repr_temperature must be positive")
        if not 0.0 < self.dct_v330_high_quantile < self.dct_v330_low_quantile < 1.0:
            raise ValueError("v3.30 requires 0 < high_quantile < low_quantile < 1")
        if self.dct_v330_final_eps_end <= 0.0:
            raise ValueError("dct_v330_final_eps_end must be positive")
        if self.dct_v330_final_eps_start < self.dct_v330_final_eps_end:
            raise ValueError("v3.30 final epsilon must anneal from larger to smaller")
        if self.dct_v330_final_eps_anneal_epochs < 1:
            raise ValueError("dct_v330_final_eps_anneal_epochs must be positive")

        dim = self.wsi_projection_dim
        self.reaggregation = ConfidenceAwareTransportReaggregation(
            dim,
            variant=variant,
            rounds=int(getattr(args, "dct_v330_rounds", 1)),
            fixed_strength=float(getattr(args, "dct_v330_feedback_strength", 0.25)),
            fixed_epsilon=float(getattr(args, "dct_v330_feedback_eps", 0.10)),
            adaptive_epsilon_start=float(
                getattr(args, "dct_v330_adaptive_eps_start", 0.50)
            ),
            adaptive_epsilon_end=float(
                getattr(args, "dct_v330_adaptive_eps_end", 0.10)
            ),
            adaptive_epsilon_anneal_epochs=int(
                getattr(args, "dct_v330_adaptive_eps_anneal_epochs", 12)
            ),
            sinkhorn_iters=int(getattr(args, "dct_v330_feedback_iters", 50)),
            marginal_strength=float(
                getattr(args, "dct_v330_marginal_strength", 0.50)
            ),
            gate_hidden_dim=int(getattr(args, "dct_v330_gate_hidden_dim", 16)),
        )
        self.prognostic_transport_prototypes = nn.Parameter(torch.empty(2, dim))
        nn.init.normal_(self.prognostic_transport_prototypes, std=0.02)
        self.register_buffer("dct_v330_risk_thresholds", torch.empty(0))
        self._last_transport_representation = None
        self._last_final_transport_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def uses_confidence(self) -> bool:
        return self.dct_v330_variant in {"confidence_gate", "prognostic_rank"}

    @property
    def uses_representation_rank(self) -> bool:
        return self.dct_v330_variant == "prognostic_rank"

    def objective_weights(self):
        return {
            "nll": 1.0,
            "transport_representation_rank": (
                self.dct_v330_repr_rank_weight if self.uses_representation_rank else 0.0
            ),
            "direction": 0.0,
        }

    def configure_train_reference(self, event_times, censorship):
        super().configure_train_reference(event_times, censorship)
        times = torch.as_tensor(event_times, dtype=torch.float32).view(-1)
        censored = torch.as_tensor(censorship, dtype=torch.float32).view(-1)
        if times.numel() != censored.numel():
            raise ValueError("event_times and censorship must have equal length")
        observed_times = times[censored < 0.5]
        if observed_times.numel() < 2:
            self.dct_v330_risk_thresholds = torch.empty(
                0, device=self.dct_stage_edges.device
            )
            return
        quantiles = torch.tensor(
            [self.dct_v330_high_quantile, self.dct_v330_low_quantile],
            dtype=observed_times.dtype,
            device=observed_times.device,
        )
        self.dct_v330_risk_thresholds = torch.quantile(
            observed_times, quantiles
        ).to(self.dct_stage_edges.device)

    def _encode_transport_slots(self, x_wsi_proj, x_omics, kwargs):
        slots_wsi, slots_omic, wsi_assignment, omic_assignment = (
            super()._encode_transport_slots(x_wsi_proj, x_omics, kwargs)
        )
        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        slots_wsi, slots_omic = self.reaggregation(
            slots_wsi,
            slots_omic,
            x_wsi_proj,
            x_omics,
            epoch=epoch,
        )
        return slots_wsi, slots_omic, wsi_assignment, omic_assignment

    def _final_epsilon_ratio(self, epoch):
        fraction = min(
            1.0,
            max(0.0, float(epoch)) / self.dct_v330_final_eps_anneal_epochs,
        )
        return self.dct_v330_final_eps_start + fraction * (
            self.dct_v330_final_eps_end - self.dct_v330_final_eps_start
        )

    def _plans_from_cost_tensor(self, costs, rows, cols, epoch, *, replay_fixed=False):
        if not self.uses_confidence:
            return super()._plans_from_cost_tensor(
                costs, rows, cols, epoch, replay_fixed=replay_fixed
            )
        if replay_fixed:
            raise RuntimeError("v3.30 confidence variants do not replay stale couplings")

        ratio = self._final_epsilon_ratio(epoch)
        plans, distances, entropies, errors, raw_epsilons, geometry_tvs = (
            [], [], [], [], [], []
        )
        for stage_idx in range(self.spt_num_stages):
            stage_plans, stage_distances = [], []
            for cost_idx in range(costs.size(2)):
                raw_cost = costs[:, stage_idx, cost_idx]
                mean = raw_cost.mean(dim=(1, 2), keepdim=True)
                std = raw_cost.std(
                    dim=(1, 2), keepdim=True, unbiased=False
                ).clamp_min(1e-4)
                standardized = (raw_cost - mean) / std
                epsilon = standardized.new_full((standardized.size(0), 1, 1), ratio)
                plan = ConfidenceAwareTransportReaggregation._log_sinkhorn(
                    standardized,
                    rows[:, stage_idx],
                    cols[:, stage_idx],
                    epsilon,
                    self.ot_iter,
                )
                plan = self._project_coupling(
                    plan, rows[:, stage_idx], cols[:, stage_idx]
                )
                stage_plans.append(plan)
                stage_distances.append((plan * raw_cost).sum(dim=(1, 2)))
                flat = plan.flatten(1)
                entropy = -(
                    flat.clamp_min(1e-8) * flat.clamp_min(1e-8).log()
                ).sum(dim=-1) / math.log(max(2, flat.size(-1)))
                entropies.append(entropy)
                row_error = (
                    plan.sum(dim=-1) - rows[:, stage_idx]
                ).abs().amax(dim=-1)
                col_error = (
                    plan.sum(dim=-2) - cols[:, stage_idx]
                ).abs().amax(dim=-1)
                errors.append(torch.maximum(row_error, col_error))
                raw_epsilons.append(std.flatten() * ratio)
            for first in range(len(stage_plans)):
                for second in range(first + 1, len(stage_plans)):
                    geometry_tvs.append(
                        0.5
                        * (stage_plans[first] - stage_plans[second])
                        .abs()
                        .sum(dim=(1, 2))
                    )
            plans.append(tuple(stage_plans))
            distances.append(torch.stack(stage_distances, dim=1).mean())

        self._last_final_transport_diagnostics = {
            "final_plan_entropy": torch.cat(entropies).mean().detach(),
            "final_marginal_error": torch.cat(errors).max().detach(),
            "final_effective_epsilon": torch.cat(raw_epsilons).mean().detach(),
            "final_geometry_plan_tv": (
                torch.cat(geometry_tvs).mean().detach()
                if geometry_tvs
                else costs.new_zeros(())
            ),
        }
        return plans, torch.stack(distances).mean()

    def _encode_logits_from_plans(self, slots_wsi, slots_omic, plans):
        tokens = self._selected_stage_events(slots_wsi, slots_omic, plans)
        tokens = tokens + self.stage_embedding.unsqueeze(0)
        tokens = self.event_norm(self.event_encoder(tokens))
        self._last_transport_representation = tokens.mean(dim=1)
        event_logits = self.event_hazard(tokens)
        gate = torch.softmax(self.event_gate(tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)
        return logits, gate

    def _transport_representation_rank_loss(self, representation, event_time, censorship):
        if self.dct_v330_risk_thresholds.numel() != 2:
            return representation.sum() * 0.0, representation.new_zeros(())
        times = event_time.float().view(-1)
        censoring = censorship.float().view(-1)
        high_threshold, low_threshold = self.dct_v330_risk_thresholds.to(times.device)
        high_risk = (censoring < 0.5) & (times <= high_threshold)
        low_risk = (times >= low_threshold) & ~high_risk
        selected = high_risk | low_risk
        count = selected.sum().to(representation.dtype)
        if not bool(selected.any()):
            return representation.sum() * 0.0, count

        normalized = F.normalize(representation[selected], dim=-1)
        prototypes = F.normalize(self.prognostic_transport_prototypes, dim=-1)
        logits = normalized @ prototypes.transpose(0, 1)
        logits = logits / self.dct_v330_repr_temperature
        targets = high_risk[selected].long()
        losses = F.cross_entropy(logits, targets, reduction="none")

        low_query = torch.full_like(times, float(low_threshold))
        query_times = torch.where(high_risk, times, low_query)
        ipcw = self._ipcw(query_times).clamp_max(self.dct_ipcw_max_weight)
        weights = ipcw[selected]
        loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        return loss, count

    def get_extra_state(self):
        feedback = self.reaggregation
        return {
            "version": "3.30-closed-loop-prognostic-transport",
            "variant": self.dct_v330_variant,
            "feedback_rounds": feedback.rounds,
            "fixed_feedback_strength": feedback.fixed_strength,
            "marginal_strength": feedback.marginal_strength,
            "representation_rank_weight": self.dct_v330_repr_rank_weight,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("DCT v3.30 checkpoint configuration mismatch")

    def forward(self, **kwargs):
        if kwargs.get("wsi_missing", False) or kwargs.get("omic_missing", False):
            raise ValueError("DCT v3.30 currently requires both WSI and omics")
        x_wsi = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        slots_wsi, slots_omic, _, _ = self._encode_transport_slots(
            x_wsi, x_omics, kwargs
        )
        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        costs, rows, cols, _ = self._cost_tensor(slots_wsi, slots_omic)
        plans, distance = self._plans_from_cost_tensor(
            costs, rows, cols, epoch, replay_fixed=False
        )
        logits, event_gate = self._encode_logits_from_plans(
            slots_wsi, slots_omic, plans
        )

        rank_loss = costs.new_zeros(())
        rank_count = costs.new_zeros(())
        if (
            self.training
            and self.uses_representation_rank
            and kwargs.get("event_time") is not None
            and kwargs.get("c") is not None
        ):
            rank_loss, rank_count = self._transport_representation_rank_loss(
                self._last_transport_representation,
                kwargs["event_time"],
                kwargs["c"],
            )
        auxiliary = self.dct_v330_repr_rank_weight * rank_loss

        diagnostics = {
            f"v330_{name}": value
            for name, value in self.reaggregation.last_diagnostics.items()
        }
        diagnostics.update(
            {
                f"v330_{name}": value
                for name, value in self._last_final_transport_diagnostics.items()
            }
        )
        diagnostics.update(
            v330_ot_distance=distance.detach(),
            v330_representation_rank=rank_loss.detach(),
            v330_representation_rank_samples=rank_count.detach(),
        )
        if self.training:
            self.last_training_losses = diagnostics
            self.last_explanations = None
        else:
            self.last_explanations = {
                **diagnostics,
                "factual_risk": self._risk(logits).detach(),
                "event_gate": event_gate.detach(),
                "feedback_plan": (
                    None if self.reaggregation.last_plan is None else self.reaggregation.last_plan
                ),
                "feedback_wsi_gate": (
                    None
                    if self.reaggregation.last_gates is None
                    else self.reaggregation.last_gates["wsi"]
                ),
                "feedback_omic_gate": (
                    None
                    if self.reaggregation.last_gates is None
                    else self.reaggregation.last_gates["omics"]
                ),
            }
        return logits, auxiliary
