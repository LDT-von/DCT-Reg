"""Confidence-aware closed-loop transport feedback for patient-specific slots.

The transport plan is used to construct cross-modal context that changes how
each modality re-reads its original encoded tokens.  Confidence is computed
from plan entropy, marginal error, cross-modal agreement and slot uncertainty;
it never uses survival labels or validation outcomes.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from survot_rank.research.components.slot_attention import _log_sinkhorn_assign
from survot_rank.research.components.transport_guided_slot_reaggregation import (
    SlotRereader,
)


class ConfidenceAwareTransportReaggregation(nn.Module):
    """Re-read original tokens under fixed or confidence-aware OT feedback."""

    VARIANTS = (
        "baseline",
        "self_update",
        "ot_feedback",
        "confidence_gate",
        "prognostic_rank",
    )

    def __init__(
        self,
        dim: int,
        *,
        variant: str = "prognostic_rank",
        rounds: int = 1,
        fixed_strength: float = 0.25,
        fixed_epsilon: float = 0.10,
        adaptive_epsilon_start: float = 0.50,
        adaptive_epsilon_end: float = 0.10,
        adaptive_epsilon_anneal_epochs: int = 12,
        sinkhorn_iters: int = 50,
        marginal_strength: float = 0.50,
        gate_hidden_dim: int = 16,
    ):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 2:
            raise ValueError("dim must be an integer >= 2")
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        if not isinstance(rounds, int) or isinstance(rounds, bool) or not 1 <= rounds <= 8:
            raise ValueError("rounds must be an integer in [1, 8]")
        if not isinstance(sinkhorn_iters, int) or isinstance(sinkhorn_iters, bool):
            raise ValueError("sinkhorn_iters must be an integer")
        if not 1 <= sinkhorn_iters <= 1000:
            raise ValueError("sinkhorn_iters must be in [1, 1000]")
        for name, value in (
            ("fixed_strength", fixed_strength),
            ("marginal_strength", marginal_strength),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name, value in (
            ("fixed_epsilon", fixed_epsilon),
            ("adaptive_epsilon_start", adaptive_epsilon_start),
            ("adaptive_epsilon_end", adaptive_epsilon_end),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if adaptive_epsilon_start < adaptive_epsilon_end:
            raise ValueError("adaptive epsilon must anneal from a larger value to a smaller value")
        if (
            not isinstance(adaptive_epsilon_anneal_epochs, int)
            or isinstance(adaptive_epsilon_anneal_epochs, bool)
            or adaptive_epsilon_anneal_epochs < 1
        ):
            raise ValueError("adaptive_epsilon_anneal_epochs must be a positive integer")
        if not isinstance(gate_hidden_dim, int) or isinstance(gate_hidden_dim, bool) or gate_hidden_dim < 2:
            raise ValueError("gate_hidden_dim must be an integer >= 2")

        self.dim = dim
        self.variant = variant
        self.rounds = rounds
        self.fixed_strength = float(fixed_strength)
        self.fixed_epsilon = float(fixed_epsilon)
        self.adaptive_epsilon_start = float(adaptive_epsilon_start)
        self.adaptive_epsilon_end = float(adaptive_epsilon_end)
        self.adaptive_epsilon_anneal_epochs = adaptive_epsilon_anneal_epochs
        self.sinkhorn_iters = sinkhorn_iters
        self.marginal_strength = float(marginal_strength)

        self.wsi_reader = SlotRereader(dim)
        self.omic_reader = SlotRereader(dim)
        self.wsi_marginal_score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.omic_marginal_score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.wsi_confidence_gate = self._make_confidence_gate(gate_hidden_dim)
        self.omic_confidence_gate = self._make_confidence_gate(gate_hidden_dim)
        self._initialize_gate(self.wsi_confidence_gate, self.fixed_strength)
        self._initialize_gate(self.omic_confidence_gate, self.fixed_strength)

        self.last_plan = None
        self.last_rows = None
        self.last_cols = None
        self.last_gates = None
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @staticmethod
    def _make_confidence_gate(hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _initialize_gate(gate: nn.Sequential, strength: float) -> None:
        output = gate[-1]
        nn.init.zeros_(output.weight)
        if strength <= 0.0:
            bias = -12.0
        elif strength >= 1.0:
            bias = 12.0
        else:
            bias = math.log(strength / (1.0 - strength))
        nn.init.constant_(output.bias, bias)

    @property
    def uses_confidence(self) -> bool:
        return self.variant in {"confidence_gate", "prognostic_rank"}

    def _validate(self, slots, tokens, mask, name):
        if slots.ndim != 3 or tokens.ndim != 3:
            raise ValueError(f"{name}: expected [batch, count, dim] tensors")
        if (
            slots.size(0) != tokens.size(0)
            or slots.size(-1) != self.dim
            or tokens.size(-1) != self.dim
            or min(*slots.shape, *tokens.shape) < 1
        ):
            raise ValueError(f"{name}: incompatible or empty dimensions")
        if slots.device != tokens.device or slots.dtype != tokens.dtype:
            raise ValueError(f"{name}: slots and tokens must share device and dtype")
        if not slots.is_floating_point() or not torch.isfinite(slots).all():
            raise ValueError(f"{name}: slots must be finite floating tensors")
        if mask is None:
            mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool or mask.device != tokens.device:
            raise ValueError(f"{name}: token mask must be bool [batch, tokens] on the input device")
        if not mask.any(dim=1).all():
            raise ValueError(f"{name}: each patient must have at least one valid token")
        if not torch.isfinite(tokens.masked_select(mask.unsqueeze(-1))).all():
            raise ValueError(f"{name}: valid tokens must be finite")
        return mask

    def epsilon_ratio(self, epoch: int) -> float:
        fraction = min(1.0, max(0.0, float(epoch)) / self.adaptive_epsilon_anneal_epochs)
        return self.adaptive_epsilon_start + fraction * (
            self.adaptive_epsilon_end - self.adaptive_epsilon_start
        )

    def _stable_marginals(self, wsi, omic):
        learned_rows = torch.softmax(self.wsi_marginal_score(wsi).squeeze(-1), dim=-1)
        learned_cols = torch.softmax(self.omic_marginal_score(omic).squeeze(-1), dim=-1)
        uniform_rows = torch.full_like(learned_rows, 1.0 / learned_rows.size(-1))
        uniform_cols = torch.full_like(learned_cols, 1.0 / learned_cols.size(-1))
        rows = torch.lerp(uniform_rows, learned_rows, self.marginal_strength)
        cols = torch.lerp(uniform_cols, learned_cols, self.marginal_strength)
        return rows, cols

    @staticmethod
    def _log_sinkhorn(cost, rows, cols, epsilon, max_iter):
        kernel = (-cost / epsilon).clamp(min=-60.0, max=60.0)
        log_rows = rows.clamp_min(1e-8).log()
        log_cols = cols.clamp_min(1e-8).log()
        log_u = torch.zeros_like(log_rows)
        log_v = torch.zeros_like(log_cols)
        for _ in range(max_iter):
            log_u = log_rows - torch.logsumexp(kernel + log_v.unsqueeze(1), dim=2)
            log_v = log_cols - torch.logsumexp(kernel + log_u.unsqueeze(2), dim=1)
        plan = (kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)).exp()
        return torch.nan_to_num(plan, nan=0.0, posinf=1.0, neginf=0.0)

    @staticmethod
    def _marginal_error(plan, rows, cols):
        row_error = (plan.sum(dim=-1) - rows).abs().amax(dim=-1)
        col_error = (plan.sum(dim=-2) - cols).abs().amax(dim=-1)
        return torch.maximum(row_error, col_error)

    @staticmethod
    def _normalized_entropy(probabilities, normalizer):
        entropy = -(
            probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        return entropy / max(math.log(max(2, int(normalizer))), 1e-8)

    def _adaptive_plan(self, wsi, omic, epoch):
        dtype = torch.float64 if wsi.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=wsi.device.type, enabled=False):
            wsi_f = wsi.to(dtype)
            omic_f = omic.to(dtype)
            similarity = F.normalize(wsi_f, dim=-1) @ F.normalize(omic_f, dim=-1).transpose(1, 2)
            raw_cost = 1.0 - similarity
            cost_mean = raw_cost.mean(dim=(1, 2), keepdim=True)
            cost_std = raw_cost.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-4)
            standardized_cost = (raw_cost - cost_mean) / cost_std
            rows, cols = self._stable_marginals(wsi_f, omic_f)
            ratio = self.epsilon_ratio(epoch)
            epsilon = standardized_cost.new_full((standardized_cost.size(0), 1, 1), ratio)
            plan = self._log_sinkhorn(
                standardized_cost, rows, cols, epsilon, self.sinkhorn_iters
            )
            raw_epsilon = cost_std.flatten() * ratio
            return plan, rows, cols, similarity, cost_std.flatten(), raw_epsilon

    def _legacy_plan(self, wsi, omic):
        dtype = torch.float64 if wsi.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=wsi.device.type, enabled=False):
            wsi_f = wsi.to(dtype)
            omic_f = omic.to(dtype)
            similarity = F.normalize(wsi_f, dim=-1) @ F.normalize(omic_f, dim=-1).transpose(1, 2)
            plan = _log_sinkhorn_assign(
                1.0 - similarity, self.sinkhorn_iters, eps=self.fixed_epsilon
            )
            rows = torch.full(
                plan.shape[:2], 1.0 / plan.size(1), device=plan.device, dtype=plan.dtype
            )
            cols = torch.full(
                (plan.size(0), plan.size(2)),
                1.0 / plan.size(2),
                device=plan.device,
                dtype=plan.dtype,
            )
            cost_std = (1.0 - similarity).std(
                dim=(1, 2), unbiased=False
            ).clamp_min(1e-4)
            epsilon = cost_std.new_full((plan.size(0),), self.fixed_epsilon)
            return plan, rows, cols, similarity, cost_std, epsilon

    @staticmethod
    def _contexts(plan, rows, cols, wsi, omic):
        wsi_weights = plan / rows.unsqueeze(-1).clamp_min(1e-8)
        omic_weights = plan.transpose(1, 2) / cols.unsqueeze(-1).clamp_min(1e-8)
        return wsi_weights @ omic, omic_weights @ wsi

    def _confidence_gates(self, plan, rows, cols, similarity):
        wsi_conditional = plan / rows.unsqueeze(-1).clamp_min(1e-8)
        omic_conditional = plan.transpose(1, 2) / cols.unsqueeze(-1).clamp_min(1e-8)
        wsi_uncertainty = self._normalized_entropy(wsi_conditional, plan.size(2))
        omic_uncertainty = self._normalized_entropy(omic_conditional, plan.size(1))
        joint = plan.flatten(1)
        plan_entropy = self._normalized_entropy(joint, joint.size(-1))
        plan_uncertainty = 0.5 * (
            (rows * wsi_uncertainty).sum(dim=-1)
            + (cols * omic_uncertainty).sum(dim=-1)
        )
        marginal_error = self._marginal_error(plan, rows, cols)
        scaled_error = (marginal_error * max(plan.size(1), plan.size(2))).clamp(0.0, 1.0)
        wsi_consistency = (wsi_conditional * similarity).sum(dim=-1).clamp(-1.0, 1.0)
        omic_consistency = (
            omic_conditional * similarity.transpose(1, 2)
        ).sum(dim=-1).clamp(-1.0, 1.0)

        def features(slot_uncertainty, consistency):
            return torch.stack(
                (
                    plan_uncertainty.unsqueeze(-1).expand_as(slot_uncertainty),
                    slot_uncertainty,
                    (consistency + 1.0) * 0.5,
                    scaled_error.unsqueeze(-1).expand_as(slot_uncertainty),
                ),
                dim=-1,
            )

        wsi_gate = torch.sigmoid(self.wsi_confidence_gate(features(wsi_uncertainty, wsi_consistency)))
        omic_gate = torch.sigmoid(
            self.omic_confidence_gate(features(omic_uncertainty, omic_consistency))
        )
        return (
            wsi_gate,
            omic_gate,
            plan_entropy,
            plan_uncertainty,
            marginal_error,
        )

    def forward(
        self,
        wsi_slots,
        omic_slots,
        wsi_tokens,
        omic_tokens,
        *,
        epoch: int = 0,
        wsi_mask=None,
        omic_mask=None,
    ):
        self.last_plan = self.last_rows = self.last_cols = self.last_gates = None
        self.last_diagnostics = {}
        wsi_mask = self._validate(wsi_slots, wsi_tokens, wsi_mask, "wsi")
        omic_mask = self._validate(omic_slots, omic_tokens, omic_mask, "omics")
        if (
            wsi_slots.size(0) != omic_slots.size(0)
            or wsi_slots.device != omic_slots.device
            or wsi_slots.dtype != omic_slots.dtype
        ):
            raise ValueError("modalities must share batch size, device and dtype")

        if self.variant == "baseline":
            zero = wsi_slots.new_zeros(())
            self.last_diagnostics = {
                "plan_entropy": zero,
                "plan_uncertainty": zero,
                "marginal_error": zero,
                "feedback_gate_mean": zero,
                "feedback_gate_min": zero,
                "feedback_gate_max": zero,
                "feedback_gate_std": zero,
                "dependence_tv": zero,
                "cost_std": zero,
                "effective_epsilon": zero,
            }
            return wsi_slots, omic_slots

        wsi, omic = wsi_slots, omic_slots
        for _ in range(self.rounds):
            if self.variant == "self_update":
                wsi, _, _ = self.wsi_reader(wsi, wsi_tokens, None, 0.0, wsi_mask)
                omic, _, _ = self.omic_reader(omic, omic_tokens, None, 0.0, omic_mask)
                continue

            if self.uses_confidence:
                plan, rows, cols, similarity, cost_std, epsilon = self._adaptive_plan(
                    wsi, omic, epoch
                )
            else:
                plan, rows, cols, similarity, cost_std, epsilon = self._legacy_plan(wsi, omic)
            contexts = self._contexts(plan, rows, cols, wsi.to(plan.dtype), omic.to(plan.dtype))
            context_wsi, context_omic = (item.to(wsi.dtype) for item in contexts)

            if self.uses_confidence:
                (
                    gate_wsi,
                    gate_omic,
                    plan_entropy,
                    plan_uncertainty,
                    marginal_error,
                ) = self._confidence_gates(plan, rows, cols, similarity)
                candidate_wsi, _, _ = self.wsi_reader(
                    wsi, wsi_tokens, context_wsi, 1.0, wsi_mask
                )
                candidate_omic, _, _ = self.omic_reader(
                    omic, omic_tokens, context_omic, 1.0, omic_mask
                )
                wsi = wsi + gate_wsi.to(wsi.dtype) * (candidate_wsi - wsi)
                omic = omic + gate_omic.to(omic.dtype) * (candidate_omic - omic)
            else:
                candidate_wsi, _, _ = self.wsi_reader(
                    wsi, wsi_tokens, context_wsi, self.fixed_strength, wsi_mask
                )
                candidate_omic, _, _ = self.omic_reader(
                    omic, omic_tokens, context_omic, self.fixed_strength, omic_mask
                )
                wsi, omic = candidate_wsi, candidate_omic
                joint = plan.flatten(1)
                plan_entropy = self._normalized_entropy(joint, joint.size(-1))
                plan_uncertainty = plan_entropy
                marginal_error = self._marginal_error(plan, rows, cols)
                gate_wsi = plan.new_full((*rows.shape, 1), self.fixed_strength)
                gate_omic = plan.new_full((*cols.shape, 1), self.fixed_strength)

        if self.variant == "self_update":
            zero = wsi.new_zeros(())
            self.last_diagnostics = {
                "plan_entropy": zero,
                "plan_uncertainty": zero,
                "marginal_error": zero,
                "feedback_gate_mean": zero,
                "feedback_gate_min": zero,
                "feedback_gate_max": zero,
                "feedback_gate_std": zero,
                "dependence_tv": zero,
                "cost_std": zero,
                "effective_epsilon": zero,
                "wsi_slot_delta": (wsi - wsi_slots).abs().mean().detach(),
                "omic_slot_delta": (omic - omic_slots).abs().mean().detach(),
            }
            return wsi, omic

        independent = rows.unsqueeze(-1) * cols.unsqueeze(-2)
        dependence_tv = 0.5 * (plan - independent).abs().sum(dim=(1, 2))
        all_gates = torch.cat((gate_wsi.flatten(), gate_omic.flatten()))
        self.last_plan = plan.detach()
        self.last_rows = rows.detach()
        self.last_cols = cols.detach()
        self.last_gates = {"wsi": gate_wsi.detach(), "omics": gate_omic.detach()}
        self.last_diagnostics = {
            "plan_entropy": plan_entropy.mean().detach(),
            "plan_uncertainty": plan_uncertainty.mean().detach(),
            "marginal_error": marginal_error.max().detach(),
            "feedback_gate_mean": all_gates.mean().detach(),
            "feedback_gate_min": all_gates.min().detach(),
            "feedback_gate_max": all_gates.max().detach(),
            "feedback_gate_std": all_gates.std(unbiased=False).detach(),
            "dependence_tv": dependence_tv.mean().detach(),
            "cost_std": cost_std.mean().detach(),
            "effective_epsilon": epsilon.mean().detach(),
            "wsi_slot_delta": (wsi - wsi_slots).abs().mean().detach(),
            "omic_slot_delta": (omic - omic_slots).abs().mean().detach(),
        }
        return wsi, omic
