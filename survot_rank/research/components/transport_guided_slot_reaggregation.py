"""Cross-modal matching changes queries that re-read ORIGINAL modality tokens.

This is a representation-learning component, not a causal explanation method.
All controls instantiate the same parameters; only the feedback operation differs.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from survot_rank.research.components.slot_attention import _log_sinkhorn_assign


class SlotRereader(nn.Module):
    """One competitive slot update, with cross-modal context in its query."""

    def __init__(self, dim: int):
        super().__init__()
        self.slot_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.context_query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2),
                                nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, slots, tokens, context, strength, token_mask):
        # Mask BEFORE normalization: NaN padding cannot contaminate valid tokens.
        tokens = tokens.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        normalized = self.token_norm(tokens)
        query = self.query(self.slot_norm(slots))
        if context is not None:
            query = query + strength * self.context_query(self.context_norm(context))
        scores = query @ self.key(normalized).transpose(1, 2) / math.sqrt(slots.size(-1))
        assignment = scores.softmax(dim=1) * token_mask.unsqueeze(1)
        weights = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        updates = weights @ self.value(normalized)
        updated = self.gru(updates.flatten(0, 1), slots.flatten(0, 1)).view_as(slots)
        updated = updated + self.ff(updated)
        return updated, assignment, weights


class TransportGuidedSlotReaggregation(nn.Module):
    MODES = ("none", "self", "attention", "ot")

    def __init__(self, dim, *, mode="ot", rounds=1, strength=0.25,
                 epsilon=0.10, sinkhorn_iters=50):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 2:
            raise ValueError("dim must be an integer >= 2")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        for name, value, limit in (("rounds", rounds, 8),
                                   ("sinkhorn_iters", sinkhorn_iters, 1000)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
                raise ValueError(f"{name} must be an integer in [1, {limit}]")
        if isinstance(strength, bool) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be finite and in [0, 1]")
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.dim, self.mode, self.rounds = dim, mode, rounds
        self.strength, self.epsilon, self.sinkhorn_iters = strength, epsilon, sinkhorn_iters
        self.wsi_reader = SlotRereader(dim)
        self.omic_reader = SlotRereader(dim)
        self.capture_attention = False
        self.last_attention = None
        self.last_plan = None
        self.last_diagnostics = {}

    def _validate(self, slots, tokens, mask, name):
        if slots.ndim != 3 or tokens.ndim != 3:
            raise ValueError(f"{name}: expected [batch, count, dim] tensors")
        if (slots.size(0) != tokens.size(0) or slots.size(-1) != self.dim
                or tokens.size(-1) != self.dim or min(*slots.shape, *tokens.shape) < 1):
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
            raise ValueError(f"{name}: each patient must have a valid token")
        if not torch.isfinite(tokens.masked_select(mask.unsqueeze(-1))).all():
            raise ValueError(f"{name}: valid tokens must be finite")
        return mask

    def _contexts(self, wsi, omic):
        # Same cosine geometry/temperature for attention and OT. Matching uses
        # float32 under mixed precision, float64 is retained for grad checks.
        dtype = torch.float64 if wsi.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=wsi.device.type, enabled=False):
            similarity = F.normalize(wsi.to(dtype), dim=-1) @ F.normalize(
                omic.to(dtype), dim=-1).transpose(1, 2)
            if self.mode == "ot":
                plan = _log_sinkhorn_assign(1.0 - similarity, self.sinkhorn_iters,
                                            eps=self.epsilon)
                wsi_weights = plan / plan.sum(-1, keepdim=True).clamp_min(1e-8)
                reverse = plan.transpose(1, 2)
                omic_weights = reverse / reverse.sum(-1, keepdim=True).clamp_min(1e-8)
                self.last_plan = plan.detach()
                self.last_diagnostics["feedback_marginal_error"] = torch.maximum(
                    (plan.sum(-1) - 1.0 / wsi.size(1)).abs().amax(),
                    (plan.sum(-2) - 1.0 / omic.size(1)).abs().amax()).detach()
            else:
                wsi_weights = (similarity / self.epsilon).softmax(-1)
                omic_weights = (similarity.transpose(1, 2) / self.epsilon).softmax(-1)
            # No detach: survival gradients reach the matching AND re-reading.
            return ((wsi_weights @ omic.to(dtype)).to(wsi.dtype),
                    (omic_weights @ wsi.to(dtype)).to(omic.dtype))

    def forward(self, wsi_slots, omic_slots, wsi_tokens, omic_tokens,
                *, wsi_mask=None, omic_mask=None):
        self.last_plan, self.last_attention, self.last_diagnostics = None, None, {}
        wsi_mask = self._validate(wsi_slots, wsi_tokens, wsi_mask, "wsi")
        omic_mask = self._validate(omic_slots, omic_tokens, omic_mask, "omics")
        if (wsi_slots.size(0) != omic_slots.size(0) or wsi_slots.device != omic_slots.device
                or wsi_slots.dtype != omic_slots.dtype):
            raise ValueError("modalities must share batch size, device and dtype")
        if self.mode == "none":
            return wsi_slots, omic_slots
        wsi, omic = wsi_slots, omic_slots
        for _ in range(self.rounds):
            contexts = (None, None) if self.mode == "self" else self._contexts(wsi, omic)
            # Both directions consume the SAME pre-update state, not an
            # accidental sequential update of the second modality.
            new_wsi, aw, pw = self.wsi_reader(wsi, wsi_tokens, contexts[0], self.strength, wsi_mask)
            new_omic, ao, po = self.omic_reader(omic, omic_tokens, contexts[1], self.strength, omic_mask)
            wsi, omic = new_wsi, new_omic
        self.last_diagnostics.update({
            "wsi_slot_delta": (wsi - wsi_slots).abs().mean().detach(),
            "omic_slot_delta": (omic - omic_slots).abs().mean().detach(),
        })
        if self.capture_attention:
            self.last_attention = {"wsi_assignment": aw.detach(), "omic_assignment": ao.detach(),
                                   "wsi_pooling": pw.detach(), "omic_pooling": po.detach()}
        return wsi, omic
