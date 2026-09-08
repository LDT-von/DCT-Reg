#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCT v4.0 — Causal Survival Slots.

Replaces the entire OT/Sinkhorn/coupling path from DCT v3.10 with an
explicit causal hazard decomposition:

  hazard(t) = baseline(t) + Σ_k h_wsi(t,k) + Σ_l h_omic(t,l) + Σ_{k,l} h_cross(t,k,l)

Each slot independently contributes to hazard.  Counterfactual training enforces
that swapping high/low-risk patient slots produces monotonic hazard response.

Key differences from DCT v3.10:
  - NO Sinkhorn, NO OT coupling, NO cost matrices
  - Slots ARE the causal prognostic factors (not OT nodes)
  - Causal intervention loss replaces direction/dose transport objectives
  - No iterative coupling projection

Key differences from SlotSPE:
  - Slots decompose to additive hazard contributions (not just selected for prediction)
  - Explicit counterfactual training on slot swaps
  - Interpretability by design: each slot = named hazard pathway
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.components.slot_attention import MultiHeadSlotAttention
from survot_rank.research.components.omics_encoder import SNN_Block, WSI_Mlp


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder helpers (mirrors OTEHV2Survival)
# ─────────────────────────────────────────────────────────────────────────────

class EncoderMixin:
    """Reused encoder initialisation from OTEHV2Survival — not a full base class."""

    def _init_per_path_model(self, omic_sizes, omics_format):
        dim = self.wsi_projection_dim
        if omics_format == "Pathways":
            self.num_pathways = len(omic_sizes)
            self.sig_networks = nn.ModuleList([
                nn.Sequential(
                    SNN_Block(dim1=idim, dim2=dim),
                    SNN_Block(dim1=dim, dim2=dim, dropout=0.25),
                )
                for idim in omic_sizes
            ])
        elif omics_format == "GeneEmbedding":
            self.sig_networks = SNN_Block(dim1=768, dim2=dim)
        elif omics_format == "RNASeq":
            self.sig_networks = SNN_Block(dim1=self.omics_input_dim, dim2=dim)
        else:
            raise ValueError(f"Invalid omics_format: {omics_format}")

    def _encode_omics(self, kwargs):
        if self.args.rna_format == "Pathways":
            x_omic = [kwargs[f"x_omic{i}"] for i in range(1, self.num_pathways + 1)]
            h_omic = [self.sig_networks[i](feat) for i, feat in enumerate(x_omic)]
            return torch.stack(h_omic).permute(1, 0, 2)
        return self.sig_networks(kwargs["x_omics"])


# ─────────────────────────────────────────────────────────────────────────────
#  CausalHazardDecomposition
# ─────────────────────────────────────────────────────────────────────────────

class CausalHazardDecomposition(nn.Module):
    """
    Decomposes hazard into additive slot contributions.

    For each time bin t and patient p:

      η_{p,t} = b_t
                + Σ_k wsi_{p,k} · σ(Linear_wsi(slots_wsi_{p,k})_{t})
                + Σ_l omic_{p,l} · σ(Linear_omic(slots_omic_{p,l})_{t})
                + Σ_{k,l} cross_{p,k,l} · σ(Linear_cross([slots_wsi_{p,k}; slots_omic_{p,l}])_{t})

      hazard_{p,t} = σ(η_{p,t})

    Terms wsi_weight, omic_weight, cross_weight are trainable scalars that
    modulate contribution strength (shared across time, per-modality).
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_wsi_slots: int,
        num_omic_slots: int,
        dropout: float = 0.1,
        wsi_weight_init: float = 0.0,
        omic_weight_init: float = 0.0,
        cross_weight_init: float = -3.0,   # exp(-3) ≈ 0.05 — small but non-trivial
    ):
        super().__init__()
        self.num_wsi_slots = num_wsi_slots
        self.num_omic_slots = num_omic_slots
        self.num_classes = num_classes

        # Shared time baseline (learned per-bin)
        self.baseline = nn.Parameter(torch.zeros(1, num_classes))

        # WSI slot → per-time-bin hazard
        self.wsi_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.wsi_head = nn.Linear(dim, num_classes)
        self.wsi_weight = nn.Parameter(torch.tensor(wsi_weight_init))

        # Omics slot → per-time-bin hazard
        self.omic_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.omic_head = nn.Linear(dim, num_classes)
        self.omic_weight = nn.Parameter(torch.tensor(omic_weight_init))

        # Cross-modal: WSI slot × Omics slot → per-time-bin hazard
        self.cross_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_head = nn.Linear(dim, num_classes)
        self.cross_weight = nn.Parameter(torch.tensor(cross_weight_init))

    def forward(self, slots_wsi, slots_omic, return_contributions: bool = False):
        """
        Args:
            slots_wsi: [B, Kw, D]
            slots_omic: [B, Ko, D]
            return_contributions: if True, also return per-slot hazard maps

        Returns:
            hazards: [B, num_classes]   — sigmoid of total logit
            contributions: dict (if return_contributions)
        """
        B, Kw, D = slots_wsi.shape
        Ko = slots_omic.shape[1]

        # exp() ensures positive, differentiable modulation without sigmoid collapse.
        # exp(0) = 1.0, exp(-3) ≈ 0.05
        wsi_mult = self.wsi_weight.exp()     # [1]
        omic_mult = self.omic_weight.exp()   # [1]
        cross_mult = self.cross_weight.exp()  # [1]  exp(-3) ≈ 0.05

        # ── WSI slot contributions ──────────────────────────────────────────
        wsi_h = self.wsi_proj(slots_wsi)                # [B, Kw, D]
        wsi_logits = self.wsi_head(wsi_h)                # [B, Kw, T]
        wsi_contrib = torch.sigmoid(wsi_logits) * wsi_mult  # [B, Kw, T]
        wsi_total = wsi_contrib.sum(dim=1)                  # [B, T]

        # ── Omics slot contributions ────────────────────────────────────────
        omic_h = self.omic_proj(slots_omic)              # [B, Ko, D]
        omic_logits = self.omic_head(omic_h)            # [B, Ko, T]
        omic_contrib = torch.sigmoid(omic_logits) * omic_mult  # [B, Ko, T]
        omic_total = omic_contrib.sum(dim=1)              # [B, T]

        # ── Cross-modal contributions ────────────────────────────────────────
        wsi_exp = slots_wsi.unsqueeze(2).expand(B, Kw, Ko, D)   # [B, Kw, Ko, D]
        omic_exp = slots_omic.unsqueeze(1).expand(B, Kw, Ko, D) # [B, Kw, Ko, D]
        cross_in = torch.cat([wsi_exp, omic_exp], dim=-1)       # [B, Kw, Ko, D*2]
        cross_h = self.cross_proj(cross_in)                       # [B, Kw, Ko, D]
        cross_logits = self.cross_head(cross_h)                  # [B, Kw, Ko, T]
        cross_contrib = torch.sigmoid(cross_logits) * cross_mult  # [B, Kw, Ko, T]
        cross_total = cross_contrib.sum(dim=(1, 2))              # [B, T]

        # ── Total hazard logit ───────────────────────────────────────────────
        # Temperature controls how much total contribution maps to hazard.
        # With ~8 total contribution per bin, /8 → logit≈1 → hazard≈0.73. Prevents
        # collapsed hazard = 0.999 from random init.
        temperature = self._css_temperature
        total = (
            self.baseline
            + wsi_total
            + omic_total
            + cross_total
        ) / temperature  # [B, T]

        if return_contributions:
            return torch.sigmoid(total), {
                "wsi_contrib": wsi_contrib,    # [B, Kw, T]
                "omic_contrib": omic_contrib,   # [B, Ko, T]
                "cross_contrib": cross_contrib,  # [B, Kw, Ko, T]
                "wsi_total": wsi_total,         # [B, T]
                "omic_total": omic_total,       # [B, T]
                "cross_total": cross_total,       # [B, T]
            }
        return torch.sigmoid(total), {}


# ─────────────────────────────────────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────────────────────────────────────

class DCTV4CausalSurvivalSlots(nn.Module, EncoderMixin):
    """
    Causal Survival Slots — v4.0.

    Replaces OT-based transport in DCT with explicit causal hazard decomposition
    and counterfactual slot-swap training.

    The forward path is:
      WSI tokens → slot attention → WSI slots
      Omics tokens → slot attention → Omics slots
      CausalHazardDecomposition → additive hazard → survival logits

    Counterfactual (causal intervention) training:
      1. Sample pairs of patients with different risk levels
      2. Swap WSI slots → recompute hazard
      3. Penalise if hazard does NOT move toward the other patient's risk level
      4. Same for omics slot swaps and cross-modal swaps
    """

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__()
        self.args = args
        self.omic_sizes = args.omic_sizes
        self.num_classes = args.n_classes
        self.wsi_embedding_dim = args.encoding_dim
        self.wsi_projection_dim = args.wsi_projection_dim
        self.omics_input_dim = omic_input_dim

        # ── Encoders ────────────────────────────────────────────────────────
        self._init_per_path_model(self.omic_sizes, args.rna_format)
        self.wsi_mlp = WSI_Mlp(dim_in=self.wsi_embedding_dim, feat_dim=self.wsi_projection_dim)

        # ── Slot attention ───────────────────────────────────────────────────
        slot_init_mode = getattr(args, "dct_slot_init_mode", "gaussian")
        slot_eval_seed = int(getattr(args, "dct_slot_eval_seed", 1729))
        self.slot_attention_wsi = MultiHeadSlotAttention(
            dim=self.wsi_projection_dim,
            num_slots=args.slot_num_wsi,
            iters=args.slot_iters,
            heads=8,
            init_mode=slot_init_mode,
            eval_seed=slot_eval_seed,
        )
        self.slot_attention_omic = MultiHeadSlotAttention(
            dim=self.wsi_projection_dim,
            num_slots=args.slot_num_omics,
            iters=args.slot_iters,
            heads=8,
            init_mode=slot_init_mode,
            eval_seed=slot_eval_seed + 1,
        )

        # ── Causal hazard decomposition ──────────────────────────────────────
        dropout = getattr(args, "otehv2_dropout", 0.1)
        self.hazard_decomp = CausalHazardDecomposition(
            dim=self.wsi_projection_dim,
            num_classes=self.num_classes,
            num_wsi_slots=args.slot_num_wsi,
            num_omic_slots=args.slot_num_omics,
            dropout=dropout,
            wsi_weight_init=float(getattr(args, "css_wsi_weight", 0.0)),
            omic_weight_init=float(getattr(args, "css_omic_weight", 0.0)),
            cross_weight_init=float(getattr(args, "css_cross_weight", -3.0)),
        )
        # Share temperature with hazard_decomp so it can divide total logit
        self.hazard_decomp._css_temperature = float(getattr(args, "css_temperature", 8.0))

        # ── Train-reference: risk anchors for causal pairing ─────────────────
        self._css_num_stages = int(getattr(args, "dct_num_stages", 4))
        self._css_anchor_momentum = float(getattr(args, "dct_anchor_momentum", 0.95))
        self._css_risk_buffer = None   # built in configure_train_reference
        self._css_risk_buffer_ema = None

        # ── Loss weights ─────────────────────────────────────────────────────
        self.css_lambda_ipcw_rank = float(getattr(args, "css_lambda_ipcw_rank", 0.10))
        self.css_lambda_causal = float(getattr(args, "css_lambda_causal", 0.20))
        self.css_lambda_sparsity = float(getattr(args, "css_lambda_sparsity", 0.01))
        self.css_lambda_cross_consistency = float(
            getattr(args, "css_lambda_cross_consistency", 0.05)
        )
        self.css_margin = float(getattr(args, "css_margin", 0.05))
        self.css_rank_margin = float(getattr(args, "dct_ipcw_rank_margin", 0.02))
        self.css_rank_temperature = float(getattr(args, "dct_ipcw_rank_temperature", 0.5))
        self._css_temperature = float(getattr(args, "css_temperature", 8.0))
        self.css_ipcw_max = float(getattr(args, "dct_ipcw_max_weight", 10.0))

        # For multi-head slot: record last assignment for diagnostics
        self._last_wsi_assignment = None
        self._last_omic_assignment = None

    # ── Train reference (risk buffers) ──────────────────────────────────────

    def configure_train_reference(self, event_times, censorship):
        """
        Build per-stage risk anchors from training fold labels.

        We use the same risk-set logic as DCT v3.10:
          - High-risk set  = patients with observed event in that stage
          - Low-risk set    = patients surviving past stage upper bound
        We also build a continuous risk score for causal pairing.
        """
        n = len(event_times)
        t_max = float(event_times.max())
        stage_edges = np.linspace(0, t_max, self._css_num_stages + 1)

        # For each stage, record which patients are high-risk / low-risk
        self._stage_high_risk = []  # list of bool tensors [n] per stage
        self._stage_low_risk = []

        for s in range(self._css_num_stages):
            t_start = stage_edges[s]
            t_end = stage_edges[s + 1]

            # High-risk: event occurred in this bin
            high = (event_times > t_start) & (event_times <= t_end) & (censorship == 0)
            # Low-risk: survived past the bin end (censored after or uncensored event after end)
            low = (event_times > t_end) | ((event_times > t_start) & (censorship == 1) & (event_times > t_end))

            self._stage_high_risk.append(
                torch.tensor(high.astype(np.float32), dtype=torch.float32)
            )
            self._stage_low_risk.append(
                torch.tensor(low.astype(np.float32), dtype=torch.float32)
            )

        # Simple risk score for causal pairing: negated survival time (higher = more risk)
        risk_scores = -event_times.astype(np.float32)
        self._risk_score_tensor = torch.tensor(risk_scores, dtype=torch.float32)
        self._n_train = n

        # EMA buffer for stable risk-level thresholds
        risk_cpu = torch.tensor(risk_scores, dtype=torch.float32)
        self._css_risk_buffer = risk_cpu
        self._css_risk_buffer_ema = risk_cpu.clone()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_ipcw(self, y, c):
        """
        Compute approximate IPCW weights for a batch.
        y: discretized event time class [B] (long)
        c: censoring indicator [B] (float, 0=event, 1=censored)
        """
        B = y.size(0)
        w = torch.ones(B, device=y.device, dtype=y.dtype)
        # Weight uncensored patients more (they provide definitive ordering signal)
        w = torch.where(c == 0, torch.ones_like(w) * 1.0, torch.ones_like(w) * 0.5)
        return w.clamp(max=self.css_ipcw_max)

    def _slots_from_batch(self, **kwargs):
        """Encode inputs and extract slots."""
        x_wsi = kwargs["x_wsi"]
        x_wsi_proj = self.wsi_mlp(x_wsi)
        x_omics = self._encode_omics(kwargs)

        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        # Record last assignment weights for diagnostics (if the module stores them)
        self._last_wsi_assignment = getattr(self.slot_attention_wsi, 'last_token_assignment', None)
        self._last_omic_assignment = getattr(self.slot_attention_omic, 'last_token_assignment', None)

        return slots_wsi, slots_omic

    def _hazard_to_logits(self, hazard, eps=1e-7):
        """Convert hazard [0,1] per bin to logits for NLL loss compatibility."""
        # hazard = sigmoid(logit)  →  logit = logit(hazard) = log(h/(1-h))
        h = hazard.clamp(eps, 1 - eps)
        return torch.logit(h)

    # ── Causal auxiliary losses ───────────────────────────────────────────────

    def _causal_intervention_loss(
        self,
        slots_wsi,
        slots_omic,
        risk_score,
        batch_size,
    ):
        """
        Core causal training signal.

        For each causal pairing:
          1. Find pairs (i, j) where risk_i < risk_j  (i is lower-risk)
          2. Swap WSI slots: hazard_i(wsi_j) should be HIGHER than hazard_i(wsi_i)
          3. Swap omics slots: same logic
          4. Also penalise wrong-direction swaps

        The loss is margin-based hinge loss over the factual vs counterfactual risk gap.
        """
        if slots_wsi.size(0) < 4:
            return torch.zeros(1, device=slots_wsi.device, dtype=slots_wsi.dtype).squeeze()

        T = self.num_classes
        device = slots_wsi.device
        dtype = slots_wsi.dtype

        # ── Factual hazards ───────────────────────────────────────────────────
        hazard_f, _ = self.hazard_decomp(slots_wsi, slots_omic)
        # risk = -sum_t survival(t) = -sum_t product_{u≤t}(1 - h_u)
        # Use the simpler discrete risk: risk = sum_t (1 - S_t)
        # S_t = prod_{u≤t}(1 - h_u) = cumprod flip
        survival = torch.cumprod(1 - hazard_f + 1e-7, dim=1)  # [B, T] — S[t] = prod_{u<=t}(1 - h_u)
        risk_f = (1 - survival).sum(dim=1)  # [B]

        # ── Build causal pairs ────────────────────────────────────────────────
        # Use risk quantile split: top 30% = high risk, bottom 30% = low risk
        n = batch_size
        quantile_low = 0.30
        quantile_high = 0.70

        risk_sorted = risk_f.detach()
        low_thresh = torch.quantile(risk_sorted, quantile_low)
        high_thresh = torch.quantile(risk_sorted, quantile_high)

        low_mask = risk_sorted <= low_thresh    # confirmed low-risk patients
        high_mask = risk_sorted >= high_thresh  # confirmed high-risk patients

        low_idx = torch.where(low_mask)[0]
        high_idx = torch.where(high_mask)[0]

        if low_idx.numel() < 2 or high_idx.numel() < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # ── WSI slot swap ─────────────────────────────────────────────────────
        # Take 4 pairs (low_i, high_j) for efficiency
        n_pairs = min(4, low_idx.numel(), high_idx.numel())
        perm_low = low_idx[torch.randperm(low_idx.numel())[:n_pairs]]
        perm_high = high_idx[torch.randperm(high_idx.numel())[:n_pairs]]

        loss_wsi = torch.tensor(0.0, device=device, dtype=dtype)
        loss_omic = torch.tensor(0.0, device=device, dtype=dtype)
        loss_cross = torch.tensor(0.0, device=device, dtype=dtype)

        for k in range(n_pairs):
            i_low = perm_low[k]   # low-risk patient
            j_high = perm_high[k]  # high-risk patient

            # Factual risks
            r_i = risk_f[i_low]       # low risk
            r_j = risk_f[j_high]       # high risk

            # Counterfactual: swap WSI slots, keep omics
            slots_wsi_swapped = slots_wsi.clone()
            slots_wsi_swapped[i_low] = slots_wsi[j_high]
            slots_wsi_swapped[j_high] = slots_wsi[i_low]

            hazard_cf_wsi, _ = self.hazard_decomp(slots_wsi_swapped, slots_omic)
            surv_cf_wsi = torch.cumprod(1 - hazard_cf_wsi, dim=1).flip(dims=[1])
            risk_cf_wsi = (1 - surv_cf_wsi).sum(dim=1)

            # After WSI swap: patient i (low→wsi_high) should become riskier
            # Patient j (high→wsi_low) should become less risky
            delta_i_wsi = risk_cf_wsi[i_low] - r_i   # should be positive (more risk)
            delta_j_wsi = risk_cf_wsi[j_high] - r_j   # should be negative (less risk)

            # Hinge loss: penalise if delta doesn't have the right sign
            loss_wsi = loss_wsi + F.relu(self.css_margin - delta_i_wsi) + F.relu(
                self.css_margin + delta_j_wsi
            )

            # Counterfactual: swap omics slots, keep WSI
            slots_omic_swapped = slots_omic.clone()
            slots_omic_swapped[i_low] = slots_omic[j_high]
            slots_omic_swapped[j_high] = slots_omic[i_low]

            hazard_cf_omic, _ = self.hazard_decomp(slots_wsi, slots_omic_swapped)
            surv_cf_omic = torch.cumprod(1 - hazard_cf_omic, dim=1).flip(dims=[1])
            risk_cf_omic = (1 - surv_cf_omic).sum(dim=1)

            delta_i_omic = risk_cf_omic[i_low] - r_i
            delta_j_omic = risk_cf_omic[j_high] - r_j
            loss_omic = loss_omic + F.relu(self.css_margin - delta_i_omic) + F.relu(
                self.css_margin + delta_j_omic
            )

            # Cross: swap both WSI and omics together
            slots_both_swapped_wsi = slots_wsi.clone()
            slots_both_swapped_omic = slots_omic.clone()
            slots_both_swapped_wsi[i_low] = slots_wsi[j_high]
            slots_both_swapped_wsi[j_high] = slots_wsi[i_low]
            slots_both_swapped_omic[i_low] = slots_omic[j_high]
            slots_both_swapped_omic[j_high] = slots_omic[i_low]

            hazard_cf_both, _ = self.hazard_decomp(slots_both_swapped_wsi, slots_both_swapped_omic)
            surv_cf_both = torch.cumprod(1 - hazard_cf_both, dim=1).flip(dims=[1])
            risk_cf_both = (1 - surv_cf_both).sum(dim=1)

            delta_i_both = risk_cf_both[i_low] - r_i
            delta_j_both = risk_cf_both[j_high] - r_j
            loss_cross = loss_cross + F.relu(self.css_margin - delta_i_both) + F.relu(
                self.css_margin + delta_j_both
            )

        n_pairs_t = torch.tensor(n_pairs, device=device, dtype=dtype)
        loss_wsi = loss_wsi / n_pairs_t
        loss_omic = loss_omic / n_pairs_t
        loss_cross = loss_cross / n_pairs_t

        total_causal = loss_wsi + loss_omic + loss_cross

        return total_causal

    def _sparsity_loss(self, slots_wsi, slots_omic):
        """
        Encourage sparse hazard contributions.
        Penalises non-zero hazard predictions (pushes most slots toward low contribution).
        """
        hazard, contrib = self.hazard_decomp(slots_wsi, slots_omic, return_contributions=True)
        # L1 on normalized contributions
        wsi_norm = F.normalize(contrib["wsi_contrib"].mean(dim=2), dim=-1)  # [B, Kw]
        omic_norm = F.normalize(contrib["omic_contrib"].mean(dim=2), dim=-1)  # [B, Ko]
        return wsi_norm.abs().mean() + omic_norm.abs().mean()

    def _cross_consistency_loss(self, slots_wsi, slots_omic):
        """
        WSI hazard and omics hazard should be positively correlated across patients.
        If a patient has high WSI hazard, their omics hazard should also be high.
        """
        hazard, contrib = self.hazard_decomp(slots_wsi, slots_omic, return_contributions=True)
        wsi_total = contrib["wsi_total"].mean(dim=1)   # [B]
        omic_total = contrib["omic_total"].mean(dim=1)  # [B]
        # Negative Pearson = penalty
        # Using soft correlation: mean(wsi * omic) - mean(wsi) * mean(omic)
        # Simple approach: MSE between normalized wsi/omic hazard
        wsi_n = (wsi_total - wsi_total.mean()) / (wsi_total.std() + 1e-6)
        omic_n = (omic_total - omic_total.mean()) / (omic_total.std() + 1e-6)
        # Maximise correlation → minimise negative correlation
        return -(wsi_n * omic_n).mean()

    def _ipcw_ranking_loss(self, hazard, y, c, ipcw):
        """
        Pairwise ranking loss: for uncensored pairs, higher event time → higher predicted risk.
        Returns batch-normalized loss (per-patient average) for scale compatibility with NLL.
        """
        B = hazard.size(0)
        if B < 2:
            return torch.zeros(1, device=hazard.device, dtype=hazard.dtype).squeeze()

        # Risk = sum_t (1 - S_t) where S_t = prod_{u<=t}(1 - h_u)
        survival = torch.cumprod(1 - hazard + 1e-7, dim=1)  # [B, T]
        risk = (1 - survival).sum(dim=1)  # [B]

        # Enumerate informative pairs: both uncensored
        loss_sum = torch.tensor(0.0, device=hazard.device, dtype=hazard.dtype)
        n_pairs = 0

        for i in range(B):
            for j in range(i + 1, B):
                ci, cj = c[i].item(), c[j].item()
                yi, yj = y[i].item(), y[j].item()

                # Informative pair: both uncensored, different event times
                if ci == 0 and cj == 0 and yi != yj:
                    if yi < yj:
                        # i's event is earlier → i should have HIGHER risk
                        diff = risk[j] + self.css_rank_margin - risk[i]
                    else:
                        # j's event is earlier → j should have HIGHER risk
                        diff = risk[i] + self.css_rank_margin - risk[j]

                    if diff > 0:  # ranking violated
                        w = (ipcw[i] + ipcw[j]) / 2.0
                        loss_sum = loss_sum + w * diff
                        n_pairs += 1

        if n_pairs > 0:
            # Batch-normalize: divide by B so it's comparable to per-patient NLL
            loss = loss_sum / float(B)
        else:
            loss = torch.zeros(1, device=hazard.device, dtype=hazard.dtype).squeeze()

        return loss

    # ── Main forward ─────────────────────────────────────────────────────────

    def forward(self, **kwargs):
        """
        Returns:
            logits: [B, n_classes] — compatible with NLLSurvLoss
            aux_loss: scalar — causal + ranking + sparsity losses
        """
        slots_wsi, slots_omic = self._slots_from_batch(**kwargs)
        hazard, contrib = self.hazard_decomp(slots_wsi, slots_omic, return_contributions=True)
        logits = self._hazard_to_logits(hazard)  # [B, n_classes]

        if self.training:
            y = kwargs.get("y")
            c = kwargs.get("c")

            # Causal intervention loss
            batch_size = slots_wsi.size(0)
            loss_causal = self._causal_intervention_loss(slots_wsi, slots_omic, None, batch_size)

            # IPCW ranking loss
            ipcw = self._compute_ipcw(y.long(), c.float()) if (y is not None and c is not None) else torch.ones(batch_size, device=hazard.device)
            loss_rank = self._ipcw_ranking_loss(hazard, y.long(), c.float(), ipcw)

            # Sparsity
            loss_sparse = self._sparsity_loss(slots_wsi, slots_omic)

            # Cross-modal consistency
            loss_cross = self._cross_consistency_loss(slots_wsi, slots_omic)

            aux_loss = (
                self.css_lambda_causal * loss_causal
                + self.css_lambda_ipcw_rank * loss_rank
                + self.css_lambda_sparsity * loss_sparse
                + self.css_lambda_cross_consistency * loss_cross
            )
        else:
            aux_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        return logits, aux_loss

    # ── Diagnostic forward (for causal audit) ─────────────────────────────────

    def causal_audit(self, **kwargs):
        """
        Full causal audit of slot-level hazard contributions.

        Returns:
            risk: [B]            — factual risk score
            hazard: [B, T]       — hazard per time bin
            contributions: dict  — per-slot hazard maps
            slot_attention: dict — slot←token assignment weights
        """
        slots_wsi, slots_omic = self._slots_from_batch(**kwargs)
        hazard, contrib = self.hazard_decomp(slots_wsi, slots_omic, return_contributions=True)
        survival = torch.cumprod(1 - hazard + 1e-7, dim=1)  # [B, T] — S[t] = prod_{u<=t}(1 - h_u)
        risk = (1 - survival).sum(dim=1)  # [B]

        return {
            "risk": risk,
            "hazard": hazard,
            "contributions": contrib,
            "slots_wsi": slots_wsi.detach(),
            "slots_omic": slots_omic.detach(),
            "wsi_assignment": self._last_wsi_assignment,
            "omic_assignment": self._last_omic_assignment,
        }

    def counterfactual_audit(self, **kwargs):
        """
        Counterfactual audit: swap high/low risk patient slots and measure
        the resulting risk change (monotonicity check).

        Returns:
            delta_risk_wsi: [B]  — risk change after WSI slot swap
            delta_risk_omic: [B] — risk change after omics slot swap
            delta_risk_both: [B] — risk change after full slot swap
            monotonicity_wsi: scalar — fraction of pairs with correct direction
            monotonicity_omic: scalar
        """
        slots_wsi, slots_omic = self._slots_from_batch(**kwargs)

        hazard_f, _ = self.hazard_decomp(slots_wsi, slots_omic)
        # cumprod(1-h) along time dim gives [P(survive past bin0), P(survive past bin0&bin1), ...]
        # S[t] = prod_{u<=t}(1 - h_u)
        surv_f = torch.cumprod(1 - hazard_f + 1e-7, dim=1)  # [B, T]
        risk_f = (1 - surv_f).sum(dim=1)  # [B]

        B = slots_wsi.size(0)
        T = self.num_classes

        # Use factual risk quantile to identify high/low risk patients
        q70 = torch.quantile(risk_f, 0.70)
        q30 = torch.quantile(risk_f, 0.30)
        high_mask = risk_f >= q70
        low_mask = risk_f <= q30

        high_idx = torch.where(high_mask)[0]
        low_idx = torch.where(low_mask)[0]

        delta_wsi = torch.zeros(B, device=slots_wsi.device)
        delta_omic = torch.zeros(B, device=slots_wsi.device)
        delta_both = torch.zeros(B, device=slots_wsi.device)

        n_wsi_pairs = 0
        n_omic_pairs = 0
        n_both_pairs = 0

        for i in low_idx:
            for j in high_idx:
                # WSI swap
                sw = slots_wsi.clone()
                sw[i] = slots_wsi[j]
                sw[j] = slots_wsi[i]
                h_wsi, _ = self.hazard_decomp(sw, slots_omic)
                s_wsi = torch.cumprod(1 - h_wsi + 1e-7, dim=1)  # [B, T]
                r_wsi = (1 - s_wsi).sum(dim=1)
                delta_wsi[i] = delta_wsi[i] + (r_wsi[i] - risk_f[i])  # should be positive
                delta_wsi[j] = delta_wsi[j] + (r_wsi[j] - risk_f[j])  # should be negative
                n_wsi_pairs += 1

                # Omic swap
                so = slots_omic.clone()
                so[i] = slots_omic[j]
                so[j] = slots_omic[i]
                h_omic, _ = self.hazard_decomp(slots_wsi, so)
                s_omic = torch.cumprod(1 - h_omic + 1e-7, dim=1)  # [B, T]
                r_omic = (1 - s_omic).sum(dim=1)
                delta_omic[i] = delta_omic[i] + (r_omic[i] - risk_f[i])
                delta_omic[j] = delta_omic[j] + (r_omic[j] - risk_f[j])
                n_omic_pairs += 1

                # Both swap
                h_both, _ = self.hazard_decomp(sw, so)
                s_both = torch.cumprod(1 - h_both + 1e-7, dim=1)  # [B, T]
                r_both = (1 - s_both).sum(dim=1)
                delta_both[i] = delta_both[i] + (r_both[i] - risk_f[i])
                delta_both[j] = delta_both[j] + (r_both[j] - risk_f[j])
                n_both_pairs += 1

        # Normalise by number of swap partners
        n_wsi_t = max(1, n_wsi_pairs)
        n_omic_t = max(1, n_omic_pairs)
        n_both_t = max(1, n_both_pairs)
        delta_wsi = delta_wsi / n_wsi_t
        delta_omic = delta_omic / n_omic_t
        delta_both = delta_both / n_both_t

        # Monotonicity: fraction of low-risk patients whose risk INCREASED after swap
        #   AND high-risk patients whose risk DECREASED after swap
        if low_idx.numel() > 0 and high_idx.numel() > 0:
            mono_wsi = (
                (delta_wsi[low_idx] > 0).float().mean()
                + (delta_wsi[high_idx] < 0).float().mean()
            ) / 2.0
            mono_omic = (
                (delta_omic[low_idx] > 0).float().mean()
                + (delta_omic[high_idx] < 0).float().mean()
            ) / 2.0
            mono_both = (
                (delta_both[low_idx] > 0).float().mean()
                + (delta_both[high_idx] < 0).float().mean()
            ) / 2.0
        else:
            mono_wsi = mono_omic = mono_both = torch.tensor(0.0)

        return {
            "delta_risk_wsi": delta_wsi,
            "delta_risk_omic": delta_omic,
            "delta_risk_both": delta_both,
            "monotonicity_wsi": mono_wsi.item(),
            "monotonicity_omic": mono_omic.item(),
            "monotonicity_both": mono_both.item(),
            "n_low_risk": low_idx.numel(),
            "n_high_risk": high_idx.numel(),
            "n_swap_pairs": n_wsi_pairs,
        }
