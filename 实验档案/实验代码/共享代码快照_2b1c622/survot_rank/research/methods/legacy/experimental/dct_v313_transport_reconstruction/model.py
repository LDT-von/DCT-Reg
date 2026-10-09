"""DCT v3.13: transport-aware pathway reconstruction.

The factual prediction path is the frozen DCT v3.11 slot-interpretable path.
During training, v3.13 adds two equally weighted reconstruction objectives:

* omics semantic slots reconstruct the encoded pathway tokens;
* WSI slots are transported into omics slot coordinates by the factual,
  stage-wise, multi-geometry Sinkhorn plans and reconstruct the same targets.

The reconstruction target is stop-gradient.  The event gate used to combine
stages is detached so the auxiliary objective cannot repurpose the survival
gate, while gradients still flow through the factual transport plans, shared
semantic slots, and both modality encoders.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.legacy.experimental.dct_v311_slot_interpretable.model import (
    DCTV311SlotInterpretable,
)


_DEBUG_LOG_PATH = Path(os.environ.get(
    "DCT_DEBUG_LOG",
    "/data1/DCT-Reg/.cursor/debug-486347.log",
))


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    """Append a single NDJSON line to the debug log (session 486347)."""
    try:
        payload = {
            "id": f"log_{int(time.time()*1000)}_{os.getpid()}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": os.environ.get("DCT_DEBUG_RUN_ID", "ablation"),
            "hypothesisId": hypothesis_id,
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:  # pragma: no cover — never break training
        try:
            sys.stderr.write(f"[debug-log] {e}\n")
        except Exception:
            pass


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
        decoded, _ = self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        decoded = queries + decoded
        return decoded + self.output_mlp(self.output_norm(decoded))


class DCTV313TransportReconstruction(DCTV311SlotInterpretable):
    """DCT v3.13 with omics self- and transport-aware cross-reconstruction."""

    RECONSTRUCTION_WEIGHT = 0.10
    RECONSTRUCTION_SELF_FRACTION = 0.50
    RECONSTRUCTION_CROSS_FRACTION = 0.50
    RECONSTRUCTION_RAMP_START = 2
    RECONSTRUCTION_RAMP_EPOCHS = 5

    FROZEN_ARGUMENTS = dict(DCTV311SlotInterpretable.FROZEN_ARGUMENTS)
    FROZEN_ARGUMENTS.update(
        {
            "dct_v313_lambda_reconstruction": RECONSTRUCTION_WEIGHT,
            "dct_v313_reconstruction_self_fraction": RECONSTRUCTION_SELF_FRACTION,
            "dct_v313_reconstruction_cross_fraction": RECONSTRUCTION_CROSS_FRACTION,
            "dct_v313_reconstruction_ramp_start": RECONSTRUCTION_RAMP_START,
            "dct_v313_reconstruction_ramp_epochs": RECONSTRUCTION_RAMP_EPOCHS,
        }
    )

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        if str(getattr(args, "rna_format", "Pathways")) != "Pathways":
            raise ValueError(
                "DCT v3.13 reconstructs pathway tokens and requires rna_format='Pathways'"
            )
        # Respect CLI `--set` overrides: FROZEN_ARGUMENTS only stamp keys the
        # caller did not already supply (ablation studies).
        v313_frozen = {
            name: value
            for name, value in self.FROZEN_ARGUMENTS.items()
            if name not in getattr(args, "_dct_user_overrides", set())
        }
        for name, value in v313_frozen.items():
            setattr(args, name, value)
        # Ablation hooks: allow callers to override reconstruction knobs
        # via args (FROZEN_ARGUMENTS already populated, but ablation flags
        # like --dct_v313_disable_self_reconstruction=true force zero weights).
        def _as_bool(v):
            if isinstance(v, bool):
                return v
            return str(v).lower() in ("1", "true", "yes", "y", "on")

        self._ablation_disable_self = _as_bool(
            getattr(args, "dct_v313_disable_self_reconstruction", False)
        )
        self._ablation_disable_cross = _as_bool(
            getattr(args, "dct_v313_disable_cross_reconstruction", False)
        )
        self._ablation_lambda_scale = float(
            getattr(args, "dct_v313_lambda_reconstruction_scale", 1.0)
        )
        # §3.1 #3: cross-reconstruction mode.
        #
        # ``transport`` (default): pass OT-transported WSI slots to the
        # cross decoder.  ``direct``: pass the un-transported WSI slots
        # directly, isolating the OT contribution to cross reconstruction.
        cross_mode = str(getattr(args, "dct_v313_cross_mode", "transport")).lower()
        if cross_mode not in ("transport", "direct"):
            raise ValueError(
                "dct_v313_cross_mode must be 'transport' or 'direct', got "
                f"{cross_mode!r}"
            )
        self._cross_mode = cross_mode
        # §3.1 #4: OT plan mode.
        #
        # ``learned`` (default): use the Sinkhorn-projected OT plans from
        # the evidence-cost geometry.  ``independent``: replace the plan
        # with T = a bᵀ (outer product of marginals) in every consumer.
        plan_mode = str(getattr(args, "dct_v313_plan_mode", "learned")).lower()
        if plan_mode not in ("learned", "independent"):
            raise ValueError(
                "dct_v313_plan_mode must be 'learned' or 'independent', got "
                f"{plan_mode!r}"
            )
        self._plan_mode = plan_mode
        weighting_mode = str(getattr(args, "dct_v313_recon_weighting", "legacy")).lower()
        if weighting_mode not in ("legacy", "per_branch"):
            raise ValueError(
                "dct_v313_recon_weighting must be 'legacy' or 'per_branch', got "
                f"{weighting_mode!r}"
            )
        self._recon_weighting_mode = weighting_mode
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = int(self.wsi_projection_dim)
        # ------------------------------------------------------------------
        # Effective per-branch reconstruction coefficients.
        #
        # We always work with TWO independent coefficients (self/cross).
        # The same pair is consumed by `reconstruction_losses` and
        # `forward`, so the value the combiner multiplies with is exactly
        # what the audit log reports.  ``dct_v313_lambda_reconstruction_effective``
        # holds the SUM of the two — i.e. the constant that the legacy code
        # used to renormalise to (no double-down).
        #
        #   - legacy     (commit 93d8314): ``coef_i = effective_weight × fraction_i``.
        #     When one branch is disabled the legacy renormalisation keeps
        #     the SUM constant (== RECONSTRUCTION_WEIGHT), so the surviving
        #     per-branch coefficient is the half-constant 0.025 (the historic
        #     "double-down" weakness — compensated in some ablation runs by
        #     scale=2).
        #   - per_branch (new v3.13 paper recipe): ``coef_i = WEIGHT × fraction_i``
        #     independently of the disable flags.  Disabling self zeros only
        #     the self term; cross keeps its full constant 0.05.  Disable
        #     cross → only self survives at 0.05.
        #   - scale=1 is mandatory in per_branch mode; the historic scale=2
        #     compensation from the legacy recipe MUST NOT be carried over.
        # ------------------------------------------------------------------
        frac_sum = max(
            self.RECONSTRUCTION_SELF_FRACTION + self.RECONSTRUCTION_CROSS_FRACTION, 1e-8
        )
        if self._recon_weighting_mode == "legacy":
            if self._ablation_disable_self and self._ablation_disable_cross:
                legacy_renormalised_weight = 0.0
            elif self._ablation_disable_self:
                legacy_renormalised_weight = (
                    self.RECONSTRUCTION_WEIGHT
                    * self._ablation_lambda_scale
                    * self.RECONSTRUCTION_CROSS_FRACTION
                    / frac_sum
                )
            elif self._ablation_disable_cross:
                legacy_renormalised_weight = (
                    self.RECONSTRUCTION_WEIGHT
                    * self._ablation_lambda_scale
                    * self.RECONSTRUCTION_SELF_FRACTION
                    / frac_sum
                )
            else:
                legacy_renormalised_weight = (
                    self.RECONSTRUCTION_WEIGHT * self._ablation_lambda_scale
                )
            self_branch_coef = (
                0.0
                if self._ablation_disable_self
                else legacy_renormalised_weight * self.RECONSTRUCTION_SELF_FRACTION
            )
            cross_branch_coef = (
                0.0
                if self._ablation_disable_cross
                else legacy_renormalised_weight * self.RECONSTRUCTION_CROSS_FRACTION
            )
        else:  # per_branch
            if self._ablation_lambda_scale != 1.0:
                raise ValueError(
                    "dct_v313_recon_weighting='per_branch' forbids "
                    "dct_v313_lambda_reconstruction_scale != 1.0 "
                    "(got {scale}).  The legacy scale=2 compensation must NOT be "
                    "carried into the new recipe.".format(
                        scale=self._ablation_lambda_scale
                    )
                )
            self_branch_coef = (
                0.0
                if self._ablation_disable_self
                else self.RECONSTRUCTION_WEIGHT * self.RECONSTRUCTION_SELF_FRACTION
            )
            cross_branch_coef = (
                0.0
                if self._ablation_disable_cross
                else self.RECONSTRUCTION_WEIGHT * self.RECONSTRUCTION_CROSS_FRACTION
            )

        self.dct_v313_reconstruction_self_coef = self_branch_coef
        self.dct_v313_reconstruction_cross_coef = cross_branch_coef
        # ``dct_v313_lambda_reconstruction_effective`` reports the SUM of
        # per-branch coefficients — i.e. the maximum coefficient that the
        # combiner can apply to the loss before the ramp multiplier.  This
        # replaces the legacy "renormalised weight" semantics and makes the
        # audit field directly comparable to ``effective_recon_coefficients()``.
        self.dct_v313_lambda_reconstruction_effective = (
            self_branch_coef + cross_branch_coef
        )
        self.dct_v313_reconstruction_self_fraction = (
            0.0 if self._ablation_disable_self else self.RECONSTRUCTION_SELF_FRACTION
        )
        self.dct_v313_reconstruction_cross_fraction = (
            0.0 if self._ablation_disable_cross else self.RECONSTRUCTION_CROSS_FRACTION
        )
        self.dct_v313_reconstruction_ramp_start = self.RECONSTRUCTION_RAMP_START
        self.dct_v313_reconstruction_ramp_epochs = self.RECONSTRUCTION_RAMP_EPOCHS
        heads = self._compatible_num_heads(dim, int(getattr(args, "otehv2_heads", 4)))
        self.pathway_reconstruction_decoder = OmicsTokenReconstructionDecoder(
            dim=dim,
            num_pathways=int(self.num_pathways),
            num_heads=heads,
        )
        self._last_reconstruction_self = 0.0
        self._last_reconstruction_cross = 0.0
        self._last_reconstruction_total = 0.0
        self._last_reconstruction_ramp = 0.0
        # Honour ablation flags for per-slot NLL and diversity too (added Sep 29 2026
        # after discovering v3.13 hard-coded PER_SLOT_NLL_WEIGHT in aux_loss, which
        # made --set dct_v311_lambda_slot_nll=0 silently ignored).
        self.dct_v313_lambda_slot_nll_effective = float(
            getattr(args, "dct_v311_lambda_slot_nll", self.PER_SLOT_NLL_WEIGHT)
        )
        self.dct_v313_lambda_slot_diversity_effective = float(
            getattr(args, "dct_v311_lambda_slot_diversity", self.SLOT_DIVERSITY_WEIGHT)
        )
        # #region agent log — H1: verify effective weights match args
        _debug_log(
            hypothesis_id="H1",
            location="dct_v313/model.py:init",
            message="effective weights at init",
            data={
                "args_dct_v311_lambda_slot_nll": float(getattr(args, "dct_v311_lambda_slot_nll", -1.0)),
                "args_dct_v311_lambda_slot_diversity": float(getattr(args, "dct_v311_lambda_slot_diversity", -1.0)),
                "args_dct_lambda_ipcw_rank": float(getattr(args, "dct_lambda_ipcw_rank", -1.0)),
                "args_dct_v313_disable_self_reconstruction": getattr(args, "dct_v313_disable_self_reconstruction", None),
                "args_dct_v313_disable_cross_reconstruction": getattr(args, "dct_v313_disable_cross_reconstruction", None),
                "args_dct_v313_recon_weighting": getattr(args, "dct_v313_recon_weighting", None),
                "self_dct_v313_lambda_slot_nll_effective": self.dct_v313_lambda_slot_nll_effective,
                "self_dct_v313_lambda_slot_diversity_effective": self.dct_v313_lambda_slot_diversity_effective,
                "self_dct_v313_lambda_reconstruction_effective": self.dct_v313_lambda_reconstruction_effective,
                "self_dct_v313_reconstruction_self_coef": float(self.dct_v313_reconstruction_self_coef),
                "self_dct_v313_reconstruction_cross_coef": float(self.dct_v313_reconstruction_cross_coef),
                "self_dct_v313_recon_weighting_mode": self._recon_weighting_mode,
                "PER_SLOT_NLL_WEIGHT_class_const": self.PER_SLOT_NLL_WEIGHT,
                "SLOT_DIVERSITY_WEIGHT_class_const": self.SLOT_DIVERSITY_WEIGHT,
                "class_id": id(self.__class__),
            },
        )
        # #endregion agent log

    @staticmethod
    def _compatible_num_heads(dim: int, preferred: int) -> int:
        heads = max(1, min(int(preferred), int(dim)))
        while dim % heads != 0:
            heads -= 1
        return heads

    @classmethod
    def objective_weights(cls) -> dict[str, float]:
        """Return the abstract recipe weight dict (class-constant only).

        §3.1 #1: in ``per_branch`` mode the reconstruction loss is decomposed
        into per-branch keys (``self``, ``cross``) and the aggregated
        ``reconstruction`` key is *omitted*.  In ``legacy`` mode the original
        aggregated key is returned for backward compatibility.  Use the
        instance-level ``objective_weights()`` for effective post-init values
        (FROZEN override accounting).
        """
        weights = {
            "nll": cls.NLL_WEIGHT,
            "ipcw_rank": cls.IPCW_RANK_WEIGHT,
            "per_slot_nll": cls.PER_SLOT_NLL_WEIGHT,
            "slot_diversity": cls.SLOT_DIVERSITY_WEIGHT,
            "weighting_mode": "per_branch",
        }
        weights["reconstruction_self"] = (
            cls.RECONSTRUCTION_WEIGHT * cls.RECONSTRUCTION_SELF_FRACTION
        )
        weights["reconstruction_cross"] = (
            cls.RECONSTRUCTION_WEIGHT * cls.RECONSTRUCTION_CROSS_FRACTION
        )
        return weights

    def objective_weights_effective(self) -> dict[str, float]:
        """Return the *effective* objective weights actually applied at
        forward time (after FROZEN overrides and ablation flags).

        Companion to :meth:`objective_weights`.  Use this when you need the
        exact coefficients that multiplied each loss term — for audit logs,
        weight summaries, and downstream consumers that care about the
        post-construction effective weights rather than the abstract
        recipe.
        """
        weights = {
            "nll": self.NLL_WEIGHT,
            "ipcw_rank": self.dct_lambda_ipcw_rank,
            "per_slot_nll": self.dct_v313_lambda_slot_nll_effective,
            "slot_diversity": self.dct_v313_lambda_slot_diversity_effective,
            "weighting_mode": self._recon_weighting_mode,
        }
        if self._recon_weighting_mode == "per_branch":
            weights["reconstruction_self"] = float(self.dct_v313_reconstruction_self_coef)
            weights["reconstruction_cross"] = float(self.dct_v313_reconstruction_cross_coef)
        else:
            weights["reconstruction"] = self.RECONSTRUCTION_WEIGHT
            weights["reconstruction_self"] = (
                self.RECONSTRUCTION_WEIGHT * self.RECONSTRUCTION_SELF_FRACTION
            )
            weights["reconstruction_cross"] = (
                self.RECONSTRUCTION_WEIGHT * self.RECONSTRUCTION_CROSS_FRACTION
            )
        return weights

    # The v3.14 schema emitter historically invoked
    # ``self.objective_weights()`` on a live instance.  With the dual-API
    # change, ``objective_weights`` is the abstract recipe (classmethod);
    # ``objective_weights_effective`` is the post-FROZEN actual values.
    # Callers that need the abstract recipe on an instance should use
    # ``type(self).objective_weights()`` (or equivalently, call the
    # classmethod directly).

    def effective_recon_coefficients(self) -> dict[str, float]:
        """Return the *effective* per-branch coefficients actually used by
        `reconstruction_losses` after weighting mode and ablation flags are
        applied.  Always reflects what the forward pass multiplies the loss
        terms by, so it can be cross-referenced with the audit log without
        guessing the mode.
        """
        return {
            "weighting_mode": self._recon_weighting_mode,
            "self": float(self.dct_v313_reconstruction_self_coef),
            "cross": float(self.dct_v313_reconstruction_cross_coef),
            "total": float(self.dct_v313_lambda_reconstruction_effective),
            "disable_self": bool(self._ablation_disable_self),
            "disable_cross": bool(self._ablation_disable_cross),
            "lambda_scale": float(self._ablation_lambda_scale),
            "cross_mode": self._cross_mode,
            "plan_mode": self._plan_mode,
        }

    @classmethod
    def key_contributions(cls) -> list[str]:
        return [
            "Transport-aware WSI-to-omics reconstruction through factual Sinkhorn plans",
            "Omics semantic-slot self-reconstruction at pathway-token resolution",
            "Per-slot survival supervision with modality-specific diversity constraints",
            "IPCW-aware patient-risk ranking",
        ]

    # DCT-Reg main did not receive commit 982a090 from the v3.11 repository.
    # Override the historical method locally so v3.13 uses the fixed cumulative
    # discrete-time likelihood without rewriting the archived v3.11 baseline.
    def _nll_surv_per_slot(self, hazard, y_onehot, event_mask, censor_mask, ipcw):
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

    def _reconstruction_ramp(self, epoch: int) -> float:
        start = int(self.dct_v313_reconstruction_ramp_start)
        duration = int(self.dct_v313_reconstruction_ramp_epochs)
        if duration <= 0:
            return 1.0
        return min(1.0, max(0.0, (float(epoch) - float(start)) / float(duration)))

    @staticmethod
    def _batch_bool_mask(value, *, batch: int, device, default: bool) -> torch.Tensor:
        if value is None:
            return torch.full((batch,), default, dtype=torch.bool, device=device)
        if torch.is_tensor(value):
            mask = value.to(device=device, dtype=torch.bool).reshape(-1)
            if mask.numel() == 1:
                return mask.expand(batch)
            if mask.numel() != batch:
                raise ValueError(
                    f"availability mask must contain 1 or {batch} values, got {mask.numel()}"
                )
            return mask
        return torch.full((batch,), bool(value), dtype=torch.bool, device=device)

    def _omics_available_mask(self, kwargs, target: torch.Tensor) -> torch.Tensor:
        batch = target.size(0)
        explicit = kwargs.get("omic_available", kwargs.get("omics_available"))
        available = self._batch_bool_mask(
            explicit, batch=batch, device=target.device, default=True
        )
        missing = self._batch_bool_mask(
            kwargs.get("omic_missing"), batch=batch, device=target.device, default=False
        )
        return available & ~missing

    @staticmethod
    def _independent_outer_plan(
        rows: torch.Tensor,
        cols: torch.Tensor,
        num_geometries: int,
    ) -> list[tuple[torch.Tensor, ...]]:
        """§3.1 #4: build the marginal-outer-product plan T = a bᵀ.

        Args:
            rows: ``[B, S, K_w]`` row marginals (per-stage WSI mass).
            cols: ``[B, S, K_o]`` column marginals (per-stage omic mass).
            num_geometries: number of cost geometries per stage (kept to
                preserve the ``plans[stage_idx] = tuple[plan]`` contract).

        Returns:
            ``list[S]`` of ``num_geometries``-tuple of plans, each ``[B, K_w, K_o]``.
        """
        plans: list[tuple[torch.Tensor, ...]] = []
        for stage_idx in range(rows.size(1)):
            stage_plans = []
            for _ in range(num_geometries):
                plan = torch.einsum(
                    "bw,bo->bwo", rows[:, stage_idx], cols[:, stage_idx]
                )
                stage_plans.append(plan)
            plans.append(tuple(stage_plans))
        return plans

    def _plans_from_cost_tensor(self, costs, rows, cols, epoch, *, replay_fixed=False):
        """§3.1 #4: when ``_plan_mode == 'independent'``, replace the
        Sinkhorn-projected plans with the marginal outer product ``T = a bᵀ``.

        The outer product carries no transport information — both modalities
        are projected independently — but still respects the per-stage
        marginals, the multi-stage shape, and the (geometry, plan) tuple
        contract that downstream consumers expect.  Distances are computed
        honestly: the per-stage mean cost mass of the independent plan on
        each cost geometry, averaged across stages.  This makes the audit
        log transparent (you can see the cost mass that *would* be
        transported under the marginals without Sinkhorn alignment).
        """
        if self._plan_mode != "independent":
            return super()._plans_from_cost_tensor(
                costs, rows, cols, epoch, replay_fixed=replay_fixed
            )
        num_geometries = costs.size(2)
        independent_plans = self._independent_outer_plan(
            rows, cols, num_geometries
        )
        # Honest distance reporting: the cost mass the independent plan
        # would transport.  This is purely diagnostic; it never feeds the
        # gradient because the plan is consumed downstream by frozen OT
        # mass terms.
        stage_distances = []
        for stage_idx, stage_plans in enumerate(independent_plans):
            geo_distances = []
            for geo_idx, plan in enumerate(stage_plans):
                geo_distances.append(
                    (plan * costs[:, stage_idx, geo_idx]).sum(dim=(1, 2))
                )
            stage_distances.append(torch.stack(geo_distances).mean())
        distance = torch.stack(stage_distances).mean()
        return independent_plans, distance

    @staticmethod
    def _transport_wsi_to_omic(
        slots_wsi: torch.Tensor,
        factual_plans,
        stage_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(factual_plans) != stage_gate.size(1):
            raise ValueError(
                "number of transport stages must match the factual event-gate width"
            )

        transported_stages = []
        for stage_plans in factual_plans:
            if len(stage_plans) < 1:
                raise ValueError("every transport stage must contain at least one geometry")
            consensus_plan = torch.stack(tuple(stage_plans), dim=0).mean(dim=0)
            column_mass = consensus_plan.sum(dim=1).clamp_min(1e-8)
            transported = torch.einsum("bwo,bwd->bod", consensus_plan, slots_wsi)
            transported_stages.append(transported / column_mass.unsqueeze(-1))

        staged = torch.stack(transported_stages, dim=1)
        weights = stage_gate.detach()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        aggregate = torch.einsum("bs,bsod->bod", weights, staged)
        return aggregate, staged

    @staticmethod
    def _reconstruction_distance(
        prediction: torch.Tensor,
        target: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"reconstruction shape {tuple(prediction.shape)} does not match "
                f"target {tuple(target.shape)}"
            )
        if not bool(available.any()):
            return prediction.sum() * 0.0

        prediction = prediction[available]
        target = target[available].detach()
        prediction = F.layer_norm(prediction, (prediction.size(-1),))
        target = F.layer_norm(target, (target.size(-1),))
        cosine = (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()
        robust_l1 = F.smooth_l1_loss(prediction, target)
        return 0.5 * (cosine + robust_l1)

    def reconstruction_losses(
        self,
        *,
        x_omics: torch.Tensor,
        slots_wsi: torch.Tensor,
        slots_omic: torch.Tensor,
        factual_plans,
        factual_gate: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target = x_omics.detach()
        if self._ablation_disable_self:
            self_loss = slots_omic.new_zeros(())
        else:
            self_prediction = self.pathway_reconstruction_decoder(slots_omic)
            self_loss = self._reconstruction_distance(
                self_prediction, target, available
            )
        if self._ablation_disable_cross:
            cross_loss = slots_omic.new_zeros(())
            transported_wsi = slots_omic.new_zeros(
                slots_omic.size(0), self.spt_num_stages, *slots_omic.shape[1:]
            )
        else:
            if self._cross_mode == "direct":
                # §3.1 #3: bypass OT entirely; feed raw WSI slots directly
                # into the same decoder used by the transport branch.  Both
                # the OT output and the raw slots are [B, K, D] tensors —
                # only the semantic content differs (raw slots live in WSI
                # coordinate space; OT output is the WSI→omic barycentric
                # projection).  No reshape required.
                transported_wsi = slots_wsi
            else:
                transported_wsi, _ = self._transport_wsi_to_omic(
                    slots_wsi, factual_plans, factual_gate
                )
            cross_prediction = self.pathway_reconstruction_decoder(transported_wsi)
            cross_loss = self._reconstruction_distance(
                cross_prediction, target, available
            )
        # Use the explicit per-branch coefficients rather than the
        # `effective_weight × fraction` decomposition: this guarantees that
        # what the combiner multiplies the loss by is exactly what the audit
        # log stored at construction time (no silent normalisation).  When
        # the branch is disabled both its coefficient and its loss value are
        # zero, so the gradient contribution is also zero.
        total = (
            self.dct_v313_reconstruction_self_coef * self_loss
            + self.dct_v313_reconstruction_cross_coef * cross_loss
        )
        return self_loss, cross_loss, total, transported_wsi

    def forward(self, **kwargs):
        if not self.training:
            return super().forward(**kwargs)

        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        if x_omics.ndim != 3 or x_omics.size(1) != int(self.num_pathways):
            raise ValueError(
                "DCT v3.13 expected encoded pathway tokens [B, P, D] with "
                f"P={self.num_pathways}, got {tuple(x_omics.shape)}"
            )

        slots_wsi, slots_omic, _, _ = self._encode_transport_slots(
            x_wsi_proj, x_omics, kwargs
        )
        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        self._reset_ipcw_memory_for_epoch(epoch)

        factual_costs, rows, cols, _ = self._cost_tensor(slots_wsi, slots_omic)
        factual_plans, ot_distance = self._plans_from_cost_tensor(
            factual_costs, rows, cols, epoch, replay_fixed=False
        )
        self._last_factual_costs = factual_costs.detach()
        self._last_factual_rows = rows.detach()
        self._last_factual_cols = cols.detach()
        self._last_slots_wsi = slots_wsi.detach()
        self._last_slots_omic = slots_omic.detach()

        factual_logits, factual_gate = self._encode_logits_from_plans(
            slots_wsi, slots_omic, factual_plans
        )
        factual_risk = self._risk(factual_logits)

        low_weights = factual_costs.new_zeros(
            factual_costs.size(0), self.spt_num_stages
        )
        high_weights = torch.zeros_like(low_weights)
        ipcw_rank_loss = factual_costs.new_zeros(())
        event_time = kwargs.get("event_time")
        censorship = kwargs.get("c")
        if event_time is not None and censorship is not None:
            low_weights, high_weights = self._stage_membership_weights(
                event_time, censorship
            )
            if self.dct_lambda_ipcw_rank != 0.0:
                ipcw_rank_loss = self._ipcw_pairwise_ranking_loss(
                    factual_logits, event_time, censorship
                )
                self._remember_ipcw_batch(
                    factual_risk,
                    event_time.float().view(-1),
                    censorship.float().view(-1),
                )
            self._update_risk_anchors(
                factual_costs.detach(), low_weights, high_weights
            )

        per_slot_nll = factual_costs.new_zeros(())
        slot_diversity = factual_costs.new_zeros(())
        y = kwargs.get("y")
        if y is not None and event_time is not None and censorship is not None:
            per_slot_nll = self.per_slot_nll_loss(
                slots_wsi, slots_omic, y, event_time, censorship
            )
            slot_diversity = self.slot_diversity_loss(slots_wsi, slots_omic)

        available = self._omics_available_mask(kwargs, x_omics)
        reconstruction_self, reconstruction_cross, reconstruction, transported = (
            self.reconstruction_losses(
                x_omics=x_omics,
                slots_wsi=slots_wsi,
                slots_omic=slots_omic,
                factual_plans=factual_plans,
                factual_gate=factual_gate,
                available=available,
            )
        )
        ramp = self._reconstruction_ramp(epoch)
        # The combiner uses the per-branch coefficients directly so the
        # value multiplied into the loss matches the audit log exactly.
        # No additional scaling by ``effective_weight_total`` happens here
        # (that would be a double-down bug).  The ramp still scales both
        # branches uniformly.
        effective_reconstruction_weight = ramp

        aux_loss = (
            self.dct_lambda_ipcw_rank * ipcw_rank_loss
            + self.dct_v313_lambda_slot_nll_effective * per_slot_nll
            + self.dct_v313_lambda_slot_diversity_effective * slot_diversity
            + effective_reconstruction_weight * reconstruction
        )

        active_stage_fraction = (
            ((low_weights > 0).any(dim=0) & (high_weights > 0).any(dim=0))
            .to(factual_costs.dtype)
            .mean()
        )
        row_entropy = -(
            rows.clamp_min(1e-8) * rows.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(max(2, rows.size(-1)))
        col_entropy = -(
            cols.clamp_min(1e-8) * cols.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(max(2, cols.size(-1)))

        self._last_reconstruction_self = float(reconstruction_self.detach())
        self._last_reconstruction_cross = float(reconstruction_cross.detach())
        self._last_reconstruction_total = float(reconstruction.detach())
        self._last_reconstruction_ramp = ramp
        self.last_training_losses = {
            "ot": ot_distance.detach(),
            "ipcw_rank": ipcw_rank_loss.detach(),
            "v311_per_slot_nll": per_slot_nll.detach(),
            "v311_slot_diversity": slot_diversity.detach(),
            "v311_slot_variance": factual_costs.new_tensor(self._last_slot_variance),
            "v313_reconstruction_self": reconstruction_self.detach(),
            "v313_reconstruction_cross": reconstruction_cross.detach(),
            "v313_reconstruction_total": reconstruction.detach(),
            "v313_reconstruction_ramp": factual_costs.new_tensor(ramp),
            "v313_reconstruction_weight": factual_costs.new_tensor(
                effective_reconstruction_weight
            ),
            "v313_omics_available_fraction": available.to(factual_costs.dtype).mean(),
            "v313_transported_slot_norm": transported.detach().norm(dim=-1).mean(),
            "active_stage_fraction": active_stage_fraction.detach(),
            "anchor_coverage": self.risk_anchor_seen.to(factual_costs.dtype).mean().detach(),
            "evidence_marginal_entropy": torch.cat(
                [row_entropy.flatten(), col_entropy.flatten()]
            ).mean().detach(),
        }
        return factual_logits, aux_loss
