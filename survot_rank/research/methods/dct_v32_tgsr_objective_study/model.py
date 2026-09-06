"""Isolated TGSR optimization study with matched DCT-Reg objectives.

The original v3.2 four-arm experiment remains NLL-only and immutable.  This
candidate reuses the same transport-guided slot reaggregation, then selects one
of four explicit survival objectives so the source of any score change can be
identified without changing the frozen v3.10 implementation.
"""

from __future__ import annotations

import torch

from survot_rank.research.components.transport_guided_slot_reaggregation import (
    TransportGuidedSlotReaggregation,
)
from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)
from survot_rank.research.methods.dct_v310_directional_regularized_transport.model import (
    DCTV310DirectionalRegularizedTransport,
)


class DCTV32TGSRObjectiveStudy(DCTTransportInterventionConsistency):
    """TGSR with an explicit NLL/IPCW/direction/full objective choice."""

    OBJECTIVES = {
        "nll": (0.0, 0.0),
        "ipcw": (0.10, 0.0),
        "direction": (0.0, 0.05),
        "full": (0.10, 0.05),
    }

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        if str(getattr(args, "bag_loss", "nll_surv")) != "nll_surv":
            raise ValueError("TGSR objective study requires bag_loss='nll_surv'")
        objective = str(getattr(args, "dct_v32_objective", "full"))
        if objective not in self.OBJECTIVES:
            raise ValueError(f"dct_v32_objective must be one of {tuple(self.OBJECTIVES)}")

        ipcw_weight, direction_weight = self.OBJECTIVES[objective]
        values = dict(DCTV310DirectionalRegularizedTransport.FROZEN_ARGUMENTS)
        values.update(
            dct_lambda_ipcw_rank=ipcw_weight,
            dct_v38_lambda_direction=direction_weight,
            fet_lambda_sparse=0.0,
            fet_lambda_faith=0.0,
            spt_lambda_ot=0.0,
            spt_lambda_rank=0.0,
            spt_lambda_stage=0.0,
        )
        for name, value in values.items():
            setattr(args, name, value)

        super().__init__(args, omic_input_dim, omic_names, pathway_names)
        self.dct_v32_objective = objective
        self.dct_lambda_ipcw_rank = ipcw_weight
        self.dct_v38_lambda_direction = direction_weight
        self.reaggregation = TransportGuidedSlotReaggregation(
            self.wsi_projection_dim,
            mode=getattr(args, "dct_v32_feedback", "ot"),
            rounds=getattr(args, "dct_v32_rounds", 1),
            strength=getattr(args, "dct_v32_feedback_strength", 0.25),
            epsilon=getattr(args, "dct_v32_feedback_eps", 0.10),
            sinkhorn_iters=getattr(args, "dct_v32_feedback_iters", 50),
            learnable_strength=getattr(
                args, "dct_v32_learnable_feedback_strength", False
            ),
        )

    def objective_weights(self):
        ipcw_weight, direction_weight = self.OBJECTIVES[self.dct_v32_objective]
        return {
            "nll": 1.0,
            "ipcw_rank": ipcw_weight,
            "direction": direction_weight,
        }

    def get_extra_state(self):
        feedback = self.reaggregation
        return {
            "version": "3.2-tgsr-objective-study",
            "objective": self.dct_v32_objective,
            "feedback": feedback.mode,
            "rounds": feedback.rounds,
            "initial_strength": feedback.strength,
            "learnable_strength": feedback.learnable_strength,
            "epsilon": feedback.epsilon,
            "sinkhorn_iters": feedback.sinkhorn_iters,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError(
                "TGSR objective-study checkpoint configuration does not match "
                "the constructed model"
            )

    def _combine_auxiliary_objectives(
        self,
        *,
        ipcw_rank_loss,
        etar_loss,
        transport_objective,
        transport_metrics,
        epoch,
    ):
        del etar_loss, transport_metrics, epoch
        return self.dct_lambda_ipcw_rank * ipcw_rank_loss + transport_objective

    def _encode_transport_slots(self, x_wsi_proj, x_omics, kwargs):
        wsi, omic, wsi_assignment, omic_assignment = super()._encode_transport_slots(
            x_wsi_proj, x_omics, kwargs
        )
        wsi, omic = self.reaggregation(wsi, omic, x_wsi_proj, x_omics)
        return wsi, omic, wsi_assignment, omic_assignment

    def forward(self, **kwargs):
        if kwargs.get("wsi_missing", False) or kwargs.get("omic_missing", False):
            raise ValueError("TGSR objective study requires both WSI and omics modalities")
        logits, auxiliary_loss = super().forward(**kwargs)
        diagnostics = {
            f"tgsr_{name}": value.detach() if torch.is_tensor(value) else value
            for name, value in self.reaggregation.last_diagnostics.items()
        }
        if self.training:
            self.last_training_losses.update(diagnostics)
        elif self.last_explanations is not None:
            self.last_explanations.update(diagnostics)
        return logits, auxiliary_loss
