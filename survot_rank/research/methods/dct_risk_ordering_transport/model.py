"""Risk Ordering Transport candidate built on DCT's intervention path."""

from __future__ import annotations

from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)
from survot_rank.research.methods.dct_v310_directional_regularized_transport.model import (
    DCTV310DirectionalRegularizedTransport,
)


class DCTRiskOrderingTransport(DCTTransportInterventionConsistency):
    """Train ordered risk responses to low/high transport interventions.

    The factual prediction path and NLL + IPCW objective match DCT v3.10. The
    candidate adds configurable endpoint direction and midpoint dose-ordering
    constraints through fresh Sinkhorn solves. It remains a model-based
    structural intervention and does not imply an identified causal effect.
    """

    IPCW_RANK_WEIGHT = 0.10

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        if str(getattr(args, "bag_loss", "nll_surv")) != "nll_surv":
            raise ValueError("Risk Ordering Transport requires bag_loss='nll_surv'")

        direction_weight = float(getattr(args, "dct_rot_lambda_direction", 0.10))
        dose_weight = float(getattr(args, "dct_rot_lambda_dose", 0.05))
        direction_margin = float(getattr(args, "dct_rot_direction_margin", 0.02))
        dose_margin = float(getattr(args, "dct_rot_dose_margin", 0.005))
        for name, value in {
            "dct_rot_lambda_direction": direction_weight,
            "dct_rot_lambda_dose": dose_weight,
            "dct_rot_direction_margin": direction_margin,
            "dct_rot_dose_margin": dose_margin,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        values = dict(DCTV310DirectionalRegularizedTransport.FROZEN_ARGUMENTS)
        values.update(
            dct_lambda_ipcw_rank=self.IPCW_RANK_WEIGHT,
            dct_v38_lambda_direction=direction_weight,
            dct_v38_lambda_dose=dose_weight,
            dct_v38_lambda_reconfiguration=0.0,
            dct_v38_direction_margin=direction_margin,
            dct_v38_dose_margin=dose_margin,
            dct_v38_warmup_epochs=0,
            dct_v38_ramp_epochs=0,
            dct_v38_dose_every=1,
        )
        for name, value in values.items():
            setattr(args, name, value)

        super().__init__(args, omic_input_dim, omic_names, pathway_names)
        self.dct_lambda_ipcw_rank = self.IPCW_RANK_WEIGHT
        self.dct_v38_lambda_direction = direction_weight
        self.dct_v38_lambda_dose = dose_weight
        self.dct_v38_lambda_reconfiguration = 0.0

    def objective_weights(self) -> dict[str, float]:
        return {
            "nll": 1.0,
            "ipcw_rank": self.IPCW_RANK_WEIGHT,
            "direction": self.dct_v38_lambda_direction,
            "dose_ordering": self.dct_v38_lambda_dose,
        }

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
        return self.IPCW_RANK_WEIGHT * ipcw_rank_loss + transport_objective
