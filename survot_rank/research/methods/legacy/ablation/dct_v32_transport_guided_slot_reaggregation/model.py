"""Candidate v3.2: transport-guided slot reaggregation (TGSR), NLL only.

User-facing v3.2 is a new candidate identity, NOT a rename of frozen v3.10.
Reuse the DCT encoders, semantic coordinates and final stage-OT decoder, but
do not construct risk anchors or execute counterfactual prediction branches.
"""

from __future__ import annotations

from survot_rank.research.components.transport_guided_slot_reaggregation import (
    TransportGuidedSlotReaggregation,
)
from survot_rank.research.methods.distributional_counterfactual_transport.model import (
    DistributionalCounterfactualTransport,
)
from survot_rank.research.methods.dct_v310_directional_regularized_transport.model import (
    DCTV310DirectionalRegularizedTransport,
)


class DCTV32TransportGuidedSlotReaggregation(DistributionalCounterfactualTransport):
    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        if getattr(args, "bag_loss", "nll_surv") != "nll_surv":
            raise ValueError("v3.2 requires bag_loss='nll_surv'")
        # Same factual backbone settings in A/B/C/D. Disable ALL legacy
        # auxiliary objectives, independently of any old YAML/CLI overrides.
        values = dict(DCTV310DirectionalRegularizedTransport.FROZEN_ARGUMENTS)
        values.update(dct_lambda_ipcw_rank=0.0, dct_v38_lambda_direction=0.0,
                      fet_lambda_sparse=0.0, fet_lambda_faith=0.0,
                      spt_lambda_ot=0.0, spt_lambda_rank=0.0, spt_lambda_stage=0.0)
        for name, value in values.items():
            setattr(args, name, value)
        super().__init__(args, omic_input_dim, omic_names, pathway_names)
        self.reaggregation = TransportGuidedSlotReaggregation(
            self.wsi_projection_dim,
            mode=getattr(args, "dct_v32_feedback", "ot"),
            rounds=getattr(args, "dct_v32_rounds", 1),
            strength=getattr(args, "dct_v32_feedback_strength", 0.25),
            epsilon=getattr(args, "dct_v32_feedback_eps", 0.10),
            sinkhorn_iters=getattr(args, "dct_v32_feedback_iters", 50),
        )

    @classmethod
    def objective_weights(cls):
        return {"nll": 1.0, "ipcw_rank": 0.0, "direction": 0.0}

    def configure_train_reference(self, event_times, censorship):
        # Compatibility hook for the shared trainer. NLL bin fitting remains
        # the dataset's responsibility; TGSR has no IPCW or anchor reference.
        # Keeping unused buffers empty also permits standalone checkpoint load.
        del event_times, censorship

    def get_extra_state(self):
        feedback = self.reaggregation
        return {"version": "3.2-tgsr", "feedback": feedback.mode,
                "rounds": feedback.rounds, "strength": feedback.strength,
                "epsilon": feedback.epsilon, "sinkhorn_iters": feedback.sinkhorn_iters}

    def set_extra_state(self, state):
        # Equal tensor shapes do not make different feedback variants the same
        # experiment. Loading a checkpoint requires the recorded configuration.
        if state != self.get_extra_state():
            raise RuntimeError("v3.2 checkpoint configuration does not match the constructed model")

    def _encode_transport_slots(self, x_wsi_proj, x_omics, kwargs):
        wsi, omic, wa, oa = super()._encode_transport_slots(x_wsi_proj, x_omics, kwargs)
        # These are the original projected token sets, NOT the initial slots.
        # The legacy encoder has no padding-mask contract; whole-model inputs
        # must therefore be actual bags (as in the existing data loader).
        wsi, omic = self.reaggregation(wsi, omic, x_wsi_proj, x_omics)
        return wsi, omic, wa, oa

    def forward(self, **kwargs):
        if kwargs.get("wsi_missing", False) or kwargs.get("omic_missing", False):
            raise ValueError("v3.2 currently requires both WSI and omics modalities")
        x_wsi = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        wsi, omic, _, _ = self._encode_transport_slots(x_wsi, x_omics, kwargs)
        costs, rows, cols, _ = self._cost_tensor(wsi, omic)
        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        plans, distance = self._plans_from_cost_tensor(costs, rows, cols, epoch,
                                                      replay_fixed=False)
        logits, gate = self._encode_logits_from_plans(wsi, omic, plans)
        diagnostics = dict(self.reaggregation.last_diagnostics)
        diagnostics["ot"] = distance.detach()
        self.last_training_losses = diagnostics
        # Do not export low/high-risk fields: v3.2 makes no anchor/DMR claim.
        self.last_explanations = None if self.training else {
            **diagnostics, "factual_risk": self._risk(logits).detach(),
            "event_gate": gate.detach(),
            "factual_coupling_marginal_error": self._marginal_error(plans, rows, cols).detach(),
        }
        return logits, logits.sum() * 0.0
