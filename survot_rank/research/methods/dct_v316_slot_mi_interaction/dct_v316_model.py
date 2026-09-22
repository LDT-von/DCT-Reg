"""DCT v3.16 patient-specific slot interaction candidate.

The R/Up/Ug/S names are hypotheses attached to four learned interaction
channels. The implementation does not claim a formal information-theoretic
decomposition; those semantics require matched ablations or interventions.
"""

from __future__ import annotations

import torch

from survot_rank.research.methods.dct_v310_directional_regularized_transport.model import (
    DCTV310DirectionalRegularizedTransport,
)
from survot_rank.research.methods.dct_v316_slot_mi_interaction.model import (
    SlotBasedMIDecompositionBlock,
)


class DCTV316SlotMIDecomposition(DCTV310DirectionalRegularizedTransport):
    """DCT v3.16 patient-specific slot interaction candidate."""

    NLL_WEIGHT = 1.0
    IPCW_RANK_WEIGHT = 0.10
    LAMBDA_MI = 0.01

    FROZEN_ARGUMENTS = {
        "dct_lambda_ipcw_rank": IPCW_RANK_WEIGHT,
        "dct_ipcw_rank_margin": 0.02,
        "dct_ipcw_rank_temperature": 0.50,
        "dct_ipcw_max_weight": 10.0,
        "dct_ipcw_rank_memory_size": 64,
        "dct_lambda_etar": 0.0,
        "dct_lambda_listwise": 0.0,
        "dct_v38_lambda_direction": 0.0,
        "dct_v38_lambda_dose": 0.0,
        "dct_v38_lambda_reconfiguration": 0.0,
        "dct_v38_direction_margin": 0.02,
        "dct_v38_temperature": 0.05,
        "dct_v38_alpha_mid": 0.50,
        "dct_v38_alpha_full": 1.00,
        "dct_v38_warmup_epochs": 0,
        "dct_v38_ramp_epochs": 0,
        "dct_anchor_momentum": 0.90,
        "dct_evidence_cost_weight": 0.0,
        "dct_evidence_mass_floor": 0.05,
        "dct_evidence_marginal_strength": 1.0,
        "dct_geometry_reliability_strength": 0.0,
        "dct_mix_ratio": 0.0,
        "dct_slot_init_mode": "deterministic",
        "dct_fixed_coupling": False,
        "dct_random_anchors": False,
        "dct_perm_labels_seed": 0,
        "dct_stage_jitter_fraction": 0.0,
        "dct_freeze_source_prototype": "",
        "dct_v382_lambda_mgptr": 0.0,
        "dct_v382_adaptive_aux_weights": False,
    }

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        bag_loss = str(getattr(args, "bag_loss", "nll_surv"))
        if bag_loss != "nll_surv":
            raise ValueError(
                "DCT v3.16 requires bag_loss='nll_surv'; "
                f"received {bag_loss!r}"
            )

        for name, value in self.FROZEN_ARGUMENTS.items():
            setattr(args, name, value)
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        self.dct_lambda_ipcw_rank = self.IPCW_RANK_WEIGHT
        self.dct_v38_lambda_direction = 0.0
        self.dct_mix_ratio = 0.0

        # v3.16 parameters
        self.v316_lambda_mi = float(getattr(args, "v316_lambda_mi", self.LAMBDA_MI))
        if self.v316_lambda_mi < 0:
            raise ValueError("v316_lambda_mi must be non-negative")

        dim = self.wsi_projection_dim
        num_wsi_slots = int(getattr(args, "slot_num_wsi", 8))
        num_omic_slots = int(getattr(args, "slot_num_omics", 8))
        slot_iters = int(getattr(args, "slot_iters", 3))
        heads = int(getattr(args, "otehv2_heads", 4))
        hidden_dim = int(getattr(args, "v316_hidden_dim", 256))
        profile_temp = float(getattr(args, "v316_profile_temp", 1.0))

        self.slot_mi_block = SlotBasedMIDecompositionBlock(
            dim=dim,
            num_wsi_slots=num_wsi_slots,
            num_omic_slots=num_omic_slots,
            slot_iters=slot_iters,
            cross_iters=3,
            num_heads=heads,
            hidden_dim=hidden_dim,
            num_classes=int(args.n_classes),
            profile_temp=profile_temp,
        )

        # v3.16 reuses the parent only for input encoders and censoring-aware
        # ranking utilities. Freeze the inherited OT/event modules that this
        # forward path never calls so optimizers and DDP do not treat them as
        # trainable parameters with permanently missing gradients.
        active_prefixes = ("sig_networks.", "wsi_mlp.", "slot_mi_block.")
        for name, parameter in self.named_parameters():
            if not name.startswith(active_prefixes):
                parameter.requires_grad_(False)

        self.last_output = None
        self.last_mi_metrics = {}

    @staticmethod
    def _risk(logits):
        hazards = torch.sigmoid(logits)
        return -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)

    def _encode_wsi_omics(self, x_wsi_proj, x_omics):
        """用Slot MI Block编码双模态。"""
        output, mi_metrics = self.slot_mi_block(
            wsi_tokens=x_wsi_proj,
            omic_tokens=x_omics,
        )

        self.last_output = output
        self.last_mi_metrics = mi_metrics

        return output

    def _predict_events(self, output):
        """从Slot MI output预测事件。"""
        logits = output["logits"]
        return logits, output.get("profile")

    def _combine_auxiliary_objectives(
        self,
        *,
        ipcw_rank_loss,
        etar_loss,
        transport_objective,
        transport_metrics,
        epoch,
    ):
        del etar_loss, transport_objective, transport_metrics, epoch

        total = self.IPCW_RANK_WEIGHT * ipcw_rank_loss

        if self.training and self.last_mi_metrics:
            L_total = self.last_mi_metrics.get("L_total", torch.tensor(0.0, device=ipcw_rank_loss.device))
            total = total + self.v316_lambda_mi * L_total

        return total

    def forward(self, **kwargs):
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)

        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        if self.training:
            self._reset_ipcw_memory_for_epoch(epoch)

        output = self._encode_wsi_omics(x_wsi_proj, x_omics)
        logits, profile = self._predict_events(output)

        if kwargs.get("event_time") is not None and kwargs.get("c") is not None:
            if self.dct_lambda_ipcw_rank != 0.0:
                ipcw_rank_loss = self._ipcw_pairwise_ranking_loss(
                    logits, kwargs["event_time"], kwargs["c"]
                )
                ipcw_pair_count = self.last_ipcw_pair_count.to(
                    device=logits.device, dtype=logits.dtype
                )
                self._remember_ipcw_batch(
                    self._risk(logits),
                    kwargs["event_time"].float().view(-1),
                    kwargs["c"].float().view(-1),
                )
            else:
                ipcw_rank_loss = logits.new_zeros(())
                ipcw_pair_count = logits.new_zeros(())

            self.last_training_losses = {
                "ipcw_rank": ipcw_rank_loss.detach(),
                "ipcw_pairs": ipcw_pair_count.detach(),
            }

            if self.last_mi_metrics:
                for k, v in self.last_mi_metrics.items():
                    if torch.is_tensor(v):
                        self.last_training_losses[k] = v.detach()

            aux_loss = self._combine_auxiliary_objectives(
                ipcw_rank_loss=ipcw_rank_loss,
                etar_loss=logits.new_zeros(()),
                transport_objective=logits.new_zeros(()),
                transport_metrics={},
                epoch=epoch,
            )

            return logits, aux_loss

        self.last_explanations = {
            "profile": output["profile"].detach(),
            "slots_wsi": output["slots_wsi"].detach(),
            "slots_omic": output["slots_omic"].detach(),
            "slot_gate_wsi": output["slot_gate_wsi"].detach(),
            "slot_gate_omic": output["slot_gate_omic"].detach(),
            "factual_risk": self._risk(logits).detach(),
        }

        return logits, logits.new_zeros(())

    def objective_weights(self) -> dict[str, float]:
        return {
            "nll": self.NLL_WEIGHT,
            "ipcw_rank": self.IPCW_RANK_WEIGHT,
            "channel_contrastive": self.v316_lambda_mi,
        }

    @classmethod
    def key_contributions(cls) -> list[str]:
        return [
            "Patient-specific interaction routing over four learned channels",
            "Sparse Top-K WSI and Omics slot decoders on the NLL path",
            "Bidirectional slot interaction before survival prediction",
            "Censor-aware IPCW ranking plus diagonal channel-prediction contrast",
        ]

    @classmethod
    def research_question(cls) -> str:
        return (
            "患者特异的多模态交互路由能否提高生存排序，且其四通道解释能否通过消融验证？"
        )
