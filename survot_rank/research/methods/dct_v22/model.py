"""DCT v2.2 candidate: smaller existing MLP and observation-only monitoring."""
from dataclasses import fields

import torch
from torch import nn
from torch.nn import functional as F

from survot_rank.research.methods.dct_v2.losses import logit_nll
from survot_rank.research.methods.dct_v2.model import deterministic_readout
from survot_rank.research.methods.dct_v2.transport import fixed_realized_marginal_plan, realized_independent_plan
from survot_rank.research.methods.dct_v21.model import DCTV21
from survot_rank.research.methods.dct_v22.config import DCTV22Config
from survot_rank.research.methods.dct_v22.diagnostics import relation_gap_metrics, state_monitoring_metrics


class DCTV22(DCTV21):
    """Keep v2.1 transport/readout and exactly the same six objectives.

The wide, zero-WSI-dropout control keeps the parent MLP and RNG sequence
unchanged. Other arms resize this existing projection and apply dropout;
no additional encoder or prediction head is introduced.
    """
    dct_v22_monitoring_enabled = True

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        config = args if isinstance(args, DCTV22Config) else DCTV22Config.from_args(args)
        super().__init__(config, omic_input_dim, omic_names, pathway_names)
        self.args = args
        if config.wsi_hidden_dim != config.wsi_input_dim or config.wsi_dropout != 0:
            layers = [nn.Linear(config.wsi_input_dim, config.wsi_hidden_dim), nn.ReLU(inplace=False)]
            if config.wsi_dropout > 0:
                layers.append(nn.Dropout(config.wsi_dropout, inplace=False))
            layers.append(nn.Linear(config.wsi_hidden_dim, config.dim))
            self.wsi_mlp = nn.Sequential(*layers)
        self.last_monitoring_metrics = {}
        self.last_monitoring_metric_weights = {}
        self._monitoring_training_forward_count = 0

    def get_extra_state(self):
        value = super().get_extra_state()
        value.update(schema="dct_v22_standalone_v1", config=self.config.to_dict())
        return value

    def set_extra_state(self, value):
        if not isinstance(value, dict) or value.get("schema") != "dct_v22_standalone_v1":
            raise ValueError("DCT v2.2 requires its own checkpoint schema")
        if DCTV22Config.from_dict(value["config"]).to_dict() != self.config.to_dict():
            raise ValueError("construct DCT v2.2 with the exact saved recipe")

    def _cost_monitoring_parameters(self):
        """Unique active cost parameters; exclude representation/readout modules."""
        candidates = [*self.core.stage_pair_cost.parameters(), *self.core.evidence_gate.parameters()]
        # This version registers stage_embedding as a non-gradient buffer.
        # Include it only if a supported custom core makes it trainable.
        candidates.append(self.core.stage_embedding)
        seen, result = set(), []
        for value in candidates:
            if value.requires_grad and id(value) not in seen:
                result.append(value)
                seen.add(id(value))
        return tuple(result)

    def relation_loss(self, state, y, c, epoch=0):
        """The v2.1 contrast algorithm, with observation-only gap/gradient stats.

        Optional gradient norms cover stage_pair_cost, evidence_gate, and a
        trainable stage_embedding if present, for the *raw* relation loss
        before lambda/schedule weighting or AMP scaling. No event encoder or
        prediction head is included. Sampling counts training auxiliary
        forwards and neither accumulates .grad nor adds a loss.
        """
        cfg = self.config
        if cfg.relation_objective == "legacy_hinge":
            return super().relation_loss(state, y, c, epoch)
        if self.core.plan_mode == "independent":
            self.last_relation_diagnostics = {}
            return state.plans.sum()*0.
        dtype = self.event_hazard.weight.dtype
        values = {field.name: (getattr(state, field.name).detach().to(dtype)
                              if getattr(state, field.name).is_floating_point()
                              else getattr(state, field.name).detach()) for field in fields(state)}
        detached = type(state)(**values)
        with deterministic_readout(self), torch.autocast(device_type=state.plans.device.type, enabled=False):
            actual = detached.plans
            rows, cols = actual.sum(-1), actual.sum(-2)
            freedom = ((rows > cfg.min_plan_mass).sum(-1) >= 2)&((cols > cfg.min_plan_mass).sum(-1) >= 2)
            mass_weight = (self._stage_weights(detached)*(actual.sum((-2, -1))*freedom).mean(2)).sum(-1).clamp(0, 1).detach()
            if not bool((mass_weight > 0).any()):
                self.last_relation_diagnostics = {"probe_gain":actual.new_tensor(cfg.relation_probe_gain),
                                                  "effective_mass":mass_weight.mean()}
                return state.plans.sum()*0.
            costs, _, _ = self.core.costs_and_marginals(detached.wsi_slots, detached.omics_slots)
            projected = fixed_realized_marginal_plan(
                costs, rows, cols, epsilon=self.core.epsilon(epoch), iterations=cfg.projection_iters,
                tolerance=cfg.projection_tolerance, min_mass=cfg.min_plan_mass)
            quality = tuple(value.detach() for value in self._quality(detached))
            learned, _, _ = self._predict(detached, projected, frozen=True, quality=quality,
                                          gain_override=cfg.relation_probe_gain)
            with torch.no_grad():
                independent, _, _ = self._predict(detached, realized_independent_plan(actual), frozen=True,
                                                  quality=quality, gain_override=cfg.relation_probe_gain)
                reference = logit_nll(independent, y, c, cfg.nll_alpha, "none")
            factual = logit_nll(learned, y, c, cfg.nll_alpha, "none")
            gap = factual-reference.detach()
            loss = (mass_weight*cfg.relation_temperature*F.softplus(gap/cfg.relation_temperature)).mean()
        self.last_relation_diagnostics = {
            "probe_factual_nll":factual.detach().mean(), "probe_independent_nll":reference.mean(),
            "probe_nll_gap":gap.detach().mean(), "probe_gain":gap.new_tensor(cfg.relation_probe_gain),
            "effective_mass":mass_weight.mean(),
            "row_error":(projected.sum(-1)-rows).detach().abs().max(),
            "col_error":(projected.sum(-2)-cols).detach().abs().max(),
            **relation_gap_metrics(gap),
        }
        interval = cfg.relation_gradient_interval
        count = self._monitoring_training_forward_count
        if (interval > 0 and count > 0 and count % interval == 0 and self.training
                and torch.is_grad_enabled() and loss.requires_grad and cfg.relation_effective_weight(epoch) > 0):
            parameters = self._cost_monitoring_parameters()
            if parameters:
                gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                squared = sum(gradient.detach().double().square().sum() for gradient in gradients if gradient is not None)
                norm = actual.new_zeros(()) if isinstance(squared, int) else squared.sqrt().to(actual.dtype)
                self.last_relation_diagnostics.update(cost_gradient_norm_raw=norm,
                                                       cost_gradient_sample_count=actual.new_ones(()))
        return loss

    def forward(self, **kwargs):
        self.last_monitoring_metrics = {}
        self.last_monitoring_metric_weights = {}
        if self.training and kwargs.get("compute_aux", True):
            self._monitoring_training_forward_count += 1
        logits, auxiliary = super().forward(**kwargs)
        with torch.no_grad():
            state = self.last_state
            self.last_monitoring_metrics = state_monitoring_metrics(
                state, self.per_slot_hazard_wsi(state.wsi_slots),
                self.per_slot_hazard_omic(state.omics_slots), self.last_local_readout,
                self.residual_gain)
            if self.training and kwargs.get("compute_aux", True):
                self.last_monitoring_metrics.update(
                    global_nll=float(logit_nll(logits.detach(), kwargs["y"], kwargs["c"], self.config.nll_alpha).item()),
                    auxiliary_total=float(auxiliary.detach().item()))
                sampled = self.last_relation_diagnostics.get("cost_gradient_sample_count")
                self.last_monitoring_metrics["relation_cost_gradient_sample_count"] = 0. if sampled is None else float(sampled.item())
                norm = self.last_relation_diagnostics.get("cost_gradient_norm_raw")
                if norm is not None:
                    self.last_monitoring_metrics["relation_cost_gradient_norm_raw"] = float(norm.item())
            counts = {
                "wsi": float(state.wsi_mask.any(-1).sum().item()),
                "omics": float(state.omics_mask.any(-1).sum().item()),
                "transport": float(state.pair_available.sum().item()),
            }
            for name in self.last_monitoring_metrics:
                if name.startswith("wsi_"):
                    count = counts["wsi"]
                elif name.startswith("omics_"):
                    count = counts["omics"]
                elif name.startswith(("transport_", "plan_", "local_")):
                    count = counts["transport"]
                elif name.startswith("relation_cost_gradient_"):
                    count = 1.
                else:
                    count = float(logits.size(0))
                self.last_monitoring_metric_weights[name] = count
        self.last_diagnostics.update(schema="dct_v22_standalone_diagnostics_v1",
                                     monitoring=dict(self.last_monitoring_metrics))
        return logits, auxiliary
