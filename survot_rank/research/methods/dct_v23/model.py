"""DCT v2.3: masked gene recovery and held-patient benefit calibration.

Survival inference is the inherited full-input DCT path. Reconstruction is
an additional training pass; the full-input state is never reused as a donor,
receiver, cost, marginal or decoder query in the masked recovery pass.
"""
from copy import deepcopy

import torch
from torch.nn import functional as F

from survot_rank.research.methods.dct_v2.model import deterministic_readout
from survot_rank.research.methods.dct_v2.transport import pair_tokens, realized_independent_plan
from survot_rank.research.methods.dct_v22.model import DCTV22
from survot_rank.research.methods.dct_v23.config import DCTV23Config
from survot_rank.research.methods.dct_v23.reconstruction import (
    GlobalGeneSchema, RawGeneReconstructionDecoder, sample_global_mask,
    pathway_receiver_assignment, conditional_donor_context, masked_pathway_mse,
    fixed_mask_mean, validate_gain_provenance,
)


class DCTV23(DCTV22):
    """Keep native survival predictions; train recovery on genuinely hidden genes.

    Calibrated mode consumes externally cross-fitted patient-by-pathway gains.
    A pathway gain is not a true slot-pair correspondence label. It supervises
    receiver usefulness through existing bounded stage costs, while the actual
    matching still learns from reconstruction and survival objectives.
    """
    dct_v23_reconstruction_enabled = True

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        config = args if isinstance(args, DCTV23Config) else DCTV23Config.from_args(args)
        super().__init__(config, omic_input_dim, omic_names, pathway_names)
        self.args = args
        self.register_buffer("reconstruction_gene_mean", torch.empty(0))
        self.register_buffer("reconstruction_gene_std", torch.empty(0))
        self.gene_schema = None
        self.reconstruction_train_patient_ids = ()
        self._gain_targets = {}
        self._gain_provenance = None
        self.last_reconstruction = {}
        self.last_reconstruction_state = None
        self.reconstruction_decoder = None
        if config.reconstruction_mode != "off":
            # New training-only initialization does not perturb the parent's
            # construction RNG or the six-loss off-control initialization.
            with torch.random.fork_rng(devices=[]):
                self.reconstruction_decoder = RawGeneReconstructionDecoder(
                    config.dim, config.recon_hidden_dim, config.omic_sizes)
        names = getattr(args, "dct_v23_pathway_gene_ids", None)
        if names is not None:
            self.configure_reconstruction_reference(
                names, getattr(args, "dct_v23_raw_gene_mean", None),
                getattr(args, "dct_v23_raw_gene_std", None),
                getattr(args, "dct_v23_train_patient_ids", None),
                gene_ids=getattr(args, "dct_v23_gene_ids", None))
        elif omic_names is not None:
            # Metadata alone is not a train-fold normalization reference.
            self.gene_schema = GlobalGeneSchema.build(omic_names, config.omic_sizes)

    def configure_reconstruction_reference(self, pathway_gene_ids, raw_gene_mean,
                                           raw_gene_std, train_patient_ids, *, gene_ids=None):
        schema = GlobalGeneSchema.build(pathway_gene_ids, self.config.omic_sizes, gene_ids)
        if raw_gene_mean is None or raw_gene_std is None:
            raise ValueError("reconstruction needs fixed training-fold gene mean/std")
        device = self.event_hazard.weight.device
        mean = torch.as_tensor(raw_gene_mean, dtype=torch.float32, device=device).detach().flatten()
        std = torch.as_tensor(raw_gene_std, dtype=torch.float32, device=device).detach().flatten()
        if mean.shape != (len(schema.gene_ids),) or std.shape != mean.shape:
            raise ValueError("gene normalization dimensions must match actual unique IDs")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or bool((std < 0).any()):
            raise ValueError("gene normalization statistics must be finite with nonnegative std")
        ids = tuple(train_patient_ids or ())
        if not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("actual unique training patient IDs are required")
        self.gene_schema = schema
        self.reconstruction_gene_mean = mean.clone()
        # Constant training genes retain fixed coordinates rather than causing
        # division by zero; targets themselves are never learned.
        self.reconstruction_gene_std = torch.where(std > 0, std.clamp_min(1e-6), torch.ones_like(std)).clone()
        self.reconstruction_train_patient_ids = ids
        self._gain_targets, self._gain_provenance = {}, None
        return self

    def bind_gain_targets(self, gain_by_patient, provenance):
        if self.gene_schema is None or not self.reconstruction_train_patient_ids:
            raise ValueError("configure the training reconstruction reference before gains")
        validate_gain_provenance(provenance, self.gene_schema,
                                 self.reconstruction_train_patient_ids, self.config.wsi_input_dim)
        if not isinstance(gain_by_patient, dict) or set(gain_by_patient) != set(self.reconstruction_train_patient_ids):
            raise ValueError("one OOF gain vector is required for every and only training patient")
        validated = {}
        for patient, values in gain_by_patient.items():
            vector = torch.as_tensor(values, dtype=torch.float32).detach().flatten()
            if vector.shape != (self.num_pathways,) or not bool(torch.isfinite(vector).all()):
                raise ValueError("OOF gains must be finite signed [pathways] vectors")
            validated[patient] = vector.tolist()
        self._gain_targets = validated
        self._gain_provenance = deepcopy(provenance)
        return self

    def get_extra_state(self):
        state = super().get_extra_state()
        state.update(schema="dct_v23_standalone_v1", config=self.config.to_dict(),
                     reconstruction=self.config.reconstruction_mode != "off",
                     reconstruction_mode=self.config.reconstruction_mode,
                     losses=list(self.objective_weights_effective(100000).keys()),
                     loss_terms=list(self.objective_weights_effective(100000).keys()),
                     pathway_gene_ids=None if self.gene_schema is None else self.gene_schema.pathway_gene_ids,
                     train_patient_ids=self.reconstruction_train_patient_ids,
                     oof_gain_targets=deepcopy(self._gain_targets),
                     oof_gain_provenance=deepcopy(self._gain_provenance))
        return state

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("schema") != "dct_v23_standalone_v1":
            raise ValueError("DCT v2.3 requires its own checkpoint schema")
        if DCTV23Config.from_dict(state["config"]).to_dict() != self.config.to_dict():
            raise ValueError("construct DCT v2.3 with the exact saved recipe")
        names = state.get("pathway_gene_ids")
        self.gene_schema = None if names is None else GlobalGeneSchema.build(names, self.config.omic_sizes)
        self.reconstruction_train_patient_ids = tuple(state.get("train_patient_ids", ()))
        self._gain_targets, self._gain_provenance = {}, None
        if state.get("oof_gain_targets"):
            self.bind_gain_targets(state["oof_gain_targets"], state.get("oof_gain_provenance"))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ("reconstruction_gene_mean", "reconstruction_gene_std"):
            value = state_dict.get(prefix + name)
            if value is not None:
                if value.ndim != 1 or not bool(torch.isfinite(value).all()):
                    raise ValueError("invalid reconstruction reference in checkpoint")
                expected = 0 if self.gene_schema is None else len(self.gene_schema.gene_ids)
                if value.numel() not in (0, expected) or name.endswith("std") and bool((value <= 0).any()):
                    raise ValueError("invalid reconstruction reference dimensions or std")
                setattr(self, name, getattr(self, name).new_empty(value.shape))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def objective_weights_effective(self, epoch=0):
        weights = super().objective_weights_effective(epoch)
        cfg = self.config
        if cfg.reconstruction_mode != "off" and cfg.reconstruction_weight > 0:
            weights["reconstruction"] = cfg.reconstruction_weight
        if cfg.reconstruction_mode == "gain_calibrated" and cfg.calibration_weight > 0:
            weights["gain_calibration"] = cfg.calibration_weight if epoch >= cfg.calibration_warmup_epochs else 0.
        return weights

    def _masked_reconstruction(self, kwargs, *, global_mask=None, independent=False):
        if self.reconstruction_decoder is None:
            raise ValueError("reconstruction is disabled for this recipe")
        schema = self.gene_schema
        if schema is None or self.reconstruction_gene_mean.numel() == 0:
            raise ValueError("configure fixed train-fold gene identities and statistics before reconstruction")
        x = kwargs.get("x_wsi")
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("reconstruction requires x_wsi [B,N,D]")
        values = tuple(kwargs.get("x_omic" + str(index + 1)) for index in range(self.num_pathways))
        if any(not isinstance(value, torch.Tensor) or value.shape != (x.size(0), width)
               or value.device != x.device or not value.is_floating_point()
               for value, width in zip(values, self.config.omic_sizes)):
            raise ValueError("reconstruction requires actual pathway gene values")
        availability = kwargs.get("omics_mask")
        if availability is None:
            availability = torch.ones(x.size(0), self.num_pathways, dtype=torch.bool, device=x.device)
        if kwargs.get("omic_missing", False):
            availability = torch.zeros_like(availability)
        schema.validate_duplicate_values(values, availability)
        if global_mask is None:
            global_mask = sample_global_mask(x.size(0), len(schema.gene_ids), self.config.mask_ratio, x.device)
        if not isinstance(global_mask, torch.Tensor) or global_mask.device != x.device or global_mask.shape != (x.size(0), len(schema.gene_ids)):
            raise ValueError("explicit global mask shape/device does not match actual genes")
        masks = schema.expand_mask(global_mask)
        masked_kwargs = dict(kwargs)
        targets = []
        for index, (value, ids, mask) in enumerate(zip(values, schema.indices, masks)):
            ids = torch.tensor(ids, device=value.device)
            # Hidden raw values are absent before any learned computation.
            masked_kwargs["x_omic" + str(index + 1)] = value.masked_fill(mask, 0.)
            target = (value.detach().float() - self.reconstruction_gene_mean[ids]) / self.reconstruction_gene_std[ids]
            safe_target = target.masked_fill(~availability[:, index:index + 1], 0.)
            if not bool(torch.isfinite(safe_target).all()):
                raise ValueError("available raw reconstruction targets must be finite")
            targets.append(safe_target)
        wsi, omics, wm, om = self._encode(masked_kwargs)
        epoch = kwargs.get("epoch", kwargs.get("cur_epoch", getattr(self.args, "cur_epoch", 0)))
        state = self.core(wsi, omics, wsi_mask=wm, omics_mask=om, epoch=epoch)
        assignment = pathway_receiver_assignment(omics, state.omics_slots, self.config.coordinate_temperature)
        native_context = torch.bmm(assignment.to(state.omics_slots.dtype), state.omics_slots)
        donor_context = conditional_donor_context(state, state.plans, assignment).to(omics.dtype)
        native = self.reconstruction_decoder(omics, native_context, torch.zeros_like(donor_context))
        conditional = self.reconstruction_decoder(omics, native_context, donor_context)
        native_mse, counts = masked_pathway_mse(native, targets, masks, om)
        conditional_mse, _ = masked_pathway_mse(conditional, targets, masks, om)
        result = {"state": state, "assignment": assignment, "global_mask": global_mask,
                  "pathway_masks": masks, "targets": tuple(targets), "counts": counts,
                  "native_prediction": native, "conditional_prediction": conditional,
                  "pathway_native_mse": native_mse, "pathway_conditional_mse": conditional_mse,
                  "reconstruction_loss": .5 * (fixed_mask_mean(native_mse, counts) + fixed_mask_mean(conditional_mse, counts)),
                  "native_context": native_context, "donor_context": donor_context}
        if independent:
            plan = realized_independent_plan(state.plans)
            donor = conditional_donor_context(state, plan, assignment).to(omics.dtype)
            predictions = self.reconstruction_decoder(omics, native_context, donor)
            error, _ = masked_pathway_mse(predictions, targets, masks, om)
            result.update(independent_prediction=predictions, pathway_independent_mse=error,
                          independent_plan=plan)
        return result

    @torch.no_grad()
    def reconstruction_audit(self, *, global_mask=None, **kwargs):
        """Explicit held-target audit; never called by ordinary survival inference."""
        solver = deepcopy(self.core.last_solver_diagnostics)
        try:
            with deterministic_readout(self):
                result = self._masked_reconstruction(kwargs, global_mask=global_mask, independent=True)
            return result
        finally:
            self.core.last_solver_diagnostics = solver

    def gain_calibration_loss(self, reconstruction, patient_ids):
        cfg = self.config
        if not self._gain_targets:
            raise ValueError("gain_calibrated training requires audited held-patient gain labels")
        state = reconstruction["state"]
        if not isinstance(patient_ids, (list, tuple)) or len(patient_ids) != state.wsi_slots.size(0):
            raise ValueError("calibration requires actual patient_ids [B]")
        if any(patient not in self._gain_targets for patient in patient_ids):
            raise ValueError("calibration may look up only the fitted training patient identities")
        gains = state.wsi_slots.new_tensor([self._gain_targets[patient] for patient in patient_ids], dtype=torch.float32)
        with torch.autocast(device_type=state.wsi_slots.device.type, enabled=False):
            assignment = reconstruction["assignment"].detach().float()
            assignment = assignment * state.omics_mask.detach().float()[..., None]
            receiver_gain = torch.einsum("bpo,bp->bo", assignment, gains) / assignment.sum(1).clamp_min(1e-8)
            goal = torch.sigmoid(-receiver_gain / cfg.gain_scale).detach()
            contents = pair_tokens(state.wsi_slots.detach().float(), state.omics_slots.detach().float())
            predicted = torch.sigmoid(self.core.stage_pair_cost(contents)).permute(0, 3, 1, 2)
            target = goal[:, None, None, :].expand_as(predicted)
            per_patient = (predicted - target).square().mean((1, 2, 3))
            available = state.pair_available.detach().float()
            loss = (per_patient * available).sum() / available.sum().clamp_min(1)
        self.last_reconstruction.update(calibration_signed_pathway_gain=gains.detach(),
                                        calibration_receiver_gain=receiver_gain.detach(),
                                        calibration_cost_goal=goal,
                                        calibration_predicted_cost=predicted.detach())
        return loss

    def forward(self, **kwargs):
        logits, auxiliary = super().forward(**kwargs)
        self.last_reconstruction = {}
        self.last_reconstruction_state = None
        cfg = self.config
        if not self.training or not kwargs.get("compute_aux", True) or cfg.reconstruction_mode == "off":
            self.last_diagnostics["schema"] = "dct_v23_standalone_diagnostics_v1"
            return logits, auxiliary
        solver = deepcopy(self.core.last_solver_diagnostics)
        try:
            reconstruction = self._masked_reconstruction(kwargs, global_mask=kwargs.get("reconstruction_global_mask"))
        finally:
            self.core.last_solver_diagnostics = solver
        self.last_reconstruction_state = reconstruction["state"]
        self.last_reconstruction = reconstruction
        epoch = kwargs.get("epoch", kwargs.get("cur_epoch", getattr(self.args, "cur_epoch", 0)))
        weights = self.objective_weights_effective(epoch)
        raw = {"reconstruction": reconstruction["reconstruction_loss"]}
        if cfg.reconstruction_mode == "gain_calibrated":
            weight = weights.get("gain_calibration", 0.)
            raw["gain_calibration"] = self.gain_calibration_loss(reconstruction, kwargs.get("patient_ids")) if weight > 0 else logits.sum() * 0.
        auxiliary = auxiliary + sum(weights.get(name, 0.) * value for name, value in raw.items())
        for name, value in raw.items():
            weight = weights.get(name, 0.)
            self.last_training_losses.update({name: value.detach(), name + "_weighted": (weight * value).detach(),
                                             name + "_effective_weight": logits.new_tensor(weight)})
        counts = reconstruction["counts"]
        valid = counts > 0
        monitoring = {
            "reconstruction_native_mse": float(fixed_mask_mean(reconstruction["pathway_native_mse"].detach(), counts)),
            "reconstruction_conditional_mse": float(fixed_mask_mean(reconstruction["pathway_conditional_mse"].detach(), counts)),
            "reconstruction_signed_gain": float(fixed_mask_mean((reconstruction["pathway_native_mse"] - reconstruction["pathway_conditional_mse"]).detach(), counts)),
            "reconstruction_masked_gene_count": float(counts.sum()),
            "reconstruction_valid_pathway_count": float(valid.sum()),
            "auxiliary_total": float(auxiliary.detach()),
        }
        self.last_monitoring_metrics.update(monitoring)
        for name in monitoring:
            self.last_monitoring_metric_weights[name] = float(logits.size(0))
        self.last_diagnostics.update(schema="dct_v23_standalone_diagnostics_v1", monitoring=dict(self.last_monitoring_metrics))
        return logits, auxiliary
