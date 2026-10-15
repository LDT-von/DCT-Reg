"""Detached observations of actual features/plans, independent of loss values.

Every distribution is reduced within each patient before averaging patients.
Unavailable modalities are excluded, not treated as collapsed zero features.
Near duplicates use absolute raw Euclidean distance <= 1e-3; tanh saturation
uses abs(tanh(slot logits)) >= .99. These are observations, not collapse labels.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def relation_gap_metrics(gap):
    """Absolute patient gaps reveal differences hidden by signed averaging."""
    value = gap.detach()
    return {"probe_nll_gap_abs": value.abs().mean(),
            "probe_worse_fraction": (value > 0).to(value.dtype).mean()}


@torch.no_grad()
def slot_monitoring_metrics(slots, risk_logits, available, prefix, *, duplicate_distance=1e-3):
    """Measure feature separation separately from the response used in reading."""
    valid = available.detach().bool()
    metrics = {prefix + "_monitor_patient_count": float(valid.sum().item())}
    if not bool(valid.any()):
        return metrics
    # Double precision avoids low precision cancellation/overflow in raw
    # distances; no value is used by prediction or backward.
    values = slots.detach()[valid].double()
    logits = risk_logits.detach()[valid].double()
    metrics[prefix + "_slot_norm_mean"] = float(values.norm(dim=-1).mean(1).mean().item())
    response = logits.tanh()
    metrics[prefix + "_risk_response_variance"] = float(response.var(dim=1, unbiased=False).mean(-1).mean().item())
    metrics[prefix + "_tanh_saturation_fraction"] = float((response.abs() >= .99).double().mean((1, 2)).mean().item())
    count = values.size(1)
    metrics[prefix + "_slot_pair_count"] = float(count * (count - 1) // 2)
    if count < 2:
        return metrics
    indices = torch.triu_indices(count, count, 1, device=values.device)
    first, second = values[:, indices[0]], values[:, indices[1]]
    distance = (first - second).norm(dim=-1)
    metrics[prefix + "_slot_pair_dist_mean"] = float(distance.mean(1).mean().item())
    # Mean of per-patient minima, not a minimum over the entire batch.
    metrics[prefix + "_slot_pair_dist_min"] = float(distance.min(dim=1).values.mean().item())
    metrics[prefix + "_slot_near_duplicate_fraction"] = float((distance <= duplicate_distance).double().mean(1).mean().item())
    metrics[prefix + "_slot_cosine_mean"] = float(F.cosine_similarity(first, second, dim=-1).mean(1).mean().item())
    return metrics


@torch.no_grad()
def plan_monitoring_metrics(plans, pair_available):
    """Observe exchange amount and departure from same-margin independence.

Plans have shape [patient, stage, geometry, WSI slot, omics slot]. At zero
mass the normalized L1 difference is explicitly zero. A tiny nonzero plan
is assessed on its normalized probabilities, avoiding mass underflow in Q.
"""
    valid = pair_available.detach().bool()
    metrics = {"transport_monitor_patient_count": float(valid.sum().item())}
    if not bool(valid.any()):
        return metrics
    actual = plans.detach()[valid].double()
    mass = actual.sum((-2, -1))
    nonzero = mass > 0
    denominator = torch.where(nonzero, mass, torch.ones_like(mass))
    probability = actual / denominator[..., None, None]
    independent = probability.sum(-1)[..., :, None] * probability.sum(-2)[..., None, :]
    difference = (probability - independent).abs().sum((-2, -1))
    difference = torch.where(nonzero, difference, torch.zeros_like(difference))
    metrics["transport_mass_mean"] = float(mass.mean((1, 2)).mean().item())
    metrics["plan_independence_l1_normalized"] = float(difference.mean((1, 2)).mean().item())
    metrics["transport_zero_mass_fraction"] = float((~nonzero).double().mean((1, 2)).mean().item())
    return metrics


@torch.no_grad()
def state_monitoring_metrics(state, wsi_logits, omics_logits, local_readout, residual_gain):
    metrics = slot_monitoring_metrics(state.wsi_slots, wsi_logits, state.wsi_mask.any(-1), "wsi")
    metrics.update(slot_monitoring_metrics(state.omics_slots, omics_logits, state.omics_mask.any(-1), "omics"))
    metrics.update(plan_monitoring_metrics(state.plans, state.pair_available))
    valid = state.pair_available.detach().bool()
    if local_readout is not None and bool(valid.any()):
        local = local_readout.detach()[valid].double()
        metrics["local_readout_norm"] = float(local.norm(dim=-1).mean(1).mean().item())
    metrics["residual_gain"] = float(residual_gain.detach().double().tanh().item())
    return metrics
