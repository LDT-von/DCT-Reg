"""Explicit, batch-independent diversity objective for repaired v3.13.

The historical v3.11 objective is intentionally unchanged. This helper penalizes
each off-diagonal slot pair before averaging, so a distant outlier cannot conceal
duplicate pairs. It does not establish biological slot identities or guarantee
escape from exact collapse: squared distances have zero gradient when every slot
is identical. No noise or anchors are injected to hide that limitation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _nonnegative_finite(name: str, value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonnegative number") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return parsed


def _validate_modality(name: str, slots: torch.Tensor, hazards: torch.Tensor) -> None:
    if slots.ndim != 3 or hazards.ndim != 3:
        raise ValueError(f"{name}: expected slots [B, K, D] and hazards [B, K, C]")
    if slots.shape[:2] != hazards.shape[:2]:
        raise ValueError(f"{name}: slot and hazard batch/slot dimensions must match")
    if slots.size(1) < 1 or slots.size(2) < 1 or hazards.size(2) < 1:
        raise ValueError(f"{name}: slot count and feature dimensions must be positive")
    if slots.device != hazards.device:
        raise ValueError(f"{name}: slots and hazards must use the same device")
    if not slots.is_floating_point() or not hazards.is_floating_point():
        raise ValueError(f"{name}: slots and hazards must be floating-point tensors")
    if not bool(torch.isfinite(slots).all()) or not bool(torch.isfinite(hazards).all()):
        raise ValueError(f"{name}: slots and hazards must be finite")


def _modality_terms(
    slots: torch.Tensor,
    hazards: torch.Tensor,
    *,
    variance_min: float,
    variance_max: float,
    target_sq: float,
    pair_weight: float,
    duplicate_sq: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    batch_size, slot_count, _ = slots.shape
    graph_zero = slots.sum() * 0.0 + hazards.sum() * 0.0
    if batch_size == 0:
        return graph_zero, {
            "patient_count": 0,
            "slot_count": slot_count,
            "pair_count_per_patient": slot_count * (slot_count - 1),
            "hazard_variance": 0.0,
            "variance_term": 0.0,
            "pair_term": 0.0,
            "weighted_pair_term": 0.0,
            "total_term": 0.0,
            "mean_pair_distance_sq": 0.0,
            "min_pair_distance_sq": 0.0,
            "mean_pair_distance": 0.0,
            "min_pair_distance": 0.0,
            "violating_pair_fraction": 0.0,
            "duplicate_pair_fraction": 0.0,
            "full_collapse_patient_fraction": 0.0,
        }

    hazard_mean = hazards.mean(dim=1, keepdim=True)
    variance = (hazards - hazard_mean).square().mean(dim=(1, 2))
    variance_term = (F.relu(variance_min - variance) + F.relu(variance - variance_max)).mean()

    if slot_count == 1:
        # There is no pair constraint for a single slot. Preserve an autograd
        # connection rather than creating an unrelated constant tensor.
        pair_term = slots.sum() * 0.0
        pair_stats = dict(
            mean_pair_distance_sq=0.0,
            min_pair_distance_sq=0.0,
            mean_pair_distance=0.0,
            min_pair_distance=0.0,
            violating_pair_fraction=0.0,
            duplicate_pair_fraction=0.0,
            full_collapse_patient_fraction=0.0,
        )
    else:
        squared = (slots.unsqueeze(2) - slots.unsqueeze(1)).square().sum(dim=-1)
        off_diagonal = ~torch.eye(slot_count, device=slots.device, dtype=torch.bool)
        # Each patient has exactly K * (K - 1) directed off-diagonal pairs.
        # First average pairs within each patient, then average patients.
        pairs = squared[:, off_diagonal]
        pair_term = F.relu(target_sq - pairs).mean(dim=1).mean()
        detached = pairs.detach()
        pair_stats = {
            "mean_pair_distance_sq": detached.mean().item(),
            "min_pair_distance_sq": detached.min().item(),
            "mean_pair_distance": detached.sqrt().mean().item(),
            "min_pair_distance": detached.min().sqrt().item(),
            "violating_pair_fraction": (detached < target_sq).float().mean().item(),
            "duplicate_pair_fraction": (detached <= duplicate_sq).float().mean().item(),
            "full_collapse_patient_fraction": (detached == 0.0).all(dim=1).float().mean().item(),
        }

    total = variance_term + pair_weight * pair_term
    diagnostics = {
        "patient_count": batch_size,
        "slot_count": slot_count,
        "pair_count_per_patient": slot_count * (slot_count - 1),
        "hazard_variance": variance.detach().mean().item(),
        "variance_term": variance_term.detach().item(),
        "pair_term": pair_term.detach().item(),
        "weighted_pair_term": (pair_weight * pair_term).detach().item(),
        "total_term": total.detach().item(),
        **pair_stats,
    }
    return total, diagnostics


def repaired_slot_diversity_loss(
    slots_wsi: torch.Tensor,
    slots_omic: torch.Tensor,
    hazard_wsi: torch.Tensor,
    hazard_omic: torch.Tensor,
    *,
    variance_min: float,
    variance_max: float,
    target_pair_dist: float = 1.0,
    pair_weight: float = 0.5,
    duplicate_tolerance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return an equal-modality mean loss and detached diagnostic scalars.

    Hazards are probabilities, typically the sigmoid outputs of each modality's
    slot hazard head. Variance is computed separately for every patient over
    slots and hazard bins. Pair distances use the raw slot representations.
    ``target_pair_dist`` and ``duplicate_tolerance`` are Euclidean distances;
    the pair objective compares their squares with squared pair distances.
    Explicit zero values are honored. An empty batch returns a graph-connected
    zero. One slot has no pair constraint but still has the variance-band term.

    At exact equality, a positive pair loss can have zero slot gradient. The
    full-collapse diagnostic reports that condition; the objective deliberately
    does not pretend to solve this symmetry with an unrequested perturbation.
    """
    variance_min = _nonnegative_finite("variance_min", variance_min)
    variance_max = _nonnegative_finite("variance_max", variance_max)
    target_pair_dist = _nonnegative_finite("target_pair_dist", target_pair_dist)
    pair_weight = _nonnegative_finite("pair_weight", pair_weight)
    duplicate_tolerance = _nonnegative_finite("duplicate_tolerance", duplicate_tolerance)
    if variance_min > variance_max:
        raise ValueError("variance_min must be less than or equal to variance_max")
    target_sq = target_pair_dist * target_pair_dist
    duplicate_sq = duplicate_tolerance * duplicate_tolerance
    if not math.isfinite(target_sq) or not math.isfinite(duplicate_sq):
        raise ValueError("squared pair-distance thresholds must be finite")
    _validate_modality("wsi", slots_wsi, hazard_wsi)
    _validate_modality("omic", slots_omic, hazard_omic)
    if slots_wsi.size(0) != slots_omic.size(0):
        raise ValueError("both modalities must have the same batch size")
    if slots_wsi.device != slots_omic.device:
        raise ValueError("both modalities must use the same device")

    parameters = dict(
        variance_min=variance_min,
        variance_max=variance_max,
        target_sq=target_sq,
        pair_weight=pair_weight,
        duplicate_sq=duplicate_sq,
    )
    wsi_loss, wsi_stats = _modality_terms(slots_wsi, hazard_wsi, **parameters)
    omic_loss, omic_stats = _modality_terms(slots_omic, hazard_omic, **parameters)
    loss = 0.5 * (wsi_loss + omic_loss)
    return loss, {
        "empty_batch": slots_wsi.size(0) == 0,
        "target_pair_dist": target_pair_dist,
        "pair_weight": pair_weight,
        "variance_min": variance_min,
        "variance_max": variance_max,
        "duplicate_tolerance": duplicate_tolerance,
        "loss": loss.detach().item(),
        "wsi": wsi_stats,
        "omic": omic_stats,
    }
