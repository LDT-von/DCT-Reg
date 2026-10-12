from __future__ import annotations

import pytest
import torch

from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.diversity import (
    repaired_slot_diversity_loss,
)


def objective(slots_wsi, slots_omic=None, hazards_wsi=None, hazards_omic=None, **overrides):
    if slots_omic is None:
        slots_omic = slots_wsi.clone()
    if hazards_wsi is None:
        hazards_wsi = slots_wsi.new_full((*slots_wsi.shape[:2], 2), 0.5)
    if hazards_omic is None:
        hazards_omic = slots_omic.new_full((*slots_omic.shape[:2], 2), 0.5)
    parameters = dict(variance_min=0.0, variance_max=0.2, target_pair_dist=1.0, pair_weight=0.5)
    parameters.update(overrides)
    return repaired_slot_diversity_loss(
        slots_wsi, slots_omic, hazards_wsi, hazards_omic, **parameters
    )


@pytest.mark.parametrize("repeats", [2, 4, 8])
def test_repeating_patient_preserves_loss_and_scales_each_copy_gradient(repeats):
    one_wsi = torch.tensor([[[0.0], [0.2], [0.4]]], dtype=torch.float64, requires_grad=True)
    one_omic = torch.tensor([[[0.1], [0.3], [0.7]]], dtype=torch.float64, requires_grad=True)
    one_loss, one_stats = objective(one_wsi, one_omic)
    one_loss.backward()
    many_wsi = one_wsi.detach().repeat(repeats, 1, 1).requires_grad_()
    many_omic = one_omic.detach().repeat(repeats, 1, 1).requires_grad_()
    many_loss, many_stats = objective(many_wsi, many_omic)
    many_loss.backward()
    torch.testing.assert_close(many_loss, one_loss)
    for single, repeated in [(one_wsi, many_wsi), (one_omic, many_omic)]:
        torch.testing.assert_close(repeated.grad, single.grad.repeat(repeats, 1, 1) / repeats)
        torch.testing.assert_close(repeated.grad.sum(dim=0), single.grad.squeeze(0))
    for modality in ["wsi", "omic"]:
        for key in ["pair_term", "mean_pair_distance_sq", "violating_pair_fraction"]:
            assert many_stats[modality][key] == pytest.approx(one_stats[modality][key])


def test_mixed_patient_and_modality_objective_is_equal_mean_of_standalone_patients():
    wsi = torch.tensor([[[0.0], [0.2], [0.4]], [[0.0], [1.2], [3.0]]])
    omic = torch.tensor([[[0.0], [0.1]], [[0.0], [0.8]]])
    hw = torch.tensor([[[0.1], [0.2], [0.3]], [[0.1], [0.5], [0.9]]])
    ho = torch.tensor([[[0.2], [0.8]], [[0.3], [0.4]]])
    kwargs = dict(variance_min=0.03, variance_max=0.08)
    combined, stats = objective(wsi, omic, hw, ho, **kwargs)
    standalone = [objective(wsi[i:i+1], omic[i:i+1], hw[i:i+1], ho[i:i+1], **kwargs)[0] for i in range(2)]
    torch.testing.assert_close(combined, torch.stack(standalone).mean())
    assert stats["wsi"]["pair_count_per_patient"] == 6
    assert stats["omic"]["pair_count_per_patient"] == 2


def test_manual_off_diagonal_normalization_and_variance_gradient_scaling():
    # Directed distances are [0.04, 0.16, 0.04, 0.04, 0.16, 0.04].
    slots = torch.tensor([[[0.0], [0.2], [0.4]]], dtype=torch.float64)
    hazards = torch.tensor([[[0.1], [0.4], [0.8]]], dtype=torch.float64, requires_grad=True)
    loss, stats = objective(slots, hazards_wsi=hazards, hazards_omic=hazards, variance_max=0.0)
    assert stats["wsi"]["mean_pair_distance_sq"] == pytest.approx(0.08)
    assert stats["wsi"]["pair_term"] == pytest.approx(0.92)
    loss.backward()
    repeated_hazards = hazards.detach().repeat(4, 1, 1).requires_grad_()
    repeated_loss, _ = objective(
        slots.repeat(4, 1, 1), hazards_wsi=repeated_hazards,
        hazards_omic=repeated_hazards, variance_max=0.0,
    )
    repeated_loss.backward()
    torch.testing.assert_close(repeated_loss, loss)
    torch.testing.assert_close(repeated_hazards.grad, hazards.grad.repeat(4, 1, 1) / 4)


def test_duplicate_pairs_cannot_be_hidden_by_distant_outlier():
    slots = torch.tensor([[[0.0], [0.0], [0.0], [10.0]]])
    loss, stats = objective(slots)
    assert stats["wsi"]["mean_pair_distance_sq"] > 1.0
    assert stats["wsi"]["pair_term"] == pytest.approx(0.5)
    assert stats["wsi"]["duplicate_pair_fraction"] == pytest.approx(0.5)
    assert loss.item() == pytest.approx(0.25)


def test_zero_pair_weight_and_zero_target_are_preserved():
    slots = torch.zeros(1, 3, 2, requires_grad=True)
    zero_weight, stats = objective(slots, pair_weight=0.0)
    assert zero_weight.item() == 0.0
    assert stats["pair_weight"] == 0.0
    assert stats["wsi"]["pair_term"] == 1.0
    zero_weight.backward()
    torch.testing.assert_close(slots.grad, torch.zeros_like(slots))
    zero_target, stats = objective(slots, target_pair_dist=0.0)
    assert zero_target.item() == 0.0
    assert stats["wsi"]["violating_pair_fraction"] == 0.0


def test_explicit_zero_variance_bounds_are_preserved():
    slots = torch.tensor([[[0.0], [2.0]]])
    hazards = torch.tensor([[[0.0], [1.0]]])
    loss, stats = objective(slots, hazards_wsi=hazards, hazards_omic=hazards, variance_min=0.0, variance_max=0.0)
    assert loss.item() == pytest.approx(0.25)
    assert stats["variance_max"] == 0.0


def test_near_collapse_has_finite_nonzero_gradient():
    slots = torch.tensor([[[0.0, 0.0], [1e-4, 0.0], [0.0, 2e-4]]], dtype=torch.float64, requires_grad=True)
    loss, stats = objective(slots)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(slots.grad).all()
    assert slots.grad.abs().sum() > 0.0
    assert stats["wsi"]["full_collapse_patient_fraction"] == 0.0


def test_exact_collapse_reports_positive_loss_and_honest_zero_distance_gradient():
    slots = torch.zeros(2, 4, 3, requires_grad=True)
    loss, stats = objective(slots)
    loss.backward()
    assert loss.item() == pytest.approx(0.5)
    torch.testing.assert_close(slots.grad, torch.zeros_like(slots))
    assert stats["wsi"]["full_collapse_patient_fraction"] == 1.0
    assert stats["wsi"]["duplicate_pair_fraction"] == 1.0
    assert stats["wsi"]["min_pair_distance_sq"] == 0.0
    assert all(not isinstance(value, torch.Tensor) for value in stats["wsi"].values())


def test_slot_permutation_preserves_loss_diagnostics_and_gradient_order():
    slots = torch.tensor([[[0.0], [0.2], [0.9]]], requires_grad=True)
    hazards = torch.tensor([[[0.1], [0.4], [0.8]]])
    loss, stats = objective(slots, hazards_wsi=hazards, hazards_omic=hazards)
    loss.backward()
    perm = torch.tensor([2, 0, 1])
    permuted = slots.detach()[:, perm].requires_grad_()
    other_loss, other_stats = objective(permuted, hazards_wsi=hazards[:, perm], hazards_omic=hazards[:, perm])
    other_loss.backward()
    torch.testing.assert_close(other_loss, loss)
    torch.testing.assert_close(permuted.grad, slots.grad[:, perm])
    for modality in ["wsi", "omic"]:
        for key, value in stats[modality].items():
            assert other_stats[modality][key] == pytest.approx(value)


def test_one_slot_has_finite_graph_connected_zero_pair_constraint():
    slots = torch.ones(2, 1, 3, requires_grad=True)
    loss, stats = objective(slots, variance_min=0.02)
    assert loss.item() == pytest.approx(0.02)
    assert stats["wsi"]["pair_term"] == 0.0
    assert stats["wsi"]["pair_count_per_patient"] == 0
    assert stats["wsi"]["full_collapse_patient_fraction"] == 0.0
    loss.backward()
    assert slots.grad is not None
    torch.testing.assert_close(slots.grad, torch.zeros_like(slots))


def test_empty_batch_has_finite_graph_connected_zero_and_explicit_diagnostics():
    slots = torch.empty(0, 3, 2, requires_grad=True)
    loss, stats = objective(slots, variance_min=0.02)
    assert loss.item() == 0.0
    assert stats["empty_batch"] is True
    assert stats["wsi"]["patient_count"] == 0
    assert stats["wsi"]["hazard_variance"] == 0.0
    loss.backward()
    assert slots.grad is not None
    assert slots.grad.shape == slots.shape


@pytest.mark.parametrize("overrides", [
    {"pair_weight": -1.0}, {"pair_weight": float("nan")},
    {"target_pair_dist": float("inf")}, {"target_pair_dist": -0.1},
    {"variance_min": -0.1}, {"variance_max": float("nan")},
    {"variance_min": 0.3, "variance_max": 0.2},
    {"duplicate_tolerance": -0.01}, {"duplicate_tolerance": float("inf")},
    {"target_pair_dist": 1e308}, {"duplicate_tolerance": 1e308},
])
def test_invalid_hyperparameters_are_rejected(overrides):
    with pytest.raises(ValueError):
        objective(torch.zeros(1, 2, 3), **overrides)


@pytest.mark.parametrize("slots,hazards", [
    (torch.zeros(1, 0, 3), torch.zeros(1, 0, 2)),
    (torch.zeros(1, 2, 3), torch.zeros(1, 3, 2)),
    (torch.zeros(1, 2, 3), torch.zeros(1, 2, 0)),
    (torch.full((1, 2, 3), float("nan")), torch.zeros(1, 2, 2)),
])
def test_invalid_modality_shapes_or_values_are_rejected(slots, hazards):
    with pytest.raises(ValueError):
        objective(slots, hazards_wsi=hazards)
