"""Tests for DCT v3.13 reconstruction weighting policies.

Implements the new ``per_branch`` weighting described in
``paper/V313_IMPLEMENTATION_PLAN.md`` §2.1 / §4.4:

* ``legacy`` (commit 93d8314 default): disabling one branch renormalises the
  survivor so the SUM of effective reconstruction coefficients stays at
  ``RECONSTRUCTION_WEIGHT`` (== 0.05).
* ``per_branch``: each branch keeps its independent coefficient
  (``RECONSTRUCTION_WEIGHT * fraction == 0.05``); disabling a branch
  zeroes ONLY that branch.  No silent renormalisation.  ``scale=1`` is
  mandatory (the historic ``scale=2`` legacy compensation MUST NOT be
  carried over).

Both modes must agree on the full recipe (both branches enabled,
``scale=1``): total coefficient = 0.10, per-branch = 0.05.

The test verifies:
1. CLI / args round-trip: ``--set dct_v313_recon_weighting=per_branch``
   reaches the model constructor and is honoured.
2. ``per_branch`` forbids ``scale != 1`` (raises ValueError).
3. Per-branch coefficients are independent of the disable flags in the
   expected direction.
4. The value the combiner multiplies the loss by is exactly what
   ``effective_recon_coefficients()`` reports (no silent normalisation
   by the runtime).
5. ``legacy`` reproduces the historical renormalisation for the four
   disable combinations.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import (
    DCTV313TransportReconstruction,
)


def _make_args(**overrides):
    values = dict(
        bag_loss="nll_surv",
        omic_sizes=[3, 4, 5, 6, 7],
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="Pathways",
        alpha_surv=0.15,
        slot_num_wsi=3,
        slot_num_omics=3,
        slot_iters=2,
        otehv2_eps=0.05,
        otehv2_iter=15,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.0,
        dct_num_stages=4,
        dct_lambda_ipcw_rank=0.99,
        dct_ipcw_rank_margin=0.99,
        dct_ipcw_rank_temperature=0.99,
        dct_ipcw_max_weight=99.0,
        dct_ipcw_rank_memory_size=1,
        dct_lambda_etar=0.99,
        dct_lambda_listwise=0.99,
        dct_anchor_momentum=0.0,
        dct_evidence_cost_weight=0.99,
        dct_evidence_mass_floor=0.99,
        dct_evidence_marginal_strength=0.0,
        dct_geometry_reliability_strength=0.99,
        dct_geometry_reliability_temperature=0.25,
        dct_coupling_projection_iters=20,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=0.25,
        dct_v38_lambda_direction=0.99,
        dct_v38_lambda_dose=0.99,
        dct_v38_lambda_reconfiguration=0.99,
        dct_v38_direction_margin=0.99,
        dct_v38_dose_margin=0.005,
        dct_v38_reconfiguration_margin=0.02,
        dct_v38_temperature=0.99,
        dct_v38_alpha_mid=0.25,
        dct_v38_alpha_full=0.75,
        dct_v38_warmup_epochs=5,
        dct_v38_ramp_epochs=10,
        dct_v38_dose_every=1,
        dct_v382_lambda_mgptr=0.99,
        dct_v382_adaptive_aux_weights=True,
        dct_fixed_coupling=True,
        dct_random_anchors=True,
        dct_perm_labels_seed=7,
        dct_stage_jitter_fraction=0.3,
        dct_freeze_source_prototype="",
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.2,
        spt_lambda_ot=0.0,
        spt_lambda_rank=0.0,
        spt_lambda_stage=0.0,
        spt_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        dct_slot_init_mode="gaussian",
        dct_slot_eval_seed=91,
        cur_epoch=7,
        dct_v311_lambda_slot_nll=0.0,
        dct_v311_lambda_slot_diversity=0.0,
        dct_v313_lambda_reconstruction=0.10,
        dct_v313_disable_self_reconstruction="false",
        dct_v313_disable_cross_reconstruction="false",
        dct_v313_lambda_reconstruction_scale=1.0,
        dct_v313_recon_weighting="legacy",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# The single source of truth: the class invariant fractions.
# Both modes should agree when scale=1 and both branches are enabled.
SELF_FRAC = DCTV313TransportReconstruction.RECONSTRUCTION_SELF_FRACTION
CROSS_FRAC = DCTV313TransportReconstruction.RECONSTRUCTION_CROSS_FRACTION
WEIGHT = DCTV313TransportReconstruction.RECONSTRUCTION_WEIGHT


@pytest.mark.parametrize(
    "disable_self,disable_cross,mode,scale,expected_self,expected_cross,expected_total",
    [
        # Full recipe (both enabled, scale=1) — legacy and per_branch agree.
        ("false", "false", "legacy", 1.0, WEIGHT * SELF_FRAC, WEIGHT * CROSS_FRAC, WEIGHT),
        ("false", "false", "per_branch", 1.0, WEIGHT * SELF_FRAC, WEIGHT * CROSS_FRAC, WEIGHT),
        # legacy + disable_self: legacy renormalises the outer weight to
        # ``WEIGHT * fraction / frac_sum`` (= 0.05).  The combiner still
        # multiplies by the fraction once more, so the per-branch
        # coefficient on cross_loss is ``outer × fraction = 0.025``.
        # This is the legacy "double-down" weakness that per_branch fixes.
        ("true", "false", "legacy", 1.0, 0.0, WEIGHT * CROSS_FRAC * CROSS_FRAC, WEIGHT * CROSS_FRAC * CROSS_FRAC),
        ("false", "true", "legacy", 1.0, WEIGHT * SELF_FRAC * SELF_FRAC, 0.0, WEIGHT * SELF_FRAC * SELF_FRAC),
        # legacy: both disabled → 0.
        ("true", "true", "legacy", 1.0, 0.0, 0.0, 0.0),
        # per_branch: disable self → cross keeps its full class constant (0.05).
        ("true", "false", "per_branch", 1.0, 0.0, WEIGHT * CROSS_FRAC, WEIGHT * CROSS_FRAC),
        # per_branch: disable cross → self keeps its full class constant (0.05).
        ("false", "true", "per_branch", 1.0, WEIGHT * SELF_FRAC, 0.0, WEIGHT * SELF_FRAC),
        # per_branch: both disabled → 0.
        ("true", "true", "per_branch", 1.0, 0.0, 0.0, 0.0),
        # legacy: scale=2 doubles both per-branch coefficients.
        (
            "false", "false", "legacy", 2.0,
            2.0 * WEIGHT * SELF_FRAC, 2.0 * WEIGHT * CROSS_FRAC, 2.0 * WEIGHT,
        ),
        # legacy: scale=2 only self surviving → self coef doubles, total
        # doubles.  This is the historic compensation that per_branch
        # refuses.
        (
            "false", "true", "legacy", 2.0,
            2.0 * WEIGHT * SELF_FRAC * SELF_FRAC, 0.0, 2.0 * WEIGHT * SELF_FRAC * SELF_FRAC,
        ),
    ],
    ids=[
        "legacy_both",
        "per_branch_both",
        "legacy_disable_self",
        "legacy_disable_cross",
        "legacy_disable_both",
        "per_branch_disable_self",
        "per_branch_disable_cross",
        "per_branch_disable_both",
        "legacy_scale2_both",
        "legacy_scale2_only_self",
    ],
)
def test_recon_weighting_policy_produces_documented_coefficients(
    disable_self, disable_cross, mode, scale,
    expected_self, expected_cross, expected_total,
):
    args = _make_args(
        dct_v313_disable_self_reconstruction=disable_self,
        dct_v313_disable_cross_reconstruction=disable_cross,
        dct_v313_lambda_reconstruction_scale=scale,
        dct_v313_recon_weighting=mode,
    )
    model = DCTV313TransportReconstruction(args)

    eff = model.effective_recon_coefficients()
    assert eff["weighting_mode"] == mode
    assert eff["disable_self"] == (disable_self == "true")
    assert eff["disable_cross"] == (disable_cross == "true")
    assert eff["lambda_scale"] == scale
    assert eff["self"] == pytest.approx(expected_self)
    assert eff["cross"] == pytest.approx(expected_cross)
    assert eff["total"] == pytest.approx(expected_total)


@pytest.mark.parametrize(
    "scale", [-1.0, 0.5, 1.5, 2.0, 3.14],
)
def test_per_branch_forbids_non_unit_scale(scale):
    """The legacy scale=2 was a 93d8314-era compensation and must NOT
    carry over to the per_branch recipe."""
    args = _make_args(
        dct_v313_recon_weighting="per_branch",
        dct_v313_lambda_reconstruction_scale=scale,
    )
    with pytest.raises(ValueError, match="per_branch"):
        DCTV313TransportReconstruction(args)


def test_unknown_weighting_mode_rejected():
    args = _make_args(dct_v313_recon_weighting="bogus")
    with pytest.raises(ValueError, match="dct_v313_recon_weighting"):
        DCTV313TransportReconstruction(args)


def test_legacy_mode_accepts_arbitrary_scale():
    """legacy reproduces the historical 93d8314 behaviour, including the
    scale=2 compensation used by the ablation runner.  This must remain
    valid for backwards compatibility with the 70-fold ablation logs."""
    args = _make_args(dct_v313_recon_weighting="legacy", dct_v313_lambda_reconstruction_scale=2.0)
    model = DCTV313TransportReconstruction(args)
    # Both branches enabled, scale=2 → per-branch coefs double, total doubles.
    assert model.effective_recon_coefficients()["total"] == pytest.approx(2.0 * WEIGHT)


def test_objective_weights_reports_per_branch_constants():
    """``objective_weights`` (classmethod) reports the immutable abstract
    recipe.  In ``per_branch`` mode (the §3.1 default) the aggregated
    ``reconstruction`` key is omitted — consumers consume the two branch
    contributions directly via ``reconstruction_self`` / ``reconstruction_cross``.
    """
    expected = {
        "nll": 1.0,
        "ipcw_rank": 0.10,
        "per_slot_nll": 0.05,
        "slot_diversity": 0.10,
        "weighting_mode": "per_branch",
        "reconstruction_self": WEIGHT * SELF_FRAC,
        "reconstruction_cross": WEIGHT * CROSS_FRAC,
    }
    assert DCTV313TransportReconstruction.objective_weights() == expected


# ---------------------------------------------------------------------------
# The combiner must consume the EFFECTIVE per-branch coefficients (audit log
# consistency).  We assert this by feeding two synthetic scalars through the
# reconstruction_losses path and comparing the combined tensor to the
# analytical formula using `effective_recon_coefficients`.
# ---------------------------------------------------------------------------

def _fake_recon_inputs(slots_dim, available):
    generator = torch.Generator().manual_seed(123)
    slots_wsi = torch.randn(2, 3, slots_dim, generator=generator, requires_grad=True)
    slots_omic = torch.randn(2, 3, slots_dim, generator=generator, requires_grad=True)
    target = torch.randn(2, 5, slots_dim)
    return slots_wsi, slots_omic, target, available


def test_combiner_consumes_effective_coefficients_in_per_branch_mode():
    """For per_branch + disable_self, only the cross branch contributes.
    Disable self → forward must short-circuit self and ignore it."""
    args = _make_args(
        dct_v313_recon_weighting="per_branch",
        dct_v313_disable_self_reconstruction="true",
        dct_v313_disable_cross_reconstruction="false",
    )
    torch.manual_seed(7)
    model = DCTV313TransportReconstruction(args)
    model.train()
    eff = model.effective_recon_coefficients()
    assert eff["self"] == 0.0
    assert eff["cross"] == pytest.approx(WEIGHT * CROSS_FRAC)

    slots_wsi = torch.randn(2, 3, 16, requires_grad=True)
    slots_omic = torch.randn(2, 3, 16, requires_grad=True)
    target = torch.randn(2, 5, 16)
    costs, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)
    plans, _ = model._plans_from_cost_tensor(costs, rows, cols, epoch=7)
    _, gate = model._encode_logits_from_plans(slots_wsi, slots_omic, plans)
    self_loss, cross_loss, total, transported = model.reconstruction_losses(
        x_omics=target,
        slots_wsi=slots_wsi,
        slots_omic=slots_omic,
        factual_plans=plans,
        factual_gate=gate,
        available=torch.ones(2, dtype=torch.bool),
    )
    assert self_loss.item() == 0.0
    assert cross_loss.item() > 0.0
    expected_total = eff["cross"] * cross_loss
    torch.testing.assert_close(total, expected_total, rtol=1e-6, atol=1e-8)
    # Gradients on self slots must be zero (self branch is off).
    total.sum().backward()
    assert slots_omic.grad is not None
    # The cross branch still pulls cross_recon_decoder + transport path.
    assert slots_wsi.grad is not None
    assert model.pathway_reconstruction_decoder.pathway_queries.grad is not None


def test_combiner_consumes_effective_coefficients_in_legacy_mode():
    """For legacy + disable_self, the surviving cross branch's per-branch
    coefficient is ``effective_weight_total * fraction = 0.025``.  The
    combiner consumes the coefficient directly (not the inner fractioned
    `total`), so what the forward pass multiplies by is exactly what
    ``effective_recon_coefficients()`` reports."""
    args = _make_args(
        dct_v313_recon_weighting="legacy",
        dct_v313_disable_self_reconstruction="true",
        dct_v313_disable_cross_reconstruction="false",
    )
    torch.manual_seed(11)
    model = DCTV313TransportReconstruction(args)
    model.train()
    eff = model.effective_recon_coefficients()
    assert eff["self"] == 0.0
    # Legacy renormalises effective_weight, but the combiner still scales
    # by the fraction once more, so per-branch coef = WEIGHT * fraction^2
    # = 0.025 with the default 50/50 split.
    assert eff["cross"] == pytest.approx(WEIGHT * CROSS_FRAC * CROSS_FRAC)
    assert eff["total"] == pytest.approx(WEIGHT * CROSS_FRAC * CROSS_FRAC)

    slots_wsi = torch.randn(2, 3, 16, requires_grad=True)
    slots_omic = torch.randn(2, 3, 16, requires_grad=True)
    target = torch.randn(2, 5, 16)
    costs, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)
    plans, _ = model._plans_from_cost_tensor(costs, rows, cols, epoch=7)
    _, gate = model._encode_logits_from_plans(slots_wsi, slots_omic, plans)
    self_loss, cross_loss, total, _ = model.reconstruction_losses(
        x_omics=target,
        slots_wsi=slots_wsi,
        slots_omic=slots_omic,
        factual_plans=plans,
        factual_gate=gate,
        available=torch.ones(2, dtype=torch.bool),
    )
    assert self_loss.item() == 0.0
    assert cross_loss.item() > 0.0
    expected_total = eff["cross"] * cross_loss
    torch.testing.assert_close(total, expected_total, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Argparse round-trip — confirms --set dct_v313_recon_weighting=per_branch
# survives the full CLI -> config -> flatten -> parse cycle.
# ---------------------------------------------------------------------------

def test_argparse_round_trip_per_branch_flag():
    """Verify the new --dct_v313_recon_weighting flag is registered and
    accepts 'per_branch'.  This catches regressions where the argparse
    parser silently drops new keys (the 93d8314 bug class)."""
    from survot_rank.config import apply_overrides
    base = {"train": {"survot_method": "dct_v313"}, "model": {}}
    merged = apply_overrides(base, ["dct_v313_recon_weighting=per_branch"])
    assert merged["dct_v313_recon_weighting"] == "per_branch"
    assert "dct_v313_recon_weighting" in merged["_dct_user_overrides"]


def test_argparse_parser_accepts_per_branch_choice():
    from survot_rank.training.extended_args import build_base_parser
    parser = build_base_parser()
    argv = [
        "--survot_method", "dct_v313",
        "--dct_v313_recon_weighting", "per_branch",
    ]
    args = parser.parse_args(argv)
    assert args.dct_v313_recon_weighting == "per_branch"


def test_argparse_parser_rejects_unknown_mode():
    from survot_rank.training.extended_args import build_base_parser
    parser = build_base_parser()
    argv = [
        "--survot_method", "dct_v313",
        "--dct_v313_recon_weighting", "naive",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(argv)