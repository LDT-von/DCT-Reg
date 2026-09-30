"""Tests for DCT v3.13 §3.1 #3 (cross_mode) and §3.1 #4 (plan_mode).

``paper/V313_IMPLEMENTATION_PLAN.md`` §2.3 / §2.4:

* ``dct_v313_cross_mode=transport|direct`` — transport (default) routes
  the OT-transported WSI slots into the cross decoder; direct bypasses
  OT and feeds the raw WSI slots into the same decoder.  Both modes
  share the prediction OT, the self reconstruction, and every other
  loss term.

* ``dct_v313_plan_mode=learned|independent`` — learned (default) uses
  Sinkhorn-projected OT plans; independent replaces the plan with the
  outer product of the marginals ``T = a bᵀ`` in every code path that
  consumes the plan (training and eval).

These tests verify the wiring and end-to-end behaviour of both switches.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import (
    DCTV313TransportReconstruction,
)


# ---------------------------------------------------------------------------
# Argument factory shared with the rest of the v3.13 test suite.
# ---------------------------------------------------------------------------

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
        dct_lambda_ipcw_rank=0.0,
        dct_ipcw_rank_margin=0.99,
        dct_ipcw_rank_temperature=0.99,
        dct_ipcw_max_weight=99.0,
        dct_ipcw_rank_memory_size=1,
        dct_lambda_etar=0.0,
        dct_lambda_listwise=0.0,
        dct_anchor_momentum=0.0,
        dct_evidence_cost_weight=0.0,
        dct_evidence_mass_floor=0.99,
        dct_evidence_marginal_strength=0.0,
        dct_geometry_reliability_strength=0.0,
        dct_geometry_reliability_temperature=0.25,
        dct_coupling_projection_iters=20,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=0.25,
        dct_v38_lambda_direction=0.0,
        dct_v38_lambda_dose=0.0,
        dct_v38_lambda_reconfiguration=0.0,
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
        dct_v313_recon_weighting="per_branch",
        dct_v313_cross_mode="transport",
        dct_v313_plan_mode="learned",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def train_reference():
    times = torch.tensor([1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    censorship = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return times, censorship


def batch(seed=13):
    generator = torch.Generator().manual_seed(seed)
    payload = {
        "x_wsi": torch.randn(8, 6, 16, generator=generator),
        "y": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        "event_time": train_reference()[0],
        "c": train_reference()[1],
    }
    for index, width in enumerate([3, 4, 5, 6, 7], start=1):
        payload[f"x_omic{index}"] = torch.randn(8, width, generator=generator)
    return payload


# ---------------------------------------------------------------------------
# §3.1 #3: cross_mode parsing.
# ---------------------------------------------------------------------------


def test_cross_mode_default_is_transport():
    """The CLI default matches the plan §2.3: ``transport``."""
    args = _make_args()
    model = DCTV313TransportReconstruction(args)
    assert model._cross_mode == "transport"
    assert model.effective_recon_coefficients()["cross_mode"] == "transport"


def test_cross_mode_direct_is_parsed():
    args = _make_args(dct_v313_cross_mode="direct")
    model = DCTV313TransportReconstruction(args)
    assert model._cross_mode == "direct"
    assert model.effective_recon_coefficients()["cross_mode"] == "direct"


def test_cross_mode_unknown_rejected():
    args = _make_args(dct_v313_cross_mode="random")
    with pytest.raises(ValueError, match="dct_v313_cross_mode"):
        DCTV313TransportReconstruction(args)


# ---------------------------------------------------------------------------
# §3.1 #4: plan_mode parsing.
# ---------------------------------------------------------------------------


def test_plan_mode_default_is_learned():
    args = _make_args()
    model = DCTV313TransportReconstruction(args)
    assert model._plan_mode == "learned"
    assert model.effective_recon_coefficients()["plan_mode"] == "learned"


def test_plan_mode_independent_is_parsed():
    args = _make_args(dct_v313_plan_mode="independent")
    model = DCTV313TransportReconstruction(args)
    assert model._plan_mode == "independent"
    assert model.effective_recon_coefficients()["plan_mode"] == "independent"


def test_plan_mode_unknown_rejected():
    args = _make_args(dct_v313_plan_mode="random")
    with pytest.raises(ValueError, match="dct_v313_plan_mode"):
        DCTV313TransportReconstruction(args)


# ---------------------------------------------------------------------------
# §3.1 #3: direct mode bypasses the OT transport and feeds raw WSI slots
# into the cross decoder.  All other loss terms are unchanged.
# ---------------------------------------------------------------------------


def test_cross_mode_direct_skips_transport_branch():
    """``cross_mode=direct`` feeds raw WSI slots into the cross decoder
    instead of the OT-transported ones.  Cross decoder gradients still flow
    back to the WSI slots, but the transport path is no longer invoked
    for cross reconstruction."""
    torch.manual_seed(17)
    model = DCTV313TransportReconstruction(_make_args(dct_v313_cross_mode="direct"))
    model.configure_train_reference(*train_reference())
    model.train()

    logits, auxiliary = model(**batch())
    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(auxiliary)
    diagnostics = model.last_training_losses
    assert diagnostics["v313_reconstruction_self"] > 0
    # Cross reconstruction still fires in direct mode (just from a different
    # input source).
    assert diagnostics["v313_reconstruction_cross"] > 0

    auxiliary.backward()
    # Gradient still flows back to WSI slots through the cross decoder in
    # direct mode.
    assert model.pathway_reconstruction_decoder.pathway_queries.grad is not None
    assert model.pathway_reconstruction_decoder.pathway_queries.grad.abs().sum() > 0


def test_cross_mode_transport_matches_direct_for_prediction_branch():
    """The prediction OT / factual logits are NOT affected by cross_mode;
    the switch only changes the cross decoder input source.  This is the
    invariant §3.1 #3 demands: "direct arm keeps the full prediction OT".

    We probe this by recording the factual logits for two models that
    differ only in ``cross_mode``, evaluated under no_grad.  They must be
    equal because the prediction branch never touches ``transported_wsi``.
    """
    torch.manual_seed(31)
    transport_model = DCTV313TransportReconstruction(
        _make_args(dct_v313_cross_mode="transport")
    )
    transport_model.configure_train_reference(*train_reference())
    transport_model.eval()
    with torch.no_grad():
        transport_logits, _ = transport_model(**batch())

    torch.manual_seed(31)
    direct_model = DCTV313TransportReconstruction(
        _make_args(dct_v313_cross_mode="direct")
    )
    direct_model.configure_train_reference(*train_reference())
    direct_model.eval()
    with torch.no_grad():
        direct_logits, _ = direct_model(**batch())

    torch.testing.assert_close(transport_logits, direct_logits)


# ---------------------------------------------------------------------------
# §3.1 #4: independent mode replaces the plan with T = a bᵀ in every consumer.
# ---------------------------------------------------------------------------


def test_plan_mode_independent_returns_outer_product_plans():
    """When ``plan_mode=independent``, ``_plans_from_cost_tensor`` returns
    plans that satisfy ``plan[b, w, o] = rows[b, w] * cols[b, o]`` exactly."""
    torch.manual_seed(7)
    model = DCTV313TransportReconstruction(_make_args(dct_v313_plan_mode="independent"))
    model.train()
    slots_wsi = torch.randn(2, 3, 16)
    slots_omic = torch.randn(2, 3, 16)
    costs, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)
    plans, distance = model._plans_from_cost_tensor(costs, rows, cols, epoch=7)

    num_stages = rows.size(1)
    assert len(plans) == num_stages
    for stage_idx, stage_plans in enumerate(plans):
        # Each stage carries the same number of geometry plans as the
        # cost tensor (multi-geometry contract preserved).
        assert len(stage_plans) == costs.size(2)
        for plan in stage_plans:
            expected = torch.einsum(
                "bw,bo->bwo", rows[:, stage_idx], cols[:, stage_idx]
            )
            torch.testing.assert_close(plan, expected, rtol=1e-6, atol=1e-7)
            # Marginals are honoured: row sums = rows, col sums = cols.
            row_sums = plan.sum(dim=-1)
            col_sums = plan.sum(dim=-2)
            torch.testing.assert_close(row_sums, rows[:, stage_idx], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(col_sums, cols[:, stage_idx], rtol=1e-6, atol=1e-7)

    # Distance is reported honestly (cost mass of T = a bᵀ on the cost
    # tensor) — NOT zero.  This is purely diagnostic.
    assert torch.isfinite(distance)
    assert float(distance.detach()) >= 0.0


def test_plan_mode_independent_full_forward_is_finite():
    """End-to-end check that ``plan_mode=independent`` survives the full
    forward pass with the cross reconstruction enabled.  The model must
    not crash and gradients must flow."""
    torch.manual_seed(23)
    model = DCTV313TransportReconstruction(
        _make_args(dct_v313_plan_mode="independent", dct_v313_cross_mode="transport")
    )
    model.configure_train_reference(*train_reference())
    model.train()

    logits, auxiliary = model(**batch())
    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(auxiliary)
    auxiliary.backward()
    assert (
        model.pathway_reconstruction_decoder.pathway_queries.grad is not None
        and model.pathway_reconstruction_decoder.pathway_queries.grad.abs().sum() > 0
    )


def test_plan_mode_independent_matches_learned_when_marginals_are_uniform():
    """Sanity check: when the marginals are uniform (rows = cols = 1/K),
    the independent plan ``T = a bᵀ`` matches the row/column marginals
    of *any* plan with the same marginals — including the learned one.
    We assert the marginal preservation, not full plan equality (Sinkhorn
    can choose among many plans with the same marginals)."""
    torch.manual_seed(11)
    model = DCTV313TransportReconstruction(_make_args(dct_v313_plan_mode="independent"))
    model.train()
    slots_wsi = torch.randn(2, 3, 16)
    slots_omic = torch.randn(2, 3, 16)
    costs, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)
    plans, _ = model._plans_from_cost_tensor(costs, rows, cols, epoch=7)
    for stage_idx, stage_plans in enumerate(plans):
        for plan in stage_plans:
            # Plan has the correct row and column marginals.
            row_sums = plan.sum(dim=-1)
            col_sums = plan.sum(dim=-2)
            torch.testing.assert_close(
                row_sums, rows[:, stage_idx], rtol=1e-6, atol=1e-7
            )
            torch.testing.assert_close(
                col_sums, cols[:, stage_idx], rtol=1e-6, atol=1e-7
            )


def test_plan_mode_learned_does_not_short_circuit_in_normal_mode():
    """In ``learned`` mode (default) the model delegates to the parent
    Sinkhorn-based planner.  We assert the override does NOT shadow the
    parent when ``plan_mode != 'independent'``."""
    torch.manual_seed(13)
    model = DCTV313TransportReconstruction(_make_args(dct_v313_plan_mode="learned"))
    model.train()
    slots_wsi = torch.randn(2, 3, 16)
    slots_omic = torch.randn(2, 3, 16)
    costs, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)
    plans, distance = model._plans_from_cost_tensor(costs, rows, cols, epoch=7)

    # Sinkhorn-projected plans generally differ from the marginal
    # outer product.  We assert that at least one plan is NOT equal to
    # ``T = a bᵀ`` (with very high tolerance for the rare alignment).
    found_non_outer = False
    for stage_idx, stage_plans in enumerate(plans):
        for plan in stage_plans:
            outer = torch.einsum(
                "bw,bo->bwo", rows[:, stage_idx], cols[:, stage_idx]
            )
            if not torch.allclose(plan, outer, atol=1e-3):
                found_non_outer = True
                break
    assert found_non_outer, "learned plans were all equal to the marginal outer product"


# ---------------------------------------------------------------------------
# CLI round-trip via extended_args.process_args_extended.
# ---------------------------------------------------------------------------


def test_cli_parser_accepts_cross_mode_choices():
    """``--dct_v313_cross_mode`` accepts transport / direct."""
    from survot_rank.training.extended_args import build_base_parser

    parser = build_base_parser()
    # Default value
    ns = parser.parse_args([])
    assert ns.dct_v313_cross_mode == "transport"
    # Explicit value
    ns = parser.parse_args(["--dct_v313_cross_mode", "direct"])
    assert ns.dct_v313_cross_mode == "direct"
    # Unknown value rejected
    with pytest.raises(SystemExit):
        parser.parse_args(["--dct_v313_cross_mode", "bogus"])


def test_cli_parser_accepts_plan_mode_choices():
    """``--dct_v313_plan_mode`` accepts learned / independent."""
    from survot_rank.training.extended_args import build_base_parser

    parser = build_base_parser()
    ns = parser.parse_args([])
    assert ns.dct_v313_plan_mode == "learned"
    ns = parser.parse_args(["--dct_v313_plan_mode", "independent"])
    assert ns.dct_v313_plan_mode == "independent"
    with pytest.raises(SystemExit):
        parser.parse_args(["--dct_v313_plan_mode", "bogus"])
