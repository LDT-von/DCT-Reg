"""Synthetic mechanism checks; no patient training or performance claims."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from survot_rank.research.methods.dct_v22 import DCTV22, DCTV22Config
from survot_rank.research.methods.dct_v23 import DCTV23, DCTV23Config
from survot_rank.research.methods.dct_v23.reconstruction import (
    GlobalGeneSchema, conditional_donor_context, sample_global_mask,
)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(**changes):
    return replace(DCTV23Config(
        omic_sizes=(3, 3), dim=8, wsi_input_dim=6, wsi_hidden_dim=4,
        wsi_dropout=0., dropout=0., num_classes=3, wsi_slots=2, omics_slots=2,
        stages=2, heads=2, event_layers=1, slot_iters=1, slot_heads=2,
        slot_dim_head=4, slot_hidden_dim=16, recon_hidden_dim=5,
        ipcw_weight=0., slot_weight=0., diversity_weight=0., relation_weight=0., native_weight=0.,
    ), **changes)


def inputs(batch=2):
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(batch, 3, generator=generator)
    b = torch.randn(batch, 3, generator=generator)
    b[:, 0] = a[:, 2]
    return dict(x_wsi=torch.randn(batch, 5, 6, generator=generator), x_omic1=a, x_omic2=b,
                wsi_mask=torch.ones(batch, 5, dtype=torch.bool),
                omics_mask=torch.ones(batch, 2, dtype=torch.bool),
                patient_ids=["p" + str(i) for i in range(batch)],
                y=torch.arange(batch) % 3, c=torch.arange(batch).float() % 2,
                event_time=torch.arange(1, batch + 1).float(), epoch=5)


def model(**changes):
    torch.manual_seed(23)
    instance = DCTV23(config(**changes))
    instance.configure_reconstruction_reference([["A", "B", "C"], ["C", "D", "E"]],
                                                 [0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.], ["p0", "p1"])
    return instance


def bind(instance, gains=None):
    instance.bind_gain_targets(gains or {"p0": [1., -1.], "p1": [-.5, .5]},
                               provenance())
    return instance


def provenance():
    # Fully explicit synthetic OOF identity witnesses, not real patient data.
    return {"schema": "dct_v23_oof_pathway_gain_v1", "train_patient_ids": ["p0", "p1"],
            "held_out_gain": True, "no_survival_labels": True,
            "gene_ids": ["A", "B", "C", "D", "E"],
            "pathway_gene_ids": [["A", "B", "C"], ["C", "D", "E"]],
            "teacher_type": "fixed_coordinate_ridge_native_vs_conditional",
            "gain_definition": "native_masked_mse_minus_conditional_masked_mse_teacher_standardized_raw_genes",
            "raw_expression_sha256": "0" * 64, "pooled_wsi_sha256": "1" * 64,
            "global_masks": [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
            "teachers": [{"teacher_id": 0, "fit_patient_ids": ["p1"], "score_patient_ids": ["p0"],
                          "normalization_fit_patient_ids": ["p1"], "gene_mean": [0.] * 5, "gene_std": [1.] * 5,
                          "wsi_mean": [0.] * 6, "wsi_std": [1.] * 6},
                         {"teacher_id": 1, "fit_patient_ids": ["p0"], "score_patient_ids": ["p1"],
                          "normalization_fit_patient_ids": ["p0"], "gene_mean": [0.] * 5, "gene_std": [1.] * 5,
                          "wsi_mean": [0.] * 6, "wsi_std": [1.] * 6}]}


def test_off_keeps_parent_weights_rng_logits_plans_and_six_objective_behavior():
    cfg = config(reconstruction_mode="off", dropout=.15, wsi_dropout=.2)
    parent_config = DCTV22Config(**{name: value for name, value in cfg.to_dict().items()
                                  if name in DCTV22Config.__dataclass_fields__})
    torch.manual_seed(100)
    parent = DCTV22(parent_config)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(100)
    new = DCTV23(cfg)
    assert torch.equal(rng, torch.get_rng_state())
    assert dict(parent.named_parameters()).keys() == dict(new.named_parameters()).keys()
    for name, value in parent.named_parameters():
        torch.testing.assert_close(value, dict(new.named_parameters())[name], atol=0, rtol=0)
    args = inputs()
    torch.manual_seed(55)
    old_result = parent.train()(**args)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(55)
    result = new.train()(**args)
    assert torch.equal(old_rng, torch.get_rng_state())
    for old, actual in zip(old_result, result):
        torch.testing.assert_close(old, actual, atol=0, rtol=0)
    torch.testing.assert_close(parent.last_state.plans, new.last_state.plans, atol=0, rtol=0)
    assert "reconstruction" not in new.objective_weights_effective(5)


def test_enabled_decoder_initialization_keeps_parent_initialization_rng():
    torch.manual_seed(100)
    off = DCTV23(config(reconstruction_mode="off"))
    rng = torch.get_rng_state().clone()
    torch.manual_seed(100)
    on = DCTV23(config())
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in off.named_parameters():
        torch.testing.assert_close(value, dict(on.named_parameters())[name], atol=0, rtol=0)


def test_gene_schema_global_mask_hides_all_pathway_copies():
    schema = GlobalGeneSchema.build([["A", "B", "C"], ["C", "D", "E"]], (3, 3))
    assert schema.gene_ids == ("A", "B", "C", "D", "E")
    mask = torch.tensor([[False, False, True, False, False]])
    a, b = schema.expand_mask(mask)
    assert a.tolist() == [[False, False, True]]
    assert b.tolist() == [[True, False, False]]


@pytest.mark.parametrize("names", [None, [["A", "A", "C"], ["C", "D", "E"]], [["A"], ["B"]]])
def test_missing_or_invalid_actual_gene_ids_are_rejected(names):
    with pytest.raises(ValueError):
        GlobalGeneSchema.build(names, (3, 3))


def test_mismatched_duplicate_raw_gene_values_are_rejected():
    args = inputs()
    args["x_omic2"][:, 0] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        model().reconstruction_audit(**args)


def test_hidden_raw_target_never_enters_encoder_slots_cost_marginals_queries_or_predictions():
    instance = model().eval()
    args = inputs()
    mask = torch.tensor([[True, False, True, False, False], [False, False, True, False, True]])
    before = instance.reconstruction_audit(**args, global_mask=mask)
    changed = deepcopy(args)
    changed["x_omic1"][:, 2] += 100.
    changed["x_omic2"][:, 0] += 100.
    changed["x_omic1"][0, 0] -= 50.
    changed["x_omic2"][1, 2] += 75.
    after = instance.reconstruction_audit(**changed, global_mask=mask)
    for name in ("wsi_slots", "omics_slots", "plans", "row_marginals", "col_marginals", "omics_tokens"):
        torch.testing.assert_close(getattr(before["state"], name), getattr(after["state"], name), atol=0, rtol=0)
    for name in ("native_prediction", "conditional_prediction", "independent_prediction"):
        for x, y in zip(before[name], after[name]):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    torch.testing.assert_close(before["assignment"], after["assignment"], atol=0, rtol=0)
    assert not torch.equal(before["targets"][0], after["targets"][0])


def test_reconstruction_reference_is_fixed_raw_input_gene_coordinates():
    instance = model()
    instance.configure_reconstruction_reference([["A", "B", "C"], ["C", "D", "E"]],
                                                 [1., 2., 3., 4., 5.], [2., 2., 2., 2., 2.], ["p0", "p1"])
    result = instance.reconstruction_audit(**inputs(), global_mask=torch.ones(2, 5, dtype=torch.bool))
    expected = (inputs()["x_omic2"] - torch.tensor([3., 4., 5.])) / 2.
    torch.testing.assert_close(result["targets"][1], expected)
    assert not instance.reconstruction_gene_mean.requires_grad
    assert not instance.reconstruction_gene_std.requires_grad


def test_reconstruction_forward_keeps_full_input_survival_state_and_adds_absolute_error():
    instance = model().train()
    args = inputs()
    mask = torch.tensor([[True, False, True, False, False], [False, False, True, False, True]])
    torch.manual_seed(57)
    with torch.no_grad():
        baseline, _ = instance(**args, compute_aux=False)
    torch.manual_seed(57)
    logits, auxiliary = instance(**args, reconstruction_global_mask=mask)
    torch.testing.assert_close(logits, baseline, atol=0, rtol=0)
    assert instance.last_state is not instance.last_reconstruction_state
    assert not torch.equal(instance.last_state.omics_tokens, instance.last_reconstruction_state.omics_tokens)
    torch.testing.assert_close(auxiliary, instance.last_reconstruction["reconstruction_loss"] * .05)
    assert auxiliary.item() > 0
    auxiliary.backward()
    assert sum(float(parameter.grad.abs().sum()) for parameter in instance.reconstruction_decoder.parameters() if parameter.grad is not None) > 0
    assert sum(float(parameter.grad.abs().sum()) for parameter in instance.sig_networks.parameters() if parameter.grad is not None) > 0
    assert sum(float(parameter.grad.abs().sum()) for parameter in instance.core.stage_pair_cost.parameters() if parameter.grad is not None) > 0


def test_plan_sensitive_recovery_does_not_reduce_to_global_donor_pool():
    from types import SimpleNamespace
    state = SimpleNamespace(wsi_slots=torch.tensor([[[1., 0.], [0., 2.]]]), stage_weights=torch.ones(1, 1))
    p = torch.tensor([[[[[.5, 0.], [0., .5]]]]])
    q = torch.full_like(p, .25)
    assignment = torch.eye(2).unsqueeze(0)
    learned = conditional_donor_context(state, p, assignment)
    independent = conditional_donor_context(state, q, assignment)
    assert not torch.equal(learned, independent)
    torch.testing.assert_close(learned, state.wsi_slots)
    torch.testing.assert_close(conditional_donor_context(state, p * .1, assignment), learned)


def test_independent_audit_has_same_actual_rows_columns_mass_and_shared_decoder():
    result = model().reconstruction_audit(**inputs(), global_mask=torch.ones(2, 5, dtype=torch.bool))
    p, q = result["state"].plans, result["independent_plan"]
    torch.testing.assert_close(p.sum(-1), q.sum(-1))
    torch.testing.assert_close(p.sum(-2), q.sum(-2))
    torch.testing.assert_close(p.sum((-1, -2)), q.sum((-1, -2)))
    assert len(result["independent_prediction"]) == 2


def test_calibration_preserves_negative_gains_and_has_only_existing_cost_parameter_gradients():
    instance = bind(model(reconstruction_mode="gain_calibrated")).train()
    args = inputs()
    instance(**args, reconstruction_global_mask=torch.ones(2, 5, dtype=torch.bool))
    instance.zero_grad(set_to_none=True)
    loss = instance.gain_calibration_loss(instance.last_reconstruction, args["patient_ids"])
    loss.backward()
    assert instance.last_reconstruction["calibration_signed_pathway_gain"].min() < 0
    assert instance.last_reconstruction["calibration_signed_pathway_gain"].max() > 0
    cost_grads = []
    for name, parameter in instance.named_parameters():
        if name.startswith("core.stage_pair_cost."):
            cost_grads.append(parameter.grad)
        else:
            assert parameter.grad is None, name
    assert sum(float(value.abs().sum()) for value in cost_grads if value is not None) > 0


def test_negative_gain_has_larger_absolute_cost_goal_than_positive_gain():
    instance = bind(model(reconstruction_mode="gain_calibrated"), {"p0": [1., 1.], "p1": [-1., -1.]}).train()
    instance(**inputs(), reconstruction_global_mask=torch.ones(2, 5, dtype=torch.bool))
    goals = instance.last_reconstruction["calibration_cost_goal"]
    assert (goals[1] > .5).all()
    assert (goals[0] < .5).all()


def test_missing_oof_labels_or_nontraining_identity_fails_calibrated_aux_only():
    instance = model(reconstruction_mode="gain_calibrated").train()
    with pytest.raises(ValueError, match="held-patient"):
        instance(**inputs())
    bind(instance)
    args = inputs()
    args["patient_ids"] = ["validation-patient", "p1"]
    with pytest.raises(ValueError, match="training patient"):
        instance(**args)
    instance.eval()
    instance(**args)


def test_calibration_warmup_is_explicit_and_does_not_lookup_gains():
    instance = model(reconstruction_mode="gain_calibrated", calibration_warmup_epochs=10).train()
    instance(**inputs())
    assert instance.last_training_losses["gain_calibration_effective_weight"].item() == 0


def test_prediction_never_samples_masks_reads_gain_table_or_runs_decoder():
    instance = model(reconstruction_mode="gain_calibrated").eval()
    instance._gain_targets = {"sentinel": object()}
    seen = []
    hook = instance.reconstruction_decoder.register_forward_hook(lambda *args: seen.append(True))
    args = inputs()
    args.pop("patient_ids")
    rng = torch.get_rng_state().clone()
    logits, auxiliary = instance(**args)
    hook.remove()
    assert not seen and auxiliary.item() == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.isfinite(logits).all()


def test_checkpoint_restores_fixed_training_reference_and_signed_oof_targets():
    instance = bind(model(reconstruction_mode="gain_calibrated")).eval()
    state = deepcopy(instance.state_dict())
    restored = DCTV23(instance.config)
    restored.load_state_dict(state)
    restored.eval()
    assert restored.gene_schema == instance.gene_schema
    assert restored._gain_targets == instance._gain_targets
    torch.testing.assert_close(restored.reconstruction_gene_std, instance.reconstruction_gene_std)
    a, _ = instance(**inputs())
    b, _ = restored(**inputs())
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    bad = deepcopy(state)
    bad["_extra_state"]["schema"] = "dct_v22_standalone_v1"
    with pytest.raises(ValueError, match="own checkpoint"):
        restored.load_state_dict(bad)


def test_checkpoint_loss_metadata_matches_active_mode_and_zero_variance_reference_is_safe():
    # This fixture explicitly sets all five inherited auxiliary weights to 0.
    assert model(reconstruction_mode="off").get_extra_state()["losses"] == ["global_nll"]
    assert model().get_extra_state()["losses"] == ["global_nll", "reconstruction"]
    assert model(reconstruction_mode="gain_calibrated").get_extra_state()["losses"] == [
        "global_nll", "reconstruction", "gain_calibration"]
    instance = model()
    instance.configure_reconstruction_reference([["A", "B", "C"], ["C", "D", "E"]],
                                                 [0.] * 5, [0., 1., 1., 1., 1.], ["p0", "p1"])
    assert instance.reconstruction_gene_std[0].item() == 1.


def test_reference_normalization_order_and_train_patient_ids_are_checked():
    instance = model()
    with pytest.raises(ValueError, match="order"):
        instance.configure_reconstruction_reference([["A", "B", "C"], ["C", "D", "E"]], [0.] * 5, [1.] * 5,
                                                    ["p0", "p1"], gene_ids=["E", "D", "C", "B", "A"])
    with pytest.raises(ValueError, match="patient"):
        instance.configure_reconstruction_reference([["A", "B", "C"], ["C", "D", "E"]], [0.] * 5, [1.] * 5, ["p0", "p0"])


@pytest.mark.parametrize("attack", ["fit_query_overlap", "normalizer_query", "wrong_genes", "wrong_pathways",
                                    "missing_witnesses", "duplicate_query", "bad_source_hash", "uncovered_genes"])
def test_calibration_rejects_false_oof_flags_without_valid_identity_and_coordinate_witnesses(attack):
    witness = provenance()
    if attack == "fit_query_overlap":
        witness["teachers"][0]["fit_patient_ids"] = ["p0", "p1"]
    elif attack == "normalizer_query":
        witness["teachers"][0]["normalization_fit_patient_ids"] = ["p0", "p1"]
    elif attack == "wrong_genes":
        witness["gene_ids"].reverse()
    elif attack == "wrong_pathways":
        witness["pathway_gene_ids"].reverse()
    elif attack == "missing_witnesses":
        witness.pop("teachers")
    elif attack == "duplicate_query":
        witness["teachers"][1]["score_patient_ids"] = ["p0"]
        witness["teachers"][1]["fit_patient_ids"] = ["p1"]
    elif attack == "bad_source_hash":
        witness["raw_expression_sha256"] = "allegedly-audited"
    elif attack == "uncovered_genes":
        witness["global_masks"] = [[1, 0, 0, 0, 0]]
    with pytest.raises(ValueError):
        model(reconstruction_mode="gain_calibrated").bind_gain_targets({"p0": [1., -1.], "p1": [-.5, .5]}, witness)


@pytest.mark.parametrize("change", [{"mask_ratio": 0.}, {"gain_scale": 0.}, {"reconstruction_mode": "magic"},
                                   {"recon_hidden_dim": 0}, {"calibration_weight": float("nan")}])
def test_config_rejects_invalid_new_fields(change):
    with pytest.raises(ValueError):
        config(**change)


def test_cpu_autocast_recovery_calibration_and_backward_are_finite():
    instance = bind(model(reconstruction_mode="gain_calibrated")).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = instance.forward_with_losses(**inputs())
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in instance.parameters() if parameter.grad is not None)


def test_random_mask_guarantees_hidden_and_visible_unique_gene_each_patient():
    mask = sample_global_mask(200, 2, .99, torch.device("cpu"))
    assert mask.any(-1).all() and (~mask).any(-1).all()


def test_missing_modality_does_not_reconstruct_nan_targets_or_calibrate_costs():
    instance = bind(model(reconstruction_mode="gain_calibrated")).train()
    args = inputs()
    args["omics_mask"][1] = False
    args["x_omic1"][1] = float("nan")
    args["x_omic2"][1] = float("nan")
    result = instance.forward_with_losses(**args)
    assert torch.isfinite(result["loss"])
    assert (instance.last_reconstruction["counts"][1] == 0).all()
