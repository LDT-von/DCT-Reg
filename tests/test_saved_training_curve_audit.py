"""Pure saved-CSV arithmetic checks; no models or patient data are loaded."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("saved_curve_audit", ROOT / "scripts/audits/audit_saved_training_curves.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def saved(tmp_path, contents, name="summary.csv"):
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")
    return path


def test_population_std_rejects_inconsistent_report_and_keeps_hash(tmp_path):
    source = saved(tmp_path, "epoch,fold0,fold1,mean,std\n0,0.6000,0.8000,0.7000,0.3000\n")
    original = source.read_bytes()
    report = AUDIT.audit_curves(summary_csv=source)
    assert report["aggregate_validation"]["inconsistent_count"] == 1
    issue = report["aggregate_validation"]["inconsistencies"][0]
    assert issue["field"] == "std"
    assert issue["recomputed"] == pytest.approx(0.1)
    assert report["sources"][0]["sha256"] == hashlib.sha256(original).hexdigest()
    assert source.read_bytes() == original


def test_rounding_precision_tolerance_accepts_four_decimal_csv(tmp_path):
    source = saved(tmp_path, "epoch,fold0,fold1,mean,std\n0,0.6032,0.6335,0.6184,0.0152\n")
    report = AUDIT.audit_curves(summary_csv=source)
    assert report["aggregate_validation"]["inconsistent_count"] == 0
    assert report["aggregate_validation"]["checks"][0]["tolerance"] == pytest.approx(0.0001000001)


def test_standard_deviation_convention_is_explicit(tmp_path):
    source = saved(tmp_path, "epoch,fold0,fold1,mean,std\n0,0.6000,0.8000,0.7000,0.1414\n")
    report = AUDIT.audit_curves(summary_csv=source, ddof=1)
    assert report["std_convention"]["label"] == "sample"
    assert report["corrected_aggregates_from_available_folds"][0]["std"] == pytest.approx(0.141421356)
    assert report["aggregate_validation"]["inconsistent_count"] == 0
    assert AUDIT.audit_curves(summary_csv=source, ddof=0)["aggregate_validation"]["inconsistent_count"] == 1


def test_summary_does_not_invent_training_or_mechanism_evidence(tmp_path):
    source = saved(tmp_path, "epoch,fold0,fold1,mean,std\n0,0.6000,0.8000,0.7000,0.1000\n")
    report = AUDIT.audit_curves(summary_csv=source)
    assert report["source_kind"] == "validation_cindex_summary_only"
    assert "training_prediction_loss" in report["missing_evidence"]
    assert "direct_slot_distance" in report["missing_evidence"]
    assert "effective_cost_gradient" in report["missing_evidence"]
    assert report["mechanistic_cause_established"] is False
    assert all("train_loss" not in data["available_columns"] for data in report["per_fold"].values())


def test_fold2_earlystop_cannot_see_epoch9_future_best():
    # E050 fold2 rounded scores, provided directly so the check is portable.
    scores = [0.5996, 0.6894, 0.5912, 0.6548, 0.6284, 0.6193, 0.6024, 0.6224, 0.6212, 0.6932]
    rows = [{"epoch": epoch, "val_cindex": value} for epoch, value in enumerate(scores)]
    result = AUDIT.simulate_early_stopping(rows, patience=5)
    assert result["stopped_early"] is True
    assert result["stopped_epoch"] == 6
    assert result["checkpoint_best_epoch"] == 1
    assert result["checkpoint_score"] == 0.6894
    assert len(result["trace"]) == 7
    assert all(row["epoch"] <= 6 for row in result["trace"])


def test_warmup_and_min_delta_match_controller_but_checkpoint_strict_ties():
    rows = [{"epoch": i, "val_cindex": v} for i, v in enumerate([0.9, 0.6, 0.7, 0.705, 0.705, 0.72])]
    result = AUDIT.simulate_early_stopping(rows, patience=2, warmup=2, min_delta=0.01)
    assert result["stopped_epoch"] == 4
    assert result["controller_best_epoch"] == 2
    assert result["checkpoint_best_epoch"] == 0
    assert result["checkpoint_score"] == 0.9
    assert result["trace"][0]["bad_epochs"] == 0
    tied = AUDIT.simulate_early_stopping([{"epoch": i, "val_cindex": v} for i, v in enumerate([0.5, 0.7, 0.7])], patience=0)
    assert tied["checkpoint_best_epoch"] == 1


def test_global_mean_peak_and_separate_fold_best_are_distinct(tmp_path):
    source = saved(tmp_path, "epoch,fold0,fold1,mean,std\n0,0.9000,0.4000,0.6500,0.2500\n1,0.5000,0.8000,0.6500,0.1500\n")
    report = AUDIT.audit_curves(summary_csv=source)
    assert report["global_mean_peak_complete_coverage"] == {"epoch": 0, "score": 0.65}
    assert report["independently_selected_per_fold_best"]["mean"] == pytest.approx(0.85)
    assert report["per_fold"]["fold0"]["retrospective_selected_best"]["epoch"] == 0
    assert report["per_fold"]["fold1"]["retrospective_selected_best"]["epoch"] == 1


def test_per_fold_columns_preserved_and_signed_gap_not_patient_absolute_gap(tmp_path):
    path = saved(tmp_path, "epoch,val_cindex,train_global_nll,train_reference_deficit,train_relation_probe_nll_gap,train_relation_weighted,comment\n"
                 "0,0.6,0.8,0.2,-0.000000001,0.0011,start\n1,0.65,0.4,0.3,0.000000001,0.0010,next\n", "epoch_curve_fold0.csv")
    report = AUDIT.audit_curves(fold_csvs=[path])
    rows = report["per_fold"]["fold0"]["rows"]
    assert rows[0]["train_reference_deficit"] == 0.2
    assert rows[0]["comment"] == "start"
    assert report["evidence_availability"]["training_prediction_loss"]["status"] == "recorded"
    assert report["evidence_availability"]["weighted_relation_contribution"]["status"] == "recorded"
    assert "mean_absolute_patient_relation_gap" in report["missing_evidence"]
    gap = report["signed_relation_gap_summary"][0]
    assert gap["minimum_signed_epoch_mean"] == -1e-9
    assert gap["near_zero_epoch_means_at_1e_8"] == 2
    assert gap["mean_absolute_patient_gap_recoverable"] is False


@pytest.mark.parametrize("contents, match", [
    ("epoch,fold0\n0,0.5\n0,0.6\n", "Duplicate epochs"),
    ("epoch,fold0\n1,0.5\n0,0.6\n", "increasing"),
    ("epoch,fold0\n0.5,0.5\n", "nonnegative integer"),
    ("epoch,fold0,fold0\n0,0.5,0.6\n", "Duplicate column"),
    ("epoch,fold0\n0,2.0\n", "out of"),
])
def test_bad_inputs_rejected(tmp_path, contents, match):
    source = saved(tmp_path, contents)
    with pytest.raises(ValueError, match=match):
        AUDIT.audit_curves(summary_csv=source)


def test_missing_original_metric_rejected(tmp_path):
    source = saved(tmp_path, "epoch,train_loss\n0,0.5\n", "epoch_curve_fold0.csv")
    with pytest.raises(ValueError, match="lacks selected metric"):
        AUDIT.audit_curves(fold_csvs=[source])


def test_missing_and_nonfinite_observations_mark_simulation_unavailable(tmp_path):
    source = saved(tmp_path, "epoch,fold0,fold1,mean,std\n0,0.5,0.6,0.55,0.05\n1,nan,0.7,,\n")
    report = AUDIT.audit_curves(summary_csv=source)
    assert report["per_fold"]["fold0"]["early_stopping"]["status"] == "unavailable"
    assert report["coverage_warnings"] == ["epoch 1: only 1/2 folds available"]
    assert report["global_mean_peak_complete_coverage"]["epoch"] == 0


def test_epoch_gap_disables_patience_simulation():
    result = AUDIT.simulate_early_stopping([{"epoch": 0, "val_cindex": 0.5}, {"epoch": 2, "val_cindex": 0.4}])
    assert result["status"] == "unavailable"


def test_fresh_outputs_only_and_source_never_changes(tmp_path):
    source = saved(tmp_path, "epoch,fold0,mean,std\n0,0.6000,0.6000,0.0000\n")
    original = source.read_bytes()
    report = AUDIT.audit_curves(summary_csv=source)
    target = tmp_path / "fresh"
    outputs = AUDIT.write_outputs(report, target, corrected_csv=True)
    assert json.loads(Path(outputs["report"]).read_text(encoding="utf-8"))["mechanistic_cause_established"] is False
    assert Path(outputs["corrected_aggregate_csv"]).is_file()
    with pytest.raises(FileExistsError):
        AUDIT.write_outputs(report, target)
    assert source.read_bytes() == original


def test_cli_original_directory_and_minimization(tmp_path):
    source_dir = tmp_path / "originals"
    source_dir.mkdir()
    saved(source_dir, "epoch,val_loss,train_global_nll\n0,0.8,0.9\n1,0.5,0.4\n2,0.6,0.2\n", "epoch_curve_fold0.csv")
    output_dir = tmp_path / "new_audit"
    assert AUDIT.main(["--fold-dir", str(source_dir), "--metric", "val_loss", "--mode", "min", "--patience", "1", "--output-dir", str(output_dir)]) == 0
    report = json.loads((output_dir / "saved_curve_audit.json").read_text(encoding="utf-8"))
    assert report["per_fold"]["fold0"]["early_stopping"]["checkpoint_best_epoch"] == 1
    assert report["per_fold"]["fold0"]["early_stopping"]["stopped_epoch"] == 2


def test_abs_gap_does_not_count_as_signed_gap(tmp_path):
    source = saved(tmp_path, "epoch,val_cindex,train_relation_probe_nll_gap_abs\n0,0.5,0.01\n", "epoch_curve_fold0.csv")
    report = AUDIT.audit_curves(fold_csvs=[source])
    assert report["evidence_availability"]["signed_relation_gap"]["status"] == "missing"
    assert report["evidence_availability"]["mean_absolute_patient_relation_gap"]["status"] == "recorded"
    assert report["signed_relation_gap_summary"] == []


def test_actual_v22_diagnostic_names_are_recognized_without_filling_missing_fields(tmp_path):
    columns = [
        "epoch", "val_cindex", "train_optimization_main_nll", "train_eval_main_nll",
        "train_eval_cindex", "val_main_nll", "train_wsi_slot_pair_dist_mean",
        "train_omics_slot_pair_dist_min", "train_wsi_slot_cosine_mean",
        "train_omics_slot_cosine_mean", "train_wsi_risk_response_variance",
        "train_omics_risk_response_variance", "train_plan_independence_l1_normalized",
        "train_transport_mass_mean", "train_relation_probe_nll_gap",
        "train_relation_probe_nll_gap_abs", "train_relation_cost_grad_norm",
    ]
    source = saved(tmp_path, ",".join(columns) + "\n" + ",".join(["0"] + ["0.2"] * (len(columns) - 1)) + "\n",
                   "epoch_curve_fold0.csv")
    report = AUDIT.audit_curves(fold_csvs=[source])
    for group in ("training_prediction_loss", "training_cindex", "validation_prediction_loss",
                  "direct_slot_distance", "slot_cosine_similarity", "slot_risk_response_variance",
                  "plan_distance_from_independent", "actual_transport_mass", "signed_relation_gap",
                  "mean_absolute_patient_relation_gap", "effective_cost_gradient"):
        assert report["evidence_availability"][group]["status"] == "recorded"
    assert report["evidence_availability"]["deployed_transport_gain"]["status"] == "missing"
    assert report["evidence_availability"]["local_readout_response"]["status"] == "missing"
    assert report["evidence_availability"]["weighted_relation_contribution"]["status"] == "missing"
    assert report["mechanistic_cause_established"] is False


def test_reference_deficit_never_counts_as_prediction_loss(tmp_path):
    source = saved(tmp_path, "epoch,val_cindex,train_reference_deficit\n0,0.6,0.3\n",
                   "epoch_curve_fold0.csv")
    report = AUDIT.audit_curves(fold_csvs=[source])
    assert "training_prediction_loss" in report["missing_evidence"]
    assert "training_cindex" in report["missing_evidence"]
    assert report["per_fold"]["fold0"]["rows"][0]["train_reference_deficit"] == 0.3
