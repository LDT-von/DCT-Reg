"""Temporary synthetic fixtures test evidence guards; no model scores are used."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import random
import tempfile
import unittest
import warnings
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "summarize.py"
SPEC = importlib.util.spec_from_file_location("e048_summary", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def dump_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class SyntheticFixture:
    """A fully local fake batch, expressly unrelated to real training evidence."""

    def __init__(self, root, arms=("full", "no_self"), protocols=("legacy_val",)):
        self.root = Path(root)
        self.source = self.root / "fixture_source.txt"
        self.source.write_text("SYNTHETIC TEST FIXTURE ONLY", encoding="utf-8")
        self.tasks = []
        for protocol in protocols:
            for fold in range(5):
                split = self.root / f"fixture_split_{protocol}_{fold}.csv"
                if protocol == "legacy_val":
                    split.write_text(f"train,val\nT{fold}a,P{fold}a\nT{fold}b,P{fold}b\nT{fold}c,P{fold}c\n", encoding="utf-8")
                    partition = "val"
                else:
                    split.write_text(f"inner_train,inner_val,outer_test\nT{fold}a,V{fold}a,P{fold}a\nT{fold}b,V{fold}b,P{fold}b\nT{fold}c,V{fold}c,P{fold}c\n", encoding="utf-8")
                    partition = "outer_test"
                for arm in arms:
                    task_id = f"fixture_{protocol}_{arm}_{fold}"
                    task_dir = self.root / task_id
                    task_dir.mkdir()
                    config = task_dir / "fixture_config.json"
                    dump_json(config, {"synthetic_fixture": True, "arm": arm})
                    predictions = {
                        f"P{fold}a": {"risk": 3.0, "time": 1.0, "censor": 0, "logits": [1]},
                        f"P{fold}b": {"risk": 2.0 if arm == "full" else 4.0, "time": 2.0, "censor": 0, "logits": [1]},
                        f"P{fold}c": {"risk": 1.0, "time": 3.0, "censor": 1, "logits": [1]},
                    }
                    prediction = task_dir / (f"split_{fold}_results.pkl" if partition == "val" else "outer_test_predictions.pkl")
                    prediction.write_bytes(pickle.dumps(predictions))
                    audit = task_dir / "fixture_actual_loader_labels.json"
                    dump_json(audit, {"partition": partition, "patients": {patient: {key: row[key] for key in ("time", "censor")} for patient, row in predictions.items()}})
                    checkpoint = task_dir / "fixture_checkpoint.pth"
                    checkpoint.write_bytes(b"SYNTHETIC TEST FIXTURE CHECKPOINT")
                    task = {
                        "task_id": task_id, "output_dir": str(task_dir), "study": "SYNTHETIC_FIXTURE",
                        "arm": arm, "fold": fold, "seed": 3, "protocol": protocol,
                        "config_file": self.reference(config), "split_file": self.reference(split),
                        "source_files": [self.reference(self.source)],
                        "recipe": {"data": {"fixture": True}, "train": {"max_epochs": 10}},
                    }
                    self.tasks.append(task)
                    dump_json(task_dir / "task.json", task)
                    receipt = {
                        "status": "complete", "task_id": task_id, "protocol": protocol, "metric": "cindex",
                        "training_complete": True, "completed_epochs": 10, "max_epochs": 10,
                        "evaluation_partition": partition, "prediction_patient_count": 3,
                        "score": summary.compute_cindex(predictions),
                        "prediction_artifact": {**self.reference(prediction), "role": "best_val" if partition == "val" else "outer_test"},
                        "label_audit_artifact": self.reference(audit), "checkpoint_artifact": self.reference(checkpoint),
                        "artifacts": [self.reference(path) for path in (prediction, audit, checkpoint)],
                    }
                    dump_json(task_dir / "completion.json", receipt)
        dump_json(self.root / "plan.json", {"tasks": self.tasks, "arms": list(arms), "folds": list(range(5)), "synthetic_fixture": True})

    @staticmethod
    def reference(path):
        return {"path": str(path), "sha256": summary.sha256_file(path)}

    def modify_completion(self, index, **changes):
        path = Path(self.tasks[index]["output_dir"]) / "completion.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt.update(changes)
        dump_json(path, receipt)


class CIndexFixtureTests(unittest.TestCase):
    def test_perfect_and_reversed(self):
        rows = {"a": {"risk": 2.0, "time": 1, "censor": 0}, "b": {"risk": 1.0, "time": 2, "censor": 0}}
        self.assertEqual(summary.compute_cindex(rows), 1.0)
        rows["a"]["risk"] = 0.0
        self.assertEqual(summary.compute_cindex(rows), 0.0)

    def test_risk_ties_have_half_credit(self):
        rows = {"a": {"risk": 1.0, "time": 1, "censor": 0}, "b": {"risk": 1.0 + 1e-9, "time": 2, "censor": 0}}
        self.assertEqual(summary.compute_cindex(rows), 0.5)

    def test_event_censor_same_time_compared_event_event_excluded(self):
        rows = {"a": {"risk": 3, "time": 1, "censor": 0}, "b": {"risk": 1, "time": 1, "censor": 1}, "c": {"risk": 2, "time": 1, "censor": 0}}
        self.assertEqual(summary.compute_cindex(rows), 1.0)

    def test_no_comparable_pairs_rejected(self):
        with self.assertRaises(summary.EvidenceError):
            summary.compute_cindex({"a": {"risk": 1, "time": 1, "censor": 1}})

    def test_non_finite_labels_risks_and_fractional_censor_rejected(self):
        for key, value in (("risk", float("nan")), ("time", float("inf")), ("censor", 0.2), ("time", -1)):
            rows = {"a": {"risk": 1, "time": 1, "censor": 0}, "b": {"risk": 0, "time": 2, "censor": 1}}
            rows["a"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(summary.EvidenceError):
                summary.compute_cindex(rows)

    def test_numpy_trainer_pickle_supported(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable; primitive fixtures still checked")
        with tempfile.TemporaryDirectory(prefix="e048_synthetic_fixture_") as folder:
            path = Path(folder) / "fixture.pkl"
            path.write_bytes(pickle.dumps({"a": {"risk": np.float32(1), "time": np.float64(1), "censor": np.array([0.0]), "logits": np.array([1, 2])}}))
            self.assertEqual(summary.read_predictions(path)["a"]["risk"], 1.0)

    def test_arbitrary_pickle_global_rejected(self):
        class Untrusted:
            def __reduce__(self):
                return (os.getcwd, ())
        with tempfile.TemporaryDirectory(prefix="e048_synthetic_fixture_") as folder:
            path = Path(folder) / "fixture.pkl"
            path.write_bytes(pickle.dumps(Untrusted()))
            with self.assertRaises(summary.EvidenceError):
                summary.read_predictions(path)

    def test_matches_sksurv_on_randomized_censor_and_tie_fixtures(self):
        try:
            import numpy as np
            from sksurv.metrics import concordance_index_censored
            from sksurv.exceptions import NoComparablePairException
        except ImportError:
            self.skipTest("scikit-survival unavailable; analytic fixture cases still checked")
        rng = random.Random(48)
        for fixture_index in range(200):
            # Integer times/risks intentionally yield event-time and risk ties.
            rows = {str(index): {"risk": float(rng.randrange(8)), "time": float(rng.randrange(1, 8)), "censor": rng.randrange(2)} for index in range(rng.randrange(2, 20))}
            ordered = list(rows.values())
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    expected = concordance_index_censored(
                        np.asarray([row["censor"] == 0 for row in ordered], dtype=bool),
                        np.asarray([row["time"] for row in ordered]),
                        np.asarray([row["risk"] for row in ordered]), tied_tol=1e-8,
                    )[0]
            except (NoComparablePairException, ValueError):
                with self.subTest(fixture_index=fixture_index), self.assertRaises(summary.EvidenceError):
                    summary.compute_cindex(rows)
            else:
                with self.subTest(fixture_index=fixture_index):
                    if math.isfinite(expected):
                        self.assertAlmostEqual(summary.compute_cindex(rows), expected, places=14)
                    else:
                        # Some sksurv versions return NaN for degenerate event
                        # ties. E048 deliberately rejects undefined scores.
                        with self.assertRaises(summary.EvidenceError):
                            summary.compute_cindex(rows)


class BatchFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="e048_synthetic_fixture_")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_batch_recomputed_paired_and_sample_std(self):
        SyntheticFixture(self.root)
        report = summary.summarize_batch(self.root)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["fold_results"]), 10)
        full = next(row for row in report["five_fold_summary"] if row["arm"] == "full")
        self.assertEqual(full["cindex_mean"], 1.0)
        self.assertEqual(full["cindex_sample_std"], 0.0)
        self.assertAlmostEqual(report["paired_five_fold_summary"][0]["mean_full_minus_control"], 1 / 3)

    def test_missing_fold_has_no_partial_mean(self):
        fixture = SyntheticFixture(self.root)
        (Path(fixture.tasks[0]["output_dir"]) / "completion.json").unlink()
        report = summary.summarize_batch(self.root)
        self.assertEqual(report["status"], "incomplete_or_invalid")
        full = next(row for row in report["five_fold_summary"] if row["arm"] == "full")
        self.assertIsNone(full["cindex_mean"])
        self.assertEqual(full["missing_folds"], [0])
        control = next(row for row in report["five_fold_summary"] if row["arm"] == "no_self")
        self.assertIsNone(control["cindex_mean"])
        self.assertIsNone(report["paired_five_fold_summary"][0]["mean_full_minus_control"])

    def test_one_fold_plan_reports_missing_other_folds(self):
        fixture = SyntheticFixture(self.root)
        dump_json(self.root / "plan.json", {"tasks": fixture.tasks[:2], "arms": ["full", "no_self"], "folds": [0]})
        report = summary.summarize_batch(self.root)
        self.assertEqual(report["status"], "incomplete_or_invalid")
        for row in report["five_fold_summary"]:
            self.assertIsNone(row["cindex_mean"])
            self.assertEqual(row["missing_folds"], [1, 2, 3, 4])

    def test_declared_study_entirely_missing_is_not_hidden(self):
        fixture = SyntheticFixture(self.root)
        dump_json(self.root / "plan.json", {"tasks": fixture.tasks, "selected_arms": ["full", "no_self"], "studies": ["SYNTHETIC_FIXTURE", "MISSING_FIXTURE"], "seeds": [3]})
        report = summary.summarize_batch(self.root)
        missing = [row for row in report["five_fold_summary"] if row["study"] == "MISSING_FIXTURE"]
        self.assertEqual(len(missing), 2)
        self.assertTrue(all(row["missing_folds"] == list(range(5)) and row["cindex_mean"] is None for row in missing))

    def test_repeating_evaluation_patients_across_folds_rejected(self):
        fixture = SyntheticFixture(self.root)
        # Move the two fold-1 controls onto fold-0 patient IDs while keeping
        # their internal artifacts, score and split receipts mutually aligned.
        for task in fixture.tasks[2:4]:
            task_dir = Path(task["output_dir"])
            task_path = task_dir / "task.json"
            final_task = json.loads(task_path.read_text(encoding="utf-8"))
            split_path = Path(final_task["split_file"]["path"])
            split_path.write_text("train,val\nT1a,P0a\nT1b,P0b\nT1c,P0c\n", encoding="utf-8")
            final_task["split_file"] = fixture.reference(split_path)
            dump_json(task_path, final_task)
            prediction = task_dir / "split_1_results.pkl"
            rows = pickle.loads(prediction.read_bytes())
            prediction.write_bytes(pickle.dumps({patient.replace("P1", "P0"): row for patient, row in rows.items()}))
            audit_path = task_dir / "fixture_actual_loader_labels.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["patients"] = {patient.replace("P1", "P0"): row for patient, row in audit["patients"].items()}
            dump_json(audit_path, audit)
            completion = json.loads((task_dir / "completion.json").read_text(encoding="utf-8"))
            completion["prediction_artifact"] = {**fixture.reference(prediction), "role": "best_val"}
            completion["label_audit_artifact"] = fixture.reference(audit_path)
            completion["artifacts"] = [fixture.reference(Path(item["path"])) for item in completion["artifacts"]]
            dump_json(task_dir / "completion.json", completion)
        report = summary.summarize_batch(self.root)
        self.assertTrue(any("overlap across folds" in row["error"] for row in report["errors"]))
        self.assertTrue(all(row["cindex_mean"] is None for row in report["five_fold_summary"]))

    def test_interrupted_and_short_training_rejected(self):
        fixture = SyntheticFixture(self.root)
        fixture.modify_completion(0, status="interrupted")
        fixture.modify_completion(1, completed_epochs=4)
        report = summary.summarize_batch(self.root)
        self.assertEqual(report["status"], "incomplete_or_invalid")
        self.assertTrue(any("complete training" in row["error"] for row in report["errors"]))
        self.assertTrue(any("before max_epochs" in row["error"] for row in report["errors"]))

    def test_prediction_hash_tampering_rejected(self):
        fixture = SyntheticFixture(self.root)
        task_dir = Path(fixture.tasks[0]["output_dir"])
        with (task_dir / "split_0_results.pkl").open("ab") as stream:
            stream.write(b"tamper")
        report = summary.summarize_batch(self.root)
        self.assertTrue(any("SHA-256 changed" in row["error"] for row in report["errors"]))

    def test_reported_score_not_trusted(self):
        fixture = SyntheticFixture(self.root)
        fixture.modify_completion(0, score=0.123456)
        report = summary.summarize_batch(self.root)
        self.assertTrue(any("differs from recomputed" in row["error"] for row in report["errors"]))

    def test_actual_loader_label_mismatch_rejected(self):
        fixture = SyntheticFixture(self.root)
        task_dir = Path(fixture.tasks[0]["output_dir"])
        audit_path = task_dir / "fixture_actual_loader_labels.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["patients"]["P0a"]["time"] = 9.0
        dump_json(audit_path, audit)
        completion = json.loads((task_dir / "completion.json").read_text(encoding="utf-8"))
        completion["label_audit_artifact"] = fixture.reference(audit_path)
        completion["artifacts"] = [fixture.reference(Path(item["path"])) for item in completion["artifacts"]]
        dump_json(task_dir / "completion.json", completion)
        report = summary.summarize_batch(self.root)
        self.assertTrue(any("actual loader" in row["error"] for row in report["errors"]))

    def test_common_recipe_mismatch_invalidates_entire_cohort(self):
        fixture = SyntheticFixture(self.root)
        task_path = Path(fixture.tasks[1]["output_dir"]) / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["recipe"]["train"]["max_epochs"] = 20
        dump_json(task_path, task)
        report = summary.summarize_batch(self.root)
        self.assertFalse(any(row["fold"] == 0 for row in report["fold_results"]))
        self.assertTrue(any("common data/training" in row["error"] for row in report["errors"]))

    def test_warmup_and_topk_recipe_changes_refuse_matched_means(self):
        for option, original, changed in (("recon_warmup_epochs", 5, 0), ("topk", 2, 4)):
            with self.subTest(option=option):
                case_dir = self.root / option
                case_dir.mkdir()
                fixture = SyntheticFixture(case_dir)
                for task in fixture.tasks:
                    path = Path(task["output_dir"]) / "task.json"
                    final_task = json.loads(path.read_text(encoding="utf-8"))
                    final_task["recipe"]["train"][option] = original
                    dump_json(path, final_task)
                path = Path(fixture.tasks[1]["output_dir"]) / "task.json"
                altered = json.loads(path.read_text(encoding="utf-8"))
                altered["recipe"]["train"][option] = changed
                dump_json(path, altered)
                report = summary.summarize_batch(case_dir)
                self.assertTrue(any("common data/training" in row["error"] for row in report["errors"]))
                self.assertTrue(all(row["cindex_mean"] is None for row in report["five_fold_summary"]))
                self.assertTrue(all(row["mean_full_minus_control"] is None for row in report["paired_five_fold_summary"]))

    def test_split_missing_patient_and_overlap_rejected(self):
        fixture = SyntheticFixture(self.root)
        split_path = Path(fixture.tasks[0]["split_file"]["path"])
        predictions = summary.read_predictions(Path(fixture.tasks[0]["output_dir"]) / "split_0_results.pkl")
        audit = {"partition": "val", "patients": {patient: {"time": row["time"], "censor": row["censor"]} for patient, row in predictions.items()}}
        split_path.write_text("train,val\nP0a,P0a\nT0b,P0b\nT0c,P0c\n", encoding="utf-8")
        with self.assertRaisesRegex(summary.EvidenceError, "overlap"):
            summary.verify_prediction_labels(predictions, audit, split_path, "val")
        split_path.write_text("train,val\nT0a,P0a\nT0b,P0b\n", encoding="utf-8")
        with self.assertRaisesRegex(summary.EvidenceError, "exactly cover"):
            summary.verify_prediction_labels(predictions, audit, split_path, "val")

    def test_final_predictions_rejected(self):
        fixture = SyntheticFixture(self.root)
        task_dir = Path(fixture.tasks[0]["output_dir"])
        original = task_dir / "split_0_results.pkl"
        final = task_dir / "split_0_results_final.pkl"
        original.rename(final)
        completion = json.loads((task_dir / "completion.json").read_text(encoding="utf-8"))
        completion["prediction_artifact"] = {**fixture.reference(final), "role": "best_val"}
        completion["artifacts"][0] = fixture.reference(final)
        dump_json(task_dir / "completion.json", completion)
        report = summary.summarize_batch(self.root)
        self.assertTrue(any("actual best-epoch" in row["error"] for row in report["errors"]))

    def test_protocols_separate_and_wrong_partition_rejected(self):
        fixture = SyntheticFixture(self.root, protocols=("legacy_val", "outer_test"))
        report = summary.summarize_batch(self.root)
        self.assertEqual(report["status"], "complete")
        self.assertEqual({row["protocol"] for row in report["five_fold_summary"]}, {"legacy_val", "outer_test"})
        fixture.modify_completion(10, evaluation_partition="val")
        report = summary.summarize_batch(self.root)
        self.assertTrue(any("incompatible with protocol" in row["error"] for row in report["errors"]))

    def test_duplicate_tasks_and_path_escape_rejected(self):
        fixture = SyntheticFixture(self.root)
        plan = json.loads((self.root / "plan.json").read_text(encoding="utf-8"))
        plan["tasks"].append(plan["tasks"][0])
        dump_json(self.root / "plan.json", plan)
        with self.assertRaisesRegex(summary.EvidenceError, "Duplicate"):
            summary.summarize_batch(self.root)
        plan["tasks"].pop()
        plan["tasks"][0]["output_dir"] = str(self.root.parent / "outside_fixture")
        dump_json(self.root / "plan.json", plan)
        with self.assertRaisesRegex(summary.EvidenceError, "escapes"):
            summary.summarize_batch(self.root)

    def test_new_report_directory_only_and_no_significance_claim(self):
        SyntheticFixture(self.root)
        report = summary.summarize_batch(self.root)
        output = self.root / "new_fixture_report"
        summary.write_report(report, output)
        self.assertTrue((output / "report.json").is_file())
        self.assertTrue((output / "five_fold_summary.csv").is_file())
        self.assertIn("种子与折不当作独立样本", (output / "README.md").read_text(encoding="utf-8"))
        self.assertNotIn("p_value", json.dumps(report))
        with self.assertRaises(FileExistsError):
            summary.write_report(report, output)


if __name__ == "__main__":
    unittest.main()
