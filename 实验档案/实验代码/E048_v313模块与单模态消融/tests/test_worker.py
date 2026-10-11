"""Worker contract tests with temporary labels and a fully mocked runner.

These checks never instantiate a model, load feature bags, train, or infer.
Synthetic prediction/checkpoint fixtures verify rejection and receipt logic;
they are not experiment results.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import pickle
import signal
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import yaml


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import worker
import summarize


class FixtureLoader:
    def __init__(self, dataset, batches=None):
        self.dataset = dataset
        self.batches = [object(), object()] if batches is None else batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e048_worker_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.factory = SimpleNamespace(label_col="survival", censorship_var="censorship")

    def options(self, **changes):
        values = dict(arm="full", config=str(HERE / "configs" / "base.yaml"),
                      output_dir=str(self.root / "task"), protocol="legacy_val",
                      fold=0, seed=3, gpu="0")
        values.update(changes)
        return argparse.Namespace(**values)

    def dataset(self, identifiers):
        return SimpleNamespace(label_df=pd.DataFrame({
            "case id": list(identifiers), "survival": [1.0, 2.0][:len(identifiers)],
            "censorship": [0.0, 0.0][:len(identifiers)],
        }))

    def split(self, protocol="legacy_val"):
        path = self.root / "data" / "splits" / "fixed" / "blca" / "fold_0.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = ("train,val\nP1,P3\nP2,P4\n" if protocol == "legacy_val" else
                    "inner_train,inner_val,outer_test\nP1,P3,P5\nP2,P4,P6\n")
        path.write_text(contents, encoding="utf-8")
        return path

    def test_base_config_uses_real_parser_and_retains_managed_flags(self):
        options = self.options(fold=2, seed=11, gpu="1", arm="no_self")
        parsed, config = worker.prepare_args(options)
        self.assertEqual(parsed.rna_format, "Pathways")
        self.assertEqual(parsed.encoding_dim, 1536)
        self.assertEqual(parsed.max_epochs, 30)
        self.assertEqual(parsed.dct_v313_recon_weighting, "per_branch")
        self.assertEqual(parsed.dct_v313_lambda_reconstruction, 0.10)
        self.assertEqual((parsed.k_start, parsed.k_end, parsed.seed), (2, 3, 11))
        self.assertEqual(parsed.gpu, "1")
        self.assertEqual(parsed.on_missing_wsi, "error")
        self.assertEqual(Path(parsed.results_dir), Path(options.output_dir).resolve() / "artifacts")
        self.assertIn("dct_v313_recon_weighting", parsed._dct_user_overrides)
        self.assertEqual(config["model"]["dct_v313_recon_weighting"], "per_branch")

    def test_prepare_args_rejects_truncated_or_test_only_training(self):
        base = yaml.safe_load((HERE / "configs" / "base.yaml").read_text(encoding="utf-8"))
        for key, value in (("max_epochs", 0), ("max_smoke_batches", 1),
                           ("early_stop_patience", 2), ("only_test", True),
                           ("wsi_missing", True), ("omic_missing", True)):
            with self.subTest(key=key):
                config = json.loads(json.dumps(base))
                config["train"][key] = value
                path = self.root / f"{key}.yaml"
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    worker.prepare_args(self.options(config=str(path)))

    def test_common_recipe_includes_all_shared_knobs_and_excludes_identity_and_controls(self):
        parsed, _ = worker.prepare_args(self.options())
        parsed.grad_accum_steps = 4
        parsed.future_parser_option = "included automatically"
        parsed.cur_epoch = 20
        parsed.cur_fold = 0
        parsed.max_cindex = .75
        parsed.max_cindex_epoch = 9
        recipe = worker.common_recipe(parsed)
        for key in ("warmup_epochs", "grad_accum_steps", "event_sampling_fraction",
                    "temperature", "topk_ratio", "top_k_method", "otehv2_eps",
                    "otehv2_iter", "otehv2_heads", "otehv2_layers", "otehv2_dropout",
                    "dct_num_stages", "future_parser_option"):
            self.assertIn(key, recipe)
        for key in ("results_dir", "specific_simple", "gpu", "k_start", "k_end",
                    "_dct_user_overrides", "cur_epoch", "cur_fold", "max_cindex",
                    "max_cindex_epoch", "dct_lambda_ipcw_rank",
                    "dct_v311_lambda_slot_nll", "dct_v311_lambda_slot_diversity",
                    "dct_v313_recon_weighting"):
            self.assertNotIn(key, recipe)
        parsed.warmup_epochs = recipe["warmup_epochs"] + 1
        self.assertNotEqual(worker.common_recipe(parsed), recipe)
        parsed.topk_ratio = recipe["topk_ratio"] + .1
        self.assertNotEqual(worker.common_recipe(parsed), recipe)

    def test_common_recipe_refuses_nonfinite_or_nonserializable_values(self):
        for value in (float("nan"), float("inf"), object()):
            with self.subTest(value=repr(value)):
                with self.assertRaises((ValueError, TypeError)):
                    worker.common_recipe(SimpleNamespace(custom=value))

    def test_common_recipe_normalizes_numpy_feature_dimensions(self):
        import numpy as np
        recipe = worker.common_recipe(SimpleNamespace(omic_sizes=[np.int64(3), np.int64(5)],
                                                      omic_names=np.array(["G1", "G2"])))
        self.assertEqual(recipe, {"omic_sizes": [3, 5], "omic_names": ["G1", "G2"]})

    def test_strict_loader_forwards_dataset_and_rejects_none(self):
        dataset = self.dataset(("P1", "P2"))
        original = FixtureLoader(dataset, ["good", None])
        wrapped = worker.StrictLoader(original)
        self.assertIs(wrapped.dataset, dataset)
        self.assertEqual(len(wrapped), 2)
        iterator = iter(wrapped)
        self.assertEqual(next(iterator), "good")
        with self.assertRaisesRegex(ValueError, "skipped/None"):
            next(iterator)

    def test_actual_labels_matches_exact_patient_set_and_outcomes(self):
        split = self.split()
        audit = worker.actual_labels(self.dataset(("P3", "P4")), self.factory, split, "val")
        self.assertEqual(audit, {"partition": "val", "patients": {
            "P3": {"time": 1.0, "source_time": 1.0, "censor": 0.0},
            "P4": {"time": 2.0, "source_time": 2.0, "censor": 0.0}}})
        with self.assertRaisesRegex(ValueError, "differ from split"):
            worker.actual_labels(self.dataset(("P3", "P9")), self.factory, split, "val")

    def test_label_audit_uses_actual_float32_time_and_retains_source_precision(self):
        import numpy as np
        dataset = self.dataset(("P3", "P4"))
        dataset.label_df.loc[0, "survival"] = 6.279
        audit = worker.actual_labels(dataset, self.factory, self.split(), "val")
        predictions = {"P3": {"risk": .2, "time": float(np.float32(6.279)), "censor": 0.0},
                       "P4": {"risk": .3, "time": 2.0, "censor": 0.0}}
        self.assertNotEqual(audit["patients"]["P3"]["time"], 6.279)
        self.assertEqual(audit["patients"]["P3"]["source_time"], 6.279)
        summarize.verify_prediction_labels(predictions, audit, self.split(), "val")
        predictions["P3"]["time"] += .01
        with self.assertRaises(summarize.EvidenceError):
            summarize.verify_prediction_labels(predictions, audit, self.split(), "val")

    def test_actual_labels_rejects_duplicate_or_invalid_survival_targets(self):
        split = self.split()
        with self.assertRaisesRegex(ValueError, "Repeated patient"):
            worker.actual_labels(self.dataset(("P3", "P3")), self.factory, split, "val")
        for column, value in (("survival", float("nan")), ("survival", -1.0),
                              ("censorship", .5)):
            with self.subTest(column=column, value=value):
                dataset = self.dataset(("P3", "P4"))
                dataset.label_df.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, "Invalid survival target"):
                    worker.actual_labels(dataset, self.factory, split, "val")

    def test_existing_task_artifacts_are_rejected_without_overwrite(self):
        output = Path(self.options().output_dir)
        output.mkdir()
        prior = output / "completion.json"
        prior.write_text("preserve this receipt", encoding="utf-8")
        with patch.object(worker, "prepare_args") as parse:
            with self.assertRaisesRegex(FileExistsError, "Refusing existing task artifacts"):
                worker.execute_task(self.options())
        parse.assert_not_called()
        self.assertEqual(prior.read_text(encoding="utf-8"), "preserve this receipt")

    def fixture_runner(self, options, scenario="complete"):
        """Replace the whole runner, exercising the worker's wrappers and receipts."""
        split = self.split(options.protocol)
        config = self.root / "fixture.yaml"
        config.write_text("fixture_only: true\n", encoding="utf-8")
        options.config = str(config)
        source_root = self.root / "source"
        source = source_root / "survot_rank" / "fixture.py"
        source.parent.mkdir(parents=True)
        source.write_text("# synthetic hashed source fixture\n", encoding="utf-8")
        args = SimpleNamespace(data_path=str(self.root / "data"), which_splits="fixed",
                               study="blca", max_epochs=3, seed=options.seed,
                               evaluation_protocol=options.protocol,
                               results_dir=str(Path(options.output_dir) / "artifacts"))
        train = self.dataset(("P1", "P2"))
        val = self.dataset(("P3", "P4"))
        outer = self.dataset(("P5", "P6"))
        runner = types.ModuleType("survot_rank.training.train_runner")
        originals = {name: Mock(name=name) for name in
                     ("get_model", "get_split", "evaluate", "evaluate_outer_test")}
        runner.get_model = originals["get_model"]

        def original_split(*_):
            batches = [object(), None] if scenario == "skipped_train_batch" else None
            return train, val, FixtureLoader(train, batches), FixtureLoader(val)

        def original_evaluate(args, factory, model, loader, loss_fn, survival_train):
            list(loader)  # A genuine None batch must be caught by StrictLoader.
            predictions = {str(row["case id"]): {
                "risk": 3.0 - float(row["survival"]),
                "time": float(row["survival"]), "censor": float(row["censorship"]),
            } for _, row in loader.dataset.label_df.iterrows()}
            if scenario == "prediction_label_mismatch":
                next(iter(predictions.values()))["time"] += .25
            return predictions, 1.0, 1.0, [], 0.0, 0.0, 0.0

        def original_outer(*_, **kwargs):
            return runner.evaluate(args, self.factory, object(), FixtureLoader(outer), None, None)

        runner.get_split = original_split
        runner.evaluate = original_evaluate
        runner.evaluate_outer_test = original_outer
        originals.update(get_split=original_split, evaluate=original_evaluate,
                         evaluate_outer_test=original_outer)

        def dump(payload, path):
            with Path(path).open("wb") as stream:
                pickle.dump(payload, stream)

        runner.safe_pickle_dump = dump

        def run(parsed):
            if scenario == "empty_return":
                return
            _, _, train_loader, val_loader = runner.get_split(parsed, self.factory, options.fold)
            model = runner.get_model("fixture", parsed)
            list(train_loader)
            predictions = runner.evaluate(parsed, self.factory, model, val_loader, None, None)[0]
            artifacts = Path(parsed.results_dir)
            artifacts.mkdir(parents=True)
            epochs = [0, 1] if scenario == "partial_epochs" else [0, 1, 2]
            score = .75 if scenario == "score_mismatch" else 1.0
            pd.DataFrame({"epoch": epochs, "val_cindex": [score] * len(epochs)}).to_csv(
                artifacts / f"epoch_curve_fold{options.fold}.csv", index=False)
            (artifacts / f"model_best_s{options.fold}.pth").write_bytes(b"mock checkpoint fixture")
            dump(predictions, artifacts / f"split_{options.fold}_results.pkl")
            if scenario == "interrupted_return":
                # Model the inherited runner swallowing the worker's SIGINT.
                try:
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                except KeyboardInterrupt:
                    pass
            if scenario == "changed_source":
                source.write_text("# source changed during fixture run\n", encoding="utf-8")
            if options.protocol == "outer_test" and scenario != "missing_outer":
                parsed.cur_epoch = 2
                parsed.max_cindex = 1.0
                parsed.max_cindex_epoch = 1
                # Outer evaluation rebuilds the model after training state changed.
                runner.get_model("fixture", parsed)
                runner.evaluate_outer_test(args=parsed)

        runner.run = run
        model_module = SimpleNamespace(build_model=lambda *_, **__: SimpleNamespace(
            audit=lambda: {"arm": options.arm, "self": .05, "cross": .05}))
        training = importlib.import_module("survot_rank.training")
        contexts = contextlib.ExitStack()
        contexts.enter_context(patch.object(worker, "ROOT", source_root))
        contexts.enter_context(patch.object(worker, "prepare_args", return_value=(args, {})))
        contexts.enter_context(patch.object(worker, "load_sibling", side_effect=lambda name:
                                           model_module if name == "model" else summarize))
        contexts.enter_context(patch.object(training, "train_runner", runner, create=True))
        contexts.enter_context(patch.dict(sys.modules, {"survot_rank.training.train_runner": runner}))
        contexts.enter_context(patch.dict(os.environ, {}, clear=False))
        contexts.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return contexts, runner, originals, split

    def run_fixture(self, scenario="complete", protocol="legacy_val"):
        options = self.options(protocol=protocol)
        context, runner, originals, _ = self.fixture_runner(options, scenario)
        prior_handler = signal.getsignal(signal.SIGINT)
        with context:
            if scenario == "complete":
                worker.execute_task(options)
            else:
                with self.assertRaises((ValueError, RuntimeError)):
                    worker.execute_task(options)
            for name, original in originals.items():
                self.assertIs(getattr(runner, name), original)
            self.assertIs(signal.getsignal(signal.SIGINT), prior_handler)
        receipt = json.loads((Path(options.output_dir) / "completion.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "complete" if scenario == "complete" else "failed")
        self.assertEqual(receipt["training_complete"], scenario == "complete")
        return receipt, Path(options.output_dir)

    def test_mock_complete_validation_run_writes_verified_receipt(self):
        receipt, output = self.run_fixture()
        self.assertEqual(receipt["completed_epochs"], 3)
        self.assertEqual(receipt["evaluation_partition"], "val")
        self.assertEqual(receipt["prediction_patient_count"], 2)
        self.assertEqual(receipt["prediction_artifact"]["role"], "best_val")
        task = json.loads((output / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(set(task["training_label_audit"]["patients"]), {"P1", "P2"})
        self.assertEqual(len(task["recipe"]["training_labels_sha256"]), 64)
        for record in receipt["artifacts"]:
            self.assertEqual(worker.sha256(output / record["path"]), record["sha256"])

    def test_mock_complete_outer_run_saves_actual_outer_predictions(self):
        receipt, output = self.run_fixture(protocol="outer_test")
        self.assertEqual(receipt["evaluation_partition"], "outer_test")
        self.assertEqual(receipt["prediction_artifact"]["role"], "outer_test")
        self.assertEqual(set(summarize.read_predictions(output / "outer_test_predictions.pkl")),
                         {"P5", "P6"})

    def test_mock_partial_epochs_cannot_claim_completion(self):
        receipt, _ = self.run_fixture("partial_epochs")
        self.assertIn("Partial or interrupted epoch curve", receipt["error"])

    def test_mock_swallowed_interrupt_cannot_claim_completion(self):
        receipt, _ = self.run_fixture("interrupted_return")
        self.assertIn("Interrupted training", receipt["error"])

    def test_mock_normal_return_without_artifacts_cannot_claim_completion(self):
        receipt, _ = self.run_fixture("empty_return")
        self.assertIn("Missing or ambiguous", receipt["error"])

    def test_mock_missing_outer_evaluation_cannot_claim_completion(self):
        receipt, _ = self.run_fixture("missing_outer", protocol="outer_test")
        self.assertIn("Final evaluation partition", receipt["error"])

    def test_mock_recorded_score_mismatch_cannot_claim_completion(self):
        receipt, _ = self.run_fixture("score_mismatch")
        self.assertIn("disagree with the recorded C-index", receipt["error"])

    def test_mock_changed_source_cannot_claim_completion(self):
        receipt, _ = self.run_fixture("changed_source")
        self.assertIn("Input changed during training", receipt["error"])

    def test_mock_skipped_training_batch_cannot_claim_completion(self):
        receipt, _ = self.run_fixture("skipped_train_batch")
        self.assertIn("skipped/None batch", receipt["error"])

    def test_mock_predictions_with_wrong_actual_labels_cannot_claim_completion(self):
        receipt, _ = self.run_fixture("prediction_label_mismatch")
        self.assertIn("differs from actual loader", receipt["error"])


if __name__ == "__main__":
    unittest.main()
