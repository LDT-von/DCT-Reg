"""One explicit E048 training task, using the existing patient-level runner.

This file is executed only by the operator's --execute command. A normal
runner return is insufficient: actual labels, predictions, epochs and hashes
are verified before completion.json can claim completion.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_sibling(name):
    unique = "e048_" + name
    spec = importlib.util.spec_from_file_location(unique, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    spec.loader.exec_module(module)
    return module


def artifact(path, output_dir):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(output_dir)), "sha256": sha256(path)}


def split_path(args, fold):
    return Path(args.data_path) / "splits" / args.which_splits / args.study / f"fold_{fold}.csv"


def split_ids(path, partition):
    import pandas as pd
    frame = pd.read_csv(path, dtype=str)
    if partition not in frame:
        raise ValueError(f"Split {path} is missing {partition}")
    ids = frame[partition].dropna().astype(str).tolist()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate patient IDs in {partition}: {path}")
    return set(ids)


def actual_labels(dataset, factory, path, partition):
    import numpy as np
    frame = dataset.label_df
    patients = {}
    for _, row in frame.iterrows():
        patient = str(row["case id"])
        if patient in patients:
            raise ValueError(f"Repeated patient in actual loader: {patient}")
        source_time = float(row[factory.label_col])
        # SurvivalDataset.get_label and its collate function use float32.
        # Preserve the source value as well, but verify the actual tensor target.
        time = float(np.float32(source_time))
        censor = float(row[factory.censorship_var])
        if (not math.isfinite(source_time) or source_time < 0 or not math.isfinite(time)
                or time < 0 or censor not in (0.0, 1.0)):
            raise ValueError(f"Invalid survival target for {patient}")
        patients[patient] = {"time": time, "source_time": source_time, "censor": censor}
    expected = split_ids(path, partition)
    if set(patients) != expected:
        raise ValueError(f"Actual {partition} patients differ from split: "
                         f"missing={sorted(expected-set(patients))}, extra={sorted(set(patients)-expected)}")
    return {"partition": partition, "patients": patients}


class StrictLoader:
    """Reject skipped batches instead of shifting the legacy positional IDs."""
    def __init__(self, loader):
        self._loader = loader

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __iter__(self):
        for batch in self._loader:
            if batch is None:
                raise ValueError("E048 forbids a skipped/None batch")
            yield batch


def prepare_args(options):
    from survot_rank.config import config_to_argv, flatten_config, load_config
    from survot_rank.training.extended_args import process_args_extended
    config = load_config(options.config)
    flat = flatten_config(config)
    # These controlled E048 knobs are consumed by the model constructor, not
    # forwarded as unknown flags to older versions of the common parser.
    managed = {key: flat.pop(key) for key in list(flat) if key.startswith("dct_v313_")}
    parsed = process_args_extended(config_to_argv(flat))
    for key, value in managed.items():
        setattr(parsed, key, value)
    parsed._dct_user_overrides = set(flat) | set(managed)
    parsed.survot_method = "dct_v313_transport_reconstruction"
    parsed.newslot_method = parsed.survot_method
    parsed.evaluation_protocol = options.protocol
    parsed.k_start, parsed.k_end = options.fold, options.fold + 1
    parsed.seed = options.seed
    parsed.gpu = options.gpu
    parsed.results_dir = str(Path(options.output_dir).resolve() / "artifacts")
    parsed.specific_simple = f"E048_{options.arm}_{options.protocol}_seed{options.seed}"
    parsed.on_missing_wsi = "error"
    if (parsed.max_epochs <= 0 or getattr(parsed, "max_smoke_batches", 0)
            or getattr(parsed, "smoke_k_batches", 0)):
        raise ValueError("E048 completion requires positive epochs and no truncated smoke batches")
    if getattr(parsed, "wsi_missing", False) or getattr(parsed, "omic_missing", False):
        raise ValueError("E048 paired-cohort arms require both original modalities available")
    # Full-budget runs make completion objective and comparable across arms.
    if getattr(parsed, "early_stop_patience", 0):
        raise ValueError("E048 v1 requires early_stop_patience=0")
    if getattr(parsed, "only_test", False):
        raise ValueError("E048 controls must be trained from scratch")
    return parsed, config


def common_recipe(args):
    """Capture all effective shared arguments, rather than a partial allowlist.

    Arm-specific objective knobs are audited by model.audit(); task paths and
    GPU placement do not affect the common training recipe. New parser options
    are included automatically so configuration drift cannot silently pair.
    """
    identity = {"results_dir", "specific_simple", "gpu", "k_start", "k_end",
                "_dct_user_overrides", "cur_epoch", "cur_fold", "max_cindex", "max_cindex_epoch"}
    controlled = {
        "dct_lambda_ipcw_rank", "dct_v311_lambda_slot_nll",
        "dct_v311_lambda_slot_diversity", "dct_v311_target_pair_dist",
        "dct_v311_lambda_pair_dist", "dct_v38_lambda_direction",
        "dct_v38_lambda_dose", "dct_v38_lambda_reconfiguration",
        "dct_v313_recon_weighting", "dct_v313_lambda_reconstruction_scale",
        "dct_v313_lambda_wsi_reconstruction", "dct_v313_disable_self_reconstruction",
        "dct_v313_disable_cross_reconstruction", "dct_v313_cross_mode", "dct_v313_plan_mode",
    }
    result = {key: value for key, value in vars(args).items()
              if key not in identity | controlled}
    # Fail early for non-serializable state instead of writing an incomplete audit.
    def numpy_value(value):
        if type(value).__module__.startswith("numpy") and hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Unserializable recipe argument: {type(value).__name__}")
    return json.loads(json.dumps(result, allow_nan=False, default=numpy_value))


def execute_task(options):
    import pandas as pd
    output = Path(options.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"config.yaml"}
    existing = {item.name for item in output.iterdir()}
    if existing - allowed:
        raise FileExistsError(f"Refusing existing task artifacts: {sorted(existing-allowed)}")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu
    os.environ["DCT_DEBUG_LOG"] = str(output / "constructor_debug.log")
    args, _ = prepare_args(options)
    source_split = split_path(args, options.fold)
    partitions = (("train", "val") if options.protocol == "legacy_val"
                  else ("inner_train", "inner_val", "outer_test"))
    sets = [split_ids(source_split, part) for part in partitions]
    for i, left in enumerate(sets):
        if any(left & right for right in sets[i+1:]):
            raise ValueError("Patient leakage between split partitions")
    source_paths = sorted((ROOT / "survot_rank").rglob("*.py")) + sorted(HERE.glob("*.py"))
    task_id = f"{options.protocol}/{options.arm}/{args.study}/fold{options.fold}/seed{options.seed}"
    task = {"schema_version": 1, "task_id": task_id, "arm": options.arm, "study": args.study,
            "fold": options.fold, "seed": options.seed, "protocol": options.protocol,
            "recipe": common_recipe(args),
            "config_file": {"path": str(Path(options.config).resolve()), "sha256": sha256(options.config)},
            "split_file": {"path": str(source_split.resolve()), "sha256": sha256(source_split)},
            "source_files": [{"path": str(p), "sha256": sha256(p)} for p in source_paths]}
    model_module = load_sibling("model")
    verification = load_sibling("summarize")
    from survot_rank.training import train_runner as runner
    original_model, original_split = runner.get_model, runner.get_split
    original_evaluate, original_outer = runner.evaluate, runner.evaluate_outer_test
    interrupted = False
    audit_seen = None
    training_audit = None
    model_audit = None
    outer_score = None

    def construct(method, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        nonlocal model_audit
        model = model_module.build_model(args, options.arm, omic_input_dim, omic_names, pathway_names)
        audit = model.audit()
        resolved = common_recipe(args)
        if training_audit is None:
            raise ValueError("Cannot construct E048 model before actual training labels are audited")
        training_label_bytes = json.dumps(training_audit, sort_keys=True, allow_nan=False).encode("utf-8")
        resolved["training_labels_sha256"] = hashlib.sha256(training_label_bytes).hexdigest()
        if model_audit is not None and audit != model_audit:
            raise ValueError("Train/outer-test constructors resolved different recipes")
        if model_audit is not None and resolved != task["recipe"]:
            raise ValueError("Train/outer-test constructors resolved different common arguments")
        if model_audit is None:
            # Inherited frozen constructors may update args. Record the effective
            # common settings before fitting, after those overrides have run.
            task["recipe"] = resolved
            task["training_label_audit"] = training_audit
            write_new_json(output / "task.json", task)
        model_audit = audit
        return model

    def strict_split(args, factory, fold):
        nonlocal training_audit
        train, val, train_loader, val_loader = original_split(args, factory, fold)
        train_part, val_part = partitions[:2]
        training_audit = actual_labels(train, factory, source_split, train_part)
        actual_labels(val, factory, source_split, val_part)
        return train, val, StrictLoader(train_loader), StrictLoader(val_loader)

    def audited_evaluate(args, factory, model, loader, loss_fn, survival_train):
        nonlocal audit_seen
        partition = getattr(loader.dataset, "split", None)
        # Dataset versions do not all expose the split string. Actual ID sets
        # determine the partition unambiguously; the validated CSV is disjoint.
        actual_ids = set(loader.dataset.label_df["case id"].astype(str))
        matches = [part for part in partitions[1:] if actual_ids == split_ids(source_split, part)]
        if len(matches) != 1:
            raise ValueError("Cannot identify actual evaluation partition")
        partition = matches[0]
        labels = actual_labels(loader.dataset, factory, source_split, partition)
        result = original_evaluate(args, factory, model, StrictLoader(loader), loss_fn, survival_train)
        verification.verify_prediction_labels(result[0], labels, source_split, partition)
        audit_seen = labels
        if partition == "outer_test":
            # The original runner does not persist outer-test predictions.
            # This wrapper saves exactly the one evaluation it already performed.
            runner.safe_pickle_dump(result[0], str(output / "outer_test_predictions.pkl"))
        return result

    def audited_outer(*args, **kwargs):
        nonlocal outer_score
        result = original_outer(*args, **kwargs)
        outer_score = float(result[1])
        return result

    def interrupt_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    runner.get_model, runner.get_split = construct, strict_split
    runner.evaluate, runner.evaluate_outer_test = audited_evaluate, audited_outer
    previous_signal = signal.signal(signal.SIGINT, interrupt_handler)
    try:
        runner.run(args)
        if interrupted:
            raise RuntimeError("Interrupted training is not a completed experiment")
        curves = list(output.rglob(f"epoch_curve_fold{options.fold}.csv"))
        checkpoints = list(output.rglob(f"model_best_s{options.fold}.pth"))
        if len(curves) != 1 or len(checkpoints) != 1 or model_audit is None or audit_seen is None:
            raise ValueError("Missing or ambiguous actual training artifacts")
        curve = pd.read_csv(curves[0])
        if list(curve["epoch"].astype(int)) != list(range(args.max_epochs)):
            raise ValueError("Partial or interrupted epoch curve; refusing completion")
        partition = "val" if options.protocol == "legacy_val" else "outer_test"
        if audit_seen["partition"] != partition:
            raise ValueError("Final evaluation partition did not match requested protocol")
        if partition == "val":
            predictions = list(output.rglob(f"split_{options.fold}_results.pkl"))
            if len(predictions) != 1:
                raise ValueError("Missing or ambiguous best-validation predictions")
            prediction_path = predictions[0]
            recorded_score = float(curve["val_cindex"].max())
        else:
            prediction_path = output / "outer_test_predictions.pkl"
            if outer_score is None:
                raise ValueError("Outer-test evaluation failed")
            recorded_score = outer_score
        labels_path = output / "label_audit.json"
        write_new_json(labels_path, audit_seen)
        normalized = verification.read_predictions(prediction_path)
        verification.verify_prediction_labels(normalized, audit_seen, source_split, partition)
        score = verification.compute_cindex(normalized)
        if not math.isfinite(recorded_score) or abs(score-recorded_score) > 1e-8:
            raise ValueError("Saved predictions disagree with the recorded C-index")
        # Recheck shared code/config/split hashes after training. A live edit
        # during the run invalidates the comparison instead of going unnoticed.
        for record in task["source_files"] + [task["config_file"], task["split_file"]]:
            if sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"Input changed during training: {record['path']}")
        write_new_json(output / "model_audit.json", model_audit)
        prediction_record = artifact(prediction_path, output)
        prediction_record["role"] = "best_val" if partition == "val" else "outer_test"
        labels_record, checkpoint_record = artifact(labels_path, output), artifact(checkpoints[0], output)
        complete = {"schema_version": 1, "status": "complete", "task_id": task_id,
                    "protocol": options.protocol, "metric": "cindex", "score": score,
                    "training_complete": True, "completed_epochs": len(curve), "max_epochs": args.max_epochs,
                    "evaluation_partition": partition, "prediction_patient_count": len(normalized),
                    "prediction_artifact": prediction_record, "label_audit_artifact": labels_record,
                    "checkpoint_artifact": checkpoint_record,
                    "artifacts": [prediction_record, labels_record, checkpoint_record,
                                  artifact(curves[0], output), artifact(output / "model_audit.json", output)]}
        write_new_json(output / "completion.json", complete)
        print(json.dumps({"task_id": task_id, "status": "complete", "cindex": score}))
    except BaseException as exc:
        if not (output / "completion.json").exists():
            write_new_json(output / "completion.json", {"schema_version": 1, "task_id": task_id,
                           "protocol": options.protocol, "status": "failed", "training_complete": False,
                           "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        runner.get_model, runner.get_split = original_model, original_split
        runner.evaluate, runner.evaluate_outer_test = original_evaluate, original_outer
        signal.signal(signal.SIGINT, previous_signal)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=("legacy_val", "outer_test"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    execute_task(parser.parse_args(argv))


if __name__ == "__main__":
    main()
