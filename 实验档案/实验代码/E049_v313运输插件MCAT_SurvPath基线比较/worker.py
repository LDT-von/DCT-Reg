"""One explicit E049 training task for MCAT / SurvPath with the transport plugin.

Executed only by the operator's --execute command.  Each task builds its own
native adapter from the data factory's omic_sizes and runs a full training
budget without early stopping.
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
    unique = "e049_" + name
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
        time = float(np.float32(source_time))
        censor = float(row[factory.censorship_var])
        if (not math.isfinite(source_time) or source_time < 0
                or not math.isfinite(time) or time < 0
                or censor not in (0.0, 1.0)):
            raise ValueError(f"Invalid survival target for {patient}")
        patients[patient] = {"time": time, "source_time": source_time, "censor": censor}
    expected = split_ids(path, partition)
    if set(patients) != expected:
        raise ValueError(f"Actual {partition} patients differ from split: "
                         f"missing={sorted(expected-set(patients))}, extra={sorted(set(patients)-expected)}")
    return {"partition": partition, "patients": patients}


def _mcat_omic_sizes_from_factory(factory):
    """MCAT must receive the six actual signature groups, never fabricated sizes."""
    sizes = getattr(factory, "omic_sizes", None)
    if not isinstance(sizes, (list, tuple)) or len(sizes) != 6 or any(int(x) <= 0 for x in sizes):
        raise ValueError("MCAT requires exactly six nonempty Pathways groups; use signature=six")
    return [int(x) for x in sizes]


def _survpath_omic_sizes_from_factory(factory):
    """Return SurvPath omic_sizes from the dataset factory."""
    if hasattr(factory, "omic_sizes") and factory.omic_sizes:
        return list(factory.omic_sizes)
    raise ValueError("SurvPath requires actual Pathways group sizes")


class StrictLoader:
    """Reject skipped batches instead of shifting positional IDs."""
    def __init__(self, loader):
        self._loader = loader

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __iter__(self):
        for batch in self._loader:
            if batch is None:
                raise ValueError("E049 forbids a skipped/None batch")
            yield batch


# ----------------------------------------------------------------------
# E049 arm definitions — must stay in sync with schedule_core.py
# ----------------------------------------------------------------------
ARM_DEFINITIONS = {
    "native":                    dict(enabled=False, plan_mode="learned",   reconstruction=False),
    "independent":               dict(enabled=True,  plan_mode="independent", reconstruction=False),
    "learned":                  dict(enabled=True,  plan_mode="learned",    reconstruction=False),
    "independent_three_decoders": dict(enabled=True, plan_mode="independent", reconstruction=True,
                                      decoder_mode="separate"),
    "learned_three_decoders":   dict(enabled=True,  plan_mode="learned",   reconstruction=True,
                                      decoder_mode="separate"),
    "learned_shared_decoders":  dict(enabled=True,  plan_mode="learned",   reconstruction=True,
                                      decoder_mode="shared"),
}


def build_native_adapter_for_arm(baseline, omic_sizes, arm, num_classes=4):
    """Build the native adapter for a given baseline and arm."""
    from survot_rank.plugins.transport import build_native_adapter
    from survot_rank.plugins.transport.config import TransportPluginConfig

    arm_def = ARM_DEFINITIONS.get(arm)
    if arm_def is None:
        raise ValueError(f"Unknown E049 arm: {arm}")

    config = TransportPluginConfig(
        wsi_dim=256, omics_dim=256, dim=256,
        plan_mode=arm_def["plan_mode"],
        residual_init=0.0,
        eps=0.05, eps_start=0.10, eps_anneal_epochs=12,
        sinkhorn_iters=50, projection_iters=10000, projection_tol=0.0001,
        wsi_slots=8, omics_slots=8,
        stages=4, heads=4, event_layers=2,
        slot_init_mode="deterministic",
        geometries=["cosine", "euclidean", "positive_dot"],
    )

    reconstruction = None
    if arm_def["reconstruction"]:
        reconstruction = {
            "dim": 256,
            "num_omics_tokens": len(omic_sizes),
            "heads": 4,
            "mode": arm_def["decoder_mode"],
            "self_weight": 0.05,
            "cross_weight": 0.05,
            "wsi_weight": 0.05,
            "ramp_start": 2,
            "ramp_epochs": 5,
            "wsi_chunk_size": 256,
        }

    return build_native_adapter(
        backbone_name=baseline,
        omic_sizes=omic_sizes,
        wsi_input_dim=1536,
        plugin_config=config,
        reconstruction_config=reconstruction,
        fusion="concat",
        enabled=arm_def["enabled"],
        num_classes=num_classes,
    )


def prepare_args(options):
    from survot_rank.config import config_to_argv, flatten_config, load_config
    from survot_rank.training.extended_args import process_args_extended
    config = load_config(options.config)
    flat = flatten_config(config)
    managed = {key: flat.pop(key) for key in list(flat) if key.startswith("dct_v313_")}
    parsed = process_args_extended(config_to_argv(flat))
    for key, value in managed.items():
        setattr(parsed, key, value)
    parsed._dct_user_overrides = set(flat) | set(managed)
    parsed.survot_method = "dct_v313_transport_reconstruction"
    parsed.newslot_method = parsed.survot_method
    # This experiment trains native MCAT/SurvPath adapters, not the repaired
    # DCT model; E049 writes its own provenance receipt below.
    parsed.dct_v313_revision = "legacy"
    parsed.evaluation_protocol = options.protocol
    parsed.study = options.study
    parsed.data_path = options.data_path
    parsed.which_splits = options.split_set
    parsed.rna_format = "Pathways"
    parsed.signature = "six" if options.baseline == "mcat" else "combine"
    parsed.batch_size = 1
    parsed.event_stratified_batches = False
    parsed.event_sampling_fraction = 0.0
    parsed.k_start, parsed.k_end = options.fold, options.fold + 1
    parsed.seed = options.seed
    parsed.gpu = 0  # CUDA_VISIBLE_DEVICES maps the selected physical GPU to cuda:0.
    parsed.results_dir = str(Path(options.output_dir).resolve() / "artifacts")
    parsed.specific_simple = f"E049_{options.baseline}_{options.arm}_{options.protocol}_seed{options.seed}"
    parsed.on_missing_wsi = "error"
    if getattr(parsed, "max_epochs", 0) <= 0:
        raise ValueError("E049 completion requires positive epochs")
    if getattr(parsed, "early_stop_patience", 0):
        raise ValueError("E049 v1 requires early_stop_patience=0")
    if getattr(parsed, "max_smoke_batches", 0):
        raise ValueError("E049 completion does not accept a truncated smoke run")
    if parsed.bag_loss != "nll_surv":
        raise ValueError("E049 native bridge currently supports nll_surv only")
    return parsed, config


def common_recipe(args, baseline, arm):
    """Capture all effective shared arguments."""
    identity = {"results_dir", "specific_simple", "gpu", "k_start", "k_end",
                "_dct_user_overrides", "cur_epoch", "cur_fold", "max_cindex", "max_cindex_epoch",
                "survot_method", "newslot_method", "evaluation_protocol"}
    controlled = {
        "dct_lambda_ipcw_rank", "dct_v311_lambda_slot_nll",
        "dct_v311_lambda_slot_diversity", "dct_v311_target_pair_dist",
        "dct_v311_lambda_pair_dist",
    }
    result = {key: value for key, value in vars(args).items()
              if key not in identity | controlled}
    result["e049_baseline"] = baseline
    result["e049_arm"] = arm

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
        raise FileExistsError(f"Refusing existing task artifacts: {sorted(existing - allowed)}")

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

    args, _ = prepare_args(options)
    source_split = split_path(args, options.fold)
    partitions = (("train", "val") if options.protocol == "legacy_val"
                 else ("inner_train", "inner_val", "outer_test"))
    sets = [split_ids(source_split, part) for part in partitions]
    for i, left in enumerate(sets):
        if any(left & right for right in sets[i+1:]):
            raise ValueError("Patient leakage between split partitions")

    source_paths = sorted((ROOT / "survot_rank").rglob("*.py")) + sorted(HERE.glob("*.py"))
    task_id = (f"{options.protocol}/{options.study}/{options.baseline}/{options.arm}/"
               f"fold{options.fold}/seed{options.seed}")
    task = {
        "schema_version": 1,
        "task_id": task_id,
        "baseline": options.baseline,
        "study": options.study,
        "arm": options.arm,
        "fold": options.fold,
        "seed": options.seed,
        "protocol": options.protocol,
        "recipe": {},
        "config_file": {"path": str(Path(options.config).resolve()), "sha256": sha256(options.config)},
        "split_file": {"path": str(source_split.resolve()), "sha256": sha256(source_split)},
        "source_files": [{"path": str(p), "sha256": sha256(p)} for p in source_paths],
    }

    training_audit = None
    model_audit = None
    outer_score = None
    arm_def = ARM_DEFINITIONS.get(options.arm, {})
    has_reconstruction = arm_def.get("reconstruction", False)

    # ----- Get omic_sizes from the factory -----
    from survot_rank.training import train_runner as runner
    original_get_split = runner.get_split

    def get_split_with_audit(args, factory, fold):
        nonlocal training_audit
        train, val, train_loader, val_loader = original_get_split(args, factory, fold)
        train_part = partitions[0]
        val_part = partitions[1]
        training_audit = actual_labels(train, factory, source_split, train_part)
        actual_labels(val, factory, source_split, val_part)

        # Derive omic_sizes from factory
        if options.baseline == "mcat":
            omic_sizes = _mcat_omic_sizes_from_factory(factory)
        else:
            omic_sizes = _survpath_omic_sizes_from_factory(factory)

        # Write omic_sizes into task for reproducibility
        if task.get("omic_sizes") is None:
            task["omic_sizes"] = omic_sizes

        return train, val, StrictLoader(train_loader), StrictLoader(val_loader), omic_sizes

    # Override get_split to capture omic_sizes before first training batch
    _omic_sizes_captured = None

    def patched_get_split(args, factory, fold):
        nonlocal _omic_sizes_captured
        result = get_split_with_audit(args, factory, fold)
        train, val, tr, va, omic_sizes = result
        _omic_sizes_captured = omic_sizes
        return train, val, tr, va

    runner.get_split = patched_get_split

    # ----- Build the adapter -----
    # We need the factory to get omic_sizes, so build after first get_split
    model_module = load_sibling("runner_bridge")
    verification = load_sibling("evidence")

    original_evaluate = runner.evaluate
    original_outer = runner.evaluate_outer_test
    original_get_model = runner.get_model
    interrupted = False
    audit_seen = None

    def model_constructor(method, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        nonlocal model_audit, _omic_sizes_captured
        if _omic_sizes_captured is None:
            raise RuntimeError("get_split must be called before model construction")
        omic_sizes = _omic_sizes_captured

        adapter = build_native_adapter_for_arm(
            baseline=options.baseline,
            omic_sizes=omic_sizes,
            arm=options.arm,
            num_classes=getattr(args, "num_classes", 4),
        )
        model = model_module.NativeRunnerBridge(adapter, options.baseline, omic_sizes)
        import torch
        model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        audit = {
            "baseline": options.baseline,
            "arm": options.arm,
            "omic_sizes": omic_sizes,
            "wsi_input_dim": 1536,
            "enabled": arm_def.get("enabled", True),
            "plan_mode": arm_def.get("plan_mode", "learned"),
            "reconstruction": arm_def.get("reconstruction", False),
            "decoder_mode": arm_def.get("decoder_mode", None),
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

        resolved = common_recipe(args, options.baseline, options.arm)
        if training_audit is None:
            raise ValueError("Cannot construct E049 model before actual training labels are audited")
        training_label_bytes = json.dumps(training_audit, sort_keys=True, allow_nan=False).encode("utf-8")
        resolved["training_labels_sha256"] = hashlib.sha256(training_label_bytes).hexdigest()

        if model_audit is not None and audit != model_audit:
            raise ValueError("Train/outer-test constructors resolved different recipes")
        if model_audit is not None and resolved != task["recipe"]:
            raise ValueError("Train/outer-test constructors resolved different common arguments")
        if model_audit is None:
            task["recipe"] = resolved
            task["training_label_audit"] = training_audit
            write_new_json(output / "task.json", task)
        model_audit = audit
        return model

    def audited_evaluate(args, factory, model, loader, loss_fn, survival_train):
        nonlocal audit_seen
        partition = getattr(loader.dataset, "split", None)
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

    runner.get_model = model_constructor
    runner.evaluate = audited_evaluate
    runner.evaluate_outer_test = audited_outer
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
        if not math.isfinite(recorded_score) or abs(score - recorded_score) > 1e-8:
            raise ValueError("Saved predictions disagree with the recorded C-index")

        for record in task["source_files"] + [task["config_file"], task["split_file"]]:
            if sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"Input changed during training: {record['path']}")

        write_new_json(output / "model_audit.json", model_audit)
        prediction_record = artifact(prediction_path, output)
        prediction_record["role"] = "best_val" if partition == "val" else "outer_test"
        labels_record = artifact(labels_path, output)
        checkpoint_record = artifact(checkpoints[0], output)

        complete = {
            "schema_version": 1,
            "status": "complete",
            "task_id": task_id,
            "baseline": options.baseline,
            "arm": options.arm,
            "protocol": options.protocol,
            "metric": "cindex",
            "score": score,
            "training_complete": True,
            "completed_epochs": len(curve),
            "max_epochs": args.max_epochs,
            "evaluation_partition": partition,
            "prediction_patient_count": len(normalized),
            "prediction_artifact": prediction_record,
            "label_audit_artifact": labels_record,
            "checkpoint_artifact": checkpoint_record,
            "artifacts": [
                prediction_record, labels_record, checkpoint_record,
                artifact(curves[0], output),
                artifact(output / "model_audit.json", output),
            ],
        }
        write_new_json(output / "completion.json", complete)
        print(json.dumps({"task_id": task_id, "status": "complete", "cindex": score}))

    except BaseException as exc:
        if not (output / "completion.json").exists():
            write_new_json(output / "completion.json", {
                "schema_version": 1, "task_id": task_id,
                "baseline": options.baseline, "arm": options.arm,
                "protocol": options.protocol,
                "status": "failed",
                "training_complete": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        raise
    finally:
        runner.get_model = original_get_model
        runner.get_split = original_get_split
        runner.evaluate = original_evaluate
        runner.evaluate_outer_test = original_outer
        signal.signal(signal.SIGINT, previous_signal)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=["mcat", "survpath"],
                        help="native backbone: mcat or survpath")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=("legacy_val", "outer_test"), required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--split-set", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    execute_task(parser.parse_args(argv))


if __name__ == "__main__":
    main()
