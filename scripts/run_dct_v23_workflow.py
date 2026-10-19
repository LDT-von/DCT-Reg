#!/usr/bin/env python3
"""V2-only server workflow. plan is data-free; calibrate/run are patient jobs.

Use a new output root for every batch. Historical drivers and outputs are never
patched or reused implicitly. audit reads trusted, locally generated pickles.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "dct_v23_server_workflow_v1"
OBJECTIVE_RECIPE = "dct_rg_simplified_core_20261019"
RECOVERY = ("off", "masked_plain", "gain_calibrated")
CORE = ("combined", "native_only", "transport_only", "independent")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def source_hashes():
    files = sorted((ROOT / "survot_rank").rglob("*.py"))
    files += sorted((ROOT / "common").rglob("*.py"))
    files += [Path(__file__), ROOT / "scripts/prepare_dct_v23_calibration.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def make_plan(output, *, stage, folds, seed=3, gpu="0", data_path=None,
              features_root=None, legacy_drivers=()):
    import yaml
    folds = list(folds)
    if not folds or len(folds) != len(set(folds)) or any(type(f) is not int or f not in range(5) for f in folds):
        raise ValueError("Select unique folds from 0 through 4")
    if stage not in {"recovery", "core"}:
        raise ValueError("Unknown stage")
    drivers = [Path(p).resolve(strict=True) for p in legacy_drivers]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load((ROOT / "configs/dct_rg_blca_legacy_val.yaml").read_text(encoding="utf-8"))
    # Current DCT-RG recipe only; historical configs retain their original losses.
    base["model"].update(dct_v2_lambda_ipcw_rank=0., dct_v2_lambda_slot_nll=0.,
        dct_v2_lambda_relation=0., dct_v2_lambda_native=0., dct_v21_local_readout="none")
    base["train"].update(seed=seed, gpu=str(gpu))
    if data_path:
        base["data"]["data_path"] = str(Path(data_path).resolve())
    if features_root:
        base["data"]["data_root_dir"] = str(Path(features_root).resolve())
    base["reconstruction"]["dct_v23_calibration_dir"] = str(output / "calibration")
    jobs = []
    for arm in RECOVERY if stage == "recovery" else CORE:
        for fold in folds:
            config = deepcopy(base)
            config["name"] = f"V2_simplified_core_{stage}_{arm}_fold{fold}_seed{seed}"
            config["train"].update(specific_simple=config["name"],
                results_dir=str(output / "runs" / arm / f"fold_{fold}"))
            config["split"].update(k_start=fold, k_end=fold + 1)
            if stage == "recovery":
                config["reconstruction"]["dct_v23_reconstruction_mode"] = arm
            else:
                config["reconstruction"]["dct_v23_reconstruction_mode"] = "off"
                config["model"].update(dct_v2_lambda_relation=0., dct_v2_lambda_native=0.,
                    dct_v23_prediction_path=arm if arm in {"native_only", "transport_only"} else "combined",
                    dct_v2_plan_mode="independent" if arm == "independent" else "learned")
            path = output / "configs" / arm / f"fold_{fold}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            jobs.append(dict(arm=arm, fold=fold, objective_recipe=OBJECTIVE_RECIPE,
                config=str(path.relative_to(output)), config_sha256=sha(path)))
    copies = []
    for i, driver in enumerate(drivers):
        dest = output / "original_driver" / f"{i}_{driver.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(driver, dest)
        copies.append(dict(original_path=str(driver), copy=str(dest.relative_to(output)), sha256=sha(dest)))
    manifest = dict(schema=SCHEMA, objective_recipe=OBJECTIVE_RECIPE,
        stage=stage, folds=folds, seed=seed, jobs=jobs,
        source_sha256=source_hashes(), original_drivers=copies,
        teacher_options=dict(inner_folds=3, mask_rounds=8, mask_ratio=.3,
                             native_dim=32, wsi_dim=16, ridge_alpha=10., seed=seed),
        evidence="legacy_val development; no patient execution implied by planning",
        source_root=str(ROOT), python=sys.executable)
    dump_new(output / "plan.json", manifest)
    return manifest


def read_plan(output, arms=None):
    import yaml
    output = Path(output).resolve()
    manifest = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest["source_sha256"] != source_hashes():
        raise ValueError("Source differs from the planned snapshot; make a fresh batch")
    for job in manifest["jobs"]:
        path = output / job["config"]
        if sha(path) != job["config_sha256"]:
            raise ValueError(f"Planned config changed: {path}")
    selected = [j for j in manifest["jobs"] if arms is None or j["arm"] in arms]
    if not selected or arms is not None and set(arms) - {j["arm"] for j in selected}:
        raise ValueError("Requested arms are not in this plan")
    return output, manifest, [(j, yaml.safe_load((output / j["config"]).read_text(encoding="utf-8"))) for j in selected]


def split_ids(config, fold):
    import pandas as pd
    data = config["data"]
    split = Path(data["data_path"]) / "splits" / config["split_setup"]["which_splits"] / data["study"] / f"fold_{fold}.csv"
    table = pd.read_csv(split)
    protocol = config["train"]["evaluation_protocol"]
    train, val = ("inner_train", "inner_val") if protocol == "outer_test" else ("train", "val")
    excluded = (val, "outer_test") if protocol == "outer_test" else (val,)
    ids = table[train].dropna().astype(str).tolist()
    evaluation = table[val].dropna().astype(str).tolist()
    forbidden = [str(p) for key in excluded for p in table[key].dropna()]
    if (len(ids) != len(set(ids)) or len(evaluation) != len(set(evaluation))
            or not ids or not evaluation or set(ids) & set(forbidden)):
        raise ValueError("Invalid or overlapping declared patient cohorts")
    return split, ids, evaluation, forbidden


def bind_inputs(output, jobs):
    """Server-only input binding: CSV hashes and stored-feature file metadata.

    Feature content hashes are not claimed: the original stored file sizes and
    nanosecond modification times are checked on every stage and acceptance.
    """
    config = jobs[0][1]
    data = config["data"]
    metadata = Path(data["data_path"])
    signature = data["signature"]
    signature_file = "signatures.csv" if signature == "six" else (
        "combine_signatures.csv" if signature == "all" else f"{signature}_signatures.csv")
    files = [metadata / "clinical/all" / f"{data['study']}.csv",
             metadata / "raw_rna_data_inter" / f"{data['study']}_rna_inter.csv",
             metadata / "signatures" / signature_file]
    files += [split_ids(c, j["fold"])[0] for j, c in jobs]
    feature_dir = Path(data["data_root_dir"]) / data["study"] / config["model"]["wsi_encoder"] / "pt_files"
    features = sorted(feature_dir.glob("*.pt"))
    if not features:
        raise FileNotFoundError(f"No stored real WSI features: {feature_dir}")
    binding = dict(csv_sha256={str(p): sha(p) for p in sorted(set(files))},
        wsi_file_metadata={str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in features},
        feature_verification="size and mtime_ns; not full content SHA256")
    path = Path(output) / "inputs.json"
    # Bind all plan folds even when this invocation runs only one arm.
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != binding:
            raise ValueError("Metadata or stored features changed since input binding")
    else:
        dump_new(path, binding)
    return binding


def audit_calibration(config, fold, path):
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import SurvivalDatasetFactory
    from survot_rank.research.methods.dct_v23.calibration import training_reference, raw_expression_matrix, load_calibration
    split, ids, _, excluded = split_ids(config, fold)
    data = config["data"]
    factory = SurvivalDatasetFactory(study=data["study"], data_path=data["data_path"],
        rna_format="Pathways", signature=data["signature"], n_bins=data["n_classes"],
        label_col=data["label_col"], num_genes=None, num_patches=None,
        binning_mode=config["train"]["binning_mode"], which_splits=config["split_setup"]["which_splits"])
    # Preserve the generator/trainer's clinical order for the RNA digest.
    ids = factory.clinical_df.loc[factory.clinical_df["case id"].astype(str).isin(ids), "case id"].astype(str).tolist()
    declared = split_ids(config, fold)[1]
    if set(ids) != set(declared):
        raise ValueError("Calibration patients do not resolve exactly")
    ref = training_reference(factory, ids)
    gains, provenance = load_calibration(path, patient_ids=ids, excluded_ids=excluded,
        pathway_gene_ids=ref["pathway_gene_ids"], raw_expression_sha256=ref["raw_expression_sha256"],
        split_sha256=sha(split), encoding_dim=data["encoding_dim"], wsi_encoder=config["model"]["wsi_encoder"],
        raw_expression=raw_expression_matrix(factory, ids, ref["gene_ids"]))
    if (provenance.get("fold") != fold or provenance.get("study") != data["study"]
            or provenance.get("evaluation_protocol") != config["train"]["evaluation_protocol"]
            or provenance.get("config_sha256") != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            or not provenance.get("wsi_file_fingerprints")):
        raise ValueError("Calibration has incomplete fold/data/WSI provenance")
    options = dict(inner_folds=3, mask_rounds=8, mask_ratio=config["reconstruction"]["dct_v23_mask_ratio"],
                   native_dim=32, wsi_dim=16, ridge_alpha=10., seed=config["train"]["seed"])
    if any(provenance.get(name) != value for name, value in options.items()):
        raise ValueError("Calibration teacher options differ from the planned recipe")
    return dict(path=str(path), sha256=sha(path), patient_count=len(gains),
                teacher_count=len(provenance["teachers"]), negative_gains_retained=True)


def calibrate(output):
    from scripts.prepare_dct_v23_calibration import prepare_one_fold
    output, manifest, jobs = read_plan(output, ["gain_calibrated"])
    bind_inputs(output, jobs)
    records = []
    try:
        for job, config in jobs:
            fold = job["fold"]
            path = output / "calibration" / f"fold_{fold}.json"
            if not path.exists():
                prepare_one_fold(config, fold, path.parent, manifest["teacher_options"])
            records.append(audit_calibration(config, fold, path))
    except BaseException as exc:
        write_report(output, "calibration_failure", dict(error=repr(exc), accepted_before_failure=records))
        raise
    binding = {str(Path(r["path"]).relative_to(output)): r["sha256"] for r in records}
    path = output / "calibration_binding.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != binding:
            raise ValueError("An already accepted calibration table changed")
    else:
        dump_new(path, binding)
    write_report(output, "calibration", records)
    return records


def check_calibration_binding(output, fold):
    output = Path(output)
    binding = json.loads((output / "calibration_binding.json").read_text(encoding="utf-8"))
    name = f"calibration/fold_{fold}.json"
    if binding.get(name) != sha(output / name):
        raise ValueError("Accepted calibration content changed; do not mix teacher batches")


def audit_job(config, fold):
    import numpy as np
    import pandas as pd
    import torch
    from sksurv.metrics import concordance_index_censored
    split, train_ids, evaluation, _ = split_ids(config, fold)
    checkpoints = list(Path(config["train"]["results_dir"]).rglob(f"model_best_s{fold}.pth"))
    if len(checkpoints) != 1:
        raise ValueError("Expected exactly one checkpoint under this fresh job root")
    directory = checkpoints[0].parent
    curve = pd.read_csv(directory / f"epoch_curve_fold{fold}.csv")
    epochs = config["train"]["max_epochs"]
    if not np.array_equal(curve["epoch"].to_numpy(), np.arange(epochs)):
        raise ValueError("Actual epochs do not cover the requested budget")
    for key in ("val_cindex", "val_loss"):
        if not np.isfinite(curve[key].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite curve: {key}")
    ckpt = directory / f"model_best_s{fold}.pth"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite checkpoint tensor: {key}")
    extra = state.get("_extra_state", {})
    if extra.get("schema") != "dct_v23_standalone_v1":
        raise ValueError("Checkpoint is not the v2.3 candidate")
    from survot_rank.config import config_to_argv
    from survot_rank.training.extended_args import process_args_extended
    from survot_rank.research.methods.dct_v23.config import DCTV23Config
    args = process_args_extended(config_to_argv(config))
    args.omic_sizes = extra["config"]["omic_sizes"]
    if DCTV23Config.from_args(args).to_dict() != DCTV23Config.from_dict(extra["config"]).to_dict():
        raise ValueError("Checkpoint recipe differs from planned configuration")
    if set(extra.get("train_patient_ids", [])) != set(train_ids):
        raise ValueError("Checkpoint training patients differ")
    if extra.get("reconstruction_mode") == "gain_calibrated":
        table = json.loads((Path(config["reconstruction"]["dct_v23_calibration_dir"]) / f"fold_{fold}.json").read_text(encoding="utf-8"))
        saved_gains = extra.get("oof_gain_targets", {})
        same_gains = set(saved_gains) == set(table["gains"]) and all(
            np.asarray(saved_gains[cid]).shape == np.asarray(values).shape
            and np.allclose(saved_gains[cid], values, rtol=1e-6, atol=1e-7)
            for cid, values in table["gains"].items())
        if not same_gains or extra.get("oof_gain_provenance") != table["provenance"]:
            raise ValueError("Checkpoint was trained with different recovery gains")
    from survot_rank.research.methods.dct_v23 import DCTV23
    # Prove that the saved parameters can actually load into the recorded model.
    candidate = DCTV23(DCTV23Config.from_dict(extra["config"]), omic_names=extra["pathway_gene_ids"])
    candidate.load_state_dict(state, strict=True)
    pred_path = directory / f"split_{fold}_results_final.pkl"
    with pred_path.open("rb") as handle:
        predictions = pickle.load(handle)
    if not isinstance(predictions, dict) or set(predictions) != set(evaluation):
        raise ValueError("Predictions must cover every and only declared validation patient")
    risks, times, censors = [], [], []
    for cid in evaluation:
        record = predictions[cid]
        for key in ("risk", "time", "censor", "logits"):
            if not np.isfinite(np.asarray(record[key], dtype=float)).all():
                raise ValueError(f"Nonfinite prediction for {cid}: {key}")
        if np.asarray(record["logits"]).size != config["data"]["n_classes"]:
            raise ValueError("Wrong prediction class count")
        logit = torch.as_tensor(record["logits"], dtype=torch.float64).reshape(1, -1)
        expected_risk = float(-(1-torch.sigmoid(logit)).cumprod(-1).sum())
        if not np.isclose(float(record["risk"]), expected_risk, rtol=1e-5, atol=1e-6):
            raise ValueError("Risk does not match saved survival logits")
        risks.append(float(record["risk"]))
        times.append(float(record["time"]))
        censors.append(float(record["censor"]))
    if any(c not in (0., 1.) for c in censors) or any(t < 0 for t in times):
        raise ValueError("Invalid survival endpoints")
    clinical = pd.read_csv(Path(config["data"]["data_path"]) / "clinical/all" / f"{config['data']['study']}.csv")
    label = config["data"]["label_col"]
    censor = "censorship_" + label.rsplit("_", 1)[1]
    for cid, t, c in zip(evaluation, times, censors):
        rows = clinical.loc[clinical["case id"].astype(str) == cid, [label, censor]].drop_duplicates()
        if len(rows) != 1 or not np.allclose(rows.iloc[0].to_numpy(dtype=float), [t, c], rtol=1e-5, atol=1e-5):
            raise ValueError("Saved survival endpoints disagree with clinical source")
    measured = float(concordance_index_censored(~np.asarray(censors, dtype=bool), np.asarray(times), np.asarray(risks))[0])
    peak = float(curve["val_cindex"].max())
    if not np.isclose(measured, peak, rtol=0, atol=1e-7):
        raise ValueError("Saved predictions disagree with selected validation peak")
    return dict(fold=fold, actual_epochs=len(curve), best_epoch=int(curve["val_cindex"].idxmax()),
        val_cindex=measured, val_loss_at_best=float(curve.loc[curve["val_cindex"].idxmax(), "val_loss"]),
        train_patient_ids=train_ids, val_patient_ids=evaluation, split_sha256=sha(split),
        checkpoint=str(ckpt), checkpoint_sha256=sha(ckpt), predictions=str(pred_path),
        predictions_sha256=sha(pred_path), curve_sha256=sha(directory / f"epoch_curve_fold{fold}.csv"),
        evidence="legacy_val validation peak; not held-out final performance")


def write_report(output, kind, records):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    dump_new(Path(output) / "reports" / f"{kind}_{stamp}.json", dict(schema=SCHEMA, records=records))


def run_jobs(output, arms=None):
    output, _, jobs = read_plan(output, arms)
    try:
        bind_inputs(output, read_plan(output)[2])
        # Check the entire requested queue before launching its first patient job.
        for job, config in jobs:
            if Path(config["train"]["results_dir"]).exists():
                raise FileExistsError("Run output already exists; audit it or plan a fresh batch")
            if job["arm"] == "gain_calibrated":
                check_calibration_binding(output, job["fold"])
                audit_calibration(config, job["fold"], output / "calibration" / f"fold_{job['fold']}.json")
    except BaseException as exc:
        write_report(output, "preflight_failure", dict(error=repr(exc)))
        raise
    records = []
    for job, config in jobs:
        command = [sys.executable, "-m", "survot_rank.cli", "train", "--config", str(output / job["config"])]
        log = output / "logs" / f"{job['arm']}_fold{job['fold']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config["train"]["gpu"]), PYTHONHASHSEED=str(config["train"]["seed"]))
        try:
            with log.open("x", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Training exited {result.returncode}; see {log}")
            acceptance = audit_job(config, job["fold"])
        except BaseException as exc:
            write_report(output, "failure", [dict(arm=job["arm"], fold=job["fold"], error=repr(exc), log=str(log))])
            raise
        acceptance.update(arm=job["arm"], wall_seconds=(datetime.now(timezone.utc)-started).total_seconds(), command=command)
        records.append(acceptance)
        write_report(output, "completed", [acceptance])
    return records


def audit(output, arms=None):
    output, _, jobs = read_plan(output, arms)
    bind_inputs(output, read_plan(output)[2])
    records, failed = [], False
    for job, config in jobs:
        try:
            if job["arm"] == "gain_calibrated":
                check_calibration_binding(output, job["fold"])
                audit_calibration(config, job["fold"], output / "calibration" / f"fold_{job['fold']}.json")
            records.append(dict(arm=job["arm"], status="accepted", **audit_job(config, job["fold"])))
        except Exception as exc:
            failed = True
            records.append(dict(arm=job["arm"], fold=job["fold"], status="waiting_or_failed", error=repr(exc)))
    write_report(output, "audit", records)
    if failed:
        raise RuntimeError("Some requested jobs lack valid complete artifacts; see reports")
    # Paired differences only within this manifest, never across unmatched runs.
    by_key = {(r["arm"], r["fold"]): r for r in records}
    pairs = (("off", "masked_plain"), ("masked_plain", "gain_calibrated")) if "off" in {r["arm"] for r in records} else (
        ("native_only", "combined"), ("transport_only", "combined"), ("independent", "combined"))
    deltas = []
    for left, right in pairs:
        for fold in sorted({r["fold"] for r in records}):
            a, b = by_key.get((left, fold)), by_key.get((right, fold))
            if a and b:
                if a["split_sha256"] != b["split_sha256"] or a["val_patient_ids"] != b["val_patient_ids"]:
                    raise ValueError("Paired cohorts differ")
                deltas.append(dict(left=left, right=right, fold=fold, right_minus_left=b["val_cindex"]-a["val_cindex"]))
    write_report(output, "paired", deltas)
    return records


def audit_existing(config_path, results_dir, folds, report_dir):
    """Inspect old outputs without changing them or trusting driver status.

    Old source identity is not recoverable from weights alone. An accepted
    artifact here still requires its original command/source/data provenance
    before it can be reused as a matched experiment.
    """
    import yaml
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config["train"]["results_dir"] = str(Path(results_dir).resolve())
    Path(report_dir).mkdir(parents=True, exist_ok=False)
    records, failed = [], False
    for fold in folds:
        try:
            records.append(dict(status="artifacts_accepted_source_pending", **audit_job(config, fold)))
        except Exception as exc:
            failed = True
            records.append(dict(fold=fold, status="waiting_or_failed", error=repr(exc)))
    write_report(report_dir, "existing", dict(config_path=str(config_path), config_sha256=sha(config_path),
        results_dir=str(results_dir), current_auditor_source=source_hashes(), records=records,
        reuse_status="Original driver, command, source and feature provenance must be checked separately"))
    if failed:
        raise RuntimeError("Historical artifacts incomplete or invalid; report retained")
    return records


def verify_delivery():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Delivery changed: {name}")
    print(f"Verified {len(manifest['files'])} delivered files. No patient data read.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "plan", "calibrate", "run", "audit", "audit-existing"))
    parser.add_argument("--output-root")
    parser.add_argument("--config")
    parser.add_argument("--results-dir")
    parser.add_argument("--stage", choices=("recovery", "core"), default="recovery")
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--data-path")
    parser.add_argument("--features-root")
    parser.add_argument("--legacy-driver", action="append", default=[])
    parser.add_argument("--arms", nargs="+")
    args = parser.parse_args(argv)
    if args.action == "verify":
        verify_delivery()
        return
    if not args.output_root:
        parser.error("--output-root is required for this action")
    if args.action == "audit-existing":
        if not args.config or not args.results_dir:
            parser.error("audit-existing requires the actual historical --config and --results-dir")
        records = audit_existing(args.config, args.results_dir, args.folds, args.output_root)
        print(f"Accepted {len(records)} historical artifact sets; original source still requires audit")
        return
    if args.action == "plan":
        result = make_plan(args.output_root, stage=args.stage, folds=args.folds, seed=args.seed, gpu=args.gpu,
            data_path=args.data_path, features_root=args.features_root, legacy_drivers=args.legacy_driver)
        print(f"Planned {len(result['jobs'])} jobs. No patient data read and no job started.")
    else:
        result = calibrate(args.output_root) if args.action == "calibrate" else (
            run_jobs(args.output_root, args.arms) if args.action == "run" else audit(args.output_root, args.arms))
        print(f"Accepted {len(result)} records")


if __name__ == "__main__":
    main()
