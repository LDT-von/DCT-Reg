#!/usr/bin/env python3
"""Explicit patient-data job: fit train-only, patient-held-out v23 gain teachers.

This command never launches the survival model. Running it reads training RNA
and stored WSI features and fits ridge teachers; only run with user authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd

from survot_rank.research.methods.dct_v23.calibration import raw_expression_matrix, training_reference
from survot_rank.research.methods.dct_v23.crossfit import fit_cross_patient_gains, save_gain_table


def prepare_one_fold(config, fold, output_dir, teacher_options):
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import SurvivalDatasetFactory, SurvivalDataset
    destination = Path(output_dir) / f"fold_{fold}.json"
    if destination.exists():
        raise FileExistsError(f"Calibration already exists: {destination}; use a fresh output directory")
    data, model, training = config["data"], config.get("model", {}), config["train"]
    setup = config.get("split_setup", {})
    if data.get("rna_format") != "Pathways":
        raise ValueError("DCT v23 calibration requires Pathways raw-gene inputs")
    factory = SurvivalDatasetFactory(study=data["study"], data_path=data["data_path"],
        rna_format="Pathways", signature=data["signature"], n_bins=data.get("n_classes", 4),
        label_col=data["label_col"], num_genes=None, num_patches=None,
        binning_mode=training.get("binning_mode", "global_qcut"), which_splits=setup.get("which_splits", "5fold"))
    split_path = Path(factory.data_path) / "splits" / factory.which_splits / factory.study / f"fold_{fold}.csv"
    split = pd.read_csv(split_path)
    protocol = training.get("evaluation_protocol", "legacy_val")
    train_column = "inner_train" if protocol == "outer_test" else "train"
    excluded_columns = ("inner_val", "outer_test") if protocol == "outer_test" else ("val",)
    for column in (train_column, *excluded_columns):
        if column not in split:
            raise ValueError(f"Split missing column {column}")
    declared_ids = split[train_column].dropna().astype(str).tolist()
    excluded = set(x for column in excluded_columns for x in split[column].dropna().astype(str))
    if len(declared_ids) != len(set(declared_ids)) or set(declared_ids) & excluded:
        raise ValueError("Training split has duplicates or overlaps evaluation patients")
    resolved = factory.clinical_df[factory.clinical_df["case id"].astype(str).isin(declared_ids)]
    ids = resolved["case id"].astype(str).tolist()
    if set(ids) != set(declared_ids):
        raise ValueError("Every declared training patient must resolve in clinical data")
    reference = training_reference(factory, ids)
    raw = raw_expression_matrix(factory, ids, reference["gene_ids"])
    encoder = model.get("wsi_encoder", "uni")
    wsi_dir = Path(data["data_root_dir"]) / factory.study / encoder / "pt_files"
    dataset = SurvivalDataset(factory, str(wsi_dir), train_column, fold,
        data.get("encoding_dim", 768), on_missing_wsi="error")
    pooled, fingerprints, seen_files = [], [], set()
    for patient_id in ids:
        slides = resolved.loc[resolved["case id"].astype(str) == patient_id, "wsi"].iloc[0]
        if str(slides) == "nan":
            raise ValueError(f"Calibration requires real paired WSI for {patient_id}")
        features = dataset.load_wsi(str(slides))
        if features.ndim != 2 or features.shape[0] < 1 or not bool(features.isfinite().all()):
            raise ValueError(f"Invalid stored WSI for {patient_id}")
        pooled.append(features.mean(0).numpy())
        for slide in str(slides).split(", "):
            path = dataset._resolve_wsi_feature_path(slide)
            if path is not None and str(path) not in seen_files:
                stat = Path(path).stat()
                fingerprints.append(dict(path=str(Path(path).resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
                seen_files.add(str(path))
    payload = fit_cross_patient_gains(raw, np.stack(pooled), patient_ids=ids,
        pathway_gene_ids=reference["pathway_gene_ids"], **teacher_options,
        extra_provenance=dict(fold=fold, study=factory.study, evaluation_protocol=protocol,
            train_split_column=train_column, excluded_patient_ids=sorted(excluded),
            split_path=str(split_path.resolve()), split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            encoding_dim=data.get("encoding_dim", 768), wsi_encoder=encoder,
            pathway_names=list(factory.pathway_names), wsi_file_fingerprints=fingerprints,
            wsi_pooling="mean over all available stored real patch features; no padding or sampling"))
    save_gain_table(payload, destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--mask-rounds", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=.3)
    parser.add_argument("--native-dim", type=int, default=32)
    parser.add_argument("--wsi-dim", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)
    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    settings = {name: getattr(args, name) for name in ("inner_folds", "mask_rounds", "mask_ratio", "native_dim", "wsi_dim", "ridge_alpha", "seed")}
    if len(args.folds) != len(set(args.folds)):
        raise ValueError("Duplicate fold requests are not allowed")
    for fold in args.folds:
        print(f"Saved patient-held-out calibration: {prepare_one_fold(config, fold, args.output_dir, settings)}")


if __name__ == "__main__":
    main()
