"""Prepare shared, complete-case inputs without training or model repairs.

Importing uses the standard library only. Explicit ``prepare`` reads configured
server inputs and creates an immutable, sampled cache per patient. This module
contains no training loop and imports no result from earlier local audits.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone


SCHEMA_VERSION = 1
_DEFAULTS = {
    "study": "blca", "label_col": "survival_months_dss", "signature": "combine",
    "folds": [0, 1, 2, 3, 4], "seeds": [3, 17, 29], "epochs": 30,
    "patches": 4096, "batch_size": 8, "eval_repeats": 20,
    "encoding_dim": None, "device": "cuda:0", "lr": 5e-4,
    "experiment": "reproduction",
}
_CENSORS = {"survival_months_os": "censorship_os", "survival_months_pfi": "censorship_pfi",
            "survival_months_dss": "censorship_dss", "survival_months_dfs": "censorship_dfs"}


def _config(config):
    result = {**_DEFAULTS, **dict(config)}
    for name in ("source_root", "metadata_root", "features_root"):
        value = Path(result[name]).expanduser()
        if not value.is_absolute():
            raise ValueError(f"{name} must be an absolute server path")
        result[name] = str(value.resolve())
    for name in ("epochs", "patches", "batch_size", "eval_repeats"):
        value = result[name]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        result[name] = int(value)
    for name in ("folds", "seeds"):
        values = list(result[name])
        if not values or any(isinstance(v, bool) or int(v) != v or int(v) < 0 for v in values):
            raise ValueError(f"{name} must be a nonempty list of nonnegative integers")
        result[name] = [int(v) for v in values]
        if len(set(result[name])) != len(result[name]):
            raise ValueError(f"{name} contains duplicates")
    if result["encoding_dim"] is not None:
        if int(result["encoding_dim"]) != result["encoding_dim"] or int(result["encoding_dim"]) < 1:
            raise ValueError("encoding_dim must be a positive integer or null")
        result["encoding_dim"] = int(result["encoding_dim"])
    if result["label_col"] not in _CENSORS:
        raise ValueError("Unsupported survival endpoint")
    if not math.isfinite(float(result["lr"])) or float(result["lr"]) <= 0:
        raise ValueError("lr must be positive and finite")
    result["lr"] = float(result["lr"])
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _source_hashes(config):
    root = Path(config["source_root"])
    files = []
    for directory in ("models", "utils", "dataset"):
        files.extend(p for p in (root / directory).rglob("*.py") if "__pycache__" not in p.parts)
    if not (root / "models/SlotSPE.py").is_file() or not (root / "dataset/dataset_survival.py").is_file():
        raise FileNotFoundError("Configured source_root must contain models/SlotSPE.py and dataset/dataset_survival.py")
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(files)}


def _csv_paths(config):
    root = Path(config["metadata_root"])
    study = config["study"]
    signature = "signatures.csv" if config["signature"] == "six" else (
        "combine_signatures.csv" if config["signature"] == "all" else f"{config['signature']}_signatures.csv")
    result = [root / "clinical/all" / f"{study}.csv",
              root / "raw_rna_data_inter" / f"{study}_rna_inter.csv",
              root / "signatures" / signature]
    result.extend(root / "splits/5fold" / study / f"fold_{fold}.csv" for fold in config["folds"])
    for path in result:
        if not path.is_file():
            raise FileNotFoundError(f"Required metadata absent (RNA is never substituted): {path}")
    return result


def _preparation_config(config):
    return {name: config[name] for name in ("source_root", "metadata_root", "features_root", "study",
            "label_col", "signature", "folds", "patches", "encoding_dim")}


def _dataset_module(config):
    """Load exactly the requested dataset file, independent of sys.path."""
    path = Path(config["source_root"]) / "dataset/dataset_survival.py"
    name = "_slotspe_proof_dataset_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load requested dataset module: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError("Dataset source provenance mismatch")
    return module


def _manifest(config, output_root, verify_cache=False):
    root = Path(output_root)
    path = root / "prepared/manifest.json"
    manifest = _read(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported prepared manifest schema")
    if manifest["preparation_config"] != _preparation_config(config):
        raise RuntimeError("Preparation configuration changed; use a fresh batch output directory")
    if manifest["source_hashes"] != _source_hashes(config):
        raise RuntimeError("SlotSPE source changed after preparation; use a fresh batch")
    current_csv = {str(p): _sha(p) for p in _csv_paths(config)}
    if manifest["input_csv_sha256"] != current_csv:
        raise RuntimeError("Clinical/RNA/signature/split CSV changed after preparation")
    if verify_cache:
        for patient in manifest["patients"].values():
            cache = root / patient["cache_path"]
            if not cache.is_file() or _sha(cache) != patient["cache_sha256"]:
                raise RuntimeError(f"Prepared input cache changed: {cache}")
    return manifest


def prepare(config, output_root):
    """Cache real paired inputs once, then fit separate train-only bins per fold.

    Missing clinical endpoints, RNA columns, slides, nonfinite inputs, and bags
    shorter than ``patches`` are explicitly excluded as complete cases.  Missing
    RNA CSV is a hard error.  No missing feature is replaced by a zero vector.
    An existing complete cache can be verified and reused; a partial preparation
    is deliberately not overwritten.
    """
    config = _config(config)
    root = Path(output_root)
    prepared = root / "prepared"
    if (prepared / "manifest.json").exists():
        return _manifest(config, root, verify_cache=True)
    if prepared.exists() and any(prepared.iterdir()):
        raise RuntimeError("Partial preparation exists; retain it and select a fresh output batch")
    # Dependency imports belong to explicit execution, never queue planning.
    import numpy as np
    import pandas as pd
    import torch

    csv_hashes = {str(p): _sha(p) for p in _csv_paths(config)}
    source_hashes = _source_hashes(config)
    module = _dataset_module(config)
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "cache/patients").mkdir(parents=True)
    start = time.perf_counter()
    censor = _CENSORS[config["label_col"]]

    class TrainBinningFactory(module.SurvivalDatasetFactory):
        def _setup_clinical_data(self):
            # The original factory bins before splitting.  This paired proof
            # replaces ONLY that preparation, shared by both model arms.
            path = Path(self.data_path) / "clinical/all" / f"{self.study}.csv"
            clinical = pd.read_csv(path)
            columns = ["case id", self.label_col, self.censorship_var, "wsi"]
            clinical = clinical[columns].dropna().reset_index(drop=True)
            clinical["case id"] = clinical["case id"].astype(str)
            if clinical["case id"].duplicated().any():
                raise ValueError("Duplicate clinical patient IDs")
            clinical.insert(2, "label", -1)
            self.clinical_df = clinical
            self.bins = None

    with (prepared / "factory_prepare.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        factory = TrainBinningFactory(config["study"], config["metadata_root"], "Pathways", config["label_col"],
                    signature=config["signature"], n_bins=4, num_patches=config["patches"])
    if not factory.omic_sizes or any(int(n) < 1 for n in factory.omic_sizes):
        raise ValueError("No nonempty real RNA pathways overlap the signature")
    if factory.gene_data_df.columns.duplicated().any() or factory.gene_data_df.index.duplicated().any():
        raise ValueError("RNA contains duplicate patient columns or gene indices")
    clinical_ids = set(factory.clinical_df["case id"])
    patients, exclusions, folds = {}, {}, {}
    encoding_dim = config["encoding_dim"]

    def cache_patient(row, dataset):
        nonlocal encoding_dim
        pid = str(row["case id"])
        if pid in patients:
            return None
        if pid in exclusions:
            return exclusions[pid]
        if pid not in factory.gene_data_df.columns:
            return "missing_rna_patient_column"
        event_time, event_censor = float(row[config["label_col"]]), float(row[censor])
        if not math.isfinite(event_time) or event_time < 0 or event_censor not in (0.0, 1.0):
            return "invalid_survival_endpoint"
        slide_names = [s.strip() for s in str(row["wsi"]).split(",") if s.strip()]
        if not slide_names:
            return "missing_slide_identifiers"
        # Match the requested original loader's feature filename convention.
        paths = [Path(config["features_root"]) / (s.rstrip(".svs") + ".pt") for s in slide_names]
        if not all(path.is_file() for path in paths):
            return "at_least_one_missing_slide_feature"
        # Treat zero-byte / corrupted feature files as a hard exclusion; data
        # was never complete, so record every affected patient and pick them up
        # through the standard exclusion summary.
        zero_byte = [path for path in paths if path.stat().st_size == 0]
        if zero_byte:
            return "zero_byte_or_truncated_slide_feature"
        chunks, slide_info = [], []
        for path in paths:
            before = path.stat()
            try:
                tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            except RuntimeError:
                # Old torch-save containers do not support mmap; still load a
                # single patient's slides, never the full cohort.
                tensor = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
                raise ValueError(f"Feature file must contain a nonempty [patch,channel] tensor: {path}")
            if encoding_dim is None:
                encoding_dim = int(tensor.shape[1])
            if tensor.shape[1] != encoding_dim:
                raise ValueError(f"Encoding width mismatch in {path}: {tensor.shape[1]} != {encoding_dim}")
            chunks.append(tensor)
            slide_info.append({"path": str(path.resolve()), "patch_count": int(tensor.shape[0]),
                "channels": int(tensor.shape[1]), "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns})
        sizes = [len(t) for t in chunks]
        total = sum(sizes)
        if total < config["patches"]:
            return "fewer_real_patches_than_configured_no_padding"
        sample_seed = int(hashlib.sha256((pid + "_slotspe_paired_proof_v1").encode()).hexdigest()[:8], 16)
        indices = np.sort(np.random.default_rng(sample_seed).choice(total, config["patches"], replace=False))
        cumulative = np.cumsum([0] + sizes)
        bag = torch.empty((config["patches"], encoding_dim), dtype=torch.float32)
        for index, tensor in enumerate(chunks):
            selected = (indices >= cumulative[index]) & (indices < cumulative[index + 1])
            bag[selected] = tensor[indices[selected] - cumulative[index]].float()
        omics = dataset.load_genes(pid)
        if not isinstance(omics, (list, tuple)) or len(omics) != len(factory.omic_sizes):
            raise ValueError("Original Pathways loader returned an unexpected RNA layout")
        if any(g.ndim != 1 or len(g) != int(size) for g, size in zip(omics, factory.omic_sizes)):
            raise ValueError("Original RNA pathway width changed")
        if not torch.isfinite(bag).all() or not all(torch.isfinite(g).all() for g in omics):
            return "nonfinite_selected_inputs"
        for path, info in zip(paths, slide_info):
            after = path.stat()
            if after.st_size != info["size_bytes"] or after.st_mtime_ns != info["mtime_ns"]:
                raise RuntimeError(f"Feature file changed during cache preparation: {path}")
        name = hashlib.sha256(pid.encode()).hexdigest() + ".pt"
        cache = prepared / "cache/patients" / name
        torch.save({"patient_id": pid, "wsi": bag, "omics": [g.detach().cpu().float() for g in omics]}, cache)
        patients[pid] = {"patient_id": pid, "time": event_time, "censor": event_censor,
            "cache_path": cache.relative_to(root).as_posix(), "cache_sha256": _sha(cache),
            "full_patch_count": total, "sample_seed": sample_seed,
            "patch_indices_concatenated_slide_order": indices.tolist(), "slides": slide_info,
            "selected_patch_tensor_sha256": hashlib.sha256(bag.numpy().tobytes()).hexdigest()}
        return None

    for fold in config["folds"]:
        split_path = Path(config["metadata_root"]) / "splits/5fold" / config["study"] / f"fold_{fold}.csv"
        split = pd.read_csv(split_path)
        split_ids = {}
        for kind in ("train", "val"):
            ids = split[kind].dropna().astype(str).tolist()
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate IDs in fold {fold} {kind}")
            split_ids[kind] = ids
        if set(split_ids["train"]) & set(split_ids["val"]):
            raise ValueError(f"Patient overlap in fold {fold}")
        fold_excluded, retained = [], {}
        for kind in ("train", "val"):
            for pid in split_ids[kind]:
                if pid not in clinical_ids:
                    fold_excluded.append({"patient_id": pid, "split": kind,
                                          "reason": "missing_clinical_endpoint_or_wsi_string"})
            dataset = module.SurvivalDataset(factory, config["features_root"], kind, fold, encoding_dim or 1)
            retained[kind] = []
            for _, row in dataset.label_df.iterrows():
                pid = str(row["case id"])
                reason = cache_patient(row, dataset)
                if reason:
                    exclusions[pid] = reason
                    fold_excluded.append({"patient_id": pid, "split": kind, "reason": reason})
                else:
                    retained[kind].append(pid)
            if not retained[kind]:
                raise ValueError(f"No complete-case {kind} patients remain in fold {fold}")
        event_times = [patients[pid]["time"] for pid in retained["train"] if patients[pid]["censor"] < 1]
        if len(event_times) < 4:
            raise ValueError(f"Fold {fold} has fewer than four uncensored training patients")
        _, edges = pd.qcut(event_times, q=4, labels=False, retbins=True)
        edges[0], edges[-1] = -np.inf, np.inf
        labels = {pid: int(np.searchsorted(edges[1:-1], patients[pid]["time"], side="right"))
                  for kind in ("train", "val") for pid in retained[kind]}
        folds[str(fold)] = {"fold": fold, "split_csv": str(split_path), "train": retained["train"],
            "val": retained["val"], "labels": labels, "excluded": fold_excluded,
            "original_split_counts": {kind: len(ids) for kind, ids in split_ids.items()},
            "counts": {"train": len(retained["train"]), "val": len(retained["val"]),
                       "excluded": len(fold_excluded), "overlap": 0},
            "binning": {"fit_subset": "complete_case_train_uncensored_only", "q": 4,
                "edges": ["-inf", *[float(x) for x in edges[1:-1]], "inf"], "right": False,
                "same_labels_both_arms": True, "validation_outcomes_used_for_training_bins": False}}
        print(f"Prepared fold={fold} train={len(retained['train'])} val={len(retained['val'])} excluded={len(fold_excluded)}", flush=True)
    if source_hashes != _source_hashes(config) or csv_hashes != {str(p): _sha(p) for p in _csv_paths(config)}:
        raise RuntimeError("Source or metadata changed during preparation")
    manifest = {"schema_version": SCHEMA_VERSION, "prepared_utc": _now(),
        "preparation_config": _preparation_config(config), "source_root": config["source_root"],
        "source_hashes": source_hashes, "input_csv_sha256": csv_hashes,
        "dataset_module_path": str(Path(config["source_root"]) / "dataset/dataset_survival.py"),
        "omic_sizes": [int(n) for n in factory.omic_sizes], "encoding_dim": encoding_dim,
        "patches": config["patches"], "folds": folds, "patients": patients,
        "counts": {"unique_complete_case_patients": len(patients), "per_fold": {f: v["counts"] for f, v in folds.items()}},
        "exclusion_policy": "Complete observed clinical/RNA/all slides, finite selected inputs, at least configured real patches; no zero substitution or padding",
        "patch_policy": "Fixed per-patient seeded sampling without replacement, shared by arms, folds, epochs, and evaluations",
        "feature_provenance": "Original feature file path/shape/size/mtime retained; cached tensors content-hashed; full original feature files are not content-hashed",
        "rna_transform": "Requested original Pathways loader, no added normalization",
        "scope": "Shared input preparation only; no training or performance gain asserted",
        "preparation_seconds": time.perf_counter() - start}
    _dump(prepared / "manifest.json", manifest)
    return manifest


def _rng(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch(root, manifest, fold_data, ids, device, torch):
    cache = [torch.load(Path(root) / manifest["patients"][pid]["cache_path"],
                        map_location="cpu", weights_only=True) for pid in ids]
    if any(item["patient_id"] != pid for item, pid in zip(cache, ids)):
        raise RuntimeError("Prepared cache patient identity mismatch")
    batch = {"x_wsi": torch.stack([item["wsi"] for item in cache]).to(device),
        "y": torch.tensor([fold_data["labels"][pid] for pid in ids], dtype=torch.long, device=device),
        "c": torch.tensor([manifest["patients"][pid]["censor"] for pid in ids], dtype=torch.float32, device=device),
        "omic_missing": False}
    for k in range(len(manifest["omic_sizes"])):
        batch[f"x_omic{k + 1}"] = torch.stack([item["omics"][k] for item in cache]).to(device)
    return batch
