"""Strict train-only binding for out-of-fold raw-expression benefit targets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA = "dct_v23_oof_pathway_gain_v1"


def global_gene_ids(pathway_gene_ids):
    result = list(dict.fromkeys(str(g) for genes in pathway_gene_ids for g in genes))
    if len(result) < 2 or any(not genes for genes in pathway_gene_ids):
        raise ValueError("At least two unique genes and nonempty pathways are required")
    return result


def array_digest(array):
    value = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def raw_expression_matrix(factory, patient_ids, gene_ids):
    if not factory.gene_data_df.index.is_unique or not factory.gene_data_df.columns.is_unique:
        raise ValueError("RNA gene and patient coordinates must be unique")
    values = factory.gene_data_df.loc[gene_ids, list(patient_ids)].to_numpy(dtype=np.float64).T
    if not np.isfinite(values).all():
        raise ValueError("Raw expression contains nonfinite values")
    return values


def training_reference(factory, patient_ids):
    ids = [str(x) for x in patient_ids]
    if len(ids) != len(set(ids)) or len(ids) < 2:
        raise ValueError("Unique training patients are required")
    pathways = [[str(g) for g in genes] for genes in factory.omic_names]
    genes = global_gene_ids(pathways)
    raw = raw_expression_matrix(factory, ids, genes)
    std = raw.std(axis=0)
    std = np.where(std > 1e-8, std, 1.)
    return dict(pathway_gene_ids=pathways, gene_ids=genes, raw_gene_mean=raw.mean(axis=0),
                raw_gene_std=std, train_patient_ids=ids, raw_expression_sha256=array_digest(raw))


def load_calibration(path, *, patient_ids, excluded_ids, pathway_gene_ids,
                     raw_expression_sha256, split_sha256=None, encoding_dim=None, wsi_encoder=None,
                     raw_expression=None):
    """Reject stale, mismatched, in-sample or incompletely covered teacher outputs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    provenance = payload.get("provenance", {})
    gains = payload.get("gains", {})
    expected = set(map(str, patient_ids))
    forbidden = set(map(str, excluded_ids))
    actual = provenance.get("train_patient_ids", [])
    if provenance.get("schema") != SCHEMA:
        raise ValueError("Unsupported calibration schema")
    if provenance.get("held_out_gain") is not True or provenance.get("no_survival_labels") is not True:
        raise ValueError("Calibration must be patient-held-out and survival-label-free")
    if len(actual) != len(set(actual)) or set(actual) != expected or set(gains) != expected:
        raise ValueError("Calibration must exactly cover the current training patients")
    if expected & forbidden:
        raise ValueError("Training and excluded cohorts overlap")
    paths = [[str(g) for g in genes] for genes in pathway_gene_ids]
    if provenance.get("pathway_gene_ids") != paths or provenance.get("gene_ids") != global_gene_ids(paths):
        raise ValueError("Calibration raw-gene/pathway coordinates differ")
    if provenance.get("raw_expression_sha256") != raw_expression_sha256:
        raise ValueError("Calibration RNA source changed")
    if split_sha256 is not None and provenance.get("split_sha256") != split_sha256:
        raise ValueError("Calibration split source changed")
    if encoding_dim is not None and provenance.get("encoding_dim") != int(encoding_dim):
        raise ValueError("Calibration WSI encoding dimension differs")
    if wsi_encoder is not None and provenance.get("wsi_encoder") != wsi_encoder:
        raise ValueError("Calibration WSI encoder differs")
    masks = np.asarray(provenance.get("global_masks", []))
    if (masks.ndim != 2 or masks.shape[1] != len(provenance["gene_ids"])
            or not np.isin(masks, [0, 1]).all() or not masks.astype(bool).any(0).all()
            or not (~masks.astype(bool)).any(1).all()):
        raise ValueError("Calibration masks must globally cover every gene and keep visible inputs")
    raw = None if raw_expression is None else np.asarray(raw_expression, dtype=np.float64)
    if raw is not None and (raw.shape != (len(actual), len(provenance["gene_ids"]))
                            or array_digest(raw) != raw_expression_sha256):
        raise ValueError("Calibration normalization verification raw source differs")
    covered = []
    for teacher in provenance.get("teachers", []):
        fit, score = set(teacher.get("fit_patient_ids", [])), set(teacher.get("score_patient_ids", []))
        if fit & score or fit | score != expected or fit & forbidden or score & forbidden:
            raise ValueError("Teacher cohorts do not form disjoint train-only complements")
        if set(teacher.get("normalization_fit_patient_ids", [])) != fit:
            raise ValueError("Teacher normalization must use its own fitting patients only")
        if len(fit) < 2 or not score:
            raise ValueError("Teacher fitting/scoring cohort is empty or too small")
        covered.extend(teacher["score_patient_ids"])
        for key in ("gene_mean", "gene_std"):
            values = np.asarray(teacher.get(key), dtype=np.float64)
            if values.shape != (len(provenance["gene_ids"]),) or not np.isfinite(values).all():
                raise ValueError("Teacher normalization statistics are invalid")
        if np.any(np.asarray(teacher["gene_std"]) <= 0):
            raise ValueError("Teacher standard deviations must be positive")
        if raw is not None:
            fit_indices = [list(map(str, patient_ids)).index(cid) for cid in teacher["fit_patient_ids"]]
            fit_raw = raw[fit_indices]
            expected_std = fit_raw.std(0)
            expected_std = np.where(expected_std > 1e-8, expected_std, 1.)
            if not (np.allclose(teacher["gene_mean"], fit_raw.mean(0), rtol=1e-10, atol=1e-10)
                    and np.allclose(teacher["gene_std"], expected_std, rtol=1e-10, atol=1e-10)):
                raise ValueError("Teacher normalization does not match its fitting patients")
    if len(covered) != len(expected) or set(covered) != expected:
        raise ValueError("Each patient must be scored by exactly one held-out teacher")
    for patient_id, gain in gains.items():
        values = np.asarray(gain, dtype=np.float64)
        if values.shape != (len(paths),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid pathway gains for {patient_id}")
        native = np.asarray(payload.get("native_mse", {}).get(patient_id), dtype=np.float64)
        conditional = np.asarray(payload.get("conditional_mse", {}).get(patient_id), dtype=np.float64)
        counts = np.asarray(payload.get("scored_gene_counts", {}).get(patient_id), dtype=np.float64)
        if (native.shape != values.shape or conditional.shape != values.shape or counts.shape != values.shape
                or not np.isfinite(native).all() or not np.isfinite(conditional).all()
                or np.any(native < 0) or np.any(conditional < 0) or np.any(counts < 1)
                or not np.isfinite(counts).all() or not np.allclose(values, native - conditional, rtol=1e-8, atol=1e-10)):
            raise ValueError("Gain targets must equal finite scored native-minus-conditional errors")
    for record in provenance.get("wsi_file_fingerprints", []):
        source = Path(record["path"])
        if not source.is_file():
            raise ValueError(f"Calibration WSI file missing: {source}")
        stat = source.stat()
        if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"Calibration WSI file changed: {source}")
    return gains, provenance


class PatientIdentifiedDataset(torch.utils.data.Dataset):
    """Carry source IDs through shuffled sampling and missing-sample filtering."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.label_df = dataset.label_df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return None if sample is None else (sample, str(self.label_df.iloc[index]["case id"]))


def collate_identified_pathways(batch):
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import _collate_pathways
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    return {"data": _collate_pathways([item[0] for item in valid]),
            "patient_ids": [item[1] for item in valid]}


def process_identified_batch(args, model, batch, device, *, test=False):
    """An explicit v23 path; historical forwarding sees its original payload."""
    if not isinstance(batch, dict) or set(batch) != {"data", "patient_ids"}:
        raise ValueError("DCT v2.3 requires batches carrying explicit patient IDs")
    data = batch["data"]
    y, time, c = (data[i].to(device) for i in (2, 3, 4))
    if len(batch["patient_ids"]) != len(y):
        raise ValueError("Patient IDs must match surviving batch samples")
    inputs = dict(x_wsi=data[0].to(device), cur_epoch=args.cur_epoch, event_time=time,
                  y=None if test else y, c=None if test else c,
                  wsi_missing=getattr(args, "wsi_missing", False) if test else False,
                  omic_missing=getattr(args, "omic_missing", False) if test else False)
    if not test:
        inputs["patient_ids"] = batch["patient_ids"]
    for pathway in range(len(data[1][0])):
        inputs[f"x_omic{pathway + 1}"] = torch.stack([sample[pathway] for sample in data[1]]).to(device)
    if getattr(args, "otehv2v2_use_clinical", False) and len(data) > 5:
        inputs["x_clinical"] = data[5].to(device)
    return model(**inputs), y, time, c
