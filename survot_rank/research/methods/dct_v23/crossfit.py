"""Small fixed-coordinate ridge teachers for patient-held-out reconstruction gains.

Teacher fitting is an explicit offline data job. Neither survival labels nor
latent slot identities enter the teacher inputs, outputs or partitioning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .calibration import SCHEMA, array_digest, global_gene_ids


def balanced_global_masks(gene_count, ratio, rounds, seed):
    if gene_count < 2 or not 0 < ratio < 1 or rounds < 1:
        raise ValueError("Valid global-gene mask settings are required")
    hidden = min(gene_count - 1, max(1, int(np.ceil(ratio * gene_count))))
    if hidden * rounds < gene_count:
        raise ValueError("Mask rounds must cover every global gene at least once")
    order = np.random.default_rng(seed).permutation(gene_count)
    masks = np.zeros((rounds, gene_count), dtype=bool)
    for round_id in range(rounds):
        masks[round_id, order[(np.arange(hidden) + round_id * hidden) % gene_count]] = True
    return masks


def fixed_projection(coordinates, dim, seed):
    """Coordinate-keyed signs keep features invariant to patient sampling/order."""
    if dim < 1:
        raise ValueError("Projection dimension must be positive")
    rows = []
    for coordinate in coordinates:
        key = hashlib.sha256(f"{seed}:{coordinate}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        rows.append(rng.choice([-1., 1.], size=dim))
    return np.asarray(rows, dtype=np.float64) / np.sqrt(len(coordinates))


def _ridge_predict(fit_x, fit_y, score_x, alpha):
    # Intercept is unpenalized. No query data affect centering or ridge fit.
    mean_x, mean_y = fit_x.mean(0), fit_y.mean(0)
    centered = fit_x - mean_x
    system = centered.T @ centered + float(alpha) * np.eye(centered.shape[1])
    coefficients = np.linalg.solve(system, centered.T @ (fit_y - mean_y))
    return (score_x - mean_x) @ coefficients + mean_y


def fit_cross_patient_gains(raw_genes, pooled_wsi, *, patient_ids, pathway_gene_ids,
                           inner_folds=3, mask_ratio=.3, mask_rounds=8,
                           native_dim=32, wsi_dim=16, ridge_alpha=10., seed=3,
                           extra_provenance=None):
    """Return native-minus-conditional MSE in teacher-standardized raw coordinates.

    Both arms see identical visible RNA and query masks. The conditional arm
    additionally sees fixed pooled WSI coordinates. Negative gains are retained.
    Each patient is scored only by a teacher fitted on the complementary patients.
    """
    ids = [str(x) for x in patient_ids]
    genes = global_gene_ids(pathway_gene_ids)
    raw = np.asarray(raw_genes, dtype=np.float64)
    wsi = np.asarray(pooled_wsi, dtype=np.float64)
    if len(ids) != len(set(ids)) or raw.shape != (len(ids), len(genes)):
        raise ValueError("Unique patient IDs and exact raw-gene coordinates required")
    if wsi.ndim != 2 or wsi.shape[0] != len(ids) or wsi.shape[1] < 1:
        raise ValueError("Pooled WSI must contain one fixed feature vector per patient")
    if not np.isfinite(raw).all() or not np.isfinite(wsi).all():
        raise ValueError("Teacher inputs must be finite")
    if not 2 <= inner_folds <= len(ids) or len(ids) - int(np.ceil(len(ids) / inner_folds)) < 2:
        raise ValueError("Each held-out teacher needs at least two fitting patients")
    if ridge_alpha <= 0:
        raise ValueError("Positive ridge regularization is required")
    masks = balanced_global_masks(len(genes), mask_ratio, mask_rounds, seed)
    native_projection = fixed_projection(genes, native_dim, seed + 13)
    wsi_projection = fixed_projection([f"UNI-coordinate-{i}" for i in range(wsi.shape[1])], wsi_dim, seed + 29)
    gene_index = {g: i for i, g in enumerate(genes)}
    pathway_indices = [np.asarray([gene_index[str(g)] for g in path], dtype=int) for path in pathway_gene_ids]
    error_native = np.zeros((len(ids), len(pathway_indices)))
    error_conditional = np.zeros_like(error_native)
    error_counts = np.zeros_like(error_native)
    order = np.random.default_rng(seed + 101).permutation(len(ids))
    fold_indices = np.array_split(order, inner_folds)
    teachers = []
    for teacher_id, score_indices in enumerate(fold_indices):
        fit_indices = np.setdiff1d(np.arange(len(ids)), score_indices)
        gene_mean, gene_std = raw[fit_indices].mean(0), raw[fit_indices].std(0)
        gene_std = np.where(gene_std > 1e-8, gene_std, 1.)
        wsi_mean, wsi_std = wsi[fit_indices].mean(0), wsi[fit_indices].std(0)
        wsi_std = np.where(wsi_std > 1e-8, wsi_std, 1.)
        standardized = (raw - gene_mean) / gene_std
        image_features = ((wsi - wsi_mean) / wsi_std) @ wsi_projection
        for mask in masks:
            # Hidden values are removed BEFORE any feature projection. Mask is
            # identical across this round, including all pathway gene replicas.
            visible = np.where(mask[None, :], 0., standardized)
            native_features = visible @ native_projection
            conditional_features = np.concatenate([native_features, image_features], axis=1)
            targets_fit = standardized[fit_indices][:, mask]
            pred_native = _ridge_predict(native_features[fit_indices], targets_fit, native_features[score_indices], ridge_alpha)
            pred_conditional = _ridge_predict(conditional_features[fit_indices], targets_fit, conditional_features[score_indices], ridge_alpha)
            true = standardized[score_indices][:, mask]
            native_error, conditional_error = (pred_native - true) ** 2, (pred_conditional - true) ** 2
            hidden_coordinates = np.flatnonzero(mask)
            for pathway_id, pathway in enumerate(pathway_indices):
                positions = np.flatnonzero(np.isin(hidden_coordinates, pathway))
                if positions.size:
                    error_native[score_indices, pathway_id] += native_error[:, positions].sum(1)
                    error_conditional[score_indices, pathway_id] += conditional_error[:, positions].sum(1)
                    error_counts[score_indices, pathway_id] += positions.size
        teachers.append(dict(teacher_id=teacher_id,
            fit_patient_ids=[ids[i] for i in fit_indices], score_patient_ids=[ids[i] for i in score_indices],
            normalization_fit_patient_ids=[ids[i] for i in fit_indices],
            gene_mean=gene_mean.tolist(), gene_std=gene_std.tolist(),
            wsi_mean=wsi_mean.tolist(), wsi_std=wsi_std.tolist()))
    if np.any(error_counts == 0):
        raise ValueError("Every patient/pathway must have held-out gene scoring coverage")
    native_mse, conditional_mse = error_native / error_counts, error_conditional / error_counts
    provenance = dict(extra_provenance or {})
    provenance.update(schema=SCHEMA, held_out_gain=True, no_survival_labels=True,
        train_patient_ids=ids, gene_ids=genes, pathway_gene_ids=[[str(g) for g in p] for p in pathway_gene_ids],
        raw_expression_sha256=array_digest(raw), pooled_wsi_sha256=array_digest(wsi),
        teachers=teachers, global_masks=masks.astype(int).tolist(),
        teacher_type="fixed_coordinate_ridge_native_vs_conditional",
        gain_definition="native_masked_mse_minus_conditional_masked_mse_teacher_standardized_raw_genes",
        seed=seed, inner_folds=inner_folds, mask_ratio=mask_ratio, mask_rounds=mask_rounds,
        native_dim=native_dim, wsi_dim=wsi_dim, ridge_alpha=ridge_alpha,
        raw_coordinate_semantics="stored RNA input values; upstream transform is inherited, not inferred")
    return dict(provenance=provenance, gains={cid: gain.tolist() for cid, gain in zip(ids, native_mse - conditional_mse)},
                native_mse={cid: x.tolist() for cid, x in zip(ids, native_mse)},
                conditional_mse={cid: x.tolist() for cid, x in zip(ids, conditional_mse)},
                scored_gene_counts={cid: x.astype(int).tolist() for cid, x in zip(ids, error_counts)})


def save_gain_table(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: an older calibration artifact is never overwritten.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return path
