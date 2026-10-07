#!/usr/bin/env python3
"""B1: Omics slot ↔ pathway affinity analysis.

This script takes the per-slot hazard export (per_slot_hazard.pkl) and:
  1. Loads wsi_coordinate_assignment [N, K_o=8, N_omics=199] (pathway softmax weights)
  2. For each omic slot k: computes mean weight over high-risk patients vs low-risk.
  3. Ranks pathways by slot-attention weight — shows which slot "listens to" which pathways.
  4. Cross-references with BLCA-known risk pathways (Hallmark/MSigDB names).
  5. Checks cross-fold consistency (Jaccard of top-20 pathways per slot).

Usage:
    python scripts/analyze_omic_slot_pathways.py \
        --pkl /data1/DCT-Reg/results/dct_v311_blca_uni2h/per_slot_export/per_slot_hazard.pkl \
        --sig /data1/dataset_csv/signatures/combine_signatures.csv \
        --out_dir /data1/DCT-Reg/results/dct_v311_blca_uni2h/omic_slot_analysis
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── BLCA clinically-relevant pathways (curated from MSigDB Hallmark) ──────────
BLCA_RISK_PATHWAYS = {
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_ANGIOGENESIS",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_HYPOXIA",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_KRAS_SIGNALING_UP",
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_MITOTIC_SPINDLE",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_NOTCH_SIGNALING",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_HEME_METABOLISM",
    "HALLMARK_FATTY_ACID_METABOLISM",
    "HALLMARK_CHOLESTEROL_HOMEOSTASIS",
    "HALLMARK_GLYCOLYSIS",
    "HALLMARK_UV_RESPONSE_UP",
    "HALLMARK_UV_RESPONSE_DN",
}


def load_pathway_names(sig_path):
    """Return the list of 199 pathway names from combine_signatures.csv."""
    with open(sig_path) as f:
        names = [line.strip().split(",")[0] for line in f if line.strip()]
    return names


def load_data(pkl_path):
    with open(pkl_path, "rb") as f:
        per_fold = pickle.load(f)
    return per_fold


def high_low_split(risks, top_pct=0.25):
    """Return bool mask for high-risk (top-pct) and low-risk (bottom-pct)."""
    thresh_high = np.percentile(risks, 100 * (1 - top_pct))
    thresh_low = np.percentile(risks, 100 * top_pct)
    high_mask = risks >= thresh_high
    low_mask = risks <= thresh_low
    return high_mask, low_mask


def slot_pathway_affinity(per_fold, pathway_names):
    """Compute per-fold slot × pathway affinity matrices.

    Returns:
        fold_affinity: list of [K_o, N_omics] arrays (one per fold)
        fold_risk_diff: list of [K_o, N_omics] arrays (high-risk minus low-risk)
    """
    fold_affinity = []
    fold_risk_diff = []
    for d in per_fold:
        N, K_o, N_omics = d["omic_coordinate_assignment"].shape
        # per-slot mean over patients: [K_o, N_omics]
        mean_aff = d["omic_coordinate_assignment"].mean(axis=0)
        high_mask, low_mask = high_low_split(d["risks"])
        if high_mask.sum() == 0 or low_mask.sum() == 0:
            risk_diff = np.zeros((K_o, N_omics))
        else:
            high_aff = d["omic_coordinate_assignment"][high_mask].mean(axis=0)
            low_aff  = d["omic_coordinate_assignment"][low_mask].mean(axis=0)
            risk_diff = high_aff - low_aff
        fold_affinity.append(mean_aff)
        fold_risk_diff.append(risk_diff)
    return fold_affinity, fold_risk_diff


def cross_fold_topk_jaccard(fold_affinity, pathway_names, top_k=20):
    """Check consistency: for each slot, compare top-K pathway sets across folds."""
    n_folds = len(fold_affinity)
    K_o = fold_affinity[0].shape[0]
    results = {}
    for k in range(K_o):
        top_sets = []
        for f in range(n_folds):
            aff = fold_affinity[f][k]
            top_idx = np.argsort(aff)[::-1][:top_k]
            top_sets.append(set(top_idx))
        # Jaccard: mean pairwise Jaccard
        jaccards = []
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                inter = len(top_sets[i] & top_sets[j])
                union = len(top_sets[i] | top_sets[j])
                jaccards.append(inter / union if union > 0 else 0.0)
        mean_jaccard = np.mean(jaccards)
        results[k] = {
            "mean_jaccard_top20": mean_jaccard,
            "stable": mean_jaccard >= 0.3,
        }
    return results


def blca_enrichment(fold_affinity, pathway_names, top_k=10):
    """For each slot, count how many BLCA-known pathways appear in its top-K."""
    n_folds = len(fold_affinity)
    K_o = fold_affinity[0].shape[0]
    # If pathway_names list doesn't match N_omics, build synthetic ones.
    n_omics = fold_affinity[0].shape[1]
    if len(pathway_names) != n_omics:
        pathway_names = [f"pathway_{i}" for i in range(n_omics)]
    name_to_idx = {n: i for i, n in enumerate(pathway_names)}
    results = {}
    for k in range(K_o):
        # Aggregate rank: average rank across folds
        ranks = np.zeros(n_omics)
        for f in range(n_folds):
            ranks += np.argsort(np.argsort(fold_affinity[f][k]))  # rank=0 is largest
        avg_rank = ranks / n_folds
        top_idx = np.argsort(avg_rank)[:top_k]
        top_names = [pathway_names[i] for i in top_idx]
        blca_hits = [n for n in top_names if n in BLCA_RISK_PATHWAYS]
        results[k] = {
            "top_pathways": top_names,
            "blca_hits": blca_hits,
            "blca_hit_count": len(blca_hits),
            "avg_rank_dict": dict(zip(pathway_names, avg_rank)),
        }
    return results


def write_report(out_dir, jaccard_results, blca_results, pathway_names):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If pathway_names has synthetic placeholders, mention that in the report.
    is_synthetic = all(n.startswith("pathway_") for n in pathway_names[:5])

    lines = [
        "# B1: Omics Slot ↔ Pathway Affinity Report",
        "# ==" * 30,
        "",
        "## Cross-fold Jaccard (top-20 pathways per slot):",
        f"{'Slot':>6}  {'Mean Jaccard':>14}  {'Stable (>=0.3)?':>16}",
        "-" * 50,
    ]
    for k, r in jaccard_results.items():
        lines.append(f"  Slot {k}    {r['mean_jaccard_top20']:>14.3f}  {str(r['stable']):>16}")

    if is_synthetic:
        lines += [
            "",
            "## ⚠️ Pathway names unavailable",
            "The `combine_signatures.csv` we tried to load does not match the",
            "329-pathway dictionary used at training time (BLCA Pathways mode).",
            "Below we report top-10 indices; cross-reference with the model's",
            "internal pathway_names attribute if a dictionary mapping is needed.",
        ]
    lines += ["", "## BLCA Risk Pathway Enrichment (top-10 pathways per slot):", ""]
    for k, r in blca_results.items():
        lines.append(f"Slot {k}  (BLCA hits: {r['blca_hit_count']}/10)")
        for pn in r["top_pathways"]:
            marker = " ⭐ BLCA" if pn in BLCA_RISK_PATHWAYS else ""
            lines.append(f"  - {pn}{marker}")
        lines.append("")

    report_path = out_dir / "b1_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[B1] Report → {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="B1: Omics slot ↔ pathway affinity analysis.")
    parser.add_argument("--pkl", type=str,
                        default="/data1/DCT-Reg/results/dct_v311_blca_uni2h/per_slot_export/per_slot_hazard.pkl")
    parser.add_argument("--sig", type=str,
                        default="/data1/dataset_csv/signatures/combine_signatures.csv")
    parser.add_argument("--out_dir", type=str,
                        default="/data1/DCT-Reg/results/dct_v311_blca_uni2h/omic_slot_analysis")
    args = parser.parse_args()

    print("[B1] Loading data...")
    pathway_names = load_pathway_names(args.sig)
    per_fold = load_data(args.pkl)
    print(f"[B1] {len(per_fold)} folds, K_o={per_fold[0]['K_o']}, N_omics={per_fold[0]['N_omics']}")
    print(f"[B1] Pathway names: {len(pathway_names)}")

    print("[B1] Computing slot-pathway affinity...")
    fold_affinity, fold_risk_diff = slot_pathway_affinity(per_fold, pathway_names)

    print("[B1] Cross-fold Jaccard consistency...")
    jaccard_results = cross_fold_topk_jaccard(fold_affinity, pathway_names, top_k=20)
    for k, r in jaccard_results.items():
        status = "✅" if r["stable"] else "⚠️ "
        print(f"  Slot {k}: Jaccard={r['mean_jaccard_top20']:.3f}  {status}")

    print("[B1] BLCA risk pathway enrichment...")
    blca_results = blca_enrichment(fold_affinity, pathway_names, top_k=10)
    for k, r in blca_results.items():
        print(f"  Slot {k}: {r['blca_hit_count']}/10 BLCA hits: {r['blca_hits']}")

    print("[B1] Writing report...")
    report_path = write_report(Path(args.out_dir), jaccard_results, blca_results, pathway_names)
    print(f"[B1] Done → {report_path}")


if __name__ == "__main__":
    main()
