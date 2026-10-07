#!/usr/bin/env python3
"""v3.11 Slot Interpretability Verification - Comprehensive Statistical Analysis.

This script verifies the three key claims of v3.11:

1. A1: Slot hazard vs time-bin monotonicity (C-index / Spearman ρ)
   - Good slots: hazard should increase monotonically with time-bin
   - Bad slots: hazard should be approximately flat
   
2. A2: Slot variance distribution check (5-fold validation)
   - train_v311_slot_variance should be in [0.005, 0.050]
   - We check this at inference time on validation sets
   
3. A3: Top-K slot hazard vs risk alignment
   - risk = final logits risk vs mean(hazard[:, :, -1])
   - Spearman correlation should be high if they align
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sksurv.metrics import concordance_index_censored

# ============================================================
# Paths
# ============================================================
PKL_PATH = Path("/data1/DCT-Reg/results/dct_v311_blca_uni/per_slot_export/per_slot_hazard.pkl")
OUT_DIR = Path("/data1/DCT-Reg/results/dct_v311_blca_uni/per_slot_export/interp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load Data
# ============================================================
print("=" * 60)
print("Loading per-slot hazard data...")
print("=" * 60)

with open(PKL_PATH, "rb") as f:
    per_fold: List[Dict] = pickle.load(f)

for d in per_fold:
    print(f"  Fold {d['fold']}: N={len(d['case_ids'])}, "
          f"K_w={d['K_w']}, K_o={d['K_o']}, C={d['C']}")

# ============================================================
# A1: Slot Hazard vs Time-Bin Monotonicity Analysis
# ============================================================
print("\n" + "=" * 60)
print("A1: Slot Hazard vs Time-Bin Monotonicity Analysis")
print("=" * 60)

def compute_slot_cindex(
    hazard_per_slot: np.ndarray,  # [N, K, C]
    times: np.ndarray,           # [N]
    censors: np.ndarray,         # [N]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute C-index for each slot's hazard predictions.
    
    Args:
        hazard_per_slot: hazard predictions [N, K, C]
        times: event times [N]
        censors: censorship flags [N] (0=event, 1=censored)
    
    Returns:
        cindex_per_slot: C-index for each slot [K]
        spearman_per_slot: Spearman rho for each slot [K]
    """
    from sksurv.metrics import concordance_index_censored
    
    N, K, C = hazard_per_slot.shape
    cindex_per_slot = np.zeros(K)
    spearman_per_slot = np.zeros(K)
    
    for k in range(K):
        # Sum hazard across time bins to get "risk score"
        slot_risk = hazard_per_slot[:, k, :].sum(axis=1)

        # Compute C-index (sksurv convention: y[0]=censored, y[1]=event)
        # Our censors: 0=event, 1=censored
        # sksurv expects: (censored, event_time, risk_score)
        # Higher risk score should correspond to shorter event time (higher hazard).
        # Since slot_risk = sum of hazards (higher = worse prognosis), we pass it directly.
        event_observed = censors == 0  # True=event (observed death)

        try:
            cindex, _, _, _, _ = concordance_index_censored(
                event_observed,  # sksurv: True if event was observed
                times,
                slot_risk  # Higher hazard sum = higher risk = shorter survival time
            )
            cindex_per_slot[k] = cindex
        except Exception as e:
            cindex_per_slot[k] = np.nan
        
        # Spearman correlation between slot hazard at each time bin and time
        # (if hazard increases with time, correlation should be positive)
        hazard_over_time = hazard_per_slot[:, k, :].mean(axis=0)  # [C]
        time_bins = np.arange(C)
        spearman, _ = stats.spearmanr(time_bins, hazard_over_time)
        spearman_per_slot[k] = spearman if not np.isnan(spearman) else 0.0
    
    return cindex_per_slot, spearman_per_slot


def assess_monotonicity(
    hazard_per_slot: np.ndarray,  # [N, K, C]
) -> Dict[int, str]:
    """Assess whether each slot's hazard is monotonically increasing with time.
    
    Returns classification: "good" (increasing), "bad" (flat), or "mixed"
    """
    N, K, C = hazard_per_slot.shape
    classification = {}
    
    for k in range(K):
        hazard_over_time = hazard_per_slot[:, k, :].mean(axis=0)  # [C]
        
        # Check if hazard increases from bin 0 to bin C-1
        if hazard_over_time[-1] > hazard_over_time[0] * 1.1:
            # Significant increase: hazard at last bin > 10% higher than first bin
            classification[k] = "good"
        elif abs(hazard_over_time[-1] - hazard_over_time[0]) < 0.05:
            # Nearly flat: difference < 5%
            classification[k] = "bad"
        else:
            classification[k] = "mixed"
    
    return classification


# Results storage
a1_results = []

for d in per_fold:
    fold = d["fold"]
    
    # WSI slots
    wsi_cindex, wsi_spearman = compute_slot_cindex(
        d["hazard_wsi"], d["times"], d["censors"]
    )
    wsi_class = assess_monotonicity(d["hazard_wsi"])
    
    # Omics slots
    omic_cindex, omic_spearman = compute_slot_cindex(
        d["hazard_omic"], d["times"], d["censors"]
    )
    omic_class = assess_monotonicity(d["hazard_omic"])
    
    for k in range(d["K_w"]):
        a1_results.append({
            "fold": fold,
            "modality": "wsi",
            "slot": k,
            "cindex": wsi_cindex[k],
            "spearman_time": wsi_spearman[k],
            "monotonicity": wsi_class[k],
        })
    
    for k in range(d["K_o"]):
        a1_results.append({
            "fold": fold,
            "modality": "omic",
            "slot": k,
            "cindex": omic_cindex[k],
            "spearman_time": omic_spearman[k],
            "monotonicity": omic_class[k],
        })

df_a1 = pd.DataFrame(a1_results)

# Save A1 results
a1_csv_path = OUT_DIR / "slot_cindex_per_bin.csv"
df_a1.to_csv(a1_csv_path, index=False)
print(f"\nSaved A1 results to: {a1_csv_path}")

# A1 Summary
print("\n" + "-" * 40)
print("A1 Summary: Slot Monotonicity Analysis")
print("-" * 40)

for mod in ["wsi", "omic"]:
    mod_df = df_a1[df_a1["modality"] == mod]
    good_count = (mod_df["monotonicity"] == "good").sum()
    bad_count = (mod_df["monotonicity"] == "bad").sum()
    mixed_count = (mod_df["monotonicity"] == "mixed").sum()
    total = len(mod_df)
    
    print(f"\n{mod.upper()} Slots:")
    print(f"  Good (increasing hazard):   {good_count}/{total} ({100*good_count/total:.1f}%)")
    print(f"  Bad (flat hazard):          {bad_count}/{total} ({100*bad_count/total:.1f}%)")
    print(f"  Mixed:                      {mixed_count}/{total} ({100*mixed_count/total:.1f}%)")
    print(f"  Mean C-index:               {mod_df['cindex'].mean():.4f} ± {mod_df['cindex'].std():.4f}")
    print(f"  Mean Spearman ρ (time):     {mod_df['spearman_time'].mean():.4f} ± {mod_df['spearman_time'].std():.4f}")

# ============================================================
# A2: Slot Variance Distribution Check
# ============================================================
print("\n" + "=" * 60)
print("A2: Slot Variance Distribution Check")
print("=" * 60)

VARIANCE_MIN = 0.005
VARIANCE_MAX = 0.050

a2_results = []

for d in per_fold:
    fold = d["fold"]
    
    # Compute per-sample variance across slots at each time bin
    # Then average over time bins
    hazard_all = np.concatenate([d["hazard_wsi"], d["hazard_omic"], ], axis=1)  # [N, K_total, C]
    K_total = hazard_all.shape[1]
    
    for i in range(len(d["case_ids"])):
        # Per-sample variance across slots at last time bin (most informative)
        var_last_bin = np.var(hazard_all[i, :, -1])
        
        # Per-sample variance averaged across all time bins
        var_mean = np.mean([np.var(hazard_all[i, :, c]) for c in range(hazard_all.shape[2])])
        
        a2_results.append({
            "fold": fold,
            "case_id": d["case_ids"][i],
            "variance_last_bin": var_last_bin,
            "variance_mean": var_mean,
            "in_range": VARIANCE_MIN <= var_mean <= VARIANCE_MAX,
        })

df_a2 = pd.DataFrame(a2_results)

# Save A2 results
a2_csv_path = OUT_DIR / "slot_variance_distribution.csv"
df_a2.to_csv(a2_csv_path, index=False)
print(f"\nSaved A2 results to: {a2_csv_path}")

# A2 Summary
print("\n" + "-" * 40)
print("A2 Summary: Slot Variance Distribution")
print("-" * 40)
print(f"Target range: [{VARIANCE_MIN}, {VARIANCE_MAX}]")

for fold in sorted(df_a2["fold"].unique()):
    fold_df = df_a2[df_a2["fold"] == fold]
    in_range = fold_df["in_range"].sum()
    total = len(fold_df)
    mean_var = fold_df["variance_mean"].mean()
    std_var = fold_df["variance_mean"].std()
    
    print(f"\nFold {fold}:")
    print(f"  Samples in range: {in_range}/{total} ({100*in_range/total:.1f}%)")
    print(f"  Mean variance:     {mean_var:.4f} ± {std_var:.4f}")

# Overall A2 summary
total_in_range = df_a2["in_range"].sum()
total_samples = len(df_a2)
overall_mean = df_a2["variance_mean"].mean()
overall_std = df_a2["variance_mean"].std()

print(f"\nOverall (all folds):")
print(f"  Samples in range: {total_in_range}/{total_samples} ({100*total_in_range/total_samples:.1f}%)")
print(f"  Mean variance:    {overall_mean:.4f} ± {overall_std:.4f}")

# ============================================================
# A3: Top-K Slot Hazard vs Risk Alignment
# ============================================================
print("\n" + "=" * 60)
print("A3: Top-K Slot Hazard vs Risk Alignment")
print("=" * 60)

a3_results = []

for d in per_fold:
    fold = d["fold"]
    
    # Final risk from model logits
    final_risk = d["risks"]
    
    # Mean hazard at last time bin across all slots
    hazard_all = np.concatenate([d["hazard_wsi"], d["hazard_omic"]], axis=1)  # [N, K_total, C]
    mean_hazard_last = hazard_all[:, :, -1].mean(axis=1)  # [N]
    
    # Mean hazard per slot (averaged across patients and time bins)
    K = hazard_all.shape[1]
    num_top_k = max(2, K // 2)  # Top half of slots
    
    # Get indices of top-K slots by mean hazard at last time bin
    # mean_hazard_per_slot shape: [K]
    mean_hazard_per_slot = mean_hazard_last  # This is already per-sample, not what we want
    
    # Actually, we want to find which slots have highest mean hazard across patients
    mean_hazard_across_patients = hazard_all[:, :, -1].mean(axis=0)  # [K]
    sorted_slot_indices = np.argsort(mean_hazard_across_patients)[-num_top_k:]  # Indices of top-K slots
    
    top_k_hazard = hazard_all[:, sorted_slot_indices, -1].mean(axis=1)  # [N]
    
    # Spearman correlation
    spearman_risk_hazard, p_val = stats.spearmanr(final_risk, mean_hazard_last)
    spearman_risk_topk, p_val_topk = stats.spearmanr(final_risk, top_k_hazard)
    
    # Pearson correlation
    pearson_risk_hazard, _ = stats.pearsonr(final_risk, mean_hazard_last)
    pearson_risk_topk, _ = stats.pearsonr(final_risk, top_k_hazard)
    
    a3_results.append({
        "fold": fold,
        "spearman_all_slots": spearman_risk_hazard,
        "spearman_topk": spearman_risk_topk,
        "pearson_all_slots": pearson_risk_hazard,
        "pearson_topk": pearson_risk_topk,
        "top_k": num_top_k,
        "p_val_all": p_val,
        "p_val_topk": p_val_topk,
    })

df_a3 = pd.DataFrame(a3_results)

# Save A3 results
a3_csv_path = OUT_DIR / "slot_risk_alignment.csv"
df_a3.to_csv(a3_csv_path, index=False)
print(f"\nSaved A3 results to: {a3_csv_path}")

# A3 Summary
print("\n" + "-" * 40)
print("A3 Summary: Hazard vs Risk Alignment")
print("-" * 40)

print("\nPer-Fold Results:")
print(f"{'Fold':>4} | {'Spearman (all)':>16} | {'Spearman (topK)':>16} | {'Pearson (all)':>14} | {'Pearson (topK)':>14}")
print("-" * 75)
for _, row in df_a3.iterrows():
    print(f"{row['fold']:>4} | {row['spearman_all_slots']:>16.4f} | {row['spearman_topk']:>16.4f} | "
          f"{row['pearson_all_slots']:>14.4f} | {row['pearson_topk']:>14.4f}")

print("\nOverall:")
print(f"  Mean Spearman (all slots):  {df_a3['spearman_all_slots'].mean():.4f}")
print(f"  Mean Spearman (top-K):     {df_a3['spearman_topk'].mean():.4f}")
print(f"  Mean Pearson (all slots):   {df_a3['pearson_all_slots'].mean():.4f}")
print(f"  Mean Pearson (top-K):       {df_a3['pearson_topk'].mean():.4f}")

# ============================================================
# Final Verdict
# ============================================================
print("\n" + "=" * 60)
print("VERIFICATION VERDICT")
print("=" * 60)

# A1 verdict: Most slots should be "good" (increasing hazard)
wsi_good_rate = (df_a1[df_a1["modality"] == "wsi"]["monotonicity"] == "good").mean()
omic_good_rate = (df_a1[df_a1["modality"] == "omic"]["monotonicity"] == "good").mean()

print(f"\nA1: Slot Monotonicity")
print(f"  WSI slots with increasing hazard: {100*wsi_good_rate:.1f}%")
print(f"  Omics slots with increasing hazard: {100*omic_good_rate:.1f}%")
if wsi_good_rate > 0.5 and omic_good_rate > 0.5:
    print("  ✓ PASS: Majority of slots show monotonic hazard increase")
elif wsi_good_rate + omic_good_rate > 0.4:
    print("  ⚠ PARTIAL: Some slots show monotonic hazard increase")
else:
    print("  ✗ FAIL: Most slots show flat hazard")

# A2 verdict: Most samples should have variance in target range
variance_pass_rate = df_a2["in_range"].mean()
print(f"\nA2: Slot Variance Distribution")
print(f"  Samples in target range [{VARIANCE_MIN}, {VARIANCE_MAX}]: {100*variance_pass_rate:.1f}%")
if variance_pass_rate > 0.7:
    print("  ✓ PASS: Majority of samples have variance in target range")
elif variance_pass_rate > 0.4:
    print("  ⚠ PARTIAL: Some samples have variance in target range")
else:
    print("  ✗ FAIL: Most samples have variance outside target range")

# A3 verdict: Spearman correlation should be high (alignment)
spearman_mean = df_a3["spearman_all_slots"].mean()
print(f"\nA3: Hazard vs Risk Alignment")
print(f"  Mean Spearman ρ (risk vs last-bin hazard): {spearman_mean:.4f}")
if spearman_mean > 0.5:
    print("  ✓ PASS: Strong alignment between risk and hazard")
elif spearman_mean > 0.2:
    print("  ⚠ PARTIAL: Moderate alignment between risk and hazard")
else:
    print("  ✗ FAIL: Weak alignment between risk and hazard")

print("\n" + "=" * 60)
print("Analysis complete. Results saved to:")
print(f"  - {a1_csv_path}")
print(f"  - {a2_csv_path}")
print(f"  - {a3_csv_path}")
print("=" * 60)
