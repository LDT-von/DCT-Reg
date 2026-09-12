#!/usr/bin/env python3
"""Export per-slot hazard from v3.11 BLCA PER-MODALITY DIVERSITY checkpoints (v2).

Differences from scripts/export_v311_fixed_per_slot_hazard.py:
- Default ckpt_root points to results/dct_v311_blca_uni_fixed_v2/...
- Matches the per-modality diversity fix

Usage:
    python scripts/export_v311_v2_per_slot_hazard.py --folds 0
    python scripts/export_v311_v2_per_slot_hazard.py --folds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

_REPO_ROOT = Path("/data1/DCT-Reg")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULTS = {
    "ckpt_root": "/data1/DCT-Reg/results/dct_v311_blca_uni_fixed_v2/blca/blca/SurvOTRank_dct_v311_slot_interpretable/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_v2_fold{}/",
    "data_path": "/data1/dataset_csv",
    "split_dir": "/data1/dataset_csv/splits/5fold/blca",
    "study": "blca",
    "rna_format": "Pathways",
    "signature": "combine",
    "label_col": "survival_months_dss",
    "n_classes": 4,
    "encoding_dim": 1024,
    "wsi_projection_dim": 256,
    "num_patches": 4096,
    "slot_num_wsi": 8,
    "slot_num_omics": 8,
    "slot_iters": 10,
    "topk_ratio": 0.25,
    "top_k_method": "parallel_topk_st",
    "alpha_surv": 0.5,
    "wsi_encoder": "uni",
    "data_root_dir": "/data/CPathPatchFeature",
    "n_bins": 4,
    "method": "SurvOTRank_dct_v311",
    "survot_method": "dct_v311_slot_interpretable",
    "specific_simple": "dct_v311_blca_uni_v2",
    "event_stratified_batches": False,
    "event_sampling_fraction": 0.0,
    "dct_anchor_momentum": 0.9,
    "dct_evidence_cost_weight": 0.0,
    "dct_evidence_mass_floor": 0.05,
    "dct_evidence_marginal_strength": 1.0,
    "dct_geometry_reliability_strength": 0.0,
    "dct_coupling_projection_iters": 1000,
    "dct_coupling_projection_tol": 0.0001,
    "dct_coordinate_temperature": 0.30,
    "dct_mix_ratio": 1.0,
    "dct_v38_direction_margin": 0.02,
    "dct_v38_dose_margin": 0.005,
    "dct_v38_reconfiguration_margin": 0.02,
    "dct_v38_temperature": 0.05,
    "dct_v38_alpha_mid": 0.5,
    "dct_v38_alpha_full": 1.0,
    "dct_v38_warmup_epochs": 0,
    "dct_v38_ramp_epochs": 0,
    "dct_v38_dose_every": 1,
    "dct_lambda_etar": 0.0,
    "dct_lambda_listwise": 0.0,
    "dct_v382_lambda_mgptr": 0.0,
    "dct_v382_adaptive_aux_weights": False,
    "dct_v311_lambda_slot_nll": 0.05,
    "dct_v311_lambda_slot_diversity": 0.02,
    "dct_v311_variance_min": 0.005,
    "dct_v311_variance_max": 0.05,
    "dct_v38_lambda_direction": 0.0,
    "dct_v38_lambda_dose": 0.0,
    "dct_v38_lambda_reconfiguration": 0.0,
    "dct_lambda_ipcw_rank": 0.10,
    "dct_ipcw_rank_margin": 0.02,
    "dct_ipcw_rank_temperature": 0.50,
    "dct_ipcw_max_weight": 10.0,
    "dct_ipcw_rank_memory_size": 0,
    "spt_prog_cost": 0.20,
    "rg_eps_start": 0.10,
    "rg_eps_anneal": 12,
    "dct_num_stages": 4,
    "dct_slot_init_mode": "gaussian",
    "dct_slot_eval_seed": 1729,
    "otehv2_eps": 0.05,
    "otehv2_iter": 50,
    "otehv2_heads": 4,
    "otehv2_layers": 2,
    "otehv2_dropout": 0.15,
    "max_epochs": 30,
    "grad_clip_norm": 1.0,
    "warmup_epochs": 0,
    "scheduler": "cosine",
    "lr": 0.0005,
    "opt": "adamW",
    "reg": 0.0005,
    "batch_size": 8,
    "bag_loss": "nll_surv",
    "num_workers": 0,
    "pin_memory": True,
    "seed": 3,
    "fit_bins_on_train": True,
    "binning_mode": "global_qcut",
    "dct_coupling_use_multiscale": True,
    "dct_coupling_scales": "2.0,1.0,0.5",
    "dct_use_coordinate_assignment": True,
    "dct_coordinate_assignment_kind": "softmax",
    "dct_coordinate_assignment_temperature": 0.3,
    "dct_use_geometry_reliability": False,
    "log_dir": "logs",
    "results_dir": "/data1/DCT-Reg/results/dct_v311_blca_uni_fixed_v2/blca",
    "which_splits": "5fold",
    "on_missing_wsi": "error",
    "gpu": 1,
    "is_smoke": False,
    "dct_use_jump_start": True,
    "dct_jump_start_steps": 200,
    "use_amp": False,
    "dct_anchor_init": "uniform",
    "dct_anchor_smoothing": 0.0,
    "listwise_loss": False,
    "listwise_max_pairs": 256,
    "k_start": 0,
    "k_end": 5,
    "study": "blca",
    "min_free_space_gb": 2.0,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_false" if v else "store_true", default=v)
        elif isinstance(v, list):
            parser.add_argument(f"--{k}", type=str, default=",".join(map(str, v)))
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out_dir", type=str,
                        default="/data1/DCT-Reg/results/dct_v311_blca_uni_fixed_v2/per_slot_export")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting per-slot hazard for v3.11 fixed v2 (per-modality diversity)")
    print(f"  ckpt_root: {args.ckpt_root}")
    print(f"  folds: {args.folds}")
    print(f"  out_dir: {out_dir}")

    # Reuse the original export logic via import
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from export_v311_fixed_per_slot_hazard import export_one_fold

    all_per_fold = []
    for fold in args.folds:
        print(f"\n=== Fold {fold} ===")
        try:
            data = export_one_fold(args, fold, out_dir)
            all_per_fold.append(data)
        except Exception as e:
            print(f"Fold {fold} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_per_fold:
        print("No folds succeeded")
        return 1

    # Save merged PKL/CSV
    pkl_path = out_dir / "per_slot_hazard.pkl"
    csv_path = out_dir / "per_slot_hazard.csv"

    with open(pkl_path, "wb") as f:
        pickle.dump(all_per_fold, f)
    print(f"\nSaved {len(all_per_fold)} folds to {pkl_path}")

    # Build CSV
    rows = []
    for d in all_per_fold:
        fold = d["fold"]
        case_ids = d["case_ids"]
        times = d["times"]
        censors = d["censors"]
        for i, cid in enumerate(case_ids):
            for modality in ["wsi", "omic"]:
                hazard = d[f"hazard_{modality}"]  # [N, K, C]
                K = hazard.shape[1]
                C = hazard.shape[2]
                risk = d["risks"][i]
                time_v = times[i]
                censor = censors[i]
                for k in range(K):
                    for c in range(C):
                        h = hazard[i, k, c]
                        rows.append({
                            "case_id": cid,
                            "fold": fold,
                            "modality": modality,
                            "slot": k,
                            "time_bin": c,
                            "hazard": h,
                            "risk": risk,
                            "time": time_v,
                            "censor": censor,
                        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Summary
    summary = out_dir / "summary.txt"
    summary.write_text(
        f"# v3.11 per-modality diversity per-slot hazard export\n"
        f"# ckpt_root: {args.ckpt_root}\n"
        f"# folds: {list(args.folds)}\n"
        + "\n".join(
            f"fold{d['fold']}: N={len(d['case_ids'])} K_w={d['K_w']} K_o={d['K_o']} C={d['C']}"
            for d in all_per_fold
        )
    )
    print(f"Saved summary to {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
