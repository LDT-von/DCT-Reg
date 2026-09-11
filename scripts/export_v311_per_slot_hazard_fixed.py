#!/usr/bin/env python3
"""Export per-slot hazard predictions for v3.11 BLCA 5-fold checkpoints.

Fixed version with correct paths for v3.11 training.
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Correct paths for v3.11 (wsi_encoder=uni)
DEFAULTS = {
    "ckpt_root": "/data1/DCT-Reg/results/dct_v311_blca_uni/blca/SurvOTRank_dct_v311/0.0005_b32_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_",
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
    "wsi_encoder": "uni",  # Fixed: was uni2h
    "data_root_dir": "/data/CPathPatchFeature",  # Fixed: WSI features at CPathPatchFeature, not splits dir
    "n_bins": 4,
    "method": "SurvOTRank_dct_v311",
    "survot_method": "dct_v311_slot_interpretable",
    "bag_loss": "nll_surv",
    "rg_num_events": 4,
    "spt_num_stages": 4,
    "rg_prog_cost": 0.20,
    "rg_lambda_ot": 0.06,
    "rg_lambda_rank": 0.15,
    "rg_lambda_stage": 0.02,
    "rg_eps_start": 0.10,
    "rg_eps_anneal": 12,
    "dct_num_stages": 4,
    "dct_lambda_ipcw_rank": 0.10,
    "dct_ipcw_rank_margin": 0.02,
    "dct_ipcw_rank_temperature": 0.50,
    "dct_ipcw_max_weight": 10.0,
    "dct_ipcw_rank_memory_size": 0,
    "dct_lambda_etar": 0.0,
    "dct_etar_margin": 0.02,
    "dct_etar_uncertainty_weight": 0.05,
    "dct_slot_init_mode": "gaussian",
    "dct_slot_eval_seed": 1729,
    "otehv2_eps": 0.05,
    "otehv2_iter": 20,
    "otehv2_warmup": 3,
    "otehv2_num_events": 4,
    "otehv2_heads": 4,
    "otehv2_layers": 2,
    "otehv2_dropout": 0.0,
}


def build_args(fold: int) -> argparse.Namespace:
    args = argparse.Namespace()
    for k, v in DEFAULTS.items():
        setattr(args, k, v)
    args.cur_epoch = 0
    args.batch_size = 1
    args.fit_bins_on_train = False
    args.binning_mode = "global_qcut"
    args.which_splits = "5fold"
    args.num_workers = 0
    args.fold = fold
    return args


def get_model_for_fold(args, dataset_factory, ckpt_path: Path, fold: int):
    from survot_rank.training.model_factory import get_model

    if args.rna_format == "RNASeq":
        omics_input_dim = (
            dataset_factory.num_genes
            if dataset_factory.num_genes is not None
            else dataset_factory.omic_sizes
        )
    elif args.rna_format == "GeneEmbedding":
        omics_input_dim = 768
    else:
        omics_input_dim = None

    args.omic_sizes = dataset_factory.omic_sizes
    args.omic_names = dataset_factory.omic_names
    args.pathway_names = getattr(dataset_factory, "pathway_names", None)

    model = get_model(
        method=args.survot_method,
        args=args,
        omic_input_dim=omics_input_dim,
        omic_names=args.omic_names,
        pathway_names=args.pathway_names,
    )

    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state_dict, dict) and (
        "state_dict" in state_dict or "model_state_dict" in state_dict
    ):
        state_dict = state_dict.get("state_dict", state_dict.get("model_state_dict"))
    
    # Strict loading: ensure all keys match exactly.
    model_state = model.state_dict()
    filtered_state = {}
    missing_keys = []
    unexpected_keys = []
    
    # Check for missing keys
    for k in model_state:
        if k not in state_dict:
            missing_keys.append(k)
    
    # Check for unexpected keys
    for k in state_dict:
        if k not in model_state:
            unexpected_keys.append(k)
    
    # Only load keys that exist in both
    for k in state_dict:
        if k in model_state:
            if model_state[k].shape == state_dict[k].shape:
                filtered_state[k] = state_dict[k]
            else:
                raise RuntimeError(
                    f"[f{fold}] Shape mismatch for key '{k}': "
                    f"checkpoint {state_dict[k].shape} vs model {model_state[k].shape}"
                )
    
    if missing_keys:
        print(f"[f{fold}] WARNING: {len(missing_keys)} keys missing from checkpoint: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"[f{fold}] WARNING: {len(unexpected_keys)} unexpected keys in checkpoint (skipped): {unexpected_keys[:5]}...")
    
    # Use strict=True to ensure no silent failures
    model.load_state_dict(filtered_state, strict=True)

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    model.eval()
    return model


def build_val_loader(args, dataset_factory, fold: int):
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
        SurvivalDataset,
        _collate_pathways,
    )

    wsi_path = os.path.join(args.data_root_dir, args.study, args.wsi_encoder, "pt_files")
    print(f"[DEBUG] Looking for WSI features at: {wsi_path}")
    
    test_data = SurvivalDataset(
        dataset_factory,
        wsi_path,
        "val",
        fold,
        args.encoding_dim,
        on_missing_wsi="error",
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_pathways,
        pin_memory=False,
    )
    return test_loader, test_data


def predict_fold(fold: int, ckpt_root: Path, out_dir: Path) -> Dict:
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
        SurvivalDatasetFactory,
    )

    args = build_args(fold)
    print(f"\n[f{fold}] ── building dataset factory & val loader ──")

    factory = SurvivalDatasetFactory(
        study=args.study,
        data_path=args.data_path,
        rna_format=args.rna_format,
        signature=args.signature,
        n_bins=args.n_bins,
        label_col=args.label_col,
        num_patches=args.num_patches,
        which_splits=args.which_splits,
    )

    val_loader, val_data = build_val_loader(args, factory, fold)
    val_ids = list(val_data.label_df["case id"])

    # Read labels from clinical CSV for configure_train_reference
    clinical_path = os.path.join(args.data_path, "clinical", "all", f"{args.study}.csv")
    clinical_df = pd.read_csv(clinical_path)
    censor_col = "censorship_dss"

    # Get train case IDs from fold CSV
    train_fold_csv = Path(args.split_dir) / f"fold_{fold}.csv"
    train_fold_df = pd.read_csv(train_fold_csv)
    train_case_ids = train_fold_df["train"].dropna().tolist()
    train_clinical = clinical_df[clinical_df["case id"].isin(train_case_ids)]
    train_times = train_clinical[args.label_col].to_numpy(dtype=np.float32)
    train_censors = train_clinical[censor_col].to_numpy(dtype=np.float32)

    # Get val data labels
    label_col_time = args.label_col
    val_times = val_data.label_df[label_col_time].to_numpy(dtype=np.float32)
    val_censors = val_data.label_df[censor_col].to_numpy(dtype=np.float32)
    times = val_times
    censors = val_censors
    print(f"[f{fold}] val cases: {len(val_ids)}")

    ckpt_path = ckpt_root / f"model_best_s{fold}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    print(f"[f{fold}] ── loading {ckpt_path.name} ──")

    model = get_model_for_fold(args, factory, ckpt_path, fold)

    if hasattr(model, "configure_train_reference"):
        model.configure_train_reference(train_times, train_censors)
    model.eval()

    print(f"[f{fold}] ── running eval forward passes ──")
    t0 = time.time()
    hazard_wsi_list: List[np.ndarray] = []
    hazard_omic_list: List[np.ndarray] = []
    risk_list: List[float] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            data_wsi = batch[0].to(device)
            
            # Unpack omics data similar to _unpack_data
            data_omics = batch[1]
            
            y_disc = batch[2].to(device)
            event_time = batch[3].to(device)
            c = batch[4].to(device)

            # Build input_args similar to _process_data_and_forward
            input_args = {
                "x_wsi": data_wsi,
                "cur_epoch": args.cur_epoch,
                "wsi_missing": False,
                "omic_missing": False,
                "event_time": event_time,
                "y": y_disc,
                "c": c,
            }
            
            # Add omics data with x_omic1, x_omic2, etc.
            if args.rna_format in ("Pathways", "RankedGenes"):
                omic_data_list = []
                for omic_item in data_omics:
                    for omic in omic_item:
                        omic_data_list.append(omic.to(device).unsqueeze(0))
                input_args.update(
                    {f"x_omic{index}": omic for index, omic in enumerate(omic_data_list, start=1)}
                )
            else:
                input_args["x_omics"] = data_omics
            logits, _aux = model(**input_args)

            explanations = model.last_explanations
            hwsi = explanations["per_slot_hazard_wsi"].cpu().numpy()[0]
            hom = explanations["per_slot_hazard_omic"].cpu().numpy()[0]
            hazard_wsi_list.append(hwsi)
            hazard_omic_list.append(hom)
            # Final risk: -sum over all time bins of S(t) = -sum(log S_padded[1:C+1])
            # Use the model's _risk logic but on the full hazard trajectory
            # Equivalent: risk = -sum_t S(t) = -sum_t (1 - cumsum_h(0..t-1))
            # We compute this as the negative survival probability across all bins
            with torch.no_grad():
                hazard_full = torch.from_numpy(np.concatenate([hwsi, hom], axis=0)).to(device)  # [K_total, C]
                # S_padded[t] = 1, S_padded[t+1] = S(t) = product of (1 - h(i)) for i in 0..t
                surv = torch.cumprod(1.0 - hazard_full, dim=-1)  # [K_total, C]
                # Final risk = -sum of S(t) across all time bins, averaged across slots
                final_risk = -surv.sum(dim=-1).mean().item()
            risk_list.append(final_risk)

    elapsed = time.time() - t0
    print(f"[f{fold}] elapsed={elapsed:.1f}s ({len(val_ids)} patients)")

    hazard_wsi = np.stack(hazard_wsi_list, axis=0)
    hazard_omic = np.stack(hazard_omic_list, axis=0)
    risks = np.asarray(risk_list, dtype=np.float32)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold": fold,
        "case_ids": val_ids,
        "times": times,
        "censors": censors,
        "risks": risks,
        "hazard_wsi": hazard_wsi,
        "hazard_omic": hazard_omic,
        "K_w": hazard_wsi.shape[1],
        "K_o": hazard_omic.shape[1],
        "C": hazard_wsi.shape[2],
    }


def write_csv_long(out_dir: Path, per_fold: List[Dict]) -> None:
    rows = []
    for d in per_fold:
        N, K_w, C = d["hazard_wsi"].shape
        _, K_o, _ = d["hazard_omic"].shape
        for i in range(N):
            cid = d["case_ids"][i]
            for k in range(K_w):
                for c in range(C):
                    rows.append({
                        "case_id": cid,
                        "fold": d["fold"],
                        "modality": "wsi",
                        "slot": k,
                        "time_bin": c,
                        "hazard": d["hazard_wsi"][i, k, c],
                        "risk": d["risks"][i],
                        "time": d["times"][i],
                        "censor": d["censors"][i],
                    })
            for k in range(K_o):
                for c in range(C):
                    rows.append({
                        "case_id": cid,
                        "fold": d["fold"],
                        "modality": "omic",
                        "slot": k,
                        "time_bin": c,
                        "hazard": d["hazard_omic"][i, k, c],
                        "risk": d["risks"][i],
                        "time": d["times"][i],
                        "censor": d["censors"][i],
                    })
    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_slot_hazard.csv"
    df.to_csv(csv_path, index=False)
    print(f"[export] long-form CSV → {csv_path}  ({len(df):,} rows)")


def main():
    parser = argparse.ArgumentParser(description="Export v3.11 per-slot hazard.")
    parser.add_argument("--ckpt_root", type=str, default=DEFAULTS["ckpt_root"])
    parser.add_argument("--out_root", type=str,
                        default="/data1/DCT-Reg/results/dct_v311_blca_uni/per_slot_export")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args_cli = parser.parse_args()

    ckpt_root = Path(args_cli.ckpt_root)
    out_dir = Path(args_cli.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold = []
    for fold in args_cli.folds:
        per_fold.append(predict_fold(fold, ckpt_root, out_dir))

    pkl_path = out_dir / "per_slot_hazard.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(per_fold, f)
    print(f"[export] per-fold pkl → {pkl_path}")

    write_csv_long(out_dir, per_fold)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"# v3.11 per-slot hazard export\n")
        f.write(f"# ckpt_root: {ckpt_root}\n")
        f.write(f"# folds: {args_cli.folds}\n")
        for d in per_fold:
            f.write(
                f"fold{d['fold']}: N={len(d['case_ids'])} "
                f"K_w={d['K_w']} K_o={d['K_o']} C={d['C']}\n"
            )
    print(f"[export] summary → {summary_path}")

    print("\n[export] per-fold summary (mean hazard across slots, time-bin 0):")
    for d in per_fold:
        mw = d["hazard_wsi"][:, :, 0].mean()
        mo = d["hazard_omic"][:, :, 0].mean()
        print(f"  fold{d['fold']}: hazard_wsi[:0]={mw:.3f} "
              f"hazard_omic[:0]={mo:.3f}  risk_mean={d['risks'].mean():.3f}")


if __name__ == "__main__":
    main()
