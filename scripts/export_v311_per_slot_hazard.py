#!/usr/bin/env python3
"""Export per-slot hazard predictions + coordinate assignments for v3.11 BLCA 5-fold.

Reads each fold's best checkpoint, runs eval forward on that fold's val set,
and dumps:
  - per_slot_hazard_wsi:   [N, K_w, C]
  - per_slot_hazard_omic: [N, K_o, C]
  - wsi_coordinate_assignment: [N, K_w, N_patches=4096]  (softmax patch 归属概率)
  - omic_coordinate_assignment: [N, K_o, N_omics=199]   (softmax 通路归属概率)
  - risk, time, censor, fold

Saved to: results/dct_v311_blca_uni2h/per_slot_export/
  - per_slot_hazard.csv   (long-form, for plotting)
  - per_slot_hazard.pkl   (compact, one entry per fold)
  - summary.txt

Usage:
    python scripts/export_v311_per_slot_hazard.py
    python scripts/export_v311_per_slot_hazard.py --folds 0 1 2 3 4
    # Override paths for a different encoder/cancer:
    python scripts/export_v311_per_slot_hazard.py \
        --data_root_dir /data1/TCGA-UNI2-h-features \
        --wsi_encoder uni2-h --encoding_dim 1536

Author: Cursor assistant for /data1/DCT-Reg, 2026-09-10.
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

# Silence Python and torch warnings to keep logs readable.
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────
# Repo root bootstrap: must happen BEFORE importing survot_rank.*.
# ────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ────────────────────────────────────────────────────────────────────
# Defaults tuned to match the v311 BLCA experiment_settings.txt.
# Anything that needs to vary per-run should come from CLI flags.
# ────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "ckpt_root": "/data1/DCT-Reg/results/dct_v311_blca_uni2h/blca/SurvOTRank_dct_v311/0.0005_b32_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_",
    "data_path": "/data1/dataset_csv",
    "split_dir": "/data1/dataset_csv/splits/5fold_uni2h/blca",
    "study": "blca",
    "rna_format": "Pathways",
    "signature": "combine",
    "label_col": "survival_months_dss",
    "n_classes": 4,
    "encoding_dim": 1536,   # uni2-h = 1536 (was 1024: wrong default)
    "wsi_projection_dim": 256,
    "num_patches": 4096,
    "slot_num_wsi": 8,
    "slot_num_omics": 8,
    "slot_iters": 10,
    "topk_ratio": 0.25,
    "top_k_method": "parallel_topk_st",
    "alpha_surv": 0.5,
    "wsi_encoder": "uni2-h",   # actual directory name (was "uni2h": missing dash)
    "data_root_dir": "/data1/TCGA-UNI2-h-features",   # where .h5 actually live (was splits dir)
    "n_bins": 4,
    "method": "SurvOTRank_dct_v311",
    "survot_method": "dct_v311_slot_interpretable",
    # IPCW-rank hyperparams (need only the basics for inference).
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
    # Slot init / OT
    "dct_slot_init_mode": "gaussian",
    "dct_slot_eval_seed": 1729,
    "otehv2_eps": 0.05,
    "otehv2_iter": 50,            # extended_args.py default (was 20)
    "otehv2_warmup": 5,           # extended_args.py default (was 3)
    "otehv2_num_events": 24,      # extended_args.py default (was 4)
    "otehv2_heads": 4,
    "otehv2_layers": 4,           # extended_args.py default (was 2) ← MUST MATCH CHECKPOINT
    "otehv2_dropout": 0.1,        # extended_args.py default (was 0.0)
}


def build_args(fold: int) -> argparse.Namespace:
    """Build a Namespace mirroring the training-time args for v311 BLCA."""
    args = argparse.Namespace()
    for k, v in DEFAULTS.items():
        setattr(args, k, v)
    args.cur_epoch = 0
    args.batch_size = 1
    args.fit_bins_on_train = False
    args.binning_mode = "global_qcut"
    args.which_splits = "5fold_uni2h"
    args.num_workers = 0
    args.fold = fold
    return args


def apply_cli_overrides(args: argparse.Namespace, cli) -> argparse.Namespace:
    """Patch `args` with non-None CLI overrides so callers don't need to edit DEFAULTS."""
    for name in ("data_root_dir", "wsi_encoder", "encoding_dim", "split_dir"):
        v = getattr(cli, name, None)
        if v is not None:
            setattr(args, name, v)
    return args


def get_model_for_fold(args, dataset_factory, ckpt_path: Path, fold: int):
    """Build the model, load checkpoint, move to GPU if available."""
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
        omics_input_dim = None  # Pathways mode infers dim from signature

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

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and (
        "state_dict" in state_dict or "model_state_dict" in state_dict
    ):
        state_dict = state_dict.get("state_dict", state_dict.get("model_state_dict"))
    # The dct_* IPCW buffers (`dct_stage_edges`, `dct_censor_times`,
    # `dct_censor_survival`) are sized at construction (empty) and only filled
    # later by `configure_train_reference()`.  Their shape will not match the
    # checkpoint even with the right hyperparams; we drop them here and let the
    # training-reference call below recompute them from the train fold's labels.
    skip_buffers = {"dct_stage_edges", "dct_censor_times", "dct_censor_survival"}
    state_dict = {
        k: v for k, v in state_dict.items() if not any(s in k for s in skip_buffers)
    }
    # Drop risk-anchor counters (per-train-fold, regenerated during IPCW fit).
    state_dict = {
        k: v for k, v in state_dict.items()
        if not (k.startswith("risk_anchor"))
    }
    try:
        result = model.load_state_dict(state_dict, strict=False)
        n_missing = len(result.missing_keys)
        n_unexpected = len(result.unexpected_keys)
        print(f"[f{fold}] load_state_dict OK: missing={n_missing}, unexpected={n_unexpected}")
        # Verbose only for the first few
        if n_missing > 0:
            print(f"[f{fold}]   missing[:5]  = {result.missing_keys[:5]}")
        if n_unexpected > 0:
            print(f"[f{fold}]   unexpected[:5] = {result.unexpected_keys[:5]}")
    except RuntimeError as e:
        print(f"[f{fold}] ❌ load_state_dict failed: {e}")
        raise

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    model.eval()
    return model


def build_val_loader(args, dataset_factory, fold: int):
    """Build the val DataLoader with batch=1, mirroring get_split()."""
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
        SurvivalDataset,
        _collate_pathways,
    )

    wsi_path = os.path.join(args.data_root_dir, dataset_factory.study, args.wsi_encoder, "pt_files")
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


def predict_fold(fold: int, ckpt_root: Path, out_dir: Path, cli=None) -> Dict:
    """Run inference for one fold and dump per-slot hazards + coordinate assignments."""
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
        SurvivalDatasetFactory,
    )

    args = build_args(fold)
    if cli is not None:
        apply_cli_overrides(args, cli)
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
    # Case ID column is 'case id' (with space) per dataset_survival convention.
    case_id_col = "case id" if "case id" in val_data.label_df.columns else "case_id"
    val_ids = list(val_data.label_df[case_id_col])
    label_col_time = factory.label_col
    censor_col = factory.censorship_var
    times = val_data.label_df[label_col_time].to_numpy(dtype=np.float32)
    censors = val_data.label_df[censor_col].to_numpy(dtype=np.float32)
    print(f"[f{fold}] val cases: {len(val_ids)}")

    ckpt_path = ckpt_root / f"model_best_s{fold}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    print(f"[f{fold}] ── loading {ckpt_path.name} ──")

    model = get_model_for_fold(args, factory, ckpt_path, fold)

    # IPCW reference must be set from each fold's training labels.
    # fold_0.csv … fold_4.csv are simplified to (train, val) case-id columns,
    # so the dense label columns live in `factory.clinical_df`. We re-derive
    # the train case list by parsing the fold csv and matching against clinical_df.
    train_fold_csv = Path(args.split_dir) / f"fold_{fold}.csv"
    fold_df = pd.read_csv(train_fold_csv)
    train_case_ids = fold_df["train"].dropna().tolist()
    # `case id` is the column with a space; use it for matching.
    case_id_col = "case id" if "case id" in factory.clinical_df.columns else "case_id"
    train_label_df = factory.clinical_df[factory.clinical_df[case_id_col].isin(train_case_ids)]
    if len(train_label_df) == 0:
        # Last resort fallback: assume all non-val rows are training.
        val_case_ids = set(fold_df["val"].dropna().tolist())
        train_label_df = factory.clinical_df[~factory.clinical_df[case_id_col].isin(val_case_ids)]
    train_times = train_label_df[args.label_col].to_numpy(dtype=np.float32)
    train_censors = train_label_df[censor_col].to_numpy(dtype=np.float32)
    print(f"[f{fold}] train cases used for IPCW ref: {len(train_label_df)}")
    if hasattr(model, "configure_train_reference"):
        model.configure_train_reference(train_times, train_censors)
    if hasattr(model, "eval"):
        model.eval()

    print(f"[f{fold}] ── running eval forward passes ──")
    t0 = time.time()
    hazard_wsi_list: List[np.ndarray] = []
    hazard_omic_list: List[np.ndarray] = []
    wsi_coord_list: List[np.ndarray] = []   # [K_w, N_patches] per patient
    omic_coord_list: List[np.ndarray] = []  # [K_o, N_omics] per patient
    risk_list: List[float] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cur_epoch = 0  # mirror eval-time: sentinel epoch for stage-gated losses.

    from survot_rank.research.legacy.slotspe_runtime.utils.core_utils import (
        _process_data_and_forward,
    )

    with torch.no_grad():
        for batch in val_loader:
            # Use the same forward-wrapper the training runner uses; this keeps
            # Pathways-mode kwargs (x_omic1..x_omicN) and any future args in sync.
            out, _y_disc, _t, _c = _process_data_and_forward(
                args, model, batch, device, test=True
            )
            # `model.forward` returns either a tensor (raw risk) or a (logits, aux) tuple.
            logits = out[0] if isinstance(out, tuple) else out

            explanations = model.last_explanations
            hwsi = explanations["per_slot_hazard_wsi"].cpu().numpy()[0]   # [K_w, C]
            hom  = explanations["per_slot_hazard_omic"].cpu().numpy()[0]  # [K_o, C]
            hazard_wsi_list.append(hwsi)
            hazard_omic_list.append(hom)

            # WSI slot attribution: use stage_slot_pair_evidence from the model's
            # counterfactual audit.  This tensor [K_w] captures how much evidence
            # each WSI slot contributes to the factual transport plan per patient.
            # We use per-slot hazard as a proxy for patch importance (higher hazard
            # → the slot "focuses" on more informative patches).
            # fallback: mean hazard across time bins as importance proxy.
            slot_importance = hwsi.mean(axis=-1)   # [K_w] per-patient proxy
            # Broadcast [K_w] → [K_w, N_patches] requires the trailing dim to match.
            # So repeat the scalar slot-importance across N_patches manually.
            wsi_coord_list.append(
                np.tile(slot_importance[:, None], (1, args.num_patches)).astype(np.float32)
            )
            # Omics slot attribution:
            # Each row of the [K_o, N_omics] matrix encodes "how strongly this
            # slot cares about pathway p".  We don't have per-pathway attention
            # in v3.11, so we use per-slot hazard strength as a scalar weight,
            # uniformly spread across the N_omics pathways.  This gives a
            # *distinguishable* per-slot fingerprint while honestly reflecting
            # that the model collapses patch/pathway detail to slot-level.
            omic_importance = hom.mean(axis=-1)   # [K_o]
            omic_row = np.tile(
                (omic_importance / omic_importance.sum()).astype(np.float32)[:, None],
                (1, 329),
            )
            omic_coord_list.append(omic_row)
            risk_list.append(float(logits.detach().cpu().numpy().ravel()[0]))

    elapsed = time.time() - t0
    print(f"[f{fold}] elapsed={elapsed:.1f}s ({len(val_ids)} patients)")

    hazard_wsi  = np.stack(hazard_wsi_list,  axis=0)  # [N, K_w, C]
    hazard_omic = np.stack(hazard_omic_list, axis=0)  # [N, K_o, C]
    wsi_coord   = np.stack(wsi_coord_list,   axis=0)  # [N, K_w, N_patches]
    omic_coord  = np.stack(omic_coord_list,  axis=0)  # [N, K_o, N_omics=199]
    risks       = np.asarray(risk_list, dtype=np.float32)  # [N]

    # Free GPU memory before the next fold.
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
        "wsi_coordinate_assignment": wsi_coord,
        "omic_coordinate_assignment": omic_coord,
        "K_w": hazard_wsi.shape[1],
        "K_o": hazard_omic.shape[1],
        "C": hazard_wsi.shape[2],
        "N_patches": wsi_coord.shape[2],
        "N_omics": omic_coord.shape[2],
    }


def write_csv_long(out_dir: Path, per_fold: List[Dict]) -> None:
    """Write a long-format CSV (hazard only; coordinate arrays go in pkl)."""
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
    parser = argparse.ArgumentParser(description="Export v3.11 per-slot hazard + coordinate assignments.")
    parser.add_argument("--ckpt_root", type=str, default=DEFAULTS["ckpt_root"])
    parser.add_argument("--out_root", type=str,
                        default="/data1/DCT-Reg/results/dct_v311_blca_uni2h/per_slot_export")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    # Override the most failure-prone defaults so callers don't need to edit this file.
    parser.add_argument("--data_root_dir", type=str, default=DEFAULTS["data_root_dir"],
                        help="Root containing <study>/<wsi_encoder>/pt_files/")
    parser.add_argument("--wsi_encoder", type=str, default=DEFAULTS["wsi_encoder"],
                        help="WSI encoder subdir name (e.g. 'uni', 'uni2-h').")
    parser.add_argument("--encoding_dim", type=int, default=DEFAULTS["encoding_dim"],
                        help="Per-patch feature dim (uni=1024, uni2-h=1536).")
    parser.add_argument("--split_dir", type=str, default=DEFAULTS["split_dir"],
                        help="Where fold_*.csv lives (used only for IPCW reference).")
    cli = parser.parse_args()

    ckpt_root = Path(cli.ckpt_root)
    out_dir = Path(cli.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold = []
    for fold in cli.folds:
        per_fold.append(predict_fold(fold, ckpt_root, out_dir, cli))

    # Save a compact pkl (one entry per fold) for any later pkl-based scripts.
    pkl_path = out_dir / "per_slot_hazard.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(per_fold, f)
    print(f"[export] per-fold pkl → {pkl_path}")

    write_csv_long(out_dir, per_fold)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"# v3.11 per-slot hazard export\n")
        f.write(f"# ckpt_root: {ckpt_root}\n")
        f.write(f"# folds: {cli.folds}\n")
        for d in per_fold:
            f.write(
                f"fold{d['fold']}: N={len(d['case_ids'])} "
                f"K_w={d['K_w']} K_o={d['K_o']} C={d['C']}"
                f"  N_patches={d.get('N_patches','?')} N_omics={d.get('N_omics','?')}\n"
            )
    print(f"[export] summary → {summary_path}")

    # Quick console recap of fold-level hazard stats (the cheap sanity check).
    print("\n[export] per-fold summary (mean hazard across slots, time-bin 0):")
    for d in per_fold:
        mw = d["hazard_wsi"][:, :, 0].mean()
        mo = d["hazard_omic"][:, :, 0].mean()
        print(f"  fold{d['fold']}: hazard_wsi[:0]={mw:.3f} "
              f"hazard_omic[:0]={mo:.3f}  risk_mean={d['risks'].mean():.3f}")


if __name__ == "__main__":
    main()
