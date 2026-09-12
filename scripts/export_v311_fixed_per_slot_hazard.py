#!/usr/bin/env python3
"""Export per-slot hazard from FIXED v3.11 BLCA checkpoints (uses UNI 1024d).

Differences from the default export script:
- Default ckpt_root points to results/dct_v311_blca_uni_fixed/...
- Default wsi_encoder = "uni" (NOT "uni2-h")
- Default encoding_dim = 1024 (NOT 1536)
- Default slot_num_omics = 4 (matches the v311 BLCA recipe)
- Default data_root_dir = /data1/TCGA-UNI-features (matches uni)
- Default which_splits = "5fold_uni" (NOT "5fold_uni2h")

CPU supported: model is moved to CUDA only if available; otherwise stays on CPU.

Usage:
    python scripts/export_v311_fixed_per_slot_hazard.py --folds 0
    python scripts/export_v311_fixed_per_slot_hazard.py --folds 0 1 2 3 4
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


# Fixed v3.11 BLCA recipe (UNI, 1024d)
DEFAULTS = {
    "ckpt_root": "/data1/DCT-Reg/results/dct_v311_blca_uni_fixed/blca/blca/SurvOTRank_dct_v311_slot_interpretable/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_fold{}/",
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
    "specific_simple": "dct_v311_blca_uni",
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
    "otehv2_warmup": 5,
    "otehv2_num_events": 24,
    "otehv2_heads": 4,
    "otehv2_layers": 4,
    "otehv2_dropout": 0.1,
}


def build_args(fold: int, ckpt_root: str) -> argparse.Namespace:
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
    # Resolve ckpt_root per fold
    args.ckpt_root = ckpt_root.format(fold)
    return args


def get_model_for_fold(args, dataset_factory, ckpt_path: Path, fold: int):
    from survot_rank.training.model_factory import get_model

    if args.rna_format == "RNASeq":
        omics_input_dim = dataset_factory.num_genes or dataset_factory.omic_sizes
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

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and (
        "state_dict" in state_dict or "model_state_dict" in state_dict
    ):
        state_dict = state_dict.get("state_dict", state_dict.get("model_state_dict"))
    skip_buffers = {"dct_stage_edges", "dct_censor_times", "dct_censor_survival"}
    state_dict = {k: v for k, v in state_dict.items() if not any(s in k for s in skip_buffers)}
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("risk_anchor")}
    result = model.load_state_dict(state_dict, strict=False)
    print(f"[f{fold}] load_state_dict OK: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    model.eval()
    return model


def predict_fold(fold: int, args, out_dir: Path) -> Dict:
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
        SurvivalDataset,
        SurvivalDatasetFactory,
        _collate_pathways,
    )

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

    wsi_path = os.path.join(args.data_root_dir, factory.study, args.wsi_encoder, "pt_files")
    val_data = SurvivalDataset(
        factory, wsi_path, "val", fold, args.encoding_dim, on_missing_wsi="error"
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_pathways,
        pin_memory=False,
    )

    case_id_col = "case id" if "case id" in val_data.label_df.columns else "case_id"
    val_ids = list(val_data.label_df[case_id_col])
    times = val_data.label_df[args.label_col].to_numpy(dtype=np.float32)
    censors = val_data.label_df[factory.censorship_var].to_numpy(dtype=np.float32)
    print(f"[f{fold}] val cases: {len(val_ids)}")

    ckpt_path = Path(args.ckpt_root) / f"model_best_s{fold}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    model = get_model_for_fold(args, factory, ckpt_path, fold)

    # IPCW reference from train labels
    train_fold_csv = Path(args.split_dir) / f"fold_{fold}.csv"
    fold_df = pd.read_csv(train_fold_csv)
    train_case_ids = fold_df["train"].dropna().tolist()
    case_id_col = "case id" if "case id" in factory.clinical_df.columns else "case_id"
    train_label_df = factory.clinical_df[factory.clinical_df[case_id_col].isin(train_case_ids)]
    if len(train_label_df) == 0:
        val_case_ids = set(fold_df["val"].dropna().tolist())
        train_label_df = factory.clinical_df[~factory.clinical_df[case_id_col].isin(val_case_ids)]
    train_times = train_label_df[args.label_col].to_numpy(dtype=np.float32)
    train_censors = train_label_df[factory.censorship_var].to_numpy(dtype=np.float32)
    if hasattr(model, "configure_train_reference"):
        model.configure_train_reference(train_times, train_censors)
    if hasattr(model, "eval"):
        model.eval()

    per_slot_hazard_wsi = []
    per_slot_hazard_omic = []
    risks = []
    patch_assign = []
    omic_assign = []
    print(f"[f{fold}] ── inference ──")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # v3.11 / per-slot interpretable returns 5-tuples:
            # (data, omics, label, event_time, c)
            # omics is a LIST (one entry per sample) of LIST-OF-TENSORS (per pathway).
            if len(batch) == 6:
                data, omics, _, _, _ = batch  # 6-tuple (data, omics_list, label, et, c, clinical)
            else:
                data, omics, _, _, _ = batch  # 5-tuple (data, omics_list, label, et, c)
            device = next(model.parameters()).device
            data = data.to(device)
            # Pathways format → list[B][P] (each entry: 1 tensor per pathway)
            # The pathway entry has shape [dim_i] (varying dims per pathway).
            # omics[batch_idx][pathway_idx] → 1D tensor of dim_i for that pathway+sample
            # We need to pass x_omic1, x_omic2, ... x_omicN, where each is the
            # pathway tensor for all samples in the batch.
            if isinstance(omics, list):
                # omics is list[B][P]; transpose to list[P][B]
                B = len(omics)
                P = len(omics[0])
                omics_by_pathway = []
                for pathway_index in range(P):
                    # Stack this pathway across all samples in the batch.
                    per_pathway = [omics[batch_index][pathway_index] for batch_index in range(B)]
                    omics_by_pathway.append(torch.stack(per_pathway).to(device))
            else:
                omics_by_pathway = [omics.to(device)]

            input_kwargs = {
                "x_wsi": data,
                "cur_epoch": 0,
                "wsi_missing": False,
                "omic_missing": False,
                "event_time": None,
                "c": None,
            }
            for pathway_index, pathway_tensor in enumerate(omics_by_pathway, start=1):
                input_kwargs[f"x_omic{pathway_index}"] = pathway_tensor

            out = model(**input_kwargs)
            # Per-slot hazards are computed inside forward in v3.11
            if hasattr(model, "per_slot_hazard_wsi_output"):
                psh_w = model.per_slot_hazard_wsi_output.detach().cpu().numpy()
                psh_o = model.per_slot_hazard_omic_output.detach().cpu().numpy()
                per_slot_hazard_wsi.append(psh_w)
                per_slot_hazard_omic.append(psh_o)
            elif hasattr(model, "_last_slots_wsi") and hasattr(model, "_last_slots_omic"):
                # v3.11 stores slot representations; compute hazard from them directly
                with torch.no_grad():
                    slots_wsi_t = model._last_slots_wsi
                    slots_omic_t = model._last_slots_omic
                    hazard_w = model.per_slot_hazard_wsi(slots_wsi_t).detach().cpu().numpy()
                    hazard_o = model.per_slot_hazard_omic(slots_omic_t).detach().cpu().numpy()
                # Apply sigmoid to get probabilities (consistent with v3.11 monitoring)
                hazard_w = 1.0 / (1.0 + np.exp(-hazard_w))
                hazard_o = 1.0 / (1.0 + np.exp(-hazard_o))
                per_slot_hazard_wsi.append(hazard_w)
                per_slot_hazard_omic.append(hazard_o)
            else:
                print(f"  [warn] batch {batch_idx}: cannot extract per-slot hazard; skipping")
                continue

            if "risk" in out:
                if isinstance(out["risk"], torch.Tensor):
                    risks.append(out["risk"].detach().cpu().numpy().ravel())
                else:
                    risks.append(np.array([out["risk"]]))

            if batch_idx % 10 == 0:
                print(f"  [{batch_idx}/{len(val_loader)}] processed", flush=True)

    risks = np.concatenate(risks) if risks and isinstance(risks[0], np.ndarray) else np.array(risks)
    per_slot_hazard_wsi = np.concatenate(per_slot_hazard_wsi, axis=0) if per_slot_hazard_wsi else np.empty((0,))
    per_slot_hazard_omic = np.concatenate(per_slot_hazard_omic, axis=0) if per_slot_hazard_omic else np.empty((0,))

    return {
        "fold": fold,
        "case_ids": val_ids,
        "times": times,
        "censors": censors,
        "risks": risks,
        "hazard_wsi": per_slot_hazard_wsi,
        "hazard_omic": per_slot_hazard_omic,
        "K_w": int(per_slot_hazard_wsi.shape[1]) if per_slot_hazard_wsi.size else 0,
        "K_o": int(per_slot_hazard_omic.shape[1]) if per_slot_hazard_omic.size else 0,
        "C": int(per_slot_hazard_wsi.shape[2]) if per_slot_hazard_wsi.size else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out_dir", type=str, default="/data1/DCT-Reg/results/dct_v311_blca_uni_fixed/per_slot_export")
    ap.add_argument("--ckpt_root", type=str, default=DEFAULTS["ckpt_root"])
    args_cli = ap.parse_args()

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_per_fold = []
    for fold in args_cli.folds:
        args = build_args(fold, args_cli.ckpt_root)
        t0 = time.time()
        try:
            fold_data = predict_fold(fold, args, out_dir)
            all_per_fold.append(fold_data)
            print(f"[f{fold}] ✅ done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[f{fold}] ❌ FAIL: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_per_fold:
        pkl_path = out_dir / "per_slot_hazard.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(all_per_fold, f)
        print(f"\nSaved {len(all_per_fold)} folds to {pkl_path}")


if __name__ == "__main__":
    main()
