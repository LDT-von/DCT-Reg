#!/usr/bin/env python3
"""Exp D: v3.11 per-slot hazard analysis (BLCA, 5-fold).

NOTE: v3.10 does NOT expose per-slot hazard (it was added in v3.11),
so we run this analysis on v3.11 BLCA checkpoints to characterize the
slot-level signal in the model that the redesigned audit is supposed
to capture.

Goals:
1. **Per-slot hazard signature**: For each slot, average hazard across
   time bins and patients.  Are slots distinguishable (i.e. does each
   slot encode a distinct survival pattern) or are they collapsed?

2. **Slot ranking on high vs low risk patients**: Does per-slot hazard
   order slots differently in high- vs low-event-time patients?
   This mirrors the redesigned audit's question (high anchor should
   push risk up, low anchor should push risk down).

Output:
- results/v310_proof_d_per_slot_hazard/per_fold.pkl
- results/v310_proof_d_per_slot_hazard/summary.json
- results/v310_proof_d_per_slot_hazard/REPORT.md
"""

from __future__ import annotations

import argparse
import json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Defaults for v3.11 BLCA (uni2-h)
DEFAULTS = {
    "ckpt_root": (
        "/data1/DCT-Reg/results/dct_v311_blca_uni_fixed/blca/blca/"
        "SurvOTRank_dct_v311_slot_interpretable/"
        "0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_fold{fold}/"
    ),
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
    "otehv2_iter": 50,
    "otehv2_warmup": 5,
    "otehv2_num_events": 24,
    "otehv2_heads": 4,
    "otehv2_layers": 4,
    "otehv2_dropout": 0.1,
}


def build_args(fold: int, ckpt_root_template: str) -> argparse.Namespace:
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
    args.ckpt_root = ckpt_root_template.format(fold=fold)
    return args


def predict_fold(fold: int, ckpt_root_template: str, out_dir: Path) -> Dict:
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
        SurvivalDataset,
        SurvivalDatasetFactory,
        _collate_pathways,
    )
    from survot_rank.training.model_factory import get_model
    from survot_rank.research.legacy.slotspe_runtime.utils.core_utils import (
        _process_data_and_forward,
    )

    args = build_args(fold, ckpt_root_template)
    ckpt_path = Path(args.ckpt_root) / f"model_best_s{fold}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    print(f"\n[f{fold}] building factory / val loader ...")
    factory = SurvivalDatasetFactory(
        study=args.study,
        data_path=args.data_path,
        rna_format=args.rna_format,
        signature=args.signature,
        n_bins=args.n_classes,
        label_col=args.label_col,
        num_patches=args.num_patches,
        which_splits=args.which_splits,
    )

    if args.rna_format in ("Pathways", "RNASeq", "GeneEmbedding"):
        rna_cases = set(factory.gene_data_df.columns)
        factory.clinical_df = factory.clinical_df[
            factory.clinical_df["case id"].isin(rna_cases)
        ].reset_index(drop=True)

    wsi_path = os.path.join(
        args.data_root_dir, factory.study, args.wsi_encoder, "pt_files"
    )
    test_data = SurvivalDataset(
        factory, wsi_path, "val", fold, args.encoding_dim, on_missing_wsi="error"
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=_collate_pathways, pin_memory=False,
    )

    args.omic_sizes = factory.omic_sizes
    args.omic_names = factory.omic_names
    args.pathway_names = getattr(factory, "pathway_names", None)
    omics_input_dim = None  # Pathways mode
    model = get_model(
        method=args.survot_method, args=args,
        omic_input_dim=omics_input_dim, omic_names=args.omic_names,
        pathway_names=args.pathway_names,
    )

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and (
        "state_dict" in state_dict or "model_state_dict" in state_dict
    ):
        state_dict = state_dict.get("state_dict", state_dict.get("model_state_dict"))
    skip_buffers = {"dct_stage_edges", "dct_censor_times", "dct_censor_survival"}
    state_dict = {
        k: v for k, v in state_dict.items() if not any(s in k for s in skip_buffers)
    }
    state_dict = {
        k: v for k, v in state_dict.items()
        if not (k.startswith("risk_anchor"))
    }
    result = model.load_state_dict(state_dict, strict=False)
    print(f"[f{fold}] load: missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    case_id_col = "case id" if "case id" in factory.clinical_df.columns else "case_id"
    train_fold_csv = Path(args.split_dir) / f"fold_{fold}.csv"
    fold_df = pd.read_csv(train_fold_csv)
    train_case_ids = fold_df["train"].dropna().tolist()
    train_label_df = factory.clinical_df[factory.clinical_df[case_id_col].isin(train_case_ids)]
    train_times = train_label_df[args.label_col].to_numpy(dtype=np.float32)
    train_censors = train_label_df[factory.censorship_var].to_numpy(dtype=np.float32)
    if hasattr(model, "configure_train_reference"):
        model.configure_train_reference(train_times, train_censors)

    print(f"[f{fold}] running inference ...")
    t0 = time.time()
    hazard_wsi_list = []
    hazard_omic_list = []
    risk_list = []
    case_ids = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            case_ids.append(f"f{fold}_idx{idx}")
            out, _, _, _ = _process_data_and_forward(args, model, batch, device, test=True)
            logits = out[0] if isinstance(out, tuple) else out
            explanations = model.last_explanations
            if explanations is None:
                continue
            hwsi = explanations["per_slot_hazard_wsi"].cpu().numpy()[0]
            hom = explanations["per_slot_hazard_omic"].cpu().numpy()[0]
            hazard_wsi_list.append(hwsi)
            hazard_omic_list.append(hom)
            risk_list.append(float(logits.detach().cpu().numpy().ravel()[0]))
    elapsed = time.time() - t0
    print(f"[f{fold}] elapsed={elapsed:.1f}s, N={len(hazard_wsi_list)}")

    hazard_wsi = np.stack(hazard_wsi_list, axis=0)   # [N, K_w, C]
    hazard_omic = np.stack(hazard_omic_list, axis=0)  # [N, K_o, C]
    risk = np.asarray(risk_list)

    return {
        "fold": fold,
        "case_ids": case_ids,
        "hazard_wsi": hazard_wsi,
        "hazard_omic": hazard_omic,
        "risk": risk,
        "times": test_data.label_df[args.label_col].to_numpy(dtype=np.float32),
        "censors": test_data.label_df[factory.censorship_var].to_numpy(dtype=np.float32),
        "elapsed_sec": elapsed,
    }


def analyze_one_fold(data: Dict) -> Dict:
    """Per-fold per-slot statistics."""
    hwsi = data["hazard_wsi"]  # [N, K_w, C]
    hom = data["hazard_omic"]  # [N, K_o, C]
    times = data["times"]
    censors = data["censors"]

    slot_mean_wsi = hwsi.mean(axis=0)   # [K_w, C]
    slot_mean_om = hom.mean(axis=0)
    slot_std_wsi = slot_mean_wsi.std(axis=0)  # std across K_w
    slot_std_om = slot_mean_om.std(axis=0)
    distinguishability_wsi = float(slot_std_wsi.mean())
    distinguishability_om = float(slot_std_om.mean())

    observed = censors < 0.5
    obs_times = times[observed]
    if len(obs_times) >= 4:
        q25 = float(np.quantile(obs_times, 0.25))
        q75 = float(np.quantile(obs_times, 0.75))
    else:
        q25, q75 = 0.0, 1.0

    low_mask = times >= q75   # longest observed survival
    high_mask = (times <= q25) & observed  # shortest observed survival
    if low_mask.sum() == 0 or high_mask.sum() == 0:
        monotone_rate = float("nan")
    else:
        rate_per_bin = []
        for c in range(hwsi.shape[-1]):
            mean_high = hwsi[high_mask, :, c].mean()
            mean_low = hwsi[low_mask, :, c].mean()
            if mean_high > mean_low:
                rate_per_bin.append(1.0)
            else:
                rate_per_bin.append(0.0)
        monotone_rate = float(np.mean(rate_per_bin))

    return {
        "fold": data["fold"],
        "n_cases": int(hwsi.shape[0]),
        "distinguishability_wsi": distinguishability_wsi,
        "distinguishability_omic": distinguishability_om,
        "monotone_rate": monotone_rate,
        "slot_mean_wsi_max": float(slot_mean_wsi.max()),
        "slot_mean_wsi_min": float(slot_mean_wsi.min()),
        "slot_mean_wsi_range": float(slot_mean_wsi.max() - slot_mean_wsi.min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output-dir", default="results/v310_proof_d_per_slot_hazard"
    )
    parser.add_argument(
        "--ckpt-template",
        default=(
            "/data1/DCT-Reg/results/dct_v311_blca_uni_fixed/blca/blca/"
            "SurvOTRank_dct_v311_slot_interpretable/"
            "0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_fold{fold}/"
        ),
        help="Template path with {fold} placeholder",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = [int(x) for x in args.folds.split(",")]
    per_fold_data = {}
    analyses = {}

    for fold in folds:
        try:
            d = predict_fold(fold, args.ckpt_template, output_dir)
            per_fold_data[fold] = d
            analyses[fold] = analyze_one_fold(d)
        except Exception as e:
            import traceback
            traceback.print_exc()
            analyses[fold] = {"error": str(e)}

    pkl_path = output_dir / "per_fold.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(per_fold_data, f)
    print(f"saved {pkl_path}")

    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(analyses, f, indent=2)
    print(f"saved {json_path}")

    valid = [
        a for a in analyses.values()
        if "monotone_rate" in a and not (
            isinstance(a.get("monotone_rate"), float) and np.isnan(a["monotone_rate"])
        )
    ]
    if valid:
        agg = {
            "monotone_rate_mean": float(np.mean([a["monotone_rate"] for a in valid])),
            "monotone_rate_std": float(np.std([a["monotone_rate"] for a in valid])),
            "distinguishability_wsi_mean": float(np.mean([a["distinguishability_wsi"] for a in valid])),
            "distinguishability_wsi_std": float(np.std([a["distinguishability_wsi"] for a in valid])),
            "distinguishability_omic_mean": float(np.mean([a["distinguishability_omic"] for a in valid])),
            "distinguishability_omic_std": float(np.std([a["distinguishability_omic"] for a in valid])),
        }
    else:
        agg = {}

    with open(output_dir / "aggregate_5fold.json", "w") as f:
        json.dump(agg, f, indent=2)

    lines = [
        "# v3.11 Proof D: Per-Slot Hazard Analysis (BLCA, 5-fold)\n",
        "**核心问题**:v3.11 slot-level hazard 在 high vs low event-time 患者上**是否区分**?\n",
        "(v3.10 doesn't expose per-slot hazard; this is v3.11-only.)\n",
        "\n",
        "**指标**:\n",
        "1. `distinguishability_wsi/omic` = 跨 slot 平均 hazard 的 std (越高 = slots 越 distinguishable)\n",
        "2. `monotone_rate` = 在 short-survival 患者 hazard > long-survival 患者 hazard 的 time-bin 占比\n",
        "\n",
        "## Per-fold\n",
        "\n",
        "| Fold | n_cases | distinguish_wsi | distinguish_omic | monotone_rate |",
        "|---|---|---|---|---|",
    ]
    for fold in folds:
        a = analyses.get(fold, {})
        if "monotone_rate" in a and not (isinstance(a.get("monotone_rate"), float) and np.isnan(a.get("monotone_rate", 0))):
            lines.append(
                f"| {fold} | {a['n_cases']} | {a['distinguishability_wsi']:.4f} | "
                f"{a['distinguishability_omic']:.4f} | {a['monotone_rate']:.3f} |"
            )
        else:
            lines.append(f"| {fold} | ERROR: {a.get('error','')[:30]} | | | |")

    if agg:
        lines.extend([
            "\n## 5-fold mean\n",
            "\n",
            f"- `monotone_rate` = **{agg['monotone_rate_mean']:.3f} ± {agg['monotone_rate_std']:.3f}**",
            f"- `distinguishability_wsi` = {agg['distinguishability_wsi_mean']:.4f} ± {agg['distinguishability_wsi_std']:.4f}",
            f"- `distinguishability_omic` = {agg['distinguishability_omic_mean']:.4f} ± {agg['distinguishability_omic_std']:.4f}",
            "\n",
            "## 解读\n",
            "\n",
            "- v3.11 mt = `monotone_rate > 0.5` = slot hazard 在 high-risk patient 上**显著高于** low-risk patient。",
            "- 这个数字是 v3.10 audit (info_gap_high ≈ 0) 的对应 v3.11 视角。",
            "- 若 v3.11 的 monotone_rate 显著高于 0.5 而 v3.10 audit 接近 random → 进一步说明 v3.10 的 'direction loss 失活' 是版本特异的架构问题。",
        ])
    md_path = output_dir / "REPORT.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"saved {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
