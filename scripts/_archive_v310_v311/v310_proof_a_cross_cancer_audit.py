#!/usr/bin/env python3
"""Exp A: Cross-cancer redesigned audit for v3.10.

Runs the redesigned dose-monotonicity audit (the same recipe as
results/audit_blca_redesign/) on v3.10 ckpts for HNSC, LUSC, SKCM.

Each cancer × fold runs in ~2 minutes on GPU.

Output: results/v310_proof_a_cross_cancer_audit/{cancer}/fold_{k}.json
+ summary across cancers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from survot_rank.cli import add_project_paths  # noqa: E402

add_project_paths()

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from redesign_audit_blca import (  # noqa: E402
    _load_model_and_loader,
    _load_parsed_args,
    run_fold,
    compute_end_to_end,
)

import numpy as np


CKPT_TEMPLATE = (
    "results/dct_v3.10/robust/final_50ep_old/{cancer}/{cancer}/"
    "SurvOTRank_dct_v310_directional_regularized_transport/"
    "0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_{cancer}_50ep"
)
CANCERS = ["hnsc", "lusc", "skcm"]


def run_one_cancer(cancer: str, folds, gpu, output_dir, parsed_args, alpha_list):
    ckpt_dir = Path(CKPT_TEMPLATE.format(cancer=cancer))
    summaries = {}
    for fold in folds:
        print(f"\n=== {cancer} fold {fold} ===")
        ckpt = str(ckpt_dir / f"model_best_s{fold}.pth")
        if not Path(ckpt).exists():
            print(f"  missing ckpt, skip")
            summaries[fold] = {"error": "missing ckpt"}
            continue
        # Mimic what redesign_audit_blca accepts
        args = argparse.Namespace(
            config=str(REPO_ROOT / "configs/dct_v310_directional_regularized_transport.yaml"),
            checkpoint=ckpt,
            fold=fold,
            epoch=50,
            output_dir=str(output_dir / cancer),
            set=[f"study={cancer}"],
        )
        t0 = time.time()
        try:
            res = run_fold(fold, args, alpha_list=alpha_list)
            res["elapsed_sec"] = round(time.time() - t0, 1)
            # Compute metrics (real vs rand)
            metrics = {}
            metrics["real_high"] = compute_end_to_end(res["real_high"], alpha_list)
            metrics["rand_high"] = compute_end_to_end(res["rand_high"], alpha_list)
            metrics["real_low"] = compute_end_to_end(res["real_low"], alpha_list)
            metrics["rand_low"] = compute_end_to_end(res["rand_low"], alpha_list)
            res["metrics"] = {
                "real_high_rate_up": metrics["real_high"]["rate_up"],
                "rand_high_rate_up": metrics["rand_high"]["rate_up"],
                "info_gap_high": metrics["real_high"]["rate_up"] - metrics["rand_high"]["rate_up"],
                "real_low_rate_down": metrics["real_low"]["rate_down"],
                "rand_low_rate_down": metrics["rand_low"]["rate_down"],
                "info_gap_low": metrics["real_low"]["rate_down"] - metrics["rand_low"]["rate_down"],
                "real_high_delta_abs_mean": metrics["real_high"]["delta_abs_mean"],
                "rand_high_delta_abs_mean": metrics["rand_high"]["delta_abs_mean"],
                "info_gap_high_delta_abs_mean": (
                    metrics["real_high"]["delta_abs_mean"]
                    - metrics["rand_high"]["delta_abs_mean"]
                ),
            }
            summaries[fold] = res
        except Exception as e:
            import traceback
            traceback.print_exc()
            summaries[fold] = {"error": str(e)}
        json_path = output_dir / cancer / f"fold_{fold}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(summaries[fold], f, indent=2)
        print(f"  saved {json_path}, elapsed {summaries[fold].get('elapsed_sec', '?')}s")
        if "metrics" in summaries[fold]:
            print(
                f"  high_rate_up={summaries[fold]['metrics'].get('real_high_rate_up', '?'):.3f}, "
                f"info_gap_high={summaries[fold]['metrics'].get('info_gap_high', '?'):.3f}, "
                f"low_rate_down={summaries[fold]['metrics'].get('real_low_rate_down', '?'):.3f}"
            )
    return summaries


def aggregate(summaries_by_cancer):
    """Aggregate per-cancer 5-fold mean of redesigned-audit metrics."""
    out = {}
    for cancer, folds in summaries_by_cancer.items():
        agg = {}
        # Collect all metric keys across folds
        all_keys = set()
        for f, fold_data in folds.items():
            if "metrics" in fold_data:
                all_keys.update(fold_data["metrics"].keys())
        for key in all_keys:
            vals = []
            for f, fold_data in folds.items():
                if "metrics" in fold_data and key in fold_data["metrics"]:
                    v = fold_data["metrics"][key]
                    if isinstance(v, (int, float)) and not (
                        isinstance(v, float) and np.isnan(v)
                    ):
                        vals.append(float(v))
            if vals:
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "per_fold": vals,
                }
        out[cancer] = agg
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cancers", default="hnsc,lusc,skcm")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--output-dir", default="results/v310_proof_a_cross_cancer_audit"
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    folds = [int(x) for x in args.folds.split(",")]
    cancers = args.cancers.split(",")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single args to reuse for setup
    first_args = argparse.Namespace(
        config=str(REPO_ROOT / "configs/dct_v310_directional_regularized_transport.yaml"),
        checkpoint="dummy",
        fold=0,
        epoch=50,
        output_dir=str(output_dir),
        set=[f"study={cancers[0]}"],
    )
    parsed = _load_parsed_args(first_args)

    summaries_by_cancer = {}
    for cancer in cancers:
        summaries_by_cancer[cancer] = run_one_cancer(
            cancer, folds, args.gpu, output_dir, parsed, alpha_list=(0.0, 1.0)
        )

    # Save aggregated
    aggregated = aggregate(summaries_by_cancer)
    with open(output_dir / "summary_5fold_by_cancer.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Markdown
    key_metrics = [
        "real_high_rate_up",
        "rand_high_rate_up",
        "info_gap_high",
        "real_low_rate_down",
        "rand_low_rate_down",
        "info_gap_low",
        "real_high_delta_abs_mean",
        "rand_high_delta_abs_mean",
        "info_gap_high_delta_abs_mean",
    ]
    lines = [
        "# v3.10 Proof A: Cross-Cancer Redesigned Audit (5-fold)\n",
        "**核心问题**:v3.10 redesigned audit 的 `info_gap ≈ 0` 是 **BLCA 特异**,还是 **跨 cancer 一致**?\n",
        "\n",
        "**实验**:与 `results/audit_blca_redesign/` 相同的脚本,跑 v3.10 训练的 HNSC, LUSC, SKCM 5 fold 审计。\n",
        "\n",
        "## 5-fold mean by cancer\n",
        "\n",
        "| metric | blca (prior) | hnsc | lusc | skcm |",
        "|---|---|---|---|---|",
    ]
    # Pre-load blca summary if available
    blca_summary_path = Path("results/audit_blca_redesign/blca/summary_5fold.json")
    blca_summary = {}
    if blca_summary_path.exists():
        blca_summary = json.load(open(blca_summary_path))
    for m in key_metrics:
        row = f"| {m} |"
        if blca_summary:
            row += f" {blca_summary.get(m, {}).get('mean', 'NA'):.4f} |" if isinstance(blca_summary.get(m), dict) else " – |"
        for cancer in cancers:
            v = aggregated.get(cancer, {}).get(m, {}).get("mean")
            row += f" {v:.4f} |" if v is not None else " – |"
        lines.append(row)

    lines.extend([
        "\n## 解读\n",
        "\n",
        "- 若 4 cancer `info_gap_high` 都 ≈ 0,**跨 cancer 一致** → v3.10 失败是架构问题。\n",
        "- 若只有 BLCA 低,其他 cancer 高 → BLCA 是 outlier,审计结论要更谨慎。\n",
    ])

    with open(output_dir / "REPORT.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {output_dir}/summary_5fold_by_cancer.json")
    print(f"saved {output_dir}/REPORT.md")


if __name__ == "__main__":
    main()
