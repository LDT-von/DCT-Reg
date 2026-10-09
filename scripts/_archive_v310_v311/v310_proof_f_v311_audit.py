#!/usr/bin/env python3
"""Exp F: Audit v3.11 BLCA using redesigned audit (5-fold).

Runs the redesigned dose-monotonicity audit on v3.11 BLCA checkpoints.
This serves two purposes:
  1. **Sanity**: Verify v3.11 audit signal is strong (info_gap_high > 0),
     confirming that the redesigned audit is meaningful and the v3.10
     failure is not a bug in the audit design.
  2. **Cross-version comparison**: v3.10 BLCA info_gap_high ≈ -0.03
     (failure), v3.11 should be ≈ +0.30 or stronger.

Output: results/v310_proof_f_v311_audit/{fold_*.json, summary, REPORT.md}
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
from survot_rank.cli import add_project_paths

add_project_paths()

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from redesign_audit_blca import (
    _load_model_and_loader,
    _load_parsed_args,
    run_fold,
    compute_end_to_end,
)

import numpy as np


V311_CKPT_TEMPLATE = (
    "results/dct_v311_blca_uni_fixed/blca/blca/"
    "SurvOTRank_dct_v311_slot_interpretable/"
    "0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_fold{fold}/"
    "model_best_s{fold}.pth"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default="results/v310_proof_f_v311_audit")
    parser.add_argument("--report-only", action="store_true",
                        help="Reuse existing fold_*.json and only regenerate the report.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = [int(x) for x in args.folds.split(",")]
    summaries = {}

    for fold in folds:
        if args.report_only:
            fp = output_dir / f"fold_{fold}.json"
            if fp.exists():
                summaries[fold] = json.load(open(fp))
            else:
                print(f"[f{fold}] report-only: missing {fp}")
            continue
        ckpt_path = V311_CKPT_TEMPLATE.format(fold=fold)
        if not Path(ckpt_path).exists():
            print(f"[f{fold}] missing ckpt: {ckpt_path}")
            continue
        run_args = argparse.Namespace(
            config=str(REPO_ROOT / "configs/dct_v310_directional_regularized_transport.yaml"),
            checkpoint=str(ckpt_path),
            fold=fold,
            epoch=30,
            output_dir=str(output_dir),
            set=["study=blca"],
        )
        print(f"\n=== Auditing v3.11 BLCA fold {fold} ===")
        t0 = time.time()
        try:
            res = run_fold(fold, run_args, alpha_list=(0.0, 1.0))
            elapsed = round(time.time() - t0, 1)
            m = {}
            m["real_high"] = compute_end_to_end(res["real_high"], (0.0, 1.0))
            m["rand_high"] = compute_end_to_end(res["rand_high"], (0.0, 1.0))
            m["real_low"] = compute_end_to_end(res["real_low"], (0.0, 1.0))
            m["rand_low"] = compute_end_to_end(res["rand_low"], (0.0, 1.0))
            res["metrics"] = {
                "real_high_rate_up": m["real_high"]["rate_up"],
                "rand_high_rate_up": m["rand_high"]["rate_up"],
                "info_gap_high": m["real_high"]["rate_up"] - m["rand_high"]["rate_up"],
                "real_low_rate_down": m["real_low"]["rate_down"],
                "rand_low_rate_down": m["rand_low"]["rate_down"],
                "info_gap_low": m["real_low"]["rate_down"] - m["rand_low"]["rate_down"],
            }
            res["elapsed_sec"] = elapsed
            summaries[fold] = res
            print(f"  info_gap_high={res['metrics']['info_gap_high']:+.3f}, "
                  f"info_gap_low={res['metrics']['info_gap_low']:+.3f}, "
                  f"elapsed={elapsed}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            summaries[fold] = {"error": str(e)}

        with open(output_dir / f"fold_{fold}.json", "w") as f:
            json.dump(summaries[fold], f, indent=2)

    # Aggregate
    METRICS = [
        "real_high_rate_up", "rand_high_rate_up", "info_gap_high",
        "real_low_rate_down", "rand_low_rate_down", "info_gap_low",
    ]
    agg = {}
    for m in METRICS:
        vals = []
        for f, fold_data in summaries.items():
            if "metrics" in fold_data and m in fold_data["metrics"]:
                v = fold_data["metrics"][m]
                if isinstance(v, (int, float)) and not np.isnan(v):
                    vals.append(float(v))
        if vals:
            agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_fold": vals}

    with open(output_dir / "summary_5fold.json", "w") as f:
        json.dump(agg, f, indent=2)

    # Compare to BLCA v3.10 5-fold mean
    blca_v310 = {}
    blca_path = Path("results/audit_blca_redesign/blca/summary_5fold.json")
    if blca_path.exists():
        blca_v310 = json.load(open(blca_path))

    # Markdown
    lines = [
        "# v3.11 Proof F: Redesigned Audit on v3.11 BLCA (5-fold)\n",
        "**核心问题**:v3.11 redesigned audit 的 `info_gap_high` 是 **正值**(模型响应 risk anchors)还是 **负值**(模型不响应)?\n",
        "**重要性**:这是 v3.10 BLCA audit 的对照,验证 redesigned audit 本身没问题 (即 v3.10 失败不是 audit bug)。\n",
        "\n",
        "## 5-fold mean\n",
        "\n",
        "| metric | v3.11 BLCA (this) | v3.10 BLCA (prior) |",
        "|---|---|---|",
    ]
    V310_KEY_MAP = {
        "info_gap_high": "info_gap_high_mean",
        "info_gap_low": "info_gap_low_mean",
        "real_high_rate_up": "real_high_rate_up_mean",
        "rand_high_rate_up": "rand_high_rate_up_mean",
        "real_low_rate_down": "real_low_rate_down_mean",
        "rand_low_rate_down": "rand_low_rate_down_mean",
    }
    for m in METRICS:
        row = f"| {m} |"
        v311 = agg.get(m)
        if v311:
            row += f" {v311['mean']:+.4f}±{v311['std']:.3f} |"
        else:
            row += " – |"
        v310 = blca_v310.get(V310_KEY_MAP.get(m, m))
        if v310 is not None:
            if isinstance(v310, dict):
                row += f" {v310['mean']:+.4f}±{v310['std']:.3f} |"
            else:
                row += f" {v310:+.4f} |"
        else:
            row += " – |"
        lines.append(row)

    # Per-fold detail
    lines.extend([
        "\n## Per-fold info_gap_high detail\n",
        "\n",
        "| fold | info_gap_high | real_high_rate_up | rand_high_rate_up |",
        "|---|---|---|---|",
    ])
    for f in folds:
        s = summaries.get(f, {})
        if "metrics" in s:
            lines.append(
                f"| {f} | {s['metrics']['info_gap_high']:+.4f} | "
                f"{s['metrics']['real_high_rate_up']:.3f} | "
                f"{s['metrics']['rand_high_rate_up']:.3f} |"
            )

    # Cross-cancer / cross-version summary
    hnsc_v310 = {}
    hnsc_path = Path("results/v310_proof_a_cross_cancer_audit/summary_5fold_by_cancer.json")
    if hnsc_path.exists():
        hnsc_v310 = json.load(open(hnsc_path))

    lines.extend([
        "\n## 跨 version 对比\n",
        "\n",
        "| setup | info_gap_high | info_gap_low |",
        "|---|---|---|",
    ])
    v310_gap_h = None
    v310_gap_l = None
    blca_igh = blca_v310.get("info_gap_high_mean")
    if isinstance(blca_igh, dict):
        v310_gap_h = blca_igh.get("mean")
    elif isinstance(blca_igh, (int, float)):
        v310_gap_h = float(blca_igh)
    blca_igl = blca_v310.get("info_gap_low_mean")
    if isinstance(blca_igl, dict):
        v310_gap_l = blca_igl.get("mean")
    elif isinstance(blca_igl, (int, float)):
        v310_gap_l = float(blca_igl)
    v311_gap_h = agg.get("info_gap_high", {}).get("mean")
    v311_gap_l = agg.get("info_gap_low", {}).get("mean")

    def _fmt(v):
        return f"{v:+.4f}" if isinstance(v, (int, float)) else "NA"

    lines.append(f"| v3.10 BLCA 5-fold | {_fmt(v310_gap_h)} | {_fmt(v310_gap_l)} |")
    lines.append(f"| v3.10 HNSC 5-fold (prior) | {_fmt(hnsc_v310.get('hnsc', {}).get('info_gap_high', {}).get('mean'))} | {_fmt(hnsc_v310.get('hnsc', {}).get('info_gap_low', {}).get('mean'))} |")
    lines.append(f"| v3.10 LUSC 5-fold (prior) | {_fmt(hnsc_v310.get('lusc', {}).get('info_gap_high', {}).get('mean'))} | {_fmt(hnsc_v310.get('lusc', {}).get('info_gap_low', {}).get('mean'))} |")
    lines.append(f"| v3.10 SKCM 5-fold (prior) | {_fmt(hnsc_v310.get('skcm', {}).get('info_gap_high', {}).get('mean'))} | {_fmt(hnsc_v310.get('skcm', {}).get('info_gap_low', {}).get('mean'))} |")
    lines.append(f"| **v3.11 BLCA (this)** | **{_fmt(v311_gap_h)}** | **{_fmt(v311_gap_l)}** |")

    lines.extend([
        "\n## 解读\n",
        "\n",
        "- v3.11 是 v3.10 的直接后续,主要差异是增加 per-slot hazard 暴露。",
        "- 若 v3.11 info_gap_high >> 0 (与 random 显著不同),**证明 redesigned audit 在 v3.11 上工作正常**。",
        "- 那么 v3.10 在 4 cancer × 5 fold 上 info_gap_high ≈ 0 就是 **架构/版本级** 现象,不是 audit 设计问题。",
        "\n",
        "## 输出\n",
        "\n",
        f"- `{output_dir}/fold_*.json` — 每 fold audit",
        f"- `{output_dir}/summary_5fold.json` — 5-fold mean",
        f"- `{output_dir}/REPORT.md` — 本报告",
    ])
    with open(output_dir / "REPORT.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
