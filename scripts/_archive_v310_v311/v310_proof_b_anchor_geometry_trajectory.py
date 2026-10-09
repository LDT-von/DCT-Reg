#!/usr/bin/env python3
"""Exp B: anchor geometry trajectory from v3.10 training logs.

Each v3.10 training log (log_start_<k>_end_<k+1>.txt) records per-epoch
metrics. We parse them and look at how v38_high_risk_gain / v38_low_risk_gain
evolve over 50 epochs — i.e. is there any sign of learning?

Combined with Proof 1's csv analysis (which confirmed gain never moves),
this script shows the SAME trajectory from the LOG files (independent source).

Also reads: anchor_coverage, plan_shift (TV distance) over training.

Output: results/v310_proof_b_anchor_geometry_trajectory/{REPORT.md, json, png}
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR_TEMPLATE = (
    "results/dct_v3.10/robust/final_50ep_old/{cancer}/{cancer}/"
    "SurvOTRank_dct_v310_directional_regularized_transport/"
    "0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_{cancer}_50ep"
)

EPOCH_RE = re.compile(
    r"\[Epoch\s+(\d+)\].*?"
    r"v38_direction=(-?[\d.eE+-]+)\s+"
    r"v38_dose=(-?[\d.eE+-]+)\s+"
    r"v38_reconfiguration=(-?[\d.eE+-]+)\s+"
    r"v38_total=(-?[\d.eE+-]+)\s+"
    r"v38_loss_scale=(-?[\d.eE+-]+)\s+"
    r"v38_active_stage_fraction=(-?[\d.eE+-]+)\s+"
    r"v38_high_risk_gain=(-?[\d.eE+-]+)\s+"
    r"v38_low_risk_gain=(-?[\d.eE+-]+)\s+"
    r"v38_high_plan_shift=(-?[\d.eE+-]+)\s+"
    r"v38_low_plan_shift=(-?[\d.eE+-]+)\s+"
    r"v38_finite=(-?[\d.eE+-]+)"
)

CANCERS = ["blca", "hnsc", "lusc", "skcm"]
OUTPUT_DIR = Path("results/v310_proof_b_anchor_geometry_trajectory")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_log(log_path):
    if not log_path.exists():
        return []
    txt = log_path.read_text()
    epochs = []
    for m in EPOCH_RE.finditer(txt):
        # Groups: 1=epoch, 2=direction, 3=dose, 4=reconfig, 5=total,
        #         6=loss_scale, 7=active_stage_fraction,
        #         8=high_risk_gain, 9=low_risk_gain,
        #         10=high_plan_shift, 11=low_plan_shift, 12=finite
        epochs.append({
            "epoch": int(m.group(1)),
            "direction": float(m.group(2)),
            "active_stage_fraction": float(m.group(7)),
            "high_gain": float(m.group(8)),
            "low_gain": float(m.group(9)),
            "high_shift": float(m.group(10)),
            "low_shift": float(m.group(11)),
            "finite": float(m.group(12)),
        })
    return epochs


def main():
    all_data = {}
    for cancer in CANCERS:
        log_dir = Path(LOG_DIR_TEMPLATE.format(cancer=cancer))
        all_data[cancer] = {}
        # Each log_start_<k>_end_<k+1>.txt corresponds to fold k
        for log_path in sorted(log_dir.glob("log_start_*.txt")):
            # parse filename: log_start_<k>_end_<k+1>.txt
            name = log_path.stem  # log_start_0_end_1
            parts = name.split("_")
            try:
                fold = int(parts[2])
            except (IndexError, ValueError):
                continue
            epochs = parse_log(log_path)
            all_data[cancer].setdefault(fold, []).extend(epochs)

    # Save raw data
    with open(OUTPUT_DIR / "raw_parsed.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"saved {OUTPUT_DIR / 'raw_parsed.json'}")

    # Aggregate per-cancer across folds
    summary = {}
    for cancer in CANCERS:
        all_eps = []
        for fold in range(5):
            all_eps.extend(all_data[cancer].get(fold, []))
        if not all_eps:
            continue
        # group by epoch
        by_epoch = {}
        for e in all_eps:
            ep = e["epoch"]
            by_epoch.setdefault(ep, []).append(e)
        agg = {}
        for ep, lst in sorted(by_epoch.items()):
            agg[ep] = {
                "high_gain_mean": float(np.mean([x["high_gain"] for x in lst])),
                "high_gain_std": float(np.std([x["high_gain"] for x in lst])),
                "low_gain_mean": float(np.mean([x["low_gain"] for x in lst])),
                "low_gain_std": float(np.std([x["low_gain"] for x in lst])),
                "direction_mean": float(np.mean([x["direction"] for x in lst])),
                "high_shift_mean": float(np.mean([x["high_shift"] for x in lst])),
                "low_shift_mean": float(np.mean([x["low_shift"] for x in lst])),
                "active_stage_fraction_mean": float(np.mean([x["active_stage_fraction"] for x in lst])),
            }
        summary[cancer] = agg
    with open(OUTPUT_DIR / "summary_by_cancer.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot 4 cancer × 4 metrics: direction, high_gain, low_gain, active_stage_fraction
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True)
    for col, cancer in enumerate(CANCERS):
        agg = summary.get(cancer, {})
        if not agg:
            continue
        epochs = sorted(agg.keys())
        for row, (metric, label) in enumerate([
            ("direction_mean", "direction loss"),
            ("high_gain_mean", "high_risk_gain"),
            ("low_gain_mean", "low_risk_gain"),
            ("active_stage_fraction_mean", "active_stage_fraction"),
        ]):
            y = [agg[e][metric] for e in epochs]
            axes[row, col].plot(epochs, y, "b-", lw=0.8)
            axes[row, col].set_title(f"{cancer}: {label}", fontsize=10)
            axes[row, col].grid(True, alpha=0.3)
            if metric == "direction_mean":
                axes[row, col].set_yscale("log")
            if row == 3:
                axes[row, col].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_4x4.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTPUT_DIR / 'trajectory_4x4.png'}")

    # Markdown
    lines = [
        "# v3.10 Proof B: Anchor Geometry / Gain Trajectory across 4 cancers (BLCA, HNSC, LUSC, SKCM)\n",
        "**核心问题**:v3.10 训练过程中,`high_risk_gain` / `low_risk_gain` / `direction` 怎么演化?\n",
        "**说明**:直接从训练 log (`log_start_*.txt`) 解析每 epoch metric。这与 Proof 1 用 `epoch_curve_*.csv` 是独立来源。\n",
        "\n",
        "## 关键观察\n",
        "\n",
        "### direction loss 5-fold mean(epoch 0 vs epoch 49)\n",
        "\n",
        "| cancer | epoch 0 | epoch 49 | Δ |",
        "|---|---|---|---|",
    ]
    for cancer in CANCERS:
        agg = summary.get(cancer, {})
        if not agg:
            continue
        first = agg.get(0, {}).get("direction_mean")
        last = agg.get(max(agg.keys()), {}).get("direction_mean")
        if first is not None and last is not None:
            lines.append(
                f"| {cancer} | {first:.5f} | {last:.5f} | {last - first:+.5f} |"
            )

    lines.extend([
        "\n### high_risk_gain 5-fold mean(epoch 0 vs epoch 49)\n",
        "\n",
        "| cancer | epoch 0 | epoch 49 | Δ |",
        "|---|---|---|---|",
    ])
    for cancer in CANCERS:
        agg = summary.get(cancer, {})
        if not agg:
            continue
        first = agg.get(0, {}).get("high_gain_mean")
        last = agg.get(max(agg.keys()), {}).get("high_gain_mean")
        if first is not None and last is not None:
            lines.append(
                f"| {cancer} | {first:+.5f} | {last:+.5f} | {last - first:+.5f} |"
            )

    lines.extend([
        "\n### low_risk_gain 5-fold mean(epoch 0 vs epoch 49)\n",
        "\n",
        "| cancer | epoch 0 | epoch 49 | Δ |",
        "|---|---|---|---|",
    ])
    for cancer in CANCERS:
        agg = summary.get(cancer, {})
        if not agg:
            continue
        first = agg.get(0, {}).get("low_gain_mean")
        last = agg.get(max(agg.keys()), {}).get("low_gain_mean")
        if first is not None and last is not None:
            lines.append(
                f"| {cancer} | {first:+.5f} | {last:+.5f} | {last - first:+.5f} |"
            )

    # Monotone slope test: fit linear regression per cancer per metric
    lines.extend([
        "\n### 训练期 trend(线性拟合斜率 per cancer)\n",
        "\n",
        "| cancer | direction | high_gain | low_gain | high_shift |",
        "|---|---|---|---|---|",
    ])
    for cancer in CANCERS:
        agg = summary.get(cancer, {})
        if not agg:
            continue
        eps = np.array(sorted(agg.keys()), dtype=np.float64)
        if len(eps) < 5:
            continue
        slopes = {}
        for metric in [
            "direction_mean", "high_gain_mean", "low_gain_mean", "high_shift_mean",
        ]:
            y = np.array([agg[int(e)][metric] for e in eps])
            slope, _ = np.polyfit(eps, y, 1)
            slopes[metric] = slope
        lines.append(
            f"| {cancer} | {slopes['direction_mean']:+.6f} | "
            f"{slopes['high_gain_mean']:+.6f} | {slopes['low_gain_mean']:+.6f} | "
            f"{slopes['high_shift_mean']:+.6f} |"
        )

    lines.extend([
        "\n## 解读\n",
        "\n",
        "1. **direction loss 下降斜率**:若 4 cancer 都是负(数值下降)→ 模型在拟合 NLL+IPCW 时 direction 项**自然下降**(不再是 lambda=0.05 的 constant)。\n",
        "2. **gain 训练期 slope**:若 4 cancer 都是 0 → direction loss 50 epoch 内**没有推动 gain**。\n",
        "3. **plan_shift**:是 transport plan 的 total variation distance,反映 factual→anchor 的耦合变化。\n",
        "\n",
        "## 输出\n",
        "\n",
        "- `trajectory_4x4.png` — 4 cancer × 4 metric 网格图\n",
        "- `raw_parsed.json` — 解析的所有 epoch 原始数据\n",
        "- `summary_by_cancer.json` — per-cancer 5-fold mean per epoch\n",
    ])
    with open(OUTPUT_DIR / "REPORT.md", "w") as f:
        f.write("\n".join(lines))
    print(f"saved {OUTPUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
