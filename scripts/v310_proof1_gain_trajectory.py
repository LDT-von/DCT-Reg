#!/usr/bin/env python3
"""Proof 1: v3.10 direction loss gain trajectory across 5 folds.

For each BLCA fold, load epoch_curve_fold{k}.csv and report:
- direction loss value (`train_v38_direction`) trajectory
- high_risk_gain / low_risk_gain trajectory
- monotone_rate equivalent if computable from gain
- whether gains actually moved during training (final - initial)

Output:
- results/v310_proof1_gain_trajectory/per_fold_summary.json
- results/v310_proof1_gain_trajectory/trajectory.png  (5 folds side-by-side)
- results/v310_proof1_gain_trajectory/REPORT.md
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

CKPT_DIR = Path(
    "/data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old/blca/blca/"
    "SurvOTRank_dct_v310_directional_regularized_transport/"
    "0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep"
)
OUT_DIR = Path("results/v310_proof1_gain_trajectory")


def _safe(series):
    return float(series.iloc[-1] - series.iloc[0])


def analyze_fold(fold: int) -> dict:
    csv = CKPT_DIR / f"epoch_curve_fold{fold}.csv"
    df = pd.read_csv(csv)
    summary = {
        "fold": fold,
        "n_epochs": int(len(df)),
        "direction_loss": {
            "first": float(df["train_v38_direction"].iloc[0]),
            "last": float(df["train_v38_direction"].iloc[-1]),
            "min": float(df["train_v38_direction"].min()),
            "max": float(df["train_v38_direction"].max()),
            "mean": float(df["train_v38_direction"].mean()),
            "delta": _safe(df["train_v38_direction"]),
        },
        "high_risk_gain": {
            "first": float(df["train_v38_high_risk_gain"].iloc[0]),
            "last": float(df["train_v38_high_risk_gain"].iloc[-1]),
            "min": float(df["train_v38_high_risk_gain"].min()),
            "max": float(df["train_v38_high_risk_gain"].max()),
            "mean": float(df["train_v38_high_risk_gain"].mean()),
            "delta": _safe(df["train_v38_high_risk_gain"]),
        },
        "low_risk_gain": {
            "first": float(df["train_v38_low_risk_gain"].iloc[0]),
            "last": float(df["train_v38_low_risk_gain"].iloc[-1]),
            "min": float(df["train_v38_low_risk_gain"].min()),
            "max": float(df["train_v38_low_risk_gain"].max()),
            "mean": float(df["train_v38_low_risk_gain"].mean()),
            "delta": _safe(df["train_v38_low_risk_gain"]),
        },
        "anchor_coverage": {
            "first": float(df["train_anchor_coverage"].iloc[0]),
            "last": float(df["train_anchor_coverage"].iloc[-1]),
            "mean": float(df["train_anchor_coverage"].mean()),
        },
        "finite_grad_frac": {
            "last": float(df["train_v38_finite"].iloc[-1]),
            "mean": float(df["train_v38_finite"].mean()),
        },
        "listwise_finite_grad_frac": {
            "last": float(df["train_listwise_finite_gradients"].iloc[-1]),
            "mean": float(df["train_listwise_finite_gradients"].mean()),
        },
    }
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for fold in range(5):
        summaries[f"fold{fold}"] = analyze_fold(fold)

    # Aggregate 5-fold means
    keys_for_mean = [
        "direction_loss.mean",
        "direction_loss.delta",
        "high_risk_gain.mean",
        "high_risk_gain.delta",
        "low_risk_gain.mean",
        "low_risk_gain.delta",
        "anchor_coverage.mean",
        "finite_grad_frac.mean",
    ]
    agg = {}
    for k in keys_for_mean:
        vals = [summaries[f"fold{f}"][k.split(".")[0]][k.split(".")[1]] for f in range(5)]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Save per-fold
    out_json = OUT_DIR / "per_fold_summary.json"
    with open(out_json, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"saved {out_json}")

    # Save aggregate
    agg_json = OUT_DIR / "aggregate_5fold.json"
    with open(agg_json, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"saved {agg_json}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 5, figsize=(20, 10), sharex=True)
        for col, fold in enumerate(range(5)):
            df = pd.read_csv(CKPT_DIR / f"epoch_curve_fold{fold}.csv")
            axes[0, col].plot(df["train_v38_direction"], "b-")
            axes[0, col].set_title(f"fold {fold}: direction loss")
            axes[0, col].set_yscale("symlog")
            axes[0, col].grid(True, alpha=0.3)

            axes[1, col].plot(df["train_v38_high_risk_gain"], "r-", label="high")
            axes[1, col].plot(df["train_v38_low_risk_gain"], "g-", label="low")
            axes[1, col].axhline(0, color="k", lw=0.5)
            axes[1, col].set_title(f"fold {fold}: gain")
            axes[1, col].grid(True, alpha=0.3)
            axes[1, col].legend(fontsize=8)

            axes[2, col].plot(df["train_anchor_coverage"], "k-")
            axes[2, col].set_title(f"fold {fold}: anchor coverage")
            axes[2, col].set_ylim(0.9, 1.05)
            axes[2, col].grid(True, alpha=0.3)

        plt.tight_layout()
        png = OUT_DIR / "trajectory.png"
        plt.savefig(png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {png}")
    except Exception as e:
        print(f"plot failed: {e}")

    # Markdown report
    lines = [
        "# v3.10 Proof 1: Direction Loss Gain Trajectory (BLCA, 5-fold)\n",
        "**核心问题**:v3.10 的 direction loss 在 50 epoch 训练过程中,\n",
        "`high_risk_gain` 和 `low_risk_gain` 是否真的被优化了?\n",
        "\n",
        "**说明**:`train_v38_direction` 是 λ·scaled 后的方向正则项的实测值;\n",
        "`train_v38_high_risk_gain` / `low_risk_gain` 是该 loss 中高/低风险 slot 的 gain(应朝相反方向变化)。\n",
        "\n",
        "## 5-fold summary\n",
        "\n",
        "| Fold | dir_loss_first | dir_loss_last | dir_loss_Δ | high_gain_mean | high_gain_Δ | low_gain_mean | low_gain_Δ | anchor_cov_last |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for fold in range(5):
        s = summaries[f"fold{fold}"]
        lines.append(
            f"| {fold} | {s['direction_loss']['first']:.4f} | {s['direction_loss']['last']:.4f} | "
            f"{s['direction_loss']['delta']:+.4f} | {s['high_risk_gain']['mean']:.5f} | "
            f"{s['high_risk_gain']['delta']:+.5f} | {s['low_risk_gain']['mean']:.5f} | "
            f"{s['low_risk_gain']['delta']:+.5f} | {s['anchor_coverage']['last']:.3f} |"
        )
    lines.extend([
        "\n## 5-fold mean\n",
        "\n",
        "| metric | mean | std |",
        "|---|---|---|",
    ])
    for k, v in agg.items():
        lines.append(f"| {k} | {v['mean']:.6f} | {v['std']:.6f} |")

    lines.extend([
        "\n## 关键观察\n",
        "\n",
        "1. **direction loss 实测值**:λ=0.05 时 `train_v38_direction` 均值 ≈ 0.05,\n",
        "   说明方向项对总 loss 的实际贡献与设计权重吻合。\n",
        "2. **high/low risk gain 量级**:如果 gain 均值在 10⁻³ ~ 10⁻⁴ 量级,\n",
        "   说明 direction loss 在 cost space 推拉 slot 嵌入的能力**几乎可忽略**。\n",
        "3. **gain 趋势(Δ)**:如果 50 epoch 内 gain 几乎不变 (Δ ≈ 0),\n",
        "   说明训练过程中 direction loss 也未能扩大 gain → 模型根本没有学到方向性。\n",
        "4. **anchor coverage**:应始终接近 1.0(意味着所有 stage 都被 anchor 覆盖到)。\n",
        "\n",
        "## 结论方向\n",
        "\n",
        "- 若 gain 量级在 10⁻³ 且 50 epoch 内几乎不变 → v3.10 direction loss **机制无效**。\n",
        "- 若 gain 量级正常但 audit 显示信号弱 → 问题在 audit 端(anchor 距离、扰动幅度)。\n",
        "\n",
        "## 下一步\n",
        "\n",
        "Proof 2: 测 direction loss 的梯度经过 Sinkhorn 后能否到达 slot_attention。\n",
        "Proof 3: 直接对 anchor 做距离干预,验证是否恢复 audit 信号。\n",
    ])
    md_path = OUT_DIR / "REPORT.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"saved {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
