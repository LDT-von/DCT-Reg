"""E053 figures and result tables: v3.13-aligned BLCA three-arm reproduction."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data1/DCT-Reg")
ARCH = ROOT / "实验档案/表格图片/E053_v313同口径_BLCA复刻"
OUT = ARCH / "out"
V313_CURVE = (ROOT / "results/dct_v313_blca_uni2h/blca/blca/"
              "SurvOTRank_dct_v313_transport_reconstruction/"
              "0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v313_blca_uni2h/"
              "epoch_curve_fold0.csv")

ARM_COLOR = {"native": "#254B61", "learned": "#C1553B", "independent": "#4E8B6B"}
ARM_LABEL = {"native": "native", "learned": "learned", "independent": "independent"}


def load_curves():
    curves = {}
    for arm in ARM_COLOR:
        curves[arm] = json.load(open(OUT / arm / "curve.json"))
    v313 = [float(r["val_cindex"]) for r in csv.DictReader(open(V313_CURVE))]
    return curves, v313


def fig_epoch_curves(curves, v313):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for arm, curve in curves.items():
        y = [e["val_cindex_mean"] for e in curve]
        ax.plot(range(1, len(y) + 1), y, marker="o", ms=3.5, lw=1.8,
                color=ARM_COLOR[arm], label=f"E053 {ARM_LABEL[arm]}")
    ax.plot(range(1, len(v313) + 1), v313, marker="s", ms=3.5, lw=1.8, ls="--",
            color="#666666", label="v3.13 reference (original run)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val c-index")
    ax.set_title("E053 three-arm validation c-index vs v3.13 (fold 0, seed 3)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    ax.set_ylim(0.45, 0.70)

    ax = axes[1]
    v = np.array(v313)
    for arm, curve in curves.items():
        e = np.array([x["val_cindex_mean"] for x in curve])
        ax.plot(v, e, "o", ms=4, alpha=.75, color=ARM_COLOR[arm],
                label=f"{ARM_LABEL[arm]}  r={np.corrcoef(v, e)[0,1]:.3f}")
    lim = [min(v.min(), 0.45), max(v.max(), 0.68)]
    ax.plot(lim, lim, color="#333333", lw=1, ls="--", label="identity")
    ax.set_xlabel("v3.13 val c-index")
    ax.set_ylabel("E053 val c-index")
    ax.set_title("Per-epoch agreement with v3.13")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ARCH / "E053_fig1_epoch_curves_vs_v313.png", dpi=150)
    plt.close(fig)


def fig_arm_summary(curves):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    summaries = json.load(open(OUT / "paired_summary.json"))["summaries"]
    arms = list(ARM_COLOR)
    means = [summaries[a]["cindex_mean"] for a in arms]
    stds = [summaries[a]["cindex_repeat_std"] for a in arms]
    ax.bar([ARM_LABEL[a] for a in arms], means, yerr=stds, capsize=5,
           color=[ARM_COLOR[a] for a in arms], alpha=.9)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + .006, f"{m:.4f}\n±{s:.4f}", ha="center", fontsize=9)
    ax.axhline(0.6576, color="#666666", ls="--", lw=1.2)
    ax.text(2.42, 0.6576, "v3.13 peak\n0.6576", fontsize=8, va="center", color="#444444")
    ax.set_ylim(0.5, 0.72)
    ax.set_ylabel("val c-index (mean of 5 repeats)")
    ax.set_title("E053 arm comparison (legacy_val selection, fold 0, seed 3)")
    ax.grid(alpha=.3, axis="y")

    ax = axes[1]
    for arm, curve in curves.items():
        y = [e["val_cindex_mean"] for e in curve]
        ax.plot(range(1, len(y) + 1), [e["train_total"] for e in curve],
                lw=1.6, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.set_xlabel("epoch")
    ax.set_ylabel("train total loss")
    ax.set_title("Training loss by arm")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ARCH / "E053_fig2_arm_summary.png", dpi=150)
    plt.close(fig)


def fig_transport_effect():
    """Learned arm: transport on vs bypassed, same weights."""
    ctl = json.load(open(OUT / "learned" / "same_weights_controls.json"))
    rows = list(csv.DictReader(open(OUT / "learned" / "predictions.csv")))
    def risks(path):
        rs = list(csv.DictReader(open(path)))
        key = {(r["patient_id"], r["repeat"]): float(r["risk"]) for r in rs}
        return key
    base = risks(OUT / "learned" / "predictions.csv")
    byp = risks(OUT / "learned" / "same_weights_bypass_predictions.csv")
    common = sorted(set(base) & set(byp))
    delta = np.array([base[k] - byp[k] for k in common])
    spread = np.std(list(base.values()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    labels = ["transport ON\n(learned)", "transport ON\n(independent)", "transport BYPASS"]
    vals = [ctl["independent"]["cindex_mean"], ctl["independent"]["cindex_mean"],
            ctl["bypass"]["cindex_mean"]]
    vals[1] = ctl["independent"]["cindex_mean"]
    ax.bar(["same_weights\nindependent", "same_weights\nbypass"], 
           [ctl["independent"]["cindex_mean"], ctl["bypass"]["cindex_mean"]],
           color=["#8FB8A8", "#B0B0B0"], alpha=.9)
    for i, (k, v) in enumerate(ctl.items()):
        ax.text(i, v["cindex_mean"] + .004, f"{v['cindex_mean']:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.55, 0.66)
    ax.set_ylabel("val c-index")
    ax.set_title("Learned weights: transport ON vs BYPASS\n(identical weights, plan mode differs)")
    ax.grid(alpha=.3, axis="y")

    ax = axes[1]
    ax.hist(delta, bins=40, color="#C1553B", alpha=.85)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("risk(transport ON) - risk(bypass)")
    ax.set_ylabel("patients x repeats")
    ax.set_title(f"Per-patient risk change from transport\nmax|Δ|={np.abs(delta).max():.2e}, "
                 f"risk spread={spread:.2f} ({np.abs(delta).max()/spread*100:.2f}% of spread)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(ARCH / "E053_fig3_transport_inference_effect.png", dpi=150)
    plt.close(fig)


def write_tables(curves, v313):
    summaries = json.load(open(OUT / "paired_summary.json"))["summaries"]
    lines = []
    lines.append("# E053 v3.13 同口径 BLCA 复刻 — 结果表\n")
    lines.append("生成时间: 2026-10-21 | 癌种: BLCA | fold 0 | seed 3 | 划分: splits/5fold_uni2h\n")
    lines.append("\n## 表 1 三臂结果 (legacy_val 选点, val = 76 患者)\n")
    lines.append("| 臂 | val c-index (5 repeat 均值) | repeat std | NLL | 选中 epoch | 参数增量 | 训练秒 |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in ("native", "learned", "independent"):
        s = summaries[a]
        lines.append(f"| {a} | {s['cindex_mean']:.4f} | {s['cindex_repeat_std']:.4f} | "
                     f"{s['nll_mean']:.4f} | {s.get('selected_epoch','-')} | "
                     f"{s['extra_parameters']:+d} | {s['train_seconds']:.0f} |")
    lines.append(f"\n- learned − native = {summaries and json.load(open(OUT/'paired_summary.json'))['learned_minus_native']:+.4f}")
    lines.append(f"- learned − independent = {json.load(open(OUT/'paired_summary.json'))['learned_minus_independent']:+.4f}")
    lines.append("\n## 表 2 与 v3.13 原产物对照 (native 臂, 逐 epoch val c-index)\n")
    lines.append("epoch 统一按 1 起算显示；v3.13 原始 CSV 的 epoch 列是 0 起算，E053 curve 的 epoch 列是 1 起算，"
                 "此处已按训练轮序一一对应（第 k 行对应第 k 轮训练后评估）。\n")
    lines.append("| 轮次 | v3.13 | E053 native | 差 |")
    lines.append("|---|---|---|---|")
    e = [x["val_cindex_mean"] for x in curves["native"]]
    for i in range(30):
        lines.append(f"| {i+1} | {v313[i]:.4f} | {e[i]:.4f} | {e[i]-v313[i]:+.4f} |")
    lines.append(f"\n- v3.13 峰值 {max(v313):.4f} @第 {int(np.argmax(v313))+1} 轮；"
                 f"E053 峰值 {max(e):.4f} @第 {int(np.argmax(e))+1} 轮")
    lines.append(f"- v3.13 末轮 {v313[-1]:.4f}；E053 末轮 {e[-1]:.4f}")
    lines.append(f"- 逐 epoch 平均绝对差 {np.mean([abs(a-b) for a,b in zip(v313,e)]):.4f}，"
                 f"pearson r = {np.corrcoef(v313,e)[0,1]:.3f}")
    lines.append("\n## 表 3 口径对齐核验\n")
    lines.append("| 项 | v3.13 | E053 | 一致 |")
    lines.append("|---|---|---|---|")
    lines.append("| features_root | /data1/TCGA-UNI2-h-features/blca/uni2-h/pt_files | 同 | 是 |")
    lines.append("| 特征格式 | .h5 (UNI2-h) | .h5 (UNI2-h) | 是 |")
    lines.append("| split_csv | splits/5fold_uni2h/blca/fold_0.csv | 同 | 是 |")
    lines.append("| train / val 患者数 | 304 / 76 | 304 / 76 | 是 |")
    lines.append("| 排除患者 | 0 | 0 | 是 |")
    lines.append("| num_patches | 2048 | 2048 | 是 |")
    lines.append("| encoding_dim | 1536 | 1536 | 是 |")
    lines.append("| label_col | survival_months_dss | 同 | 是 |")
    lines.append("| 选点规则 | max val cindex over epochs | 同 | 是 |")
    lines.append("| 评估重复次数 | 1 | 5 | 否 (E053 多重复取均值) |")
    lines.append("| 短 bag 处理 | zero-pad 到 2048 | zero-pad 到 2048 | 是 |")
    (ARCH / "E053_结果表.md").write_text("\n".join(lines), encoding="utf-8")
    print("tables written")


if __name__ == "__main__":
    curves, v313 = load_curves()
    fig_epoch_curves(curves, v313)
    fig_arm_summary(curves)
    fig_transport_effect()
    write_tables(curves, v313)
    print("figures written")
