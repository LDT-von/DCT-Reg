#!/usr/bin/env python3
"""
夜间大任务脚本 - 完整论文材料生成

任务：
1. 等待所有审计完成
2. 为所有癌种生成 KM 曲线
3. 收集所有癌种审计结果到汇总表
4. 生成所有癌种的 Figure 3（剂量响应曲线）
5. 最终汇总报告

用法:
    python3 scripts/run_night_tasks.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

REPO_ROOT = Path("/data1/DCT-Reg")
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = REPO_ROOT / "paper_outputs"


# ============================================================================
# 工具函数
# ============================================================================

def wait_for_audits(completion_file: Path, timeout_min: int = 30) -> bool:
    """等待所有审计任务完成。"""
    print(f"\n⏳ 等待审计任务完成 (timeout={timeout_min}min)...")

    start = time.time()
    timeout_sec = timeout_min * 60

    # 监控的审计目录
    audit_dirs = [
        RESULTS_DIR / "audit_hnsc",
        RESULTS_DIR / "audit_skcm",
        RESULTS_DIR / "audit_lusc",
        RESULTS_DIR / "audit_kirc",
    ]

    while time.time() - start < timeout_sec:
        all_done = True
        for d in audit_dirs:
            if not d.exists():
                all_done = False
                continue
            # 检查所有 fold 是否有 audit_metrics
            cancer = d.name.replace("audit_", "")
            folds_expected = {"hnsc": 5, "skcm": 5, "lusc": 2, "kirc": 2}[cancer]
            for fold in range(folds_expected):
                metrics_file = d / cancer / f"fold_{fold}" / "audit_metrics_fold{fold}.json"
                if not metrics_file.exists():
                    all_done = False
                    break

        if all_done:
            print("✅ 所有审计任务完成")
            return True

        time.sleep(15)
        elapsed = (time.time() - start) / 60
        print(f"  ⏱️  {elapsed:.1f}min 已过...")

    print(f"⚠ 超时 ({timeout_min}min)，继续执行")
    return False


def collect_cancer_audit(cancer: str, output_dir: Path) -> dict:
    """收集某癌种的所有审计结果。"""
    audit_dir = RESULTS_DIR / f"audit_{cancer}" / cancer
    summary = {}

    if not audit_dir.exists():
        return summary

    for fold_dir in sorted(audit_dir.glob("fold_*")):
        fold = int(fold_dir.name.replace("fold_", ""))
        metrics_file = fold_dir / f"audit_metrics_fold{fold}.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            summary[fold] = {
                "dcr": metrics.get("direction_consistency", {}).get("correct_rate"),
                "chance_gap": metrics.get("direction_consistency", {}).get("chance_gap"),
                "n_cases": metrics.get("n_cases"),
                "reconfig_tv": metrics.get("reconfiguration", {}).get("mean_tv"),
                "factual_distance_low": metrics.get("mean_factual_distance_to_low_anchor"),
                "factual_distance_high": metrics.get("mean_factual_distance_to_high_anchor"),
            }

    return summary


def collect_all_cancer_audits() -> dict:
    """收集所有癌种的审计结果。"""
    print("\n📊 收集所有癌种审计结果")

    all_results = {}
    cancers = ["hnsc", "skcm", "lusc", "kirc"]

    for cancer in cancers:
        summary = collect_cancer_audit(cancer, OUTPUT_DIR)
        if summary:
            # 计算平均
            dcr_vals = [s["dcr"] for s in summary.values() if s["dcr"] is not None]
            mean_dcr = np.mean(dcr_vals) if dcr_vals else None
            std_dcr = np.std(dcr_vals) if dcr_vals else None
            all_results[cancer] = {
                "per_fold": summary,
                "mean_dcr": float(mean_dcr) if mean_dcr else None,
                "std_dcr": float(std_dcr) if std_dcr else None,
                "n_folds": len(summary),
            }
            print(f"  ✓ {cancer.upper()}: {len(summary)} folds, "
                  f"mean DCR = {mean_dcr:.3f} ± {std_dcr:.3f}" if mean_dcr else f"  ✓ {cancer.upper()}: {len(summary)} folds")

    # BLCA 已有的数据
    blca_path = RESULTS_DIR / "audit_best_epochs_blca" / "audit_summary.json"
    if blca_path.exists():
        with open(blca_path) as f:
            blca_summary = json.load(f)
        all_results["blca"] = {
            "per_fold_summary": blca_summary["folds"],
            "mean_dcr": blca_summary["dcr_mean"],
            "std_dcr": blca_summary["dcr_std"],
            "n_folds": blca_summary["num_folds"],
        }
        print(f"  ✓ BLCA: 5 folds, mean DCR = {blca_summary['dcr_mean']:.3f} ± {blca_summary['dcr_std']:.3f}")

    return all_results


def generate_cross_cancer_table4(all_audits: dict, output_dir: Path) -> str:
    """生成 Table 4 跨癌种扩展版。"""
    print("\n📋 生成 Table 4 (跨癌种扩展)")

    lines = ["# Table 4: Cross-Cancer Mechanism Audit\n"]
    lines.append("| Cancer | N Folds | Mean DCR | ± std | Status |")
    lines.append("|--------|---------|----------|-------|--------|")

    cancer_labels = {
        "blca": "BLCA (Bladder)",
        "hnsc": "HNSC (Head & Neck)",
        "kirc": "KIRC (Kidney)",
        "lusc": "LUSC (Lung Squamous)",
        "skcm": "SKCM (Skin Melanoma)",
    }

    for cancer, data in all_audits.items():
        if "per_fold" in data and data["mean_dcr"] is not None:
            label = cancer_labels.get(cancer, cancer.upper())
            status = "✅ >0.5" if data["mean_dcr"] > 0.5 else "❌ ≤0.5"
            lines.append(f"| {label} | {data['n_folds']} | {data['mean_dcr']:.3f} | {data['std_dcr']:.3f} | {status} |")
        else:
            label = cancer_labels.get(cancer, cancer.upper())
            lines.append(f"| {label} | - | - | - | N/A |")

    lines.append("")
    lines.append("**注**: 0.5 = 随机基线；>0.5 表示机制响应超过随机")
    lines.append("")

    out_path = output_dir / "table4_cross_cancer_audit.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {out_path}")
    return str(out_path)


# ============================================================================
# KM 曲线 (所有癌种)
# ============================================================================

def generate_km_for_all_cancers(output_dir: Path, predictions_base: Path) -> bool:
    """为所有癌种生成 KM 曲线。"""
    print("\n📊 生成 KM 曲线 (所有癌种)")

    # 检查每个癌种的 predictions.csv 是否存在
    cancers_with_predictions = []
    cancers = ["blca", "hnsc", "kirc", "lusc", "skcm"]

    # 寻找 predictions.csv
    base = REPO_ROOT / "results" / "dct_v3.10_experiments" / "robust" / "full"
    for cancer in cancers:
        pred_files = list(base.glob(f"{cancer}/{cancer}/SurvOTRank*/evidence/fold_*/predictions.csv"))
        if pred_files:
            cancers_with_predictions.append((cancer, len(pred_files)))
            print(f"  ✓ {cancer.upper()}: {len(pred_files)} fold predictions")

    if not cancers_with_predictions:
        print("  ⚠ 没有 predictions 数据")
        return False

    # 用脚本生成 KM 图
    km_script = REPO_ROOT / "scripts" / "kaplan_meier_analysis.py"
    all_reports = []

    for cancer, n_preds in cancers_with_predictions:
        output_path = output_dir / f"km_curves_{cancer}"
        # 检查现有 KM 报告
        existing_km = REPO_ROOT / "results" / "kaplan_meier_analysis"
        if existing_km.exists():
            existing_files = list(existing_km.glob(f"*{cancer}*"))
            if not existing_files:
                print(f"  ⚠ {cancer.upper()} 没有 KM 数据，跳过")
                continue

        # 修改 KM 脚本以支持 cancer 参数 - 直接调用
        cmd = [
            sys.executable, str(km_script),
            "--output_dir", str(output_path),
            "--cancer", cancer,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT))
            if result.returncode == 0:
                all_reports.append(f"  ✓ {cancer.upper()}: KM 曲线生成成功")
            else:
                all_reports.append(f"  ⚠ {cancer.upper()}: KM 失败 - {result.stderr[:200]}")
        except Exception as e:
            all_reports.append(f"  ⚠ {cancer.upper()}: KM 异常 - {e}")

    print("\n".join(all_reports))
    return True


def generate_dose_response_for_cancer(cancer: str, output_dir: Path) -> bool:
    """为指定癌种生成 Figure 3 (剂量响应曲线)。"""
    print(f"\n📈 Figure 3: {cancer.upper()} 剂量响应曲线")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pickle

    # 寻找 dose_sweep 数据
    if cancer == "blca":
        base = RESULTS_DIR / "dct_v3.10_experiments" / "robust" / "full" / "blca"
    else:
        # 其他癌种
        base = RESULTS_DIR / "dct_v3.10_experiments" / "robust" / "full" / cancer

    if not base.exists():
        print(f"  ⚠ 没有 {cancer} 实验数据")
        return False

    # 找 dose_sweep.pkl
    sweep_files = list(base.glob(f"{cancer}/SurvOTRank*/evidence/fold_*/proof_transport_dependency/dose_both_directions/dose_sweep.pkl"))

    if not sweep_files:
        print(f"  ⚠ {cancer.upper()} 没有 dose_sweep 数据 (审计未完成)")
        return False

    # 加载所有 folds
    all_low, all_high, alphas = [], [], None
    for path in sorted(sweep_files):
        with open(path, "rb") as f:
            data = pickle.load(f)
        all_low.append(np.array(data["low_risks"]))
        all_high.append(np.array(data["high_risks"]))
        alphas = np.array(data["alphas"])

    if not all_low:
        return False

    low_mean = np.mean([lr.mean(axis=0) for lr in all_low], axis=0)
    low_std = np.std([lr.mean(axis=0) for lr in all_low], axis=0) / np.sqrt(len(all_low))
    high_mean = np.mean([hr.mean(axis=0) for hr in all_high], axis=0)
    high_std = np.std([hr.mean(axis=0) for hr in all_high], axis=0) / np.sqrt(len(all_high))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(alphas, low_mean, "b-o", linewidth=2, markersize=6, label="Low-risk anchor")
    ax.fill_between(alphas, low_mean - low_std, low_mean + low_std, alpha=0.2, color="blue")
    ax.plot(alphas, high_mean, "r-s", linewidth=2, markersize=6, label="High-risk anchor")
    ax.fill_between(alphas, high_mean - high_std, high_mean + high_std, alpha=0.2, color="red")
    ax.set_xlabel("Intervention strength α", fontsize=12)
    ax.set_ylabel("Predicted risk", fontsize=12)
    ax.set_title(f"(a) Dose-Response: Absolute Risk ({cancer.upper()})", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(alphas, low_mean - low_mean[0], "b-o", linewidth=2, markersize=6, label="Low-risk anchor")
    ax.fill_between(alphas, low_mean - low_mean[0] - low_std, low_mean - low_mean[0] + low_std, alpha=0.2, color="blue")
    ax.plot(alphas, high_mean - high_mean[0], "r-s", linewidth=2, markersize=6, label="High-risk anchor")
    ax.fill_between(alphas, high_mean - high_mean[0] - high_std, high_mean - high_mean[0] + high_std, alpha=0.2, color="red")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Intervention strength α", fontsize=12)
    ax.set_ylabel("Δ Risk from baseline", fontsize=12)
    ax.set_title(f"(b) Dose-Response: Risk Change ({cancer.upper()})", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Figure 3: Counterfactual Dose-Response ({cancer.upper()}, {len(sweep_files)} folds)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = output_dir / f"figure3_dose_response_{cancer}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")
    return True


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 70)
    print(f"🌙 夜间大任务启动 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: 等待审计完成
    wait_for_audits(Path("/tmp/audits_done"), timeout_min=60)

    # Step 2: 收集所有癌种审计
    all_audits = collect_all_cancer_audits()

    # 保存
    with open(OUTPUT_DIR / "all_cancer_audits.json", "w") as f:
        json.dump(all_audits, f, indent=2, default=str)

    # Step 3: 生成 Table 4
    generate_cross_cancer_table4(all_audits, OUTPUT_DIR)

    # Step 4: 生成 KM 曲线
    generate_km_for_all_cancers(OUTPUT_DIR, REPO_ROOT / "results")

    # Step 5: 生成所有癌种 Figure 3
    print("\n📈 生成所有 Figure 3 (剂量响应曲线)")
    cancers = ["blca", "hnsc", "kirc", "lusc", "skcm"]
    for cancer in cancers:
        generate_dose_response_for_cancer(cancer, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("✅ 夜间任务完成!")
    print(f"📁 输出: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
