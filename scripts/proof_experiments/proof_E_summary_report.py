#!/usr/bin/env python3
"""Proof E — 综合证明报告 (v3.11 fixed v2: per-modality diversity).

汇总所有 proof experiments 的结果到一份人类可读的 markdown 报告。
- A: v3.11 vs v3.10 配方对比
- B: Slot variance 约束 (WSI/Omics 分开检查)
- C: IPCW rank loss 行为
- D: Slot 可视化（如果有 figure）
- 训练指标 (per-modality variance from logs)

Usage:
    python scripts/proof_experiments/proof_E_summary_report.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

REPO = Path("/data1/DCT-Reg")
LOG_DIR = REPO / "logs/v311_blca_uni_fixed_v2"
PROOF_DIR = REPO / "results"
OUT = REPO / "PROOF_RESULTS_v2.md"


def parse_log(path: Path) -> Dict:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    out = {
        "epochs": [],
        "val_cindex": [],
        "best_cindex": None,
        "best_epoch": None,
    }
    for line in text.split("\n"):
        if "[Epoch" not in line or "train_loss=" not in line:
            continue
        m_epoch = re.search(r"\[Epoch (\d+)\]", line)
        if not m_epoch:
            continue
        epoch = int(m_epoch.group(1))
        row = {"epoch": epoch}
        for f in [
            "train_loss", "train_cindex", "ot", "ipcw_rank",
            "v311_per_slot_nll", "v311_slot_diversity",
            "v311_slot_variance", "v311_slot_variance_wsi",
            "v311_slot_variance_omic",
            "v311_per_slot_nll_lambda", "v311_slot_diversity_lambda",
            "active_stage_fraction", "anchor_coverage",
            "evidence_marginal_entropy", "listwise_finite_gradients",
        ]:
            m = re.search(rf"{f}=([\d.\-e]+)", line)
            row[f] = float(m.group(1)) if m else None
        out["epochs"].append(row)

    val_lines = re.findall(r"\[Epoch (\d+)\] val cindex=([\d.]+)", text)
    if val_lines:
        for ep, c in val_lines:
            out["val_cindex"].append((int(ep), float(c)))
        best = max(out["val_cindex"], key=lambda x: x[1])
        out["best_cindex"] = best[1]
        out["best_epoch"] = best[0]

    return out


def assess_per_modality(per_fold: Dict[int, Dict]) -> Dict:
    """Analyze per-modality variance from logs."""
    VAR_MIN, VAR_MAX = 0.005, 0.05
    out = {}
    for fold, data in per_fold.items():
        rows = data["epochs"]
        if not rows:
            continue
        # Last 5 epochs
        last5 = rows[-5:]

        var_wsi = [r["v311_slot_variance_wsi"] for r in last5
                   if r.get("v311_slot_variance_wsi") is not None]
        var_omic = [r["v311_slot_variance_omic"] for r in last5
                    if r.get("v311_slot_variance_omic") is not None]

        out[fold] = {
            "var_wsi_last5": var_wsi,
            "var_omic_last5": var_omic,
            "var_wsi_mean": float(np.mean(var_wsi)) if var_wsi else None,
            "var_omic_mean": float(np.mean(var_omic)) if var_omic else None,
            "var_wsi_in_band": all(VAR_MIN <= v <= VAR_MAX for v in var_wsi) if var_wsi else None,
            "var_omic_in_band": all(VAR_MIN <= v <= VAR_MAX for v in var_omic) if var_omic else None,
            "best_cindex": data.get("best_cindex"),
            "best_epoch": data.get("best_epoch"),
        }
    return out


def main():
    print("=" * 70)
    print("PROOF E — v3.11 fixed v2 (per-modality diversity) Summary Report")
    print("=" * 70)

    # Parse all fold logs
    per_fold = {}
    for fold in range(5):
        data = parse_log(LOG_DIR / f"fold{fold}.log")
        if data.get("epochs"):
            per_fold[fold] = data
    print(f"\nParsed {len(per_fold)} fold logs from {LOG_DIR}")

    if not per_fold:
        print("ERROR: No fold logs found. Training may not have started yet.")
        return 1

    assessments = assess_per_modality(per_fold)

    # Load Proof A, B, C JSON
    proof_a = json.load(open(PROOF_DIR / "proof_experiment_A.json")) \
        if (PROOF_DIR / "proof_experiment_A.json").exists() else None
    proof_b = json.load(open(PROOF_DIR / "proof_experiment_B.json")) \
        if (PROOF_DIR / "proof_experiment_B.json").exists() else None
    proof_c = json.load(open(PROOF_DIR / "proof_experiment_C.json")) \
        if (PROOF_DIR / "proof_experiment_C.json").exists() else None

    # Build report
    md = ["# v3.11 Fixed v2 — Per-Modality Diversity — Proof Summary\n\n"]
    md.append("> v3.11 fixed v2 的核心改动:  `slot_diversity_loss` 改为 per-modality\n")
    md.append("> 分别约束 WSI 和 Omics 槽的 variance，避免合并时掩盖坍缩\n\n")
    md.append("数据来源: `results/dct_v311_blca_uni_fixed_v2` (5-fold, 30 epoch each)\n\n")
    md.append("---\n\n")

    # Per-modality variance
    md.append("## Per-Modality Variance (修复后，关键验证点)\n\n")
    md.append("**目标**：`v311_slot_variance_wsi ∈ [0.005, 0.05]` AND `v311_slot_variance_omic ∈ [0.005, 0.05]`\n\n")
    md.append("**判定**：训练末段（前 5 epoch）每模态平均 variance 落入目标区间 = PASS\n\n")
    md.append("| Fold | var_wsi (last5) | var_omic (last5) | WSI 在范围? | Omics 在范围? | Best val_c |\n")
    md.append("|------|-----------------|------------------|------------|---------------|-----------|\n")

    n_pass = 0
    for fold in sorted(assessments.keys()):
        d = assessments[fold]
        if d["var_wsi_last5"]:
            wsi_str = ", ".join(f"{v:.4f}" for v in d["var_wsi_last5"])
        else:
            wsi_str = "N/A"
        if d["var_omic_last5"]:
            omic_str = ", ".join(f"{v:.4f}" for v in d["var_omic_last5"])
        else:
            omic_str = "N/A"

        wsi_pass = "✅" if d["var_wsi_in_band"] else "❌"
        omic_pass = "✅" if d["var_omic_in_band"] else "❌"

        if d["var_wsi_in_band"] and d["var_omic_in_band"]:
            n_pass += 1

        best_c = f"{d['best_cindex']:.4f}" if d["best_cindex"] else "N/A"
        best_e = f"@{d['best_epoch']}" if d["best_epoch"] is not None else ""
        md.append(f"| {fold} | {wsi_str} | {omic_str} | {wsi_pass} | {omic_pass} | "
                  f"{best_c}{best_e} |\n")

    md.append(f"\n**结果**：{n_pass}/{len(assessments)} folds 通过双模态约束\n\n")
    if n_pass == len(assessments):
        md.append("✅ **PER-MODALITY DIVERSITY 真正工作**\n\n")
    elif n_pass >= 3:
        md.append("⚠️ **部分通过** — 需要进一步调参\n\n")
    else:
        md.append("❌ **修复未完全生效** — 仍需调查\n\n")

    md.append("---\n\n")

    # Cross-link proof experiments
    md.append("## Proof 实验交叉验证\n\n")

    md.append("### Proof A — 配方对比 (v3.10 vs v3.11)\n\n")
    if proof_a:
        # Try to extract a summary
        recipes = proof_a.get("by_recipe", {})
        if recipes:
            md.append("| 配方 | N folds | Mean C-index |\n")
            md.append("|------|---------|-------------|\n")
            for recipe_name, d in recipes.items():
                if "mean_cindex" in d:
                    n = d.get("n_folds", "?")
                    md.append(f"| {recipe_name} | {n} | {d['mean_cindex']:.4f} |\n")
        md.append("\n")

    md.append("### Proof B — Slot variance constraint (per-modality, 新版)\n\n")
    if proof_b:
        md.append(f"- 约束目标: variance ∈ [{proof_b['constraint']['v_min']}, {proof_b['constraint']['v_max']}]\n")
        md.append(f"- folds 总数: {proof_b['n_folds']}\n")
        md.append(f"- WSI 通过: {proof_b['n_folds_wsi_in_band']}/{proof_b['n_folds']}\n")
        md.append(f"- Omics 通过: {proof_b['n_folds_omic_in_band']}/{proof_b['n_folds']}\n")
        md.append(f"- 双模态都通过: {proof_b['n_folds_both_in_band']}/{proof_b['n_folds']}\n")
        md.append(f"- 总体: **{proof_b['verdict']}**\n\n")
    else:
        md.append("(尚未跑 — 等待训练完成后跑 `proof_B_variance_constraint_FIXED.py`，但需要更新到 fixed_v2 路径)\n\n")

    md.append("### Proof C — IPCW rank loss 行为\n\n")
    if proof_c:
        md.append(f"- {json.dumps(proof_c, indent=2)[:500]}\n")
        md.append("\n")
    else:
        md.append("(尚未跑)\n\n")

    # Verdict
    md.append("---\n\n## 结论\n\n")
    if n_pass == len(assessments) and len(assessments) >= 4:
        md.append("✅ **v3.11 fixed v2 训练成功**:\n")
        md.append("- Per-modality diversity 约束在所有 fold 上生效\n")
        md.append("- WSI 槽未坍缩（variance 在区间内）\n")
        md.append("- Omics 槽未坍缩（variance 在区间内）\n")
        md.append("- 修复了原版的「合并 variance 掩盖 WSI 坍缩」的 bug\n\n")
    else:
        md.append("⚠️ 训练仍在进行中或部分 fold 未通过 — 请检查日志\n\n")

    report = "".join(md)
    OUT.write_text(report)
    print(f"\n✅ Report saved to {OUT}")
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
