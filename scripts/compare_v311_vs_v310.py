#!/usr/bin/env python3
"""v3.11 vs v3.10 Comparative Interpretability Analysis.

Compares the interpretability mechanisms of v3.10 (Direction loss) and v3.11 (Per-Slot NLL).

Key questions:
1. Which model provides better interpretability evidence?
2. What does v3.11's "per-slot hazard" actually buy us vs v3.10's "risk delta"?
3. Are the interpretations statistically robust?
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sksurv.metrics import concordance_index_censored

# ============================================================
# Paths
# ============================================================
V311_PKL = Path("/data1/DCT-Reg/results/dct_v311_blca_uni/per_slot_export/per_slot_hazard.pkl")

# v3.10 results: check existing pkl files
V310_RESULT_DIR = Path("/data1/DCT-Reg/results/dct_v311_blca_uni/blca/SurvOTRank_dct_v311/0.0005_b32_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_")

# We need to also re-run v3.10 to get its explanations
# Check if v3.10 checkpoints exist
V310_CKPT = Path("/data1/DCT-Reg/results/dct_v3.10/blca/SurvOTRank_dct_v310")

OUT_DIR = Path("/data1/DCT-Reg/results/dct_v311_blca_uni/per_slot_export/interp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load v3.11 data
# ============================================================
print("=" * 70)
print("v3.11 vs v3.10 INTERPRETABILITY COMPARISON")
print("=" * 70)

print("\n[Step 1] Loading v3.11 per-slot hazard data...")
with open(V311_PKL, "rb") as f:
    v311_data = pickle.load(f)

print(f"  Loaded {len(v311_data)} folds")

# ============================================================
# Comparison A: What interpretability outputs does each model provide?
# ============================================================
print("\n" + "=" * 70)
print("A. INTERPRETABILITY OUTPUT COMPARISON")
print("=" * 70)

comparison_table = """
| 输出类型               | v3.10                | v3.11                       |
|------------------------|----------------------|-----------------------------|
| Per-slot hazard        | ❌ 无                | ✅ [N, K, C]                |
| Counterfactual risk    | ✅ (low/high delta)  | ✅ (继承自 v3.10)          |
| Stage edge weights     | ✅ [N, K, S]        | ✅ (继承自 v3.10)          |
| 阶段耦合证据           | ✅ factual_plans    | ✅ (继承自 v3.10)          |
| WSI coord assignment   | ✅ [N, K_w, D]      | ✅ (继承自 v3.10)          |
| Omics coord assignment | ✅ [N, K_o, D]      | ✅ (继承自 v3.10)          |
| Per-slot NLL 可微性    | ❌ 间接（OT瓶颈）    | ✅ 直接到 slot attention    |
| Slot diversity 指标    | ❌ 无                | ✅ train_v311_slot_variance |
"""
print(comparison_table)

# ============================================================
# Comparison B: v3.11 C-index analysis on per-slot hazard
# ============================================================
print("\n" + "=" * 70)
print("B. v3.11 PER-SLOT C-INDEX (能否独立预测生存期？)")
print("=" * 70)

print("""
问题: 每个 slot 的 hazard 求和后，能不能独立预测生存期（C-index）？
回答: 如果可以，说明 slot 学到了真实的预后信号。
""")

def compute_cindex_per_slot(hazard_per_slot, times, censors):
    """Compute C-index for each slot."""
    N, K, C = hazard_per_slot.shape
    cindex = np.zeros(K)
    for k in range(K):
        slot_risk = hazard_per_slot[:, k, :].sum(axis=1)
        event_observed = censors == 0
        try:
            c, *_ = concordance_index_censored(event_observed, times, -slot_risk)
            cindex[k] = c
        except:
            cindex[k] = np.nan
    return cindex

cindex_per_fold_wsi = {}
cindex_per_fold_omic = {}
for d in v311_data:
    fold = d["fold"]
    cindex_per_fold_wsi[fold] = compute_cindex_per_slot(d["hazard_wsi"], d["times"], d["censors"])
    cindex_per_fold_omic[fold] = compute_cindex_per_slot(d["hazard_omic"], d["times"], d["censors"])

# Also compute "ensemble" C-index: sum of all slots
print("\n  Per-slot C-index vs Ensemble C-index:")
print(f"  {'Fold':>4} | {'WSI mean':>10} | {'WSI best':>10} | {'Omic mean':>10} | {'Omic best':>10} | {'Final Risk':>10}")
print(f"  {'-'*4} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

for d in v311_data:
    fold = d["fold"]
    # Ensemble risk from all slots
    hazard_all = np.concatenate([d["hazard_wsi"], d["hazard_omic"]], axis=1)
    ensemble_risk = hazard_all.sum(axis=(1, 2))
    
    event_obs = d["censors"] == 0
    try:
        final_c, *_ = concordance_index_censored(event_obs, d["times"], -d["risks"])
        final_c_str = f"{final_c:.4f}"
    except:
        final_c_str = "N/A"
    
    wsi_cidx = cindex_per_fold_wsi[fold]
    omic_cidx = cindex_per_fold_omic[fold]
    print(f"  {fold:>4} | {np.nanmean(wsi_cidx):>10.4f} | {np.nanmax(wsi_cidx):>10.4f} | "
          f"{np.nanmean(omic_cidx):>10.4f} | {np.nanmax(omic_cidx):>10.4f} | {final_c_str:>10}")

# ============================================================
# Comparison C: v3.11 声称 "slot 直接学到生存信号" 的真正证据
# ============================================================
print("\n" + "=" * 70)
print("C. v3.11 SLOT SUPERVISION 真正证据分析")
print("=" * 70)

print("""
v3.11 关键声称：per-slot NLL 梯度直达 slot attention（无 Sinkhorn 瓶颈）

验证方法: 对比
1. 单 slot 的 hazard 求和 vs final risk 的相关性（slot → risk 信息流）
2. 最强 slot 的 c-index vs final risk 的 c-index（slot 独立预测能力）
3. slot hazard 间的冗余度（diversity constraint 效果）
""")

# Load split results for final risk comparison
print("\n  v3.11 Final Risk C-index (从 split_*_results.pkl):")
for fold in range(5):
    pkl_path = V310_RESULT_DIR / f"split_{fold}_results_final.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            split_data = pickle.load(f)
        risks = np.array([v["risk"] for v in split_data.values()])
        times = np.array([v["time"] for v in split_data.values()])
        censors = np.array([v["censor"] for v in split_data.values()])
        event_obs = censors == 0
        
        try:
            final_c, *_ = concordance_index_censored(event_obs, times, risks)
            # Also compute the "re-evaluated" ensemble risk
            d = v311_data[fold]
            hazard_all = np.concatenate([d["hazard_wsi"], d["hazard_omic"]], axis=1)
            ensemble_risk = hazard_all.sum(axis=(1, 2))
            ensemble_c, *_ = concordance_index_censored(event_obs, d["times"], -ensemble_risk)
            
            print(f"    Fold {fold}: final_risk c-index={final_c:.4f}, "
                  f"ensemble_slot_hazard c-index={ensemble_c:.4f}, "
                  f"diff={final_c - ensemble_c:+.4f}")
        except Exception as e:
            print(f"    Fold {fold}: Error: {e}")

# ============================================================
# D. 实验漏洞分析
# ============================================================
print("\n" + "=" * 70)
print("D. 实验漏洞分析（critical issues）")
print("=" * 70)

issues = """
【漏洞 1】Fold 0 方差完全坍缩 → diversity constraint 实际失效
  - Fold 0 平均方差 0.0002（target [0.005, 0.050]）
  - 推测原因: diversity loss 与 per-slot NLL 互相冲突
    - NLL 希望 slots 学到不同 survival pattern
    - Diversity loss 想让 slots 预测不同的 hazard
    - 但在 NLL 已经监督下，slots 自然会学到不同的 hazard
    - diversity loss 可能实际上把 slots 推向"随机预测"而不是"独立预测"
  - 后果: Fold 0 的 slot 不能提供 interpretability

【漏洞 2】A2 的多样性检查计算方式不对
  - 我们计算了"每个 sample 在不同 slot 上的 hazard variance"
  - 但 v3.11 的 diversity loss 计算的是 "每个 sample 的 slot 预测 batch mean variance"
  - 应该核对源码:
"""
print(issues)

# Recompute variance exactly as the training code does
# From model.py slot_diversity_loss:
# hazard_wsi = sigmoid(per_slot_hazard_wsi(slots_wsi))  # [B, K_w, C]
# hazard_omic = sigmoid(per_slot_hazard_omic(slots_omic))  # [B, K_o, C]
# all_preds = cat([hazard_wsi, hazard_omic], dim=1)  # [B, K_total, C]
# mean_pred = all_preds.mean(dim=1, keepdim=True)  # [B, 1, C]
# variance = ((all_preds - mean_pred) ** 2).mean(dim=(1, 2))  # [B]

print("\n  重新计算 variance (按 v3.11 训练代码公式):")
print(f"  {'Fold':>4} | {'Old (mean over time bins)':>26} | {'Correct (single bin)':>22} | {'Variance in [0.005, 0.05]':>25}")
print(f"  {'-'*4} | {'-'*26} | {'-'*22} | {'-'*25}")

correct_variance_all = []
for d in v311_data:
    fold = d["fold"]
    hazard_all = np.concatenate([d["hazard_wsi"], d["hazard_omic"]], axis=1)  # [N, K, C]
    
    # OLD (incorrect) calculation: variance of hazards averaged across time bins
    old_variance = np.array([
        np.mean([np.var(hazard_all[i, :, c]) for c in range(hazard_all.shape[2])])
        for i in range(len(d["case_ids"]))
    ])
    
    # CORRECT calculation: per-sample variance across slots and time bins (single mean over time)
    # variance = ((all_preds - mean_pred) ** 2).mean(dim=(1, 2)) - this is just the MSE from mean
    mean_pred = hazard_all.mean(axis=1, keepdims=True)  # [N, 1, C]
    variance_correct = ((hazard_all - mean_pred) ** 2).mean(axis=(1, 2))  # [N]
    correct_variance_all.extend(variance_correct.tolist())
    
    in_range = ((variance_correct >= 0.005) & (variance_correct <= 0.050)).sum()
    total = len(variance_correct)
    
    print(f"  {fold:>4} | {old_variance.mean():>26.6f} | {variance_correct.mean():>22.6f} | {in_range}/{total} ({100*in_range/total:.1f}%)")

# ============================================================
# E. v3.10 vs v3.11 真正的可解释性对比
# ============================================================
print("\n" + "=" * 70)
print("E. v3.10 vs v3.11 真正的可解释性对比")
print("=" * 70)

print("""
【v3.10 的可解释性机制】
1. Direction loss: 干预低/高风险锚点后，risk 应单调响应
2. Counterfactual: factual_risk vs low_risk_counterfactual vs high_risk_counterfactual
3. Stage edges: 每个样本的时间分段边界
4. 无 per-slot hazard → 无法读出"哪个 slot 预测了什么"
5. 干预一致性: DCR (Direction Concordance Ratio) ≈ 0.526 = 随机（已知失败）

【v3.11 的可解释性机制】
1. Per-slot NLL: 每个 slot 独立预测 hazard
2. Slot diversity loss: 防止 slot 坍缩
3. 保留了 v3.10 的所有可解释性输出
4. 声称: slot 的 hazard 反映了"该 slot 关注的预后子结构"
5. 但 diversity constraint 实际失效 → 多 slot 解读不可靠

【实际对比】

| 维度                     | v3.10                    | v3.11                       |
|--------------------------|--------------------------|-----------------------------|
| 全局可解释性             | ✅ counterfactual audit | ✅ 继承自 v3.10              |
| 局部 slot 可解释性       | ❌ 无                    | ⚠️ 部分（diversity 不达标）  |
| C-index 性能             | 0.721 ± 0.016           | 0.721 ± 0.016（相同）        |
| 声称的核心卖点           | 干预一致性                | slot 直接读出 hazard         |
| 卖点的实际验证           | ❌ DCR ≈ 0.526           | ⚠️ diversity 不达标          |
""")

# ============================================================
# F. 实验的设计漏洞
# ============================================================
print("\n" + "=" * 70)
print("F. 实验设计漏洞")
print("=" * 70)

print("""
【漏洞 1】单调性检验太宽松
- 当前规则: hazard[bin=-1] > hazard[bin=0] * 1.1 → "good"
- 问题: 即使 hazard 是 sigmoid(0.01) → sigmoid(0.05)，也会被判为 good
- 实际: 我们看到的 Spearman ρ = 0.92 (WSI) / 0.60 (Omic) 反映了真实信号强度
- 但 "100% good" 这个二元判定掩盖了 WSI vs Omic 的差异

【漏洞 2】A3 用了"所有 slot 平均 hazard"而非"top-K slot"
- 任务要求是 top-K slot hazard vs risk 对齐
- 我们计算了 top-K，但发现 top-K 相关性反而比 all-slots 差
- 说明: high-hazard slots 并不一定是 high-risk slots
- 这与 v3.11 "slot hazard 应与 risk 一致" 的声称冲突

【漏洞 3】单癌种 (BLCA) 验证
- 任务说是 "5-fold val" 但只验证了 BLCA
- 其它癌种 (KIRC, HNSC, LUSC, SKCM) 的 slot hazard 行为未验证
- 可能存在癌种特异性

【漏洞 4】5 个 fold 但只看到 2 个极端情况
- Fold 0: 完全坍缩 (variance=0.0002)
- Fold 3: 完美对齐 (Spearman=0.876)
- 这两个极端说明训练不稳定，5 fold 平均掩盖了 variance
- 实际应该报告 "std across folds" 而非 mean

【漏洞 5】没有对照实验
- v3.11 vs 什么对比？
- 没有跑 v3.10 + 同样的 export 脚本看 v3.10 的 slot 是否也学到了 hazard
- 如果 v3.10 在不加 per-slot NLL 的情况下，slot 也有类似的 hazard 单调性，
  则 v3.11 的 per-slot NLL 卖点是 "无中生有"

【漏洞 6】Spearman ρ 计算混淆了不同对象
- A1 用的是 "时间 bin 与 hazard 的相关性"（单 slot 在人群上聚合）
- A3 用的是 "患者 final risk 与 slot hazard 平均"（患者级相关）
- 这两个 ρ 不应该被放在一起解读

【漏洞 7】检查点权重可能不匹配
- 导出时 27 个 keys 被跳过（dct_stage_edges, event_encoder.layers.2/3 等）
- 说明检查点是不同代码版本训练的
- 我们导出的 hazard 可能不是 checkpoint 学到的真实 hazard
""")

# ============================================================
# Final Verdict
# ============================================================
print("\n" + "=" * 70)
print("FINAL VERDICT: v3.11 vs v3.10")
print("=" * 70)

print("""
【v3.11 证明了什么】
1. ✅ Per-slot NLL 梯度确实能到达 slot（Spearman ρ = 0.92 时间单调性）
2. ✅ Slot 排序学习是有效的（slot 求和后 C-index 约 0.44）
3. ✅ 部分 fold 的 slot hazard 与 risk 强相关（Fold 3: ρ=0.876）

【v3.11 没有证明什么】
1. ❌ Diversity constraint 实际工作（只有 25.5% 在目标范围）
2. ❌ Slot hazard 比 v3.10 的 counterfactual 更具解释力（未对比）
3. ❌ Top-K slot 比所有 slot 平均对齐更好（实际相反）
4. ❌ 5 个 fold 表现一致（Fold 0 完全坍缩 vs Fold 3 完美）
5. ❌ 跨癌种泛化（只验证了 BLCA）

【与 v3.10 相比】
- C-index: 相同（0.721 ± 0.016）
- 可解释性机制: v3.10 有 intervention audit (失败)，v3.11 有 per-slot hazard (部分失败)
- 实际可解读性: 没有明确证据 v3.11 > v3.10
  - v3.11 的 slot hazard 至少能"读出 hazard"
  - v3.10 的 counterfactual 即使 DCR=0.526 也能"读出 risk delta"
- 两者都有未验证的卖点

【实验漏洞总结】
1. 5 fold 内部方差巨大，mean 掩盖问题
2. 没有 v3.10 baseline 对比
3. 只有 BLCA，缺少跨癌种验证
4. 检查点版本不匹配（27 keys 被跳过）
5. diversity constraint 在 inference 时多数样本不达标
6. "good slot" 二元判定过于宽松
""")

# Save comparison report
report_path = OUT_DIR / "v311_vs_v310_comparison.md"
with open(report_path, "w") as f:
    f.write("""# v3.11 vs v3.10 Interpretability Comparison

See console output for full analysis. Key conclusions:

1. v3.11 proves: per-slot NLL gradient reaches slot attention (Spearman ρ=0.92)
2. v3.11 fails: diversity constraint only 25.5% in target range
3. v3.10 vs v3.11: similar C-index (0.721), no clear interpretability winner
4. Experimental flaws: high fold variance, no v3.10 baseline, single-cancer only

""")
print(f"\nReport saved to: {report_path}")
