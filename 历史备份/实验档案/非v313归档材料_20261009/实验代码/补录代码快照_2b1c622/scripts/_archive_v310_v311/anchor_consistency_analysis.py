#!/usr/bin/env python3
"""
实验 1.1: 锚点跨折一致性分析

目的: 验证预后锚点在不同折之间是否稳定
通过标准:
  - 同癌种跨折锚点 cosine similarity > 0.7
  - 锚点方向角度 std < 30°

如果失败: 说明锚点本身就不稳定，方向响应无从谈起
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt
import seaborn as sns

def load_anchor_from_checkpoint(checkpoint_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """从检查点提取高低风险锚点"""
    import torch
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # 尝试多种可能的键名
        state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        
        # 查找锚点
        high_anchor = None
        low_anchor = None
        
        for key in state_dict.keys():
            if 'high' in key.lower() and 'anchor' in key.lower():
                high_anchor = state_dict[key].cpu().numpy().flatten()
            elif 'low' in key.lower() and 'anchor' in key.lower():
                low_anchor = state_dict[key].cpu().numpy().flatten()
        
        if high_anchor is None or low_anchor is None:
            # 尝试从配置或其他位置提取
            print(f"⚠️  在 state_dict 中未找到锚点，尝试其他键...")
            print(f"可用键: {list(state_dict.keys())[:10]}")
            return None, None
        
        return high_anchor, low_anchor
    
    except Exception as e:
        print(f"❌ 加载检查点失败: {checkpoint_path}")
        print(f"   错误: {e}")
        return None, None


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算余弦相似度"""
    return 1 - cosine(v1, v2)


def compute_direction_angle(high: np.ndarray, low: np.ndarray) -> float:
    """计算方向向量的角度 (弧度)"""
    direction = high - low
    # 计算与第一个维度的角度（简化版）
    # 实际应该用主成分或其他参考方向
    angle = np.arctan2(direction[1] if len(direction) > 1 else 0, direction[0])
    return angle


def analyze_anchor_consistency(
    cancer: str,
    folds: List[int],
    results_base: Path,
    output_path: Path
):
    """主分析函数"""
    
    print("=" * 80)
    print("🔬 实验 1.1: 锚点跨折一致性分析")
    print("=" * 80)
    print()
    print(f"癌种: {cancer}")
    print(f"分析折: {folds}")
    print()
    
    # 收集所有锚点
    high_anchors = {}
    low_anchors = {}
    
    for fold in folds:
        # 尝试多种可能的路径
        possible_paths = [
            results_base / f"dct_v3.10_experiments/robust/full/{cancer.lower()}/uni2h/5fold_uni2h/fold_{fold}/evidence/fold_{fold}/checkpoint.pt",
            results_base / f"dct_v3.10_experiments/robust/full/{cancer.lower()}/uni2h/5fold_uni2h/fold{fold}/evidence/fold_{fold}/checkpoint.pt",
            results_base / f"dct_v3.10/{cancer.lower()}/fold_{fold}/checkpoint.pt",
            results_base / f"dct_v3.10/{cancer.lower()}/model_best_s{fold}.pth",
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            print(f"⚠️  Fold {fold}: 未找到检查点")
            print(f"   尝试的路径:")
            for p in possible_paths:
                print(f"     - {p}")
            continue
        
        print(f"Fold {fold}: 加载 {checkpoint_path}")
        high_anchor, low_anchor = load_anchor_from_checkpoint(checkpoint_path)
        
        if high_anchor is not None and low_anchor is not None:
            high_anchors[fold] = high_anchor
            low_anchors[fold] = low_anchor
            print(f"  ✓ 高风险锚点维度: {high_anchor.shape}")
            print(f"  ✓ 低风险锚点维度: {low_anchor.shape}")
        else:
            print(f"  ✗ 提取锚点失败")
    
    print()
    print(f"成功加载: {len(high_anchors)}/{len(folds)} 折")
    print()
    
    if len(high_anchors) < 2:
        print("❌ 锚点数量不足，无法进行一致性分析")
        return
    
    # 分析 1: 高风险锚点相似度矩阵
    print("=" * 80)
    print("📊 分析 1: 高风险锚点相似度")
    print("=" * 80)
    print()
    
    folds_loaded = sorted(high_anchors.keys())
    n_folds = len(folds_loaded)
    
    high_sim_matrix = np.zeros((n_folds, n_folds))
    for i, fold_i in enumerate(folds_loaded):
        for j, fold_j in enumerate(folds_loaded):
            if i == j:
                high_sim_matrix[i, j] = 1.0
            else:
                sim = compute_cosine_similarity(
                    high_anchors[fold_i],
                    high_anchors[fold_j]
                )
                high_sim_matrix[i, j] = sim
    
    # 提取上三角（不含对角线）
    high_sims = high_sim_matrix[np.triu_indices(n_folds, k=1)]
    
    print("高风险锚点相似度矩阵:")
    print()
    header = "Fold  " + "  ".join([f"F{f}" for f in folds_loaded])
    print(header)
    print("-" * len(header))
    for i, fold in enumerate(folds_loaded):
        row = f"F{fold}   " + "  ".join([f"{high_sim_matrix[i, j]:.3f}" for j in range(n_folds)])
        print(row)
    print()
    print(f"平均相似度: {high_sims.mean():.4f} ± {high_sims.std():.4f}")
    print(f"最小相似度: {high_sims.min():.4f}")
    print(f"最大相似度: {high_sims.max():.4f}")
    print()
    
    # 分析 2: 低风险锚点相似度矩阵
    print("=" * 80)
    print("📊 分析 2: 低风险锚点相似度")
    print("=" * 80)
    print()
    
    low_sim_matrix = np.zeros((n_folds, n_folds))
    for i, fold_i in enumerate(folds_loaded):
        for j, fold_j in enumerate(folds_loaded):
            if i == j:
                low_sim_matrix[i, j] = 1.0
            else:
                sim = compute_cosine_similarity(
                    low_anchors[fold_i],
                    low_anchors[fold_j]
                )
                low_sim_matrix[i, j] = sim
    
    low_sims = low_sim_matrix[np.triu_indices(n_folds, k=1)]
    
    print("低风险锚点相似度矩阵:")
    print()
    print(header)
    print("-" * len(header))
    for i, fold in enumerate(folds_loaded):
        row = f"F{fold}   " + "  ".join([f"{low_sim_matrix[i, j]:.3f}" for j in range(n_folds)])
        print(row)
    print()
    print(f"平均相似度: {low_sims.mean():.4f} ± {low_sims.std():.4f}")
    print(f"最小相似度: {low_sims.min():.4f}")
    print(f"最大相似度: {low_sims.max():.4f}")
    print()
    
    # 分析 3: 方向向量的角度稳定性
    print("=" * 80)
    print("📊 分析 3: 方向向量稳定性")
    print("=" * 80)
    print()
    
    directions = {}
    angles = []
    
    for fold in folds_loaded:
        direction = high_anchors[fold] - low_anchors[fold]
        directions[fold] = direction
        angle = compute_direction_angle(high_anchors[fold], low_anchors[fold])
        angles.append(angle)
        print(f"Fold {fold}: 方向角度 = {np.degrees(angle):.2f}°")
    
    print()
    
    # 计算方向向量之间的相似度
    direction_sims = []
    for i, fold_i in enumerate(folds_loaded):
        for j in range(i+1, n_folds):
            fold_j = folds_loaded[j]
            sim = compute_cosine_similarity(directions[fold_i], directions[fold_j])
            direction_sims.append(sim)
    
    print(f"方向向量平均相似度: {np.mean(direction_sims):.4f} ± {np.std(direction_sims):.4f}")
    print(f"方向角度标准差: {np.degrees(np.std(angles)):.2f}°")
    print()
    
    # 分析 4: 锚点质心和离散度
    print("=" * 80)
    print("📊 分析 4: 锚点质心和离散度")
    print("=" * 80)
    print()
    
    high_anchors_array = np.array([high_anchors[f] for f in folds_loaded])
    low_anchors_array = np.array([low_anchors[f] for f in folds_loaded])
    
    high_centroid = high_anchors_array.mean(axis=0)
    low_centroid = low_anchors_array.mean(axis=0)
    
    high_std = high_anchors_array.std(axis=0).mean()
    low_std = low_anchors_array.std(axis=0).mean()
    
    print(f"高风险锚点离散度 (平均 std): {high_std:.4f}")
    print(f"低风险锚点离散度 (平均 std): {low_std:.4f}")
    print()
    
    # 计算每个锚点到质心的距离
    print("各折锚点到质心的距离:")
    for fold in folds_loaded:
        high_dist = np.linalg.norm(high_anchors[fold] - high_centroid)
        low_dist = np.linalg.norm(low_anchors[fold] - low_centroid)
        print(f"  Fold {fold}: 高风险 {high_dist:.4f}, 低风险 {low_dist:.4f}")
    
    print()
    
    # 判定结果
    print("=" * 80)
    print("🎯 判定结果")
    print("=" * 80)
    print()
    
    high_pass = high_sims.mean() > 0.7
    low_pass = low_sims.mean() > 0.7
    direction_pass = np.degrees(np.std(angles)) < 30
    
    print(f"{'指标':<30} {'阈值':<15} {'实际值':<15} {'状态'}")
    print("-" * 75)
    print(f"{'高风险锚点相似度':<30} {'>0.70':<15} {f'{high_sims.mean():.4f}':<15} {'✅ 通过' if high_pass else '❌ 失败'}")
    print(f"{'低风险锚点相似度':<30} {'>0.70':<15} {f'{low_sims.mean():.4f}':<15} {'✅ 通过' if low_pass else '❌ 失败'}")
    print(f"{'方向角度稳定性':<30} {'<30°':<15} {f'{np.degrees(np.std(angles)):.2f}°':<15} {'✅ 通过' if direction_pass else '❌ 失败'}")
    print()
    
    overall_pass = high_pass and low_pass and direction_pass
    
    if overall_pass:
        print("✅ 总体判定: 通过 - 锚点跨折一致性良好")
    else:
        print("❌ 总体判定: 失败 - 锚点跨折一致性不足")
        print()
        print("⚠️  诊断:")
        if not high_pass:
            print("  - 高风险锚点不稳定，需要改进提取方法")
        if not low_pass:
            print("  - 低风险锚点不稳定，需要改进提取方法")
        if not direction_pass:
            print("  - 方向向量不稳定，可能导致方向响应失效")
    
    print()
    
    # 保存结果
    results = {
        "cancer": cancer,
        "folds_analyzed": folds_loaded,
        "high_risk_anchors": {
            "mean_similarity": float(high_sims.mean()),
            "std_similarity": float(high_sims.std()),
            "min_similarity": float(high_sims.min()),
            "max_similarity": float(high_sims.max()),
            "pass_threshold": high_pass
        },
        "low_risk_anchors": {
            "mean_similarity": float(low_sims.mean()),
            "std_similarity": float(low_sims.std()),
            "min_similarity": float(low_sims.min()),
            "max_similarity": float(low_sims.max()),
            "pass_threshold": low_pass
        },
        "direction_stability": {
            "mean_direction_similarity": float(np.mean(direction_sims)),
            "angle_std_degrees": float(np.degrees(np.std(angles))),
            "pass_threshold": direction_pass
        },
        "overall_pass": overall_pass
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ 结果已保存: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="锚点跨折一致性分析")
    parser.add_argument("--cancer", type=str, default="BLCA", help="癌种")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4", help="折索引，逗号分隔")
    parser.add_argument("--results_base", type=str, default="/data1/DCT-Reg/results", help="结果根目录")
    parser.add_argument("--output", type=str, default="/data1/DCT-Reg/results/anchor_diagnostics/anchor_consistency_report.json", help="输出路径")
    
    args = parser.parse_args()
    
    folds = [int(f.strip()) for f in args.folds.split(',')]
    
    analyze_anchor_consistency(
        cancer=args.cancer,
        folds=folds,
        results_base=Path(args.results_base),
        output_path=Path(args.output)
    )


if __name__ == "__main__":
    main()
