#!/usr/bin/env python3
"""
实验 1.3: 锚点分离度分析

目的: 验证高低风险锚点是否足够分离
方法: 计算锚点间距离、归一化距离、使用锚点距离作为分类器的性能

通过标准:
  - 归一化距离 > 2.0 (足够分离)
  - 用锚点距离作为分类器，AUC > 0.65

如果失败: 锚点太接近，无法提供有效的方向指导
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def load_anchor_from_checkpoint(checkpoint_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """从检查点提取高低风险锚点"""
    import torch
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        
        high_anchor = None
        low_anchor = None
        
        for key in state_dict.keys():
            if 'high' in key.lower() and 'anchor' in key.lower():
                high_anchor = state_dict[key].cpu().numpy().flatten()
            elif 'low' in key.lower() and 'anchor' in key.lower():
                low_anchor = state_dict[key].cpu().numpy().flatten()
        
        return high_anchor, low_anchor
    
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return None, None


def load_training_features_and_labels(
    cancer: str,
    fold: int,
    results_base: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载训练集特征和风险标签
    
    返回: (features, risk_labels)
    risk_labels: 1 = 高风险, 0 = 低风险
    """
    import torch
    
    # 尝试加载 split_manifest.json 获取训练集样本
    possible_evidence_dirs = [
        results_base / f"dct_v3.10_experiments/robust/full/{cancer.lower()}/uni2h/5fold_uni2h/fold_{fold}/evidence/fold_{fold}",
        results_base / f"dct_v3.10/{cancer.lower()}/fold_{fold}",
    ]
    
    evidence_dir = None
    for path in possible_evidence_dirs:
        if path.exists():
            evidence_dir = path
            break
    
    if evidence_dir is None:
        return None, None
    
    # 加载 split_manifest
    split_manifest_path = evidence_dir / "split_manifest.json"
    if not split_manifest_path.exists():
        print(f"⚠️  未找到 split_manifest.json")
        return None, None
    
    with open(split_manifest_path) as f:
        split_manifest = json.load(f)
    
    # 从 manifest 中提取训练集风险分组
    train_high_risk = split_manifest.get('train_high_risk_patients', [])
    train_low_risk = split_manifest.get('train_low_risk_patients', [])
    
    if not train_high_risk or not train_low_risk:
        print(f"⚠️  split_manifest 中缺少风险分组")
        return None, None
    
    print(f"✓ 训练集高风险样本: {len(train_high_risk)}")
    print(f"✓ 训练集低风险样本: {len(train_low_risk)}")
    
    # 注意: 这里需要实际加载训练集特征
    # 由于没有保存训练集特征，这里使用模拟数据
    
    return None, None


def analyze_anchor_separation(
    cancer: str,
    folds: List[int],
    results_base: Path,
    output_path: Path
):
    """主分析函数"""
    
    print("=" * 80)
    print("🔬 实验 1.3: 锚点分离度分析")
    print("=" * 80)
    print()
    print(f"癌种: {cancer}")
    print(f"分析折: {folds}")
    print()
    
    # 收集所有锚点
    anchors_data = {}
    
    for fold in folds:
        possible_paths = [
            results_base / f"dct_v3.10_experiments/robust/full/{cancer.lower()}/uni2h/5fold_uni2h/fold_{fold}/evidence/fold_{fold}/checkpoint.pt",
            results_base / f"dct_v3.10/{cancer.lower()}/fold_{fold}/checkpoint.pt",
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            print(f"⚠️  Fold {fold}: 未找到检查点")
            continue
        
        high_anchor, low_anchor = load_anchor_from_checkpoint(checkpoint_path)
        
        if high_anchor is not None and low_anchor is not None:
            anchors_data[fold] = {
                'high': high_anchor,
                'low': low_anchor
            }
            print(f"Fold {fold}: ✓ 加载成功")
    
    print()
    print(f"成功加载: {len(anchors_data)}/{len(folds)} 折")
    print()
    
    if len(anchors_data) == 0:
        print("❌ 无可用数据")
        return
    
    # 分析 1: 锚点间欧氏距离
    print("=" * 80)
    print("📊 分析 1: 锚点间欧氏距离")
    print("=" * 80)
    print()
    
    distances = {}
    normalized_distances = {}
    
    for fold, anchors in anchors_data.items():
        high = anchors['high']
        low = anchors['low']
        
        # 欧氏距离
        euclidean_dist = np.linalg.norm(high - low)
        
        # 归一化距离 (相对于特征维度的平方根)
        feature_dim = len(high)
        normalized_dist = euclidean_dist / np.sqrt(feature_dim)
        
        distances[fold] = euclidean_dist
        normalized_distances[fold] = normalized_dist
        
        print(f"Fold {fold}:")
        print(f"  特征维度: {feature_dim}")
        print(f"  欧氏距离: {euclidean_dist:.4f}")
        print(f"  归一化距离: {normalized_dist:.4f}")
        print()
    
    mean_dist = np.mean(list(distances.values()))
    std_dist = np.std(list(distances.values()))
    mean_norm_dist = np.mean(list(normalized_distances.values()))
    std_norm_dist = np.std(list(normalized_distances.values()))
    
    print(f"平均欧氏距离: {mean_dist:.4f} ± {std_dist:.4f}")
    print(f"平均归一化距离: {mean_norm_dist:.4f} ± {std_norm_dist:.4f}")
    print()
    
    # 分析 2: 锚点间余弦距离
    print("=" * 80)
    print("📊 分析 2: 锚点间余弦距离")
    print("=" * 80)
    print()
    
    from scipy.spatial.distance import cosine
    
    cosine_distances = {}
    
    for fold, anchors in anchors_data.items():
        high = anchors['high']
        low = anchors['low']
        
        cos_dist = cosine(high, low)
        cosine_distances[fold] = cos_dist
        
        print(f"Fold {fold}: 余弦距离 = {cos_dist:.4f} (相似度 = {1-cos_dist:.4f})")
    
    print()
    mean_cos_dist = np.mean(list(cosine_distances.values()))
    print(f"平均余弦距离: {mean_cos_dist:.4f}")
    print(f"平均余弦相似度: {1-mean_cos_dist:.4f}")
    print()
    
    # 分析 3: 锚点的范数
    print("=" * 80)
    print("📊 分析 3: 锚点的 L2 范数")
    print("=" * 80)
    print()
    
    high_norms = {}
    low_norms = {}
    
    for fold, anchors in anchors_data.items():
        high = anchors['high']
        low = anchors['low']
        
        high_norm = np.linalg.norm(high)
        low_norm = np.linalg.norm(low)
        
        high_norms[fold] = high_norm
        low_norms[fold] = low_norm
        
        print(f"Fold {fold}:")
        print(f"  高风险锚点范数: {high_norm:.4f}")
        print(f"  低风险锚点范数: {low_norm:.4f}")
        print(f"  范数比: {high_norm/low_norm:.4f}")
        print()
    
    mean_high_norm = np.mean(list(high_norms.values()))
    mean_low_norm = np.mean(list(low_norms.values()))
    
    print(f"平均高风险锚点范数: {mean_high_norm:.4f}")
    print(f"平均低风险锚点范数: {mean_low_norm:.4f}")
    print()
    
    # 分析 4: 使用锚点距离作为分类器 (模拟)
    print("=" * 80)
    print("📊 分析 4: 锚点距离作为分类器的性能 (模拟)")
    print("=" * 80)
    print()
    
    print("⚠️  注意: 由于未保存训练集特征，使用模拟数据评估")
    print()
    
    # 模拟: 生成一些样本，看锚点距离能否区分高低风险
    simulated_aucs = []
    
    for fold, anchors in anchors_data.items():
        high = anchors['high']
        low = anchors['low']
        
        # 模拟高风险样本 (接近高风险锚点)
        n_high = 50
        high_samples = high + np.random.randn(n_high, len(high)) * 0.5
        
        # 模拟低风险样本 (接近低风险锚点)
        n_low = 50
        low_samples = low + np.random.randn(n_low, len(low)) * 0.5
        
        # 合并
        all_samples = np.vstack([high_samples, low_samples])
        all_labels = np.hstack([np.ones(n_high), np.zeros(n_low)])
        
        # 计算到高风险锚点的距离作为风险分数
        dists_to_high = np.linalg.norm(all_samples - high, axis=1)
        # 距离越小，风险越高，所以取负
        risk_scores = -dists_to_high
        
        # 计算 AUC
        try:
            auc = roc_auc_score(all_labels, risk_scores)
            simulated_aucs.append(auc)
            print(f"Fold {fold}: 模拟 AUC = {auc:.4f}")
        except:
            print(f"Fold {fold}: 计算 AUC 失败")
    
    print()
    if simulated_aucs:
        print(f"平均模拟 AUC: {np.mean(simulated_aucs):.4f} ± {np.std(simulated_aucs):.4f}")
        print()
        print("💡 解释: 这是理想情况下的性能（样本在锚点周围生成）")
        print("   实际性能需要真实训练集数据验证")
    print()
    
    # 分析 5: 锚点在特征空间的位置分析
    print("=" * 80)
    print("📊 分析 5: 锚点相对位置")
    print("=" * 80)
    print()
    
    for fold, anchors in anchors_data.items():
        high = anchors['high']
        low = anchors['low']
        
        # 计算均值向量的方向
        mean_vector = (high + low) / 2
        direction_vector = high - low
        
        # 方向向量与均值向量的夹角
        cos_angle = np.dot(direction_vector, mean_vector) / (
            np.linalg.norm(direction_vector) * np.linalg.norm(mean_vector) + 1e-8
        )
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        print(f"Fold {fold}:")
        print(f"  中心点范数: {np.linalg.norm(mean_vector):.4f}")
        print(f"  方向向量范数: {np.linalg.norm(direction_vector):.4f}")
        print(f"  方向-中心夹角: {angle_deg:.2f}°")
        print()
    
    # 判定结果
    print("=" * 80)
    print("🎯 判定结果")
    print("=" * 80)
    print()
    
    pass_normalized_dist = mean_norm_dist > 2.0
    pass_cosine_dist = mean_cos_dist > 0.2  # 余弦距离 > 0.2 表示不太相似
    pass_simulated_auc = np.mean(simulated_aucs) > 0.65 if simulated_aucs else False
    
    print(f"{'指标':<40} {'阈值':<15} {'实际值':<20} {'状态'}")
    print("-" * 85)
    print(f"{'归一化距离':<40} {'>2.0':<15} {f'{mean_norm_dist:.4f}':<20} {'✅ 通过' if pass_normalized_dist else '❌ 失败'}")
    print(f"{'余弦距离':<40} {'>0.2':<15} {f'{mean_cos_dist:.4f}':<20} {'✅ 通过' if pass_cosine_dist else '❌ 失败'}")
    
    if simulated_aucs:
        print(f"{'模拟分类器 AUC':<40} {'>0.65':<15} {f'{np.mean(simulated_aucs):.4f}':<20} {'✅ 通过' if pass_simulated_auc else '❌ 失败'}")
    else:
        print(f"{'模拟分类器 AUC':<40} {'>0.65':<15} {'N/A':<20} {'⚠️  无法评估'}")
    
    print()
    
    overall_pass = pass_normalized_dist and pass_cosine_dist
    
    if overall_pass:
        print("✅ 总体判定: 通过 - 锚点分离度足够")
        print()
        print("💡 解释:")
        print("   高低风险锚点在特征空间中有足够的分离")
        print("   这为方向传输提供了有效的指导")
    else:
        print("❌ 总体判定: 失败 - 锚点分离度不足")
        print()
        print("⚠️  诊断:")
        if not pass_normalized_dist:
            print("   - 归一化距离过小，锚点太接近")
            print("   - 可能原因: 高低风险样本在特征空间中重叠")
        if not pass_cosine_dist:
            print("   - 余弦距离过小，锚点方向太相似")
            print("   - 可能原因: 锚点只是范数不同，方向一致")
        print()
        print("   建议:")
        print("     - 增加锚点提取时的风险分组对比度")
        print("     - 使用更极端的风险分位数 (如 10% vs 90%)")
        print("     - 验证特征空间是否包含预后信息")
    
    print()
    
    # 保存结果
    results = {
        "cancer": cancer,
        "folds_analyzed": list(anchors_data.keys()),
        "euclidean_distance": {
            "mean": float(mean_dist),
            "std": float(std_dist),
            "by_fold": {str(k): float(v) for k, v in distances.items()}
        },
        "normalized_distance": {
            "mean": float(mean_norm_dist),
            "std": float(std_norm_dist),
            "pass_threshold": pass_normalized_dist,
            "by_fold": {str(k): float(v) for k, v in normalized_distances.items()}
        },
        "cosine_distance": {
            "mean": float(mean_cos_dist),
            "cosine_similarity": float(1 - mean_cos_dist),
            "pass_threshold": pass_cosine_dist,
            "by_fold": {str(k): float(v) for k, v in cosine_distances.items()}
        },
        "anchor_norms": {
            "high_risk_mean": float(mean_high_norm),
            "low_risk_mean": float(mean_low_norm),
            "by_fold": {
                str(k): {"high": float(high_norms[k]), "low": float(low_norms[k])}
                for k in high_norms.keys()
            }
        },
        "simulated_classifier": {
            "mean_auc": float(np.mean(simulated_aucs)) if simulated_aucs else None,
            "std_auc": float(np.std(simulated_aucs)) if simulated_aucs else None,
            "pass_threshold": pass_simulated_auc,
            "note": "Based on simulated samples around anchors"
        },
        "overall_pass": overall_pass
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ 结果已保存: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="锚点分离度分析")
    parser.add_argument("--cancer", type=str, default="BLCA", help="癌种")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4", help="折索引，逗号分隔")
    parser.add_argument("--results_base", type=str, default="/data1/DCT-Reg/results", help="结果根目录")
    parser.add_argument("--output", type=str, default="/data1/DCT-Reg/results/anchor_diagnostics/anchor_separation_report.json", help="输出路径")
    
    args = parser.parse_args()
    
    folds = [int(f.strip()) for f in args.folds.split(',')]
    
    analyze_anchor_separation(
        cancer=args.cancer,
        folds=folds,
        results_base=Path(args.results_base),
        output_path=Path(args.output)
    )


if __name__ == "__main__":
    main()
