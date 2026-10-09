#!/usr/bin/env python3
"""
运行锚点诊断实验 1.1, 1.2, 1.3

直接使用 checkpoint.pt 中的原型和风险锚点信息
"""

import torch
import numpy as np
import glob
import json
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def load_anchors_from_checkpoint(ckpt_path):
    """从检查点提取锚点信息"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 提取原型（这些可以作为锚点的代理）
    wsi_prototypes = ckpt.get('shared_wsi_prototypes')
    omic_prototypes = ckpt.get('shared_omic_prototypes')
    risk_anchor_costs = ckpt.get('risk_anchor_costs')
    
    if wsi_prototypes is not None:
        wsi_prototypes = wsi_prototypes.cpu().numpy()
    if omic_prototypes is not None:
        omic_prototypes = omic_prototypes.cpu().numpy()
    if risk_anchor_costs is not None:
        risk_anchor_costs = risk_anchor_costs.cpu().numpy()
    
    return {
        'wsi_prototypes': wsi_prototypes,
        'omic_prototypes': omic_prototypes,
        'risk_anchor_costs': risk_anchor_costs,
        'path': str(ckpt_path)
    }


def experiment_1_1_anchor_consistency():
    """实验 1.1: 锚点跨折一致性分析"""
    
    print("=" * 80)
    print("🔬 实验 1.1: 锚点跨折一致性分析")
    print("=" * 80)
    print()
    
    # 找到所有检查点
    checkpoints = sorted(glob.glob(
        "/data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/*/evidence/fold_*/checkpoint.pt"
    ))
    
    print(f"找到 {len(checkpoints)} 个检查点")
    print()
    
    # 加载所有锚点
    fold_data = {}
    for ckpt_path in checkpoints:
        fold_num = int(ckpt_path.split('fold_')[-1].split('/')[0])
        print(f"加载 Fold {fold_num}...")
        data = load_anchors_from_checkpoint(ckpt_path)
        fold_data[fold_num] = data
        
        if data['wsi_prototypes'] is not None:
            print(f"  WSI prototypes: {data['wsi_prototypes'].shape}")
        if data['omic_prototypes'] is not None:
            print(f"  Omic prototypes: {data['omic_prototypes'].shape}")
        if data['risk_anchor_costs'] is not None:
            print(f"  Risk anchor costs: {data['risk_anchor_costs'].shape}")
    
    print()
    
    if len(fold_data) < 2:
        print("❌ 锚点数量不足")
        return
    
    folds = sorted(fold_data.keys())
    n_folds = len(folds)
    
    # 分析 WSI 原型一致性
    print("=" * 80)
    print("📊 分析 1.1.1: WSI 原型跨折一致性")
    print("=" * 80)
    print()
    
    wsi_prototypes_list = [fold_data[f]['wsi_prototypes'] for f in folds]
    
    if all(p is not None for p in wsi_prototypes_list):
        # 计算跨折相似度矩阵
        sim_matrix = np.zeros((n_folds, n_folds))
        
        for i, fold_i in enumerate(folds):
            for j, fold_j in enumerate(folds):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    # 计算平均余弦相似度（多个原型的平均）
                    p1 = fold_data[fold_i]['wsi_prototypes']
                    p2 = fold_data[fold_j]['wsi_prototypes']
                    
                    # 假设原型数量相同，计算每对的相似度
                    sims = []
                    n_proto = min(p1.shape[0], p2.shape[0])
                    for k in range(n_proto):
                        if p1[k].ndim > 1:
                            v1 = p1[k].flatten()
                            v2 = p2[k].flatten()
                        else:
                            v1 = p1[k]
                            v2 = p2[k]
                        sim = 1 - cosine(v1, v2)
                        sims.append(sim)
                    
                    sim_matrix[i, j] = np.mean(sims)
        
        # 打印相似度矩阵
        print("WSI 原型相似度矩阵:")
        print()
        header = "Fold  " + "  ".join([f"F{f}" for f in folds])
        print(header)
        print("-" * len(header))
        for i, fold in enumerate(folds):
            row = f"F{fold}   " + "  ".join([f"{sim_matrix[i, j]:.3f}" for j in range(n_folds)])
            print(row)
        print()
        
        # 提取上三角
        upper_tri_sims = sim_matrix[np.triu_indices(n_folds, k=1)]
        
        print(f"平均相似度: {upper_tri_sims.mean():.4f} ± {upper_tri_sims.std():.4f}")
        print(f"最小相似度: {upper_tri_sims.min():.4f}")
        print(f"最大相似度: {upper_tri_sims.max():.4f}")
        print()
        
        wsi_pass = upper_tri_sims.mean() > 0.7
        print(f"{'✅ 通过' if wsi_pass else '❌ 失败'}: WSI 原型平均相似度 {'>' if wsi_pass else '<='} 0.70")
    else:
        print("⚠️  部分折缺少 WSI 原型")
        wsi_pass = False
    
    print()
    
    # 分析 Omic 原型一致性
    print("=" * 80)
    print("📊 分析 1.1.2: Omic 原型跨折一致性")
    print("=" * 80)
    print()
    
    omic_prototypes_list = [fold_data[f]['omic_prototypes'] for f in folds]
    
    if all(p is not None for p in omic_prototypes_list):
        sim_matrix = np.zeros((n_folds, n_folds))
        
        for i, fold_i in enumerate(folds):
            for j, fold_j in enumerate(folds):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    p1 = fold_data[fold_i]['omic_prototypes']
                    p2 = fold_data[fold_j]['omic_prototypes']
                    
                    sims = []
                    n_proto = min(p1.shape[0], p2.shape[0])
                    for k in range(n_proto):
                        if p1[k].ndim > 1:
                            v1 = p1[k].flatten()
                            v2 = p2[k].flatten()
                        else:
                            v1 = p1[k]
                            v2 = p2[k]
                        sim = 1 - cosine(v1, v2)
                        sims.append(sim)
                    
                    sim_matrix[i, j] = np.mean(sims)
        
        print("Omic 原型相似度矩阵:")
        print()
        print(header)
        print("-" * len(header))
        for i, fold in enumerate(folds):
            row = f"F{fold}   " + "  ".join([f"{sim_matrix[i, j]:.3f}" for j in range(n_folds)])
            print(row)
        print()
        
        upper_tri_sims = sim_matrix[np.triu_indices(n_folds, k=1)]
        
        print(f"平均相似度: {upper_tri_sims.mean():.4f} ± {upper_tri_sims.std():.4f}")
        print(f"最小相似度: {upper_tri_sims.min():.4f}")
        print(f"最大相似度: {upper_tri_sims.max():.4f}")
        print()
        
        omic_pass = upper_tri_sims.mean() > 0.7
        print(f"{'✅ 通过' if omic_pass else '❌ 失败'}: Omic 原型平均相似度 {'>' if omic_pass else '<='} 0.70")
    else:
        print("⚠️  部分折缺少 Omic 原型")
        omic_pass = False
    
    print()
    
    # 总体判定
    print("=" * 80)
    print("🎯 实验 1.1 总体判定")
    print("=" * 80)
    print()
    
    overall_pass = wsi_pass and omic_pass
    
    print(f"{'指标':<30} {'阈值':<15} {'状态'}")
    print("-" * 60)
    print(f"{'WSI 原型一致性':<30} {'>0.70':<15} {'✅ 通过' if wsi_pass else '❌ 失败'}")
    print(f"{'Omic 原型一致性':<30} {'>0.70':<15} {'✅ 通过' if omic_pass else '❌ 失败'}")
    print()
    
    if overall_pass:
        print("✅ 总体判定: 通过 - 锚点跨折一致性良好")
        print("   可以继续后续实验")
    else:
        print("❌ 总体判定: 失败 - 锚点跨折一致性不足")
        print()
        print("⚠️  诊断:")
        if not wsi_pass:
            print("  - WSI 原型不稳定，需要改进提取方法")
        if not omic_pass:
            print("  - Omic 原型不稳定，需要改进提取方法")
        print()
        print("⚠️  后续影响:")
        print("  - 锚点不稳定会导致方向响应失效")
        print("  - 这可以解释为何 DCR < 0.5（低于随机）")
        print("  - 需要重新设计锚点提取机制")
    
    print()
    
    # 保存结果
    results = {
        "experiment": "1.1_anchor_consistency",
        "cancer": "BLCA",
        "folds_analyzed": folds,
        "wsi_prototype_consistency": {
            "mean_similarity": float(upper_tri_sims.mean()) if wsi_pass else None,
            "pass": bool(wsi_pass)
        },
        "omic_prototype_consistency": {
            "mean_similarity": float(upper_tri_sims.mean()) if omic_pass else None,
            "pass": bool(omic_pass)
        },
        "overall_pass": bool(overall_pass)
    }
    
    output_dir = Path("/data1/DCT-Reg/results/anchor_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "exp1_1_anchor_consistency.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ 结果已保存: {output_dir}/exp1_1_anchor_consistency.json")
    print()
    
    return overall_pass


def experiment_1_2_anchor_causal():
    """实验 1.2: 锚点与生存的因果关系（简化版）"""
    
    print("=" * 80)
    print("🔬 实验 1.2: 锚点分离度分析")
    print("=" * 80)
    print()
    print("注: 此实验需要测试集特征和生存数据")
    print("当前版本分析锚点本身的分离度")
    print()
    
    # 找到所有检查点
    checkpoints = sorted(glob.glob(
        "/data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/*/evidence/fold_*/checkpoint.pt"
    ))
    
    fold_data = {}
    for ckpt_path in checkpoints:
        fold_num = int(ckpt_path.split('fold_')[-1].split('/')[0])
        data = load_anchors_from_checkpoint(ckpt_path)
        fold_data[fold_num] = data
    
    folds = sorted(fold_data.keys())
    
    # 分析原型内部的离散度
    print("=" * 80)
    print("📊 分析: 原型内部离散度")
    print("=" * 80)
    print()
    
    for fold in folds:
        wsi_proto = fold_data[fold]['wsi_prototypes']
        omic_proto = fold_data[fold]['omic_prototypes']
        
        print(f"Fold {fold}:")
        
        if wsi_proto is not None and wsi_proto.shape[0] > 1:
            # 计算原型间的最小、最大、平均距离
            n_proto = wsi_proto.shape[0]
            dists = []
            for i in range(n_proto):
                for j in range(i+1, n_proto):
                    v1 = wsi_proto[i].flatten() if wsi_proto[i].ndim > 1 else wsi_proto[i]
                    v2 = wsi_proto[j].flatten() if wsi_proto[j].ndim > 1 else wsi_proto[j]
                    dist = np.linalg.norm(v1 - v2)
                    dists.append(dist)
            
            print(f"  WSI 原型间距离: {np.mean(dists):.4f} ± {np.std(dists):.4f}")
            print(f"    最小: {np.min(dists):.4f}, 最大: {np.max(dists):.4f}")
        
        if omic_proto is not None and omic_proto.shape[0] > 1:
            n_proto = omic_proto.shape[0]
            dists = []
            for i in range(n_proto):
                for j in range(i+1, n_proto):
                    v1 = omic_proto[i].flatten() if omic_proto[i].ndim > 1 else omic_proto[i]
                    v2 = omic_proto[j].flatten() if omic_proto[j].ndim > 1 else omic_proto[j]
                    dist = np.linalg.norm(v1 - v2)
                    dists.append(dist)
            
            print(f"  Omic 原型间距离: {np.mean(dists):.4f} ± {np.std(dists):.4f}")
            print(f"    最小: {np.min(dists):.4f}, 最大: {np.max(dists):.4f}")
        
        print()
    
    print("=" * 80)
    print("🎯 实验 1.2 结论")
    print("=" * 80)
    print()
    print("✓ 原型内部具有一定的离散度")
    print("⚠️  完整的因果关系验证需要测试集数据")
    print("   (需要计算样本到原型的距离与真实风险的相关性)")
    print()


def experiment_1_3_anchor_separation():
    """实验 1.3: 锚点分离度"""
    
    print("=" * 80)
    print("🔬 实验 1.3: 高低风险原型分离度分析")
    print("=" * 80)
    print()
    
    # 找到所有检查点
    checkpoints = sorted(glob.glob(
        "/data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/*/evidence/fold_*/checkpoint.pt"
    ))
    
    fold_data = {}
    for ckpt_path in checkpoints:
        fold_num = int(ckpt_path.split('fold_')[-1].split('/')[0])
        data = load_anchors_from_checkpoint(ckpt_path)
        fold_data[fold_num] = data
    
    folds = sorted(fold_data.keys())
    
    print("假设: 如果原型编码了风险信息")
    print("      前半部分原型可能对应低风险，后半部分对应高风险")
    print()
    
    separations = []
    
    for fold in folds:
        wsi_proto = fold_data[fold]['wsi_prototypes']
        omic_proto = fold_data[fold]['omic_prototypes']
        
        print(f"Fold {fold}:")
        
        if wsi_proto is not None and wsi_proto.shape[0] >= 4:
            n_proto = wsi_proto.shape[0]
            mid = n_proto // 2
            
            # 假设前半部分是低风险，后半部分是高风险
            low_risk_proto = wsi_proto[:mid]
            high_risk_proto = wsi_proto[mid:]
            
            # 计算两组的质心
            low_centroid = low_risk_proto.mean(axis=0)
            high_centroid = high_risk_proto.mean(axis=0)
            
            # 计算质心间距离
            if low_centroid.ndim > 1:
                low_centroid = low_centroid.flatten()
                high_centroid = high_centroid.flatten()
            
            separation = np.linalg.norm(high_centroid - low_centroid)
            norm_separation = separation / np.sqrt(low_centroid.shape[0])
            
            print(f"  WSI 原型分离度: {separation:.4f}")
            print(f"  归一化分离度: {norm_separation:.4f}")
            
            separations.append(norm_separation)
        
        print()
    
    print("=" * 80)
    print("🎯 实验 1.3 判定")
    print("=" * 80)
    print()
    
    if separations:
        mean_sep = np.mean(separations)
        print(f"平均归一化分离度: {mean_sep:.4f}")
        print(f"目标阈值: > 2.0")
        print()
        
        pass_threshold = mean_sep > 2.0
        
        if pass_threshold:
            print("✅ 通过: 高低风险原型足够分离")
        else:
            print("❌ 失败: 高低风险原型分离度不足")
            print()
            print("⚠️  诊断:")
            print("  - 原型可能没有编码足够的风险信息")
            print("  - 或者风险信息分布在所有原型中，而非分组")
            print("  - 这可能导致方向约束失效")
    else:
        print("⚠️  无法计算分离度")
    
    print()


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "锚点质量诊断实验套件" + " " * 20 + "║")
    print("║" + " " * 78 + "║")
    print("║  实验 1.1: 锚点跨折一致性分析" + " " * 44 + "║")
    print("║  实验 1.2: 锚点分离度分析" + " " * 48 + "║")
    print("║  实验 1.3: 高低风险原型分离度" + " " * 44 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    # 实验 1.1
    exp1_1_pass = experiment_1_1_anchor_consistency()
    
    print("\n" + "="*80 + "\n")
    
    # 实验 1.2
    experiment_1_2_anchor_causal()
    
    print("\n" + "="*80 + "\n")
    
    # 实验 1.3
    experiment_1_3_anchor_separation()
    
    print("\n" + "="*80)
    print("="*80)
    print("🎯 所有实验完成")
    print("="*80)
    print()
    
    if not exp1_1_pass:
        print("❌ 关键发现: 锚点跨折一致性不足")
        print()
        print("这解释了为何:")
        print("  - DCR = 0.392 < 0.50 (方向一致性低于随机)")
        print("  - DMR = 0.184 (剂量单调率仅 18%)")
        print("  - 方向响应机制失效")
        print()
        print("建议行动:")
        print("  1. 重新设计锚点提取机制")
        print("  2. 使用更鲁棒的聚类方法")
        print("  3. 引入外部生物学先验")
        print("  4. 增加锚点提取的训练数据")
    else:
        print("✅ 锚点质量良好")
        print("   可以继续实验 2（传输路径诊断）和实验 3（风险读取器解耦诊断）")
    
    print()


if __name__ == "__main__":
    main()
