#!/usr/bin/env python3
"""
理想锚点验证实验

目的：用高质量的预定义锚点证明方向传输机制本身是有效的
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter
import pickle
import torch
import argparse
from typing import Dict, List, Tuple

def extract_survival_based_anchors(data_csv, omic_features_dir, wsi_features_dir, k=2):
    """
    方法1：基于生存时间聚类提取锚点
    
    这样提取的锚点应该天然具有风险分组特性
    """
    df = pd.read_csv(data_csv)
    
    # 加载所有样本的特征
    all_omic_features = []
    all_wsi_features = []
    valid_samples = []
    
    print(f"加载特征从 {len(df)} 个样本...")
    for idx, row in df.iterrows():
        case_id = row['case_id']
        
        # 加载 omic 特征
        omic_file = Path(omic_features_dir) / f"{case_id}.pt"
        wsi_file = Path(wsi_features_dir) / f"{case_id}.pt"
        
        if omic_file.exists() and wsi_file.exists():
            omic_feat = torch.load(omic_file, map_location='cpu')
            wsi_feat = torch.load(wsi_file, map_location='cpu')
            
            # 处理不同形状
            if omic_feat.dim() > 1:
                omic_feat = omic_feat.mean(dim=0)
            if wsi_feat.dim() > 1:
                wsi_feat = wsi_feat.mean(dim=0)
            
            all_omic_features.append(omic_feat.numpy())
            all_wsi_features.append(wsi_feat.numpy())
            valid_samples.append({
                'case_id': case_id,
                'survival_months': row['survival_months'],
                'censorship': row['censorship']
            })
    
    print(f"✓ 成功加载 {len(valid_samples)} 个样本")
    all_omic_features = np.array(all_omic_features)
    all_wsi_features = np.array(all_wsi_features)
    
    # 使用 K-means 在特征空间聚类
    # 然后按聚类的平均生存时间排序，定义高低风险
    print(f"对 Omic 特征聚类 (dim={all_omic_features.shape[1]})...")
    kmeans_omic = KMeans(n_clusters=k, random_state=42, n_init=50)
    clusters_omic = kmeans_omic.fit_predict(all_omic_features)
    
    print(f"对 WSI 特征聚类 (dim={all_wsi_features.shape[1]})...")
    kmeans_wsi = KMeans(n_clusters=k, random_state=42, n_init=50)
    clusters_wsi = kmeans_wsi.fit_predict(all_wsi_features)
    
    # 计算每个聚类的中位生存时间
    samples_df = pd.DataFrame(valid_samples)
    samples_df['cluster_omic'] = clusters_omic
    samples_df['cluster_wsi'] = clusters_wsi
    
    cluster_survival_omic = []
    cluster_survival_wsi = []
    
    print("\n=== Omic 聚类生存分析 ===")
    for i in range(k):
        mask_omic = samples_df['cluster_omic'] == i
        median_surv_omic = samples_df[mask_omic]['survival_months'].median()
        n_samples = mask_omic.sum()
        cluster_survival_omic.append((i, median_surv_omic))
        print(f"  簇 {i}: n={n_samples}, 中位生存={median_surv_omic:.1f}月")
    
    print("\n=== WSI 聚类生存分析 ===")
    for i in range(k):
        mask_wsi = samples_df['cluster_wsi'] == i
        median_surv_wsi = samples_df[mask_wsi]['survival_months'].median()
        n_samples = mask_wsi.sum()
        cluster_survival_wsi.append((i, median_surv_wsi))
        print(f"  簇 {i}: n={n_samples}, 中位生存={median_surv_wsi:.1f}月")
    
    # 按生存时间排序：生存短的=高风险，生存长的=低风险
    cluster_survival_omic.sort(key=lambda x: x[1])  # 升序
    cluster_survival_wsi.sort(key=lambda x: x[1])
    
    print(f"\n✓ Omic 风险排序: {[c[0] for c in cluster_survival_omic]} (低→高生存)")
    print(f"✓ WSI 风险排序: {[c[0] for c in cluster_survival_wsi]}")
    
    # 重新排列质心：index 0 = 高风险，index 1 = 低风险
    omic_anchors_sorted = kmeans_omic.cluster_centers_[[c[0] for c in cluster_survival_omic]]
    wsi_anchors_sorted = kmeans_wsi.cluster_centers_[[c[0] for c in cluster_survival_wsi]]
    
    return {
        'omic_anchors': omic_anchors_sorted,  # shape: (k, omic_dim)
        'wsi_anchors': wsi_anchors_sorted,    # shape: (k, wsi_dim)
        'cluster_info_omic': cluster_survival_omic,
        'cluster_info_wsi': cluster_survival_wsi,
        'n_samples': len(valid_samples)
    }


def extract_biology_prior_anchors(gene_expr_file, pathway_db="immune_proliferation"):
    """
    方法2：使用生物学先验定义锚点
    
    例如：免疫通路 vs 增殖通路
    """
    # TODO: 实现基于通路数据库的锚点提取
    # 可以用 GSEA、MSigDB 等
    pass


def validate_anchor_quality(anchors, folds_data):
    """
    验证锚点质量：
    1. 跨折一致性
    2. 风险分离度
    """
    # 在每个 fold 上提取锚点，计算相似度
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="提取基于生存时间的高质量锚点"
    )
    parser.add_argument("--data_csv", required=True, 
                        help="数据CSV路径，如 data/dataset_csv/blca_all_clean.csv")
    parser.add_argument("--omic_dir", required=True,
                        help="Omic 特征目录")
    parser.add_argument("--wsi_dir", required=True,
                        help="WSI 特征目录")
    parser.add_argument("--output", default="ideal_anchors_blca.pkl",
                        help="输出锚点文件路径")
    parser.add_argument("--k", type=int, default=2,
                        help="聚类数量（默认2=高/低风险）")
    args = parser.parse_args()
    
    print("=" * 70)
    print("理想锚点提取实验 - 基于生存时间聚类")
    print("=" * 70)
    print(f"数据: {args.data_csv}")
    print(f"Omic: {args.omic_dir}")
    print(f"WSI: {args.wsi_dir}")
    print(f"输出: {args.output}")
    print()
    
    anchors = extract_survival_based_anchors(
        args.data_csv, 
        args.omic_dir, 
        args.wsi_dir,
        k=args.k
    )
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(anchors, f)
    
    print("\n" + "=" * 70)
    print(f"✓ 锚点已保存到 {output_path}")
    print(f"  Omic anchors: {anchors['omic_anchors'].shape}")
    print(f"  WSI anchors: {anchors['wsi_anchors'].shape}")
    print(f"  样本数: {anchors['n_samples']}")
    print("=" * 70)
