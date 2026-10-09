#!/usr/bin/env python3
"""
从已训练的检查点中提取锚点

这比重新从特征聚类更简单，我们直接用模型学到的锚点
"""

import torch
import pickle
import argparse
from pathlib import Path
import numpy as np


def extract_anchors_from_checkpoint(checkpoint_path):
    """从检查点中提取锚点统计"""
    print(f"加载检查点: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 查找锚点相关的键
    anchor_keys = [k for k in state_dict.keys() if 'anchor' in k.lower()]
    
    print(f"\n找到 {len(anchor_keys)} 个锚点相关的键:")
    for k in anchor_keys:
        tensor = state_dict[k]
        print(f"  {k}: {tensor.shape}")
    
    # 提取 risk_anchor_costs (这是关键！)
    if 'risk_anchor_costs' in state_dict:
        risk_anchor_costs = state_dict['risk_anchor_costs']
        risk_anchor_seen = state_dict.get('risk_anchor_seen', None)
        
        print(f"\n✓ 提取 risk_anchor_costs: {risk_anchor_costs.shape}")
        print(f"  形状解释: [num_stages, 2, geometry, wsi_slots, omic_slots]")
        
        if risk_anchor_seen is not None:
            print(f"✓ 提取 risk_anchor_seen: {risk_anchor_seen.shape}")
            coverage = risk_anchor_seen.float().mean().item()
            print(f"  锚点覆盖率: {coverage*100:.1f}%")
        
        return {
            'risk_anchor_costs': risk_anchor_costs.numpy(),
            'risk_anchor_seen': risk_anchor_seen.numpy() if risk_anchor_seen is not None else None,
            'checkpoint_path': str(checkpoint_path)
        }
    else:
        raise KeyError("检查点中未找到 'risk_anchor_costs'")


def compute_anchor_quality(anchors):
    """计算锚点质量指标"""
    costs = torch.from_numpy(anchors['risk_anchor_costs'])
    
    # costs shape: [stages, 2, geometry, wsi_slots, omic_slots]
    num_stages = costs.shape[0]
    
    quality_metrics = {}
    
    for stage_idx in range(num_stages):
        low_risk = costs[stage_idx, 0]   # [geometry, wsi_slots, omic_slots]
        high_risk = costs[stage_idx, 1]
        
        # 计算高低风险锚点之间的分离度
        # 低风险应该有较低的 cost，高风险应该有较高的 cost
        separation = (high_risk - low_risk).mean().item()
        
        quality_metrics[f'stage{stage_idx}_separation'] = separation
    
    return quality_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从训练好的检查点中提取锚点"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="检查点路径，如 results_v310/blca/fold0/s_0_checkpoint.pt")
    parser.add_argument("--output", required=True,
                        help="输出锚点文件路径")
    args = parser.parse_args()
    
    print("=" * 70)
    print("从检查点提取锚点")
    print("=" * 70)
    
    anchors = extract_anchors_from_checkpoint(args.checkpoint)
    
    # 计算质量
    print("\n锚点质量分析:")
    quality = compute_anchor_quality(anchors)
    for metric, value in quality.items():
        print(f"  {metric}: {value:.4f}")
    
    anchors['quality_metrics'] = quality
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(anchors, f)
    
    print(f"\n✓ 锚点已保存到 {output_path}")
    print("=" * 70)
