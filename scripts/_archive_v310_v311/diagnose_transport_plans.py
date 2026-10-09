#!/usr/bin/env python3
"""
传输计划诊断脚本
快速加载 checkpoint，分析传输计划的统计特性
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import json
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️ matplotlib/seaborn not available, skipping visualization")

def calculate_entropy(plan):
    """计算传输计划的熵"""
    flat = plan.flatten()
    prob = flat / (flat.sum() + 1e-8)
    entropy = -(prob * torch.log(prob + 1e-8)).sum()
    max_entropy = np.log(len(flat))
    normalized_entropy = entropy / max_entropy
    return entropy.item(), normalized_entropy.item()

def calculate_gini(plan):
    """计算 Gini 系数（不均匀程度）"""
    flat = plan.flatten().cpu().numpy()
    flat = np.sort(flat)
    n = len(flat)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * flat)) / (n * np.sum(flat)) - (n + 1) / n
    return gini

def calculate_top_k_concentration(plan, k=10):
    """计算 Top-k 质量集中度"""
    flat = plan.flatten()
    total = flat.sum()
    topk_values, _ = torch.topk(flat, k=min(k, len(flat)))
    concentration = topk_values.sum() / (total + 1e-8)
    return concentration.item()

def analyze_single_plan(plan):
    """分析单个传输计划"""
    entropy, norm_entropy = calculate_entropy(plan)
    gini = calculate_gini(plan)
    
    stats = {
        'entropy': entropy,
        'normalized_entropy': norm_entropy,
        'gini': gini,
        'mean': plan.mean().item(),
        'std': plan.std().item(),
        'max': plan.max().item(),
        'min': plan.min().item(),
        'top1_concentration': calculate_top_k_concentration(plan, k=1),
        'top5_concentration': calculate_top_k_concentration(plan, k=5),
        'top10_concentration': calculate_top_k_concentration(plan, k=10),
        'top20_concentration': calculate_top_k_concentration(plan, k=20),
    }
    
    return stats

def visualize_transport_plan(plan, save_path, title="Transport Plan"):
    """可视化传输计划热图"""
    if not HAS_PLOTTING:
        print(f"⚠️ 跳过可视化（matplotlib/seaborn 不可用）")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 绘制热图
    sns.heatmap(plan.cpu().numpy(), 
                annot=True, 
                fmt='.4f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Transport Mass'},
                linewidths=0.5)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Omic Slots', fontsize=12)
    plt.ylabel('WSI Slots', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 热图已保存: {save_path}")

def diagnose_checkpoint(checkpoint_path, output_dir):
    """诊断 checkpoint 中的传输计划"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("传输计划诊断")
    print("=" * 80)
    
    # 1. 加载 checkpoint
    print(f"\n📂 加载 checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint 加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 2. 查找传输计划相关的参数
    print(f"\n🔍 检查 checkpoint 结构...")
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 查找所有与传输相关的参数
    transport_params = {}
    for key in state_dict.keys():
        if any(keyword in key.lower() for keyword in ['transport', 'sinkhorn', 'cost', 'plan']):
            transport_params[key] = state_dict[key]
    
    print(f"\n找到 {len(transport_params)} 个传输相关参数:")
    for key in transport_params.keys():
        print(f"  - {key}")
    
    # 3. 尝试提取传输计划（如果有保存）
    # 注意：这里我们假设需要前向传播来生成传输计划
    # 但先检查是否有直接保存的计划
    
    if 'transport_plans' in checkpoint:
        plans = checkpoint['transport_plans']
        print(f"\n✅ 找到直接保存的传输计划")
    else:
        print(f"\n⚠️ Checkpoint 中没有直接保存传输计划")
        print(f"   需要加载完整模型并在数据上前向传播")
        print(f"   继续分析模型参数...")
    
    # 4. 分析成本矩阵生成器的参数
    print(f"\n📊 分析传输模块参数统计...")
    
    param_stats = {}
    for key, param in transport_params.items():
        if param.dim() > 0:
            param_stats[key] = {
                'shape': list(param.shape),
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
            }
    
    # 保存参数统计
    stats_path = output_dir / "transport_params_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(param_stats, f, indent=2)
    
    print(f"✅ 参数统计已保存: {stats_path}")
    
    # 5. 生成诊断报告
    report = {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_keys': list(checkpoint.keys()),
        'transport_params': list(transport_params.keys()),
        'param_stats': param_stats,
        'has_saved_plans': 'transport_plans' in checkpoint,
        'status': 'need_forward_pass',
        'recommendation': 'Run forward pass on validation data to extract transport plans'
    }
    
    report_path = output_dir / "diagnosis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ 诊断报告已保存: {report_path}")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)
    
    print(f"\n💡 下一步:")
    print(f"   需要运行模型在验证集上前向传播，提取实际的传输计划")
    print(f"   使用: python scripts/extract_transport_plans.py")
    
    return report

if __name__ == "__main__":
    # 示例用法
    checkpoint_path = "/data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/ucec/ucec/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_full_ucec_50ep/evidence/fold_1/checkpoint.pt"
    
    output_dir = "/data1/DCT-Reg/transport_diagnosis"
    
    diagnose_checkpoint(checkpoint_path, output_dir)
