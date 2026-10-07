#!/usr/bin/env python3
"""可视化E4连续干预审计结果的脚本"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def plot_intervention_curves(
    df: pd.DataFrame,
    output_dir: Path,
    study: str,
    fold: int,
    num_examples: int = 10
):
    """绘制干预曲线：α vs 预测风险"""
    
    # 随机选择一些患者展示
    patients = df['patient_id'].unique()
    np.random.seed(42)
    example_patients = np.random.choice(patients, min(num_examples, len(patients)), replace=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 低风险方向
    ax = axes[0]
    for pid in example_patients:
        patient_df = df[(df['patient_id'] == pid) & (df['direction'] == 'low_risk')]
        patient_df = patient_df.sort_values('alpha')
        ax.plot(patient_df['alpha'], patient_df['risk_pred'], 
                marker='o', alpha=0.6, linewidth=2)
    
    ax.set_xlabel('Intervention Strength α (towards Low-Risk Anchor)', fontsize=12)
    ax.set_ylabel('Predicted Risk Score', fontsize=12)
    ax.set_title(f'Intervention towards Low-Risk Anchor\n{study.upper()} Fold {fold} (n={num_examples} patients)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 高风险方向
    ax = axes[1]
    for pid in example_patients:
        patient_df = df[(df['patient_id'] == pid) & (df['direction'] == 'high_risk')]
        patient_df = patient_df.sort_values('alpha')
        ax.plot(patient_df['alpha'], patient_df['risk_pred'], 
                marker='o', alpha=0.6, linewidth=2)
    
    ax.set_xlabel('Intervention Strength α (towards High-Risk Anchor)', fontsize=12)
    ax.set_ylabel('Predicted Risk Score', fontsize=12)
    ax.set_title(f'Intervention towards High-Risk Anchor\n{study.upper()} Fold {fold} (n={num_examples} patients)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{study}_fold{fold}_intervention_curves.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✅ 已保存干预曲线图: {output_dir / f'{study}_fold{fold}_intervention_curves.png'}")


def plot_aggregated_trends(df: pd.DataFrame, output_dir: Path, study: str, fold: int):
    """绘制聚合趋势：所有患者的平均风险变化"""
    
    # 计算每个α的平均风险和置信区间
    summary_low = df[df['direction'] == 'low_risk'].groupby('alpha')['risk_pred'].agg(['mean', 'std', 'sem'])
    summary_high = df[df['direction'] == 'high_risk'].groupby('alpha')['risk_pred'].agg(['mean', 'std', 'sem'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 低风险方向
    ax.plot(summary_low.index, summary_low['mean'], 
            marker='o', linewidth=3, markersize=8, label='Towards Low-Risk Anchor', color='#22D3EE')
    ax.fill_between(summary_low.index, 
                     summary_low['mean'] - 1.96 * summary_low['sem'],
                     summary_low['mean'] + 1.96 * summary_low['sem'],
                     alpha=0.2, color='#22D3EE')
    
    # 高风险方向
    ax.plot(summary_high.index, summary_high['mean'], 
            marker='s', linewidth=3, markersize=8, label='Towards High-Risk Anchor', color='#F97316')
    ax.fill_between(summary_high.index, 
                     summary_high['mean'] - 1.96 * summary_high['sem'],
                     summary_high['mean'] + 1.96 * summary_high['sem'],
                     alpha=0.2, color='#F97316')
    
    ax.set_xlabel('Intervention Strength α', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Predicted Risk Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Directional Consistency: Mean Risk vs Intervention Strength\n{study.upper()} Fold {fold}', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=summary_low.loc[0.0, 'mean'], color='gray', linestyle='--', 
               alpha=0.5, label='Baseline (α=0)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{study}_fold{fold}_aggregated_trends.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✅ 已保存聚合趋势图: {output_dir / f'{study}_fold{fold}_aggregated_trends.png'}")


def plot_direction_consistency_metrics(df: pd.DataFrame, output_dir: Path, study: str, fold: int):
    """绘制方向一致性指标"""
    
    patients = df['patient_id'].unique()
    
    # 计算每个患者的单调性
    monotonic_low = []
    monotonic_high = []
    
    for pid in patients:
        patient_df = df[df['patient_id'] == pid].sort_values('alpha')
        
        # 低风险方向：风险应该下降
        low_risk_df = patient_df[patient_df['direction'] == 'low_risk']
        risks = low_risk_df['risk_pred'].values
        is_monotonic_low = all(risks[i] >= risks[i+1] for i in range(len(risks)-1)) if len(risks) > 1 else False
        monotonic_low.append(is_monotonic_low)
        
        # 高风险方向：风险应该上升
        high_risk_df = patient_df[patient_df['direction'] == 'high_risk']
        risks = high_risk_df['risk_pred'].values
        is_monotonic_high = all(risks[i] <= risks[i+1] for i in range(len(risks)-1)) if len(risks) > 1 else False
        monotonic_high.append(is_monotonic_high)
    
    # 绘制条形图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = {
        'Towards\nLow-Risk': np.mean(monotonic_low) * 100,
        'Towards\nHigh-Risk': np.mean(monotonic_high) * 100
    }
    
    bars = ax.bar(metrics.keys(), metrics.values(), 
                   color=['#22D3EE', '#F97316'], alpha=0.8, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Monotonic Response Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Direction Consistency: Monotonic Response Rates\n{study.upper()} Fold {fold}', 
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{study}_fold{fold}_monotonicity.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✅ 已保存单调性图: {output_dir / f'{study}_fold{fold}_monotonicity.png'}")


def main():
    parser = argparse.ArgumentParser(description="可视化E4干预审计结果")
    parser.add_argument('--input', type=str, required=True,
                        help='E4实验输出的CSV文件')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='可视化输出目录')
    parser.add_argument('--study', type=str, required=True,
                        help='癌症类型')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold编号')
    parser.add_argument('--num-examples', type=int, default=10,
                        help='展示的样例患者数量')
    
    args = parser.parse_args()
    
    # 读取数据
    df = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("E4 干预审计可视化")
    print("="*80)
    print(f"输入: {args.input}")
    print(f"输出目录: {output_dir}")
    print(f"数据: {len(df)} 行, {len(df['patient_id'].unique())} 患者")
    print("="*80)
    
    # 生成可视化
    plot_intervention_curves(df, output_dir, args.study, args.fold, args.num_examples)
    plot_aggregated_trends(df, output_dir, args.study, args.fold)
    plot_direction_consistency_metrics(df, output_dir, args.study, args.fold)
    
    print("\n✅ 所有可视化已完成！")


if __name__ == '__main__':
    main()
