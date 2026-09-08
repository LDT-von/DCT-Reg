#!/usr/bin/env python3
"""
分析固定锚点实验的结果

对比：
1. DCT v3.10 原始版本（学习锚点）
2. DCT v3.10 固定锚点版本（从原始版本提取的锚点）

关键问题：固定锚点后，DCR 和 DMR 是否提升？
"""

import sys
import pickle
import torch
import pandas as pd
from pathlib import Path
import numpy as np


def load_audit_results(result_path):
    """加载审计结果"""
    if not Path(result_path).exists():
        return None
    
    with open(result_path, 'rb') as f:
        return pickle.load(f)


def extract_direction_metrics(audit_data):
    """从审计数据中提取方向一致性指标"""
    if audit_data is None:
        return None
    
    metrics = {}
    
    # 提取 DCR (Direction Consistency Rate)
    if 'dcr_per_stage' in audit_data:
        dcr_per_stage = audit_data['dcr_per_stage']
        metrics['dcr_mean'] = np.mean(dcr_per_stage)
        metrics['dcr_std'] = np.std(dcr_per_stage)
        metrics['dcr_per_stage'] = dcr_per_stage
    
    # 提取 DMR (Direction Margin Rate)
    if 'dmr_per_stage' in audit_data:
        dmr_per_stage = audit_data['dmr_per_stage']
        metrics['dmr_mean'] = np.mean(dmr_per_stage)
        metrics['dmr_std'] = np.std(dmr_per_stage)
        metrics['dmr_per_stage'] = dmr_per_stage
    
    # 提取高低风险增益
    if 'high_risk_gain' in audit_data:
        metrics['high_risk_gain_mean'] = np.mean(audit_data['high_risk_gain'])
        metrics['low_risk_gain_mean'] = np.mean(audit_data['low_risk_gain'])
    
    return metrics


def compare_experiments(original_dir, fixed_anchor_dir, cancer='blca', fold=0):
    """对比两个实验的结果"""
    
    print("=" * 70)
    print(f"对比实验：{cancer} fold{fold}")
    print("=" * 70)
    
    # 1. 查找原始版本的审计结果
    original_pattern = f"{original_dir}/**/fold_{fold}/*audit*.pkl"
    import glob
    original_files = glob.glob(original_pattern, recursive=True)
    
    print(f"\n原始版本审计文件:")
    for f in original_files[:5]:
        print(f"  {f}")
    
    # 2. 查找固定锚点版本的审计结果
    fixed_pattern = f"{fixed_anchor_dir}/**/fold_{fold}/*audit*.pkl"
    fixed_files = glob.glob(fixed_pattern, recursive=True)
    
    print(f"\n固定锚点版本审计文件:")
    for f in fixed_files[:5]:
        print(f"  {f}")
    
    # 3. 加载并对比
    if original_files:
        original_audit = load_audit_results(original_files[0])
        original_metrics = extract_direction_metrics(original_audit)
    else:
        print("\n⚠️ 未找到原始版本的审计结果")
        original_metrics = None
    
    if fixed_files:
        fixed_audit = load_audit_results(fixed_files[0])
        fixed_metrics = extract_direction_metrics(fixed_audit)
    else:
        print("\n⚠️ 未找到固定锚点版本的审计结果")
        fixed_metrics = None
    
    # 4. 打印对比结果
    print("\n" + "=" * 70)
    print("方向一致性对比")
    print("=" * 70)
    
    if original_metrics and fixed_metrics:
        print(f"\n{'指标':<30} {'原始版本':>15} {'固定锚点':>15} {'变化':>15}")
        print("-" * 75)
        
        if 'dcr_mean' in original_metrics and 'dcr_mean' in fixed_metrics:
            dcr_orig = original_metrics['dcr_mean']
            dcr_fixed = fixed_metrics['dcr_mean']
            dcr_change = dcr_fixed - dcr_orig
            print(f"{'DCR (方向一致性率)':<30} {dcr_orig:>15.4f} {dcr_fixed:>15.4f} {dcr_change:>+15.4f}")
        
        if 'dmr_mean' in original_metrics and 'dmr_mean' in fixed_metrics:
            dmr_orig = original_metrics['dmr_mean']
            dmr_fixed = fixed_metrics['dmr_mean']
            dmr_change = dmr_fixed - dmr_orig
            print(f"{'DMR (方向边际率)':<30} {dmr_orig:>15.4f} {dmr_fixed:>15.4f} {dmr_change:>+15.4f}")
        
        if 'high_risk_gain_mean' in original_metrics and 'high_risk_gain_mean' in fixed_metrics:
            hr_orig = original_metrics['high_risk_gain_mean']
            hr_fixed = fixed_metrics['high_risk_gain_mean']
            hr_change = hr_fixed - hr_orig
            print(f"{'高风险增益':<30} {hr_orig:>15.4f} {hr_fixed:>15.4f} {hr_change:>+15.4f}")
        
        if 'low_risk_gain_mean' in original_metrics and 'low_risk_gain_mean' in fixed_metrics:
            lr_orig = original_metrics['low_risk_gain_mean']
            lr_fixed = fixed_metrics['low_risk_gain_mean']
            lr_change = lr_fixed - lr_orig
            print(f"{'低风险增益':<30} {lr_orig:>15.4f} {lr_fixed:>15.4f} {lr_change:>+15.4f}")
        
        print("\n" + "=" * 70)
        print("结论")
        print("=" * 70)
        
        if 'dcr_mean' in original_metrics and 'dcr_mean' in fixed_metrics:
            if dcr_fixed > dcr_orig + 0.05:
                print("✅ 固定锚点显著提升了方向一致性！")
                print("   → 证明：方向传输机制有效，问题在于锚点质量")
            elif dcr_fixed > dcr_orig:
                print("⚠️ 固定锚点略微提升了方向一致性")
                print("   → 可能需要更高质量的锚点")
            else:
                print("❌ 固定锚点没有提升方向一致性")
                print("   → 需要重新审视方向传输机制或锚点定义")
    
    else:
        print("\n⚠️ 缺少对比数据，无法生成报告")
    
    return original_metrics, fixed_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="分析固定锚点实验结果"
    )
    parser.add_argument("--original-dir", 
                        default="results/backups/direction_only_frozen_bug_20260903_173509",
                        help="原始版本结果目录")
    parser.add_argument("--fixed-dir",
                        default="results_fixed_anchors",
                        help="固定锚点版本结果目录")
    parser.add_argument("--cancer", default="blca")
    parser.add_argument("--fold", type=int, default=0)
    
    args = parser.parse_args()
    
    compare_experiments(
        args.original_dir,
        args.fixed_dir,
        args.cancer,
        args.fold
    )


if __name__ == "__main__":
    main()
