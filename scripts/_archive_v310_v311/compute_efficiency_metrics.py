#!/usr/bin/env python3
"""
计算模型的计算效率指标：
1. 参数量统计
2. 训练时间提取
3. 推理时间测量（可选）
4. 显存占用（可选）
"""

import os
import sys
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

def count_parameters_from_checkpoint(ckpt_path):
    """从checkpoint统计参数量（不需要加载模型定义）"""
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        total_params = sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel'))
        trainable_params = total_params  # 假设全部可训练
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'size_mb': os.path.getsize(ckpt_path) / (1024 * 1024)
        }
    except ImportError:
        # 如果没有torch，使用文件大小估算
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        # 粗略估算：每个参数约4字节（float32），加上一些开销
        estimated_params = int(size_mb * 1024 * 1024 * 0.8 / 4)
        return {
            'total': estimated_params,
            'trainable': estimated_params,
            'size_mb': size_mb,
            'estimated': True
        }

def extract_training_time(epoch_curve_path):
    """从 epoch_curve_fold*.csv 提取训练时间"""
    if not os.path.exists(epoch_curve_path):
        return None
    
    df = pd.read_csv(epoch_curve_path)
    
    # 检查是否有 epoch_time 或 train_time 列
    time_cols = [c for c in df.columns if 'time' in c.lower() and 'epoch' in c.lower()]
    
    if time_cols:
        total_time = df[time_cols[0]].sum()
        avg_epoch_time = df[time_cols[0]].mean()
    else:
        # 如果没有时间列，返回 None
        total_time = None
        avg_epoch_time = None
    
    return {
        'total_epochs': len(df),
        'total_time_seconds': total_time,
        'avg_epoch_time_seconds': avg_epoch_time,
        'has_time_column': time_cols != []
    }

def analyze_model_efficiency(results_dir, model_pattern='model_best_s*.pth'):
    """分析模型的计算效率"""
    results = {
        'checkpoints': [],
        'training_curves': [],
        'summary': {}
    }
    
    # 查找所有 checkpoint
    results_path = Path(results_dir)
    checkpoint_files = list(results_path.rglob(model_pattern))
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    param_counts = []
    checkpoint_sizes = []
    
    for ckpt_path in checkpoint_files:
        print(f"Processing: {ckpt_path}")
        params = count_parameters_from_checkpoint(str(ckpt_path))
        
        results['checkpoints'].append({
            'path': str(ckpt_path.relative_to(results_path)),
            'params': params
        })
        
        param_counts.append(params['total'])
        checkpoint_sizes.append(params['size_mb'])
    
    # 查找所有 epoch_curve 文件
    curve_files = list(results_path.rglob('epoch_curve_fold*.csv'))
    
    print(f"\nFound {len(curve_files)} training curve files")
    
    training_times = []
    
    for curve_path in curve_files:
        print(f"Processing: {curve_path}")
        time_info = extract_training_time(str(curve_path))
        
        if time_info:
            results['training_curves'].append({
                'path': str(curve_path.relative_to(results_path)),
                'time_info': time_info
            })
            
            if time_info['total_time_seconds'] is not None:
                training_times.append(time_info['total_time_seconds'])
    
    # 生成摘要
    if param_counts:
        results['summary']['parameters'] = {
            'mean': int(np.mean(param_counts)),
            'std': int(np.std(param_counts)),
            'min': int(np.min(param_counts)),
            'max': int(np.max(param_counts)),
            'total_checkpoints': len(param_counts)
        }
        results['summary']['checkpoint_size_mb'] = {
            'mean': float(np.mean(checkpoint_sizes)),
            'std': float(np.std(checkpoint_sizes)),
            'min': float(np.min(checkpoint_sizes)),
            'max': float(np.max(checkpoint_sizes))
        }
    
    if training_times:
        results['summary']['training_time_seconds'] = {
            'mean': float(np.mean(training_times)),
            'std': float(np.std(training_times)),
            'min': float(np.min(training_times)),
            'max': float(np.max(training_times)),
            'total_curves': len(training_times)
        }
        # 转换为小时
        results['summary']['training_time_hours'] = {
            'mean': float(np.mean(training_times)) / 3600,
            'std': float(np.std(training_times)) / 3600,
            'min': float(np.min(training_times)) / 3600,
            'max': float(np.max(training_times)) / 3600
        }
    
    return results

def main():
    # DCT v3.2
    print("="*80)
    print("DCT v3.2 Efficiency Analysis")
    print("="*80)
    
    v32_results = analyze_model_efficiency('/data1/DCT-Reg/results/dct_v3.2')
    
    output_file = '/data1/DCT-Reg/results/efficiency_metrics_v32.json'
    with open(output_file, 'w') as f:
        json.dump(v32_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # 打印摘要
    if 'parameters' in v32_results['summary']:
        params = v32_results['summary']['parameters']['mean']
        print(f"\nDCT v3.2 Parameters: {params:,} ({params/1e6:.2f}M)")
    
    if 'training_time_hours' in v32_results['summary']:
        hours = v32_results['summary']['training_time_hours']['mean']
        print(f"DCT v3.2 Training Time: {hours:.2f} ± {v32_results['summary']['training_time_hours']['std']:.2f} hours")
    
    # DCT v3.10 (如果存在)
    v310_path = '/data1/DCT-Reg/results/dct_v3.10'
    if os.path.exists(v310_path):
        print("\n" + "="*80)
        print("DCT v3.10 Efficiency Analysis")
        print("="*80)
        
        v310_results = analyze_model_efficiency(v310_path)
        
        output_file = '/data1/DCT-Reg/results/efficiency_metrics_v310.json'
        with open(output_file, 'w') as f:
            json.dump(v310_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        if 'parameters' in v310_results['summary']:
            params = v310_results['summary']['parameters']['mean']
            print(f"\nDCT v3.10 Parameters: {params:,} ({params/1e6:.2f}M)")
        
        if 'training_time_hours' in v310_results['summary']:
            hours = v310_results['summary']['training_time_hours']['mean']
            print(f"DCT v3.10 Training Time: {hours:.2f} ± {v310_results['summary']['training_time_hours']['std']:.2f} hours")

if __name__ == '__main__':
    main()
