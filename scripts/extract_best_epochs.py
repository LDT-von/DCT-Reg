#!/usr/bin/env python3
"""
从审计实验中提取每个 fold 的 best epoch 信息
生成一个 best_epochs.json 文件供后续使用
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def find_best_epoch(epoch_curve_path: Path) -> Tuple[int, float]:
    """
    从 epoch_curve CSV 中找到 val_cindex 最高的 epoch
    
    Returns:
        (best_epoch, best_cindex)
    """
    df = pd.read_csv(epoch_curve_path)
    
    if 'val_cindex' not in df.columns:
        raise ValueError(f"No val_cindex column in {epoch_curve_path}")
    
    best_idx = df['val_cindex'].idxmax()
    best_cindex = df.iloc[best_idx]['val_cindex']
    
    return int(best_idx), float(best_cindex)


def extract_best_epochs_for_experiment(base_dir: Path, cancer: str) -> Dict:
    """
    提取一个实验的所有 fold 的 best epoch
    
    Args:
        base_dir: 实验基础目录，例如 /data1/DCT-Reg/results/dct_v3.10_experiments/robust/full
        cancer: 癌种名称，例如 'blca'
    
    Returns:
        {
            'cancer': 'blca',
            'base_dir': '/path/to/base',
            'experiment_dir': '/path/to/experiment',
            'folds': {
                0: {'best_epoch': 5, 'best_cindex': 0.6950, 'checkpoint': '/path/to/checkpoint_ep5.pt'},
                1: {...},
                ...
            },
            'summary': {
                'avg_cindex': 0.7175,
                'std_cindex': 0.0xxx,
                'num_folds': 5
            }
        }
    """
    cancer_dir = base_dir / cancer / cancer
    
    if not cancer_dir.exists():
        raise FileNotFoundError(f"Cancer directory not found: {cancer_dir}")
    
    # 查找实验目录（可能有多个，取第一个）
    exp_dirs = list(cancer_dir.glob("SurvOTRank_*/"))
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directory found in {cancer_dir}")
    
    # 可能还有一层参数目录
    exp_dir = exp_dirs[0]
    param_dirs = list(exp_dir.glob("0.*/"))
    if param_dirs:
        exp_dir = param_dirs[0]
    
    print(f"Found experiment: {exp_dir.name}")
    
    # 提取每个 fold 的 best epoch
    folds_info = {}
    cindices = []
    
    for fold in range(5):
        epoch_curve_path = exp_dir / f"epoch_curve_fold{fold}.csv"
        
        if not epoch_curve_path.exists():
            print(f"  ⚠️  Fold {fold}: epoch_curve not found")
            continue
        
        try:
            best_epoch, best_cindex = find_best_epoch(epoch_curve_path)
            cindices.append(best_cindex)
            
            # 查找对应的 checkpoint
            evidence_dir = exp_dir / "evidence" / f"fold_{fold}"
            checkpoint_path = evidence_dir / f"checkpoint_ep{best_epoch}.pt"
            
            # 如果没有 per-epoch checkpoint，使用最终 checkpoint
            if not checkpoint_path.exists():
                checkpoint_path = evidence_dir / "checkpoint.pt"
                print(f"  ⚠️  Fold {fold}: Using final checkpoint (no per-epoch checkpoint found)")
            
            folds_info[fold] = {
                'best_epoch': best_epoch,
                'best_cindex': best_cindex,
                'checkpoint': str(checkpoint_path),
                'epoch_curve': str(epoch_curve_path),
                'evidence_dir': str(evidence_dir)
            }
            
            print(f"  ✓ Fold {fold}: Best epoch {best_epoch} with C-index {best_cindex:.4f}")
            
        except Exception as e:
            print(f"  ✗ Fold {fold}: Error - {e}")
            continue
    
    if not cindices:
        raise ValueError(f"No valid folds found for {cancer}")
    
    # 汇总统计
    summary = {
        'avg_cindex': float(np.mean(cindices)),
        'std_cindex': float(np.std(cindices)),
        'num_folds': len(cindices),
        'min_cindex': float(np.min(cindices)),
        'max_cindex': float(np.max(cindices))
    }
    
    result = {
        'cancer': cancer,
        'base_dir': str(base_dir),
        'experiment_dir': str(exp_dir),
        'folds': folds_info,
        'summary': summary
    }
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract best epoch info from audit experiments")
    parser.add_argument("--cancer", type=str, default="blca", 
                       help="Cancer type (default: blca)")
    parser.add_argument("--base_dir", type=str,
                       default="/data1/DCT-Reg/results/dct_v3.10_experiments/robust/full",
                       help="Base directory of audit experiments")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file (default: best_epochs_{cancer}.json)")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    print("="*80)
    print(f"提取 {args.cancer.upper()} 的 Best Epoch 信息")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print()
    
    try:
        result = extract_best_epochs_for_experiment(base_dir, args.cancer)
        
        # 保存结果
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"best_epochs_{args.cancer}.json")
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print()
        print("="*80)
        print("汇总")
        print("="*80)
        summary = result['summary']
        print(f"C-index (best epoch): {summary['avg_cindex']:.4f} ± {summary['std_cindex']:.4f}")
        print(f"Range: [{summary['min_cindex']:.4f}, {summary['max_cindex']:.4f}]")
        print(f"Valid folds: {summary['num_folds']}/5")
        print()
        print(f"✓ 结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
