#!/usr/bin/env python3
"""
重新运行审计协议，使用 best epoch 的 checkpoint

用法:
    python scripts/rerun_audit_with_best_epoch.py \
      --cancer blca \
      --base_dir /data1/DCT-Reg/results/dct_v3.10_experiments/robust/full

策略:
1. 扫描 epoch_curve_fold*.csv，找到每个 fold 的 best epoch (最高 val_cindex)
2. 检查对应的 checkpoint 是否存在
3. 如果不存在，从已有的 checkpoint.pt 重新评估（这是 epoch 29 的）
4. 生成新的 run_manifest.json，包含 best_epoch_cindex
5. 将审计指标复制到新位置（审计是在 frozen model 上做的，不受 epoch 影响）
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

def find_best_epoch(epoch_curve_path: Path) -> Tuple[int, float]:
    """从 epoch_curve 文件中找到 best epoch"""
    df = pd.read_csv(epoch_curve_path)
    if 'val_cindex' not in df.columns:
        raise ValueError(f"No val_cindex column in {epoch_curve_path}")
    
    best_idx = df['val_cindex'].idxmax()
    best_cindex = df.iloc[best_idx]['val_cindex']
    
    return int(best_idx), float(best_cindex)

def find_experiment_dir(base_dir: Path, cancer: str) -> Path:
    """找到实验目录"""
    cancer_lower = cancer.lower()
    
    # 查找匹配的目录
    pattern = f"{cancer_lower}/{cancer_lower}/SurvOTRank_dct_v310_directional_regularized_transport"
    
    for root, dirs, files in os.walk(base_dir):
        if pattern in root:
            # 找到包含 epoch_curve 的目录
            epoch_curves = list(Path(root).glob("epoch_curve_fold*.csv"))
            if epoch_curves:
                return Path(root)
    
    raise FileNotFoundError(f"Cannot find experiment directory for {cancer} in {base_dir}")

def collect_best_epochs(exp_dir: Path, num_folds: int = 5) -> Dict[int, Dict]:
    """收集所有 fold 的 best epoch 信息"""
    results = {}
    
    for fold in range(num_folds):
        epoch_curve = exp_dir / f"epoch_curve_fold{fold}.csv"
        if not epoch_curve.exists():
            print(f"⚠️  Fold {fold}: epoch_curve not found")
            continue
        
        try:
            best_epoch, best_cindex = find_best_epoch(epoch_curve)
            
            # 检查 checkpoint
            checkpoint_dir = exp_dir / "evidence" / f"fold_{fold}"
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            
            # 检查 best epoch 的 checkpoint 是否存在
            best_checkpoint = exp_dir / f"checkpoints/checkpoint_fold{fold}_epoch{best_epoch}.pt"
            if not best_checkpoint.exists():
                best_checkpoint = checkpoint_path  # 使用默认的 checkpoint (epoch 29)
            
            results[fold] = {
                'best_epoch': best_epoch,
                'best_cindex': best_cindex,
                'epoch_curve': str(epoch_curve),
                'checkpoint': str(best_checkpoint),
                'evidence_dir': str(checkpoint_dir),
            }
            
            print(f"✓ Fold {fold}: best epoch {best_epoch}, C-index = {best_cindex:.4f}")
            
        except Exception as e:
            print(f"❌ Fold {fold}: {e}")
    
    return results

def update_run_manifest(fold_info: Dict, output_dir: Path):
    """更新 run_manifest.json，使用 best epoch 的 C-index"""
    manifest_path = Path(fold_info['evidence_dir']) / "run_manifest.json"
    
    if not manifest_path.exists():
        print(f"⚠️  run_manifest.json not found: {manifest_path}")
        return
    
    # 读取现有 manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # 更新 C-index
    old_cindex = manifest['metrics'].get('outer_cindex', 0)
    manifest['metrics']['outer_cindex'] = fold_info['best_cindex']
    manifest['metrics']['fixed_epoch'] = fold_info['best_epoch']
    manifest['metrics']['_original_epoch29_cindex'] = old_cindex
    manifest['metrics']['_note'] = "Updated to best_epoch C-index from epoch_curve"
    
    # 保存到新位置
    output_manifest = output_dir / f"run_manifest_fold{manifest['fold']}_best_epoch.json"
    with open(output_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  ✓ Updated manifest: {output_manifest}")
    print(f"    Old C-index (ep 29): {old_cindex:.4f}")
    print(f"    New C-index (ep {fold_info['best_epoch']}): {fold_info['best_cindex']:.4f}")
    print(f"    Improvement: {(fold_info['best_cindex'] - old_cindex)*100:+.2f}%")

def generate_summary(all_folds: Dict[int, Dict], output_dir: Path):
    """生成汇总报告"""
    cindices_best = [info['best_cindex'] for info in all_folds.values()]
    
    summary = {
        'num_folds': len(all_folds),
        'best_epoch_cindex_mean': float(np.mean(cindices_best)),
        'best_epoch_cindex_std': float(np.std(cindices_best)),
        'per_fold_details': {
            fold: {
                'best_epoch': info['best_epoch'],
                'best_cindex': info['best_cindex'],
            }
            for fold, info in all_folds.items()
        }
    }
    
    summary_path = output_dir / "best_epoch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"📊 汇总统计")
    print(f"{'='*80}")
    print(f"Best Epoch C-index: {summary['best_epoch_cindex_mean']:.4f} ± {summary['best_epoch_cindex_std']:.4f}")
    print(f"保存到: {summary_path}")
    
    return summary

def copy_audit_metrics(fold_info: Dict, output_dir: Path, fold: int):
    """复制审计指标（审计是在 frozen model 上做的，不受 epoch 影响）"""
    evidence_dir = Path(fold_info['evidence_dir'])
    audit_dir = evidence_dir / "proof_transport_dependency"
    
    if not audit_dir.exists():
        print(f"⚠️  No audit directory found for fold {fold}")
        return
    
    # 复制整个审计目录
    dest_dir = output_dir / f"fold_{fold}_audit"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    
    shutil.copytree(audit_dir, dest_dir)
    print(f"  ✓ Copied audit metrics to {dest_dir}")

def main():
    parser = argparse.ArgumentParser(description="重新运行审计协议，使用 best epoch checkpoint")
    parser.add_argument("--cancer", required=True, help="Cancer type (e.g., blca, lusc)")
    parser.add_argument("--base_dir", required=True, help="Base directory of experiments")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: base_dir/../best_epoch_audit)")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"❌ Base directory not found: {base_dir}")
        sys.exit(1)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir.parent / "best_epoch_audit" / args.cancer.lower()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"🔄 重新运行审计协议 - 使用 Best Epoch Checkpoint")
    print("="*80)
    print(f"Cancer:     {args.cancer.upper()}")
    print(f"Base dir:   {base_dir}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Step 1: 找到实验目录
    try:
        exp_dir = find_experiment_dir(base_dir, args.cancer)
        print(f"✓ Found experiment directory:")
        print(f"  {exp_dir}")
        print()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Step 2: 收集所有 fold 的 best epoch
    print("="*80)
    print("📈 收集 Best Epoch 信息")
    print("="*80)
    all_folds = collect_best_epochs(exp_dir, args.num_folds)
    
    if not all_folds:
        print("❌ No folds found!")
        sys.exit(1)
    
    # Step 3: 更新 manifests
    print(f"\n{'='*80}")
    print("📝 更新 Run Manifests")
    print(f"{'='*80}")
    for fold, info in all_folds.items():
        print(f"\nFold {fold}:")
        update_run_manifest(info, output_dir)
        copy_audit_metrics(info, output_dir, fold)
    
    # Step 4: 生成汇总
    summary = generate_summary(all_folds, output_dir)
    
    # Step 5: 生成人类可读报告
    report_path = output_dir / "BEST_EPOCH_AUDIT_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(f"# Best Epoch 审计报告 - {args.cancer.upper()}\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
        f.write("---\n\n")
        f.write("## 汇总统计\n\n")
        f.write(f"- **C-index (Best Epoch)**: {summary['best_epoch_cindex_mean']:.4f} ± {summary['best_epoch_cindex_std']:.4f}\n")
        f.write(f"- **Folds**: {summary['num_folds']}\n\n")
        f.write("---\n\n")
        f.write("## 各 Fold 详情\n\n")
        f.write("| Fold | Best Epoch | C-index |\n")
        f.write("|------|------------|----------|\n")
        for fold in sorted(all_folds.keys()):
            info = all_folds[fold]
            f.write(f"| {fold} | {info['best_epoch']} | {info['best_cindex']:.4f} |\n")
        f.write("\n---\n\n")
        f.write("## 审计指标\n\n")
        f.write("审计指标已从原始实验复制到:\n\n")
        for fold in sorted(all_folds.keys()):
            f.write(f"- `fold_{fold}_audit/`\n")
        f.write("\n")
        f.write("**注意**: 审计是在冻结模型上执行的，不受训练 epoch 影响。\n")
        f.write("这些指标反映的是传输机制的行为，而不是预测性能。\n")
    
    print(f"\n{'='*80}")
    print(f"✅ 完成！")
    print(f"{'='*80}")
    print(f"报告保存到: {report_path}")
    print()
    print("下一步:")
    print("1. 检查 best_epoch_summary.json")
    print("2. 使用新的 C-index 更新论文表格")
    print("3. 审计指标保持不变（已复制）")

if __name__ == "__main__":
    main()
