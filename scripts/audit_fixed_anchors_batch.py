#!/usr/bin/env python3
"""
批量审计固定 anchor 模型的全部 5 folds

Usage:
    python scripts/audit_fixed_anchors_batch.py --cancer blca
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

def find_fixed_anchor_checkpoints(cancer: str) -> Dict[int, tuple]:
    """找到固定 anchor 实验的 best checkpoints"""
    
    base_dir = Path("/data1/DCT-Reg/results_fixed_anchors")
    cancer_dir = base_dir / cancer / "SurvOTRank_dct_v310_fixed_anchors"
    
    # 找到参数目录
    param_dirs = list(cancer_dir.glob("0.0005_*_blca_proof"))
    if not param_dirs:
        raise FileNotFoundError(f"No parameter directory found in {cancer_dir}")
    
    exp_dir = param_dirs[0]
    print(f"📁 Found experiment directory: {exp_dir.name}")
    
    # 读取 best epochs 从日志或 epoch_curve
    checkpoints = {}
    for fold in range(5):
        epoch_curve = exp_dir / f"epoch_curve_fold{fold}.csv"
        if not epoch_curve.exists():
            print(f"✗ Fold {fold}: missing epoch_curve")
            continue
            
        # 读取 epoch curve 找到最佳 epoch
        import pandas as pd
        df = pd.read_csv(epoch_curve)
        if 'val_cindex' in df.columns:
            best_epoch = df['val_cindex'].idxmax()
            best_cindex = df['val_cindex'].iloc[best_epoch]
        elif 'cindex' in df.columns:
            best_epoch = df['cindex'].idxmax()
            best_cindex = df['cindex'].iloc[best_epoch]
        else:
            print(f"✗ Fold {fold}: cannot find cindex column")
            continue
        
        # model_best_s{fold}.pth 是最佳模型
        ckpt = exp_dir / f"model_best_s{fold}.pth"
        if ckpt.exists():
            checkpoints[fold] = (ckpt, best_epoch, best_cindex)
            print(f"✓ Fold {fold}: best cindex={best_cindex:.4f} @epoch {best_epoch}")
        else:
            print(f"✗ Fold {fold}: checkpoint not found")
    
    return checkpoints

def run_audit(cancer: str, fold: int, checkpoint: Path, output_dir: Path):
    """运行审计脚本"""
    
    output_pkl = output_dir / f"audit_fixed_fold{fold}.pkl"
    output_json = output_dir / f"audit_fixed_fold{fold}_summary.json"
    
    if output_json.exists():
        print(f"  ⏭️  Already audited, skipping")
        return True
    
    cmd = [
        "python", "scripts/e4_audit_adapted.py",
        "--checkpoint", str(checkpoint),
        "--study", cancer,
        "--fold", str(fold),
        "--output", str(output_pkl).replace('.pkl', '.csv'),
        "--alphas", "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    ]
    
    print(f"  🔍 Running audit...")
    print(f"     Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/data1/DCT-Reg",
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        if result.returncode == 0:
            print(f"  ✅ Audit completed")
            
            # 检查输出文件（改为CSV格式）
            output_csv = output_pkl.with_suffix('.csv')
            if output_csv.exists():
                print(f"     Output: {output_csv}")
                return True
            else:
                print(f"  ⚠️  Audit script succeeded but output file not found")
                return False
        else:
            print(f"  ❌ Audit failed with code {result.returncode}")
            print(f"     STDERR: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ Audit timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def summarize_results(output_dir: Path, num_folds: int):
    """汇总审计结果"""
    
    print("\n" + "="*60)
    print("📊 汇总审计结果")
    print("="*60)
    
    results = []
    for fold in range(num_folds):
        json_file = output_dir / f"audit_fixed_fold{fold}_summary.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
                results.append(data)
                
                metrics = data.get('metrics', {})
                print(f"\n📁 Fold {fold}:")
                print(f"   N patients: {metrics.get('n_patients', 'N/A')}")
                print(f"   DCR (Mon Dec): {metrics.get('monotonic_decrease_rate', 0)*100:.1f}%")
                print(f"   Mon Inc: {metrics.get('monotonic_increase_rate', 0)*100:.1f}%")
                print(f"   Mean risk change (low): {metrics.get('mean_risk_change_low', 0):.6f}")
                print(f"   Mean risk change (high): {metrics.get('mean_risk_change_high', 0):.6f}")
        else:
            print(f"\n📁 Fold {fold}: ⚠️  No results")
    
    if results:
        # 计算平均指标
        dcr_values = [r['metrics']['monotonic_decrease_rate'] for r in results if 'metrics' in r]
        inc_values = [r['metrics']['monotonic_increase_rate'] for r in results if 'metrics' in r]
        
        print(f"\n{'='*60}")
        print(f"📊 平均结果 ({len(results)} folds):")
        print(f"   Average DCR: {sum(dcr_values)/len(dcr_values)*100:.1f}%")
        print(f"   Average Mon Inc: {sum(inc_values)/len(inc_values)*100:.1f}%")
        print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description="批量审计固定 anchor 模型的全部 5 folds"
    )
    parser.add_argument("--cancer", required=True, help="Cancer type (e.g., blca)")
    parser.add_argument("--output-dir", type=Path, default=Path("results_fixed_anchors"),
                       help="Output directory for audit results")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)),
                       help="Folds to audit (default: 0 1 2 3 4)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print what would be done without running")
    
    args = parser.parse_args()
    
    print(f"🚀 批量审计固定 Anchor 模型")
    print(f"   Cancer: {args.cancer}")
    print(f"   Folds: {args.folds}")
    print(f"   Output: {args.output_dir}\n")
    
    # 查找 checkpoints
    try:
        checkpoints = find_fixed_anchor_checkpoints(args.cancer)
    except Exception as e:
        print(f"❌ Error finding checkpoints: {e}")
        return 1
    
    if not checkpoints:
        print("❌ No checkpoints found")
        return 1
    
    print(f"\n✅ Found {len(checkpoints)} checkpoints to audit\n")
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 逐个审计
    success_count = 0
    for fold in sorted(checkpoints.keys()):
        if fold not in args.folds:
            continue
            
        ckpt, best_epoch, best_cindex = checkpoints[fold]
        
        print(f"{'='*60}")
        print(f"🔍 审计 Fold {fold}")
        print(f"   Checkpoint: {ckpt.name}")
        print(f"   Best epoch: {best_epoch}")
        print(f"   Best cindex: {best_cindex:.4f}")
        print(f"{'='*60}")
        
        if args.dry_run:
            print(f"  [DRY RUN] Would audit {ckpt}")
            continue
        
        success = run_audit(args.cancer, fold, ckpt, args.output_dir)
        if success:
            success_count += 1
        
        print()
    
    # 汇总结果
    if not args.dry_run:
        summarize_results(args.output_dir, len(args.folds))
        
        print(f"\n✅ 完成！成功审计 {success_count}/{len(args.folds)} folds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
