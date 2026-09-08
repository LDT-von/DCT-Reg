#!/usr/bin/env python3
"""
使用 final_50ep_old 的 best epoch checkpoints 重新运行审计协议

这个脚本：
1. 读取 final_50ep_old 中的 best epoch checkpoints (model_best_s*.pth)
2. 使用 DCT-Audit 框架重新评估这些 checkpoints
3. 生成正确的审计指标（DCR/DMR/Plan TV）

Usage:
    python scripts/rerun_audit_on_best_checkpoints.py --cancer blca
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

def find_best_checkpoints(cancer: str) -> Dict[int, Path]:
    """找到 final_50ep_old 中的 best epoch checkpoints"""
    
    base_dir = Path("/data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old")
    cancer_dir = base_dir / cancer / cancer / "SurvOTRank_dct_v310_directional_regularized_transport"
    
    # 找到参数目录
    param_dirs = list(cancer_dir.glob("0.0005_*"))
    if not param_dirs:
        raise FileNotFoundError(f"No parameter directory found in {cancer_dir}")
    
    exp_dir = param_dirs[0]
    
    # 收集所有 model_best_s*.pth
    checkpoints = {}
    for fold in range(5):
        ckpt = exp_dir / f"model_best_s{fold}.pth"
        if ckpt.exists():
            checkpoints[fold] = ckpt
            print(f"✓ Found checkpoint for fold {fold}: {ckpt.name}")
        else:
            print(f"✗ Missing checkpoint for fold {fold}")
    
    return checkpoints

def create_audit_command(
    cancer: str,
    fold: int,
    checkpoint_path: Path,
    output_dir: Path
) -> List[str]:
    """生成运行审计的命令"""
    
    cmd = [
        sys.executable,
        "/data1/DCT-Reg/survot_rank/cli.py",
        "train",
        "--config", f"configs/dct_v310_directional_regularized_transport.yaml",
        "--set", "survot_method=dct_v310_directional_regularized_transport",
        "--set", "bag_loss=nll_surv",
        "--set", "max_epochs=0",  # 不训练，只评估
        "--set", "outer_eval_only=true",
        "--set", "formal_evidence_package=true",
        "--set", f"resume_checkpoint={checkpoint_path}",
        "--set", f"split={fold}",
        "--set", f"cancer={cancer}",
        "--set", "which_splits=5fold_uni2h",
        "--set", f"results_dir={output_dir}",
        # DCT v3.10 参数
        "--set", "dct_lambda_ipcw_rank=0.1",
        "--set", "dct_v38_lambda_direction=0.05",
        "--set", "dct_v38_lambda_dose=0.0",
        "--set", "dct_v38_lambda_reconfiguration=0.0",
        "--set", "dct_v38_warmup_epochs=0",
        "--set", "dct_v38_ramp_epochs=0",
        "--set", "dct_lambda_etar=0.0",
        "--set", "dct_lambda_listwise=0.0",
        "--set", "dct_mix_ratio=1.0",
        "--set", "dct_slot_init_mode=deterministic",
        "--set", "fit_bins_on_train=true",
        "--set", "binning_mode=global_qcut",
    ]
    
    return cmd

def run_audit_for_fold(cancer: str, fold: int, checkpoint: Path, output_dir: Path):
    """为单个 fold 运行审计协议"""
    
    print(f"\n{'='*80}")
    print(f"Running audit for {cancer.upper()} fold {fold}")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output dir: {output_dir}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成命令
    cmd = create_audit_command(cancer, fold, checkpoint, output_dir)
    
    # 运行命令
    print(f"\nRunning command:")
    print(" ".join(str(c) for c in cmd))
    print()
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/data1/DCT-Reg",
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"✓ Fold {fold} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Fold {fold} failed with error:")
        print(e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Rerun audit on best epoch checkpoints")
    parser.add_argument("--cancer", required=True, help="Cancer type (e.g., blca)")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)),
                       help="Folds to process (default: all)")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("/data1/DCT-Reg/results/dct_v3.10_audit_corrected"),
                       help="Output directory for audit results")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without running")
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"重新运行 {args.cancer.upper()} 的审计协议（使用 best epoch checkpoints）")
    print("="*80)
    print()
    
    # 查找 checkpoints
    try:
        checkpoints = find_best_checkpoints(args.cancer)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return 1
    
    if not checkpoints:
        print("✗ No checkpoints found")
        return 1
    
    print(f"\nFound {len(checkpoints)} checkpoints")
    
    # 为每个 fold 运行审计
    output_cancer_dir = args.output_dir / args.cancer
    results = {}
    
    for fold in args.folds:
        if fold not in checkpoints:
            print(f"\n⚠️  Skipping fold {fold}: checkpoint not found")
            continue
        
        fold_output = output_cancer_dir / f"fold_{fold}"
        
        if args.dry_run:
            cmd = create_audit_command(args.cancer, fold, checkpoints[fold], fold_output)
            print(f"\nFold {fold} command:")
            print(" ".join(str(c) for c in cmd))
            results[fold] = "dry-run"
        else:
            success = run_audit_for_fold(
                args.cancer,
                fold,
                checkpoints[fold],
                fold_output
            )
            results[fold] = "success" if success else "failed"
    
    # 汇总结果
    print("\n" + "="*80)
    print("汇总")
    print("="*80)
    
    for fold, status in results.items():
        status_icon = "✓" if status == "success" else "✗" if status == "failed" else "○"
        print(f"{status_icon} Fold {fold}: {status}")
    
    success_count = sum(1 for s in results.values() if s == "success")
    total_count = len(results)
    
    print(f"\n完成: {success_count}/{total_count} folds")
    
    if not args.dry_run and success_count > 0:
        summary_file = args.output_dir / f"{args.cancer}_audit_summary.json"
        summary = {
            'cancer': args.cancer,
            'total_folds': total_count,
            'successful_folds': success_count,
            'checkpoint_source': 'final_50ep_old',
            'output_dir': str(args.output_dir),
            'results': results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Summary saved to: {summary_file}")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())
