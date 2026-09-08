#!/usr/bin/env python3
"""
使用 final_50ep_old 的 best checkpoints 重新运行审计协议

这些 checkpoints 对应 C-index = 0.72 的模型
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

def find_best_checkpoints(cancer: str = "blca") -> dict:
    """找到 final_50ep_old 中的 best checkpoints"""
    
    base_dir = Path("/data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old")
    cancer_dir = base_dir / cancer / cancer
    
    # 找到实验目录
    exp_dirs = list(cancer_dir.glob("SurvOTRank_*/"))
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment found in {cancer_dir}")
    
    exp_dir = exp_dirs[0]
    param_dirs = list(exp_dir.glob("0.*/"))
    if param_dirs:
        exp_dir = param_dirs[0]
    
    checkpoints = {}
    for fold in range(5):
        ckpt_path = exp_dir / f"model_best_s{fold}.pth"
        if ckpt_path.exists():
            checkpoints[fold] = str(ckpt_path)
        else:
            print(f"⚠️  Fold {fold}: checkpoint not found at {ckpt_path}")
    
    return {
        'cancer': cancer,
        'exp_dir': str(exp_dir),
        'checkpoints': checkpoints
    }

def run_audit_for_fold(
    checkpoint_path: str,
    fold: int,
    config_path: str,
    output_dir: Path,
    gpu: int = 0
) -> bool:
    """对单个 fold 运行审计"""
    
    fold_output = output_dir / f"fold_{fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        "scripts/audit_dct_reg.py",
        "audit",
        "--config", config_path,
        "--checkpoint", checkpoint_path,
        "--fold", str(fold),
        "--output-dir", str(fold_output),
        "--gpu", str(gpu),
    ]
    
    print(f"\n{'='*80}")
    print(f"Running audit for fold {fold}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {fold_output}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/data1/DCT-Reg",
            check=True,
            capture_output=False
        )
        print(f"✓ Fold {fold} audit completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Fold {fold} audit failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run audit on best checkpoints from final_50ep_old"
    )
    parser.add_argument(
        "--cancer",
        default="blca",
        help="Cancer type (default: blca)"
    )
    parser.add_argument(
        "--config",
        default="configs/dct_v310_directional_regularized_transport.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--output-dir",
        default="results/audit_best_checkpoints",
        help="Output directory for audit results"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(5)),
        help="Which folds to audit (default: all 0-4)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("使用 Best Checkpoints 重新运行审计")
    print("="*80)
    print(f"Cancer: {args.cancer}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"GPU: {args.gpu}")
    print(f"Folds: {args.folds}")
    print()
    
    # 找到 checkpoints
    try:
        ckpt_info = find_best_checkpoints(args.cancer)
        print(f"Found checkpoints in: {ckpt_info['exp_dir']}")
        print(f"Available folds: {list(ckpt_info['checkpoints'].keys())}")
        print()
    except Exception as e:
        print(f"✗ Error finding checkpoints: {e}")
        return 1
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.cancer
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 checkpoint 信息
    info_file = output_dir / "checkpoint_info.json"
    with open(info_file, 'w') as f:
        json.dump(ckpt_info, f, indent=2)
    print(f"✓ Checkpoint info saved to: {info_file}\n")
    
    # 运行审计
    results = {}
    for fold in args.folds:
        if fold not in ckpt_info['checkpoints']:
            print(f"⚠️  Skipping fold {fold}: checkpoint not found")
            continue
        
        checkpoint_path = ckpt_info['checkpoints'][fold]
        success = run_audit_for_fold(
            checkpoint_path,
            fold,
            args.config,
            output_dir,
            args.gpu
        )
        results[fold] = success
    
    # 总结
    print("\n" + "="*80)
    print("审计完成总结")
    print("="*80)
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"成功: {successful}/{total}")
    for fold, success in sorted(results.items()):
        status = "✓" if success else "✗"
        print(f"  {status} Fold {fold}")
    print()
    print(f"Results saved in: {output_dir}")
    
    return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())
