#!/usr/bin/env python3
"""
运行固定锚点验证实验

目的：证明方向传输机制在高质量锚点下是有效的
关键指标：DCR 和 DMR 是否显著提升
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_experiment(cancer, fold, gpu=0):
    """运行单个 fold 的固定锚点实验"""
    
    # 首先需要为这个 fold 提取锚点
    print(f"\n{'='*70}")
    print(f"步骤 1: 提取 {cancer} fold{fold} 的锚点")
    print(f"{'='*70}")
    
    # 找到对应的检查点
    checkpoint_pattern = (
        f"results/backups/direction_only_frozen_bug_20260903_173509/"
        f"{cancer}/{cancer}/SurvOTRank_dct_transport_intervention_consistency/"
        f"*/evidence/fold_{fold}/checkpoint.pt"
    )
    
    import glob
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print(f"❌ 未找到 {cancer} fold{fold} 的检查点")
        print(f"   搜索路径: {checkpoint_pattern}")
        return False
    
    checkpoint_path = checkpoints[0]
    anchor_output = f"results/ideal_anchors/{cancer}_extracted_fold{fold}.pkl"
    
    # 提取锚点
    extract_cmd = [
        sys.executable,
        "scripts/extract_anchors_from_checkpoint.py",
        "--checkpoint", checkpoint_path,
        "--output", anchor_output
    ]
    
    print(f"运行: {' '.join(extract_cmd)}")
    result = subprocess.run(extract_cmd, cwd="/data1/DCT-Reg")
    
    if result.returncode != 0:
        print(f"❌ 锚点提取失败")
        return False
    
    # 步骤 2: 运行训练
    print(f"\n{'='*70}")
    print(f"步骤 2: 使用固定锚点训练 {cancer} fold{fold}")
    print(f"{'='*70}")
    
    train_cmd = [
        sys.executable,
        "main.py",
        "--config", "configs/dct_v310_fixed_anchors_blca.yaml",
        "--cancer", cancer,
        "--fold", str(fold),
        "--gpu", str(gpu),
        "--results_dir", "results_fixed_anchors",
        "--exp_code", f"dct_v310_fixed_anchors_{cancer}",
        "--seed", "3",
        "--set", f"fixed_anchors_path={anchor_output}"
    ]
    
    print(f"运行: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, cwd="/data1/DCT-Reg")
    
    if result.returncode != 0:
        print(f"❌ 训练失败")
        return False
    
    print(f"\n✓ {cancer} fold{fold} 完成")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="运行固定锚点验证实验"
    )
    parser.add_argument("--cancer", default="blca", 
                        help="癌症类型")
    parser.add_argument("--folds", type=int, nargs="+", default=[0],
                        help="要运行的 fold（如 0 1 2）")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID")
    args = parser.parse_args()
    
    print("=" * 70)
    print("固定锚点验证实验")
    print("=" * 70)
    print(f"癌症类型: {args.cancer}")
    print(f"Folds: {args.folds}")
    print(f"GPU: {args.gpu}")
    print()
    
    success_count = 0
    for fold in args.folds:
        if run_experiment(args.cancer, fold, args.gpu):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"完成: {success_count}/{len(args.folds)} 个 folds 成功")
    print("=" * 70)
    
    if success_count == len(args.folds):
        print("\n✓ 所有实验完成！")
        print("\n下一步：运行分析脚本检查 DCR/DMR 是否提升")
        print("  python scripts/analyze_fixed_anchor_results.py")
    else:
        print("\n⚠️ 部分实验失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
