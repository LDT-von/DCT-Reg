#!/usr/bin/env python3
"""
从检查点文件时间戳估算训练时间
"""

import os
import glob
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def find_checkpoint_files(base_dir="/data1/DCT-Reg/results"):
    """查找所有检查点文件"""
    patterns = [
        f"{base_dir}/**/*_fold*.pth",
        f"{base_dir}/**/s_*_checkpoint.pt",
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    return sorted(set(all_files))

def extract_version_from_path(path):
    """从路径中提取版本信息"""
    if "dct_v3.2" in path or "v32" in path:
        return "v3.2"
    elif "dct_v310" in path or "v3.10" in path:
        return "v3.10"
    return "unknown"

def extract_cancer_type(path):
    """从路径中提取癌症类型"""
    path_lower = path.lower()
    for cancer in ["blca", "brca", "gbmlgg", "kirc", "luad"]:
        if f"/{cancer}/" in path_lower or f"_{cancer}_" in path_lower:
            return cancer.upper()
    return "unknown"

def get_file_times(file_path):
    """获取文件的时间戳"""
    try:
        stat = os.stat(file_path)
        return {
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "size_mb": stat.st_size / (1024 * 1024)
        }
    except:
        return None

def group_checkpoints_by_experiment(checkpoint_files):
    """将检查点按实验分组"""
    experiments = defaultdict(list)
    
    for file in checkpoint_files:
        # 提取实验目录（检查点所在的目录）
        exp_dir = str(Path(file).parent)
        
        # 获取折数
        if "_fold" in file:
            fold_num = None
            parts = file.split("_fold")
            if len(parts) > 1:
                for char in parts[1]:
                    if char.isdigit():
                        fold_num = int(char)
                        break
            
            experiments[exp_dir].append({
                "file": file,
                "fold": fold_num,
                "times": get_file_times(file)
            })
    
    return experiments

def estimate_training_time(experiments):
    """基于检查点时间戳估算训练时间"""
    results = {}
    
    for exp_dir, checkpoints in experiments.items():
        version = extract_version_from_path(exp_dir)
        cancer = extract_cancer_type(exp_dir)
        
        # 按折数分组
        folds = defaultdict(list)
        for ckpt in checkpoints:
            if ckpt['times']:
                folds[ckpt['fold']].append(ckpt)
        
        # 估算每折的训练时间
        fold_times = []
        for fold_num, fold_ckpts in folds.items():
            if len(fold_ckpts) >= 2:
                # 排序检查点
                fold_ckpts.sort(key=lambda x: x['times']['modified'])
                
                # 计算第一个和最后一个检查点之间的时间差
                start_time = fold_ckpts[0]['times']['modified']
                end_time = fold_ckpts[-1]['times']['modified']
                duration = (end_time - start_time).total_seconds() / 3600  # 转换为小时
                
                fold_times.append({
                    "fold": fold_num,
                    "duration_hours": duration,
                    "num_checkpoints": len(fold_ckpts),
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                })
        
        if fold_times:
            results[exp_dir] = {
                "version": version,
                "cancer": cancer,
                "folds": fold_times,
                "mean_duration_hours": np.mean([f['duration_hours'] for f in fold_times]),
                "std_duration_hours": np.std([f['duration_hours'] for f in fold_times]),
                "num_folds": len(fold_times)
            }
    
    return results

def main():
    print("🔍 查找检查点文件...")
    checkpoint_files = find_checkpoint_files()
    print(f"找到 {len(checkpoint_files)} 个检查点文件")
    
    if not checkpoint_files:
        print("❌ 未找到任何检查点文件")
        return
    
    print("\n📦 按实验分组检查点...")
    experiments = group_checkpoints_by_experiment(checkpoint_files)
    print(f"找到 {len(experiments)} 个实验")
    
    print("\n⏱️  估算训练时间...")
    time_estimates = estimate_training_time(experiments)
    
    # 按版本和癌症类型汇总
    summary = defaultdict(lambda: defaultdict(list))
    
    for exp_dir, data in time_estimates.items():
        version = data['version']
        cancer = data['cancer']
        summary[version][cancer].append(data)
    
    print("\n" + "="*80)
    print("📈 训练时间估算汇总（基于检查点时间戳）")
    print("="*80)
    
    final_summary = {}
    for version in sorted(summary.keys()):
        final_summary[version] = {}
        print(f"\n🔹 DCT {version}")
        
        for cancer in sorted(summary[version].keys()):
            exps = summary[version][cancer]
            
            all_durations = []
            for exp in exps:
                all_durations.extend([f['duration_hours'] for f in exp['folds']])
            
            if all_durations:
                mean_hours = np.mean(all_durations)
                std_hours = np.std(all_durations)
                min_hours = np.min(all_durations)
                max_hours = np.max(all_durations)
                
                cancer_summary = {
                    "num_experiments": len(exps),
                    "num_folds_total": len(all_durations),
                    "mean_training_hours": float(mean_hours),
                    "std_training_hours": float(std_hours),
                    "min_training_hours": float(min_hours),
                    "max_training_hours": float(max_hours),
                }
                
                final_summary[version][cancer] = cancer_summary
                
                print(f"  {cancer}:")
                print(f"    实验数: {len(exps)}, 总折数: {len(all_durations)}")
                print(f"    训练时间: {mean_hours:.2f} ± {std_hours:.2f} 小时")
                print(f"    范围: {min_hours:.2f} - {max_hours:.2f} 小时")
                
                # 转换为更易读的格式
                if mean_hours < 1:
                    print(f"    平均: {mean_hours * 60:.1f} 分钟")
                else:
                    print(f"    平均: {mean_hours:.2f} 小时 ({mean_hours * 60:.0f} 分钟)")
    
    # 保存结果
    output_file = "/data1/DCT-Reg/results/training_time_estimates.json"
    output_data = {
        "summary": final_summary,
        "detailed_experiments": {k: {
            "version": v["version"],
            "cancer": v["cancer"],
            "mean_duration_hours": v["mean_duration_hours"],
            "std_duration_hours": v["std_duration_hours"],
            "num_folds": v["num_folds"]
        } for k, v in time_estimates.items()},
        "note": "训练时间基于检查点文件的时间戳估算，可能包含训练暂停、系统开销等",
        "method": "使用每折第一个和最后一个检查点之间的时间差"
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")
    
    print("\n" + "="*80)
    print("⚠️  注意事项")
    print("="*80)
    print("""
1. **估算方法**：
   - 使用每折第一个和最后一个检查点文件的时间戳差异
   - 这包括了所有 epochs 的训练时间

2. **可能的误差来源**：
   - 训练过程中的暂停或中断
   - 文件系统的时间戳更新延迟
   - 检查点保存之外的时间（如验证、日志写入）
   - 系统负载变化

3. **更准确的方法**：
   - 在训练脚本中显式记录时间
   - 使用 time 命令包装训练过程
   - 从 CUDA profiler 或系统监控工具获取
    """)

if __name__ == "__main__":
    main()
