#!/usr/bin/env python3
"""
提取 DCT 训练时间统计
从 epoch_curve_fold*.csv 文件中提取训练时间信息
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def find_epoch_curves(base_dir="/data1/DCT-Reg/results"):
    """查找所有 epoch_curve_fold*.csv 文件"""
    patterns = [
        f"{base_dir}/**/epoch_curve_fold*.csv",
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
    elif "dct_v38" in path or "v3.8" in path:
        return "v3.8"
    return "unknown"

def extract_cancer_type(path):
    """从路径中提取癌症类型"""
    path_lower = path.lower()
    for cancer in ["blca", "brca", "gbmlgg", "kirc", "luad"]:
        if f"/{cancer}/" in path_lower or f"_{cancer}_" in path_lower:
            return cancer.upper()
    return "unknown"

def analyze_epoch_curve(csv_path):
    """分析单个 epoch curve 文件"""
    try:
        df = pd.read_csv(csv_path)
        
        # 检查是否有时间列
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        
        info = {
            "path": csv_path,
            "num_epochs": len(df),
            "has_time_info": len(time_cols) > 0,
            "time_columns": time_cols,
            "final_val_cindex": df['val_cindex'].iloc[-1] if 'val_cindex' in df.columns else None,
            "best_val_cindex": df['val_cindex'].max() if 'val_cindex' in df.columns else None,
            "final_val_loss": df['val_loss'].iloc[-1] if 'val_loss' in df.columns else None,
        }
        
        # 如果有时间信息，提取它
        if time_cols:
            for col in time_cols:
                info[f'{col}_total'] = df[col].sum()
                info[f'{col}_per_epoch'] = df[col].mean()
        
        return info
    except Exception as e:
        print(f"Error analyzing {csv_path}: {e}")
        return None

def check_log_files_for_time():
    """检查日志文件中的训练时间"""
    log_patterns = [
        "/data1/DCT-Reg/*.log",
        "/data1/DCT-Reg/logs/*.log",
        "/data1/DCT-Reg/results/**/*.log",
    ]
    
    time_info = []
    for pattern in log_patterns:
        for log_file in glob.glob(pattern, recursive=True):
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    # 搜索常见的时间模式
                    if 'epoch' in content.lower() and ('time' in content.lower() or 'duration' in content.lower()):
                        time_info.append({
                            "log_file": log_file,
                            "contains_time": True
                        })
            except:
                pass
    
    return time_info

def main():
    print("🔍 查找 epoch_curve 文件...")
    epoch_files = find_epoch_curves()
    print(f"找到 {len(epoch_files)} 个 epoch_curve 文件")
    
    if not epoch_files:
        print("❌ 未找到任何 epoch_curve 文件")
        return
    
    # 按版本和癌症类型分组
    grouped = defaultdict(lambda: defaultdict(list))
    
    print("\n📊 分析文件...")
    for csv_file in epoch_files:
        version = extract_version_from_path(csv_file)
        cancer = extract_cancer_type(csv_file)
        
        info = analyze_epoch_curve(csv_file)
        if info:
            grouped[version][cancer].append(info)
            
            # 打印简要信息
            fold_match = None
            for part in csv_file.split('/'):
                if 'fold' in part.lower():
                    fold_match = part
                    break
            if not fold_match:
                fold_match = csv_file.split('/')[-1]
            
            status = "✅ 有时间" if info['has_time_info'] else "❌ 无时间"
            print(f"  {version:8s} {cancer:8s} {fold_match:20s} {info['num_epochs']:3d} epochs {status}")
    
    # 汇总统计
    print("\n" + "="*80)
    print("📈 训练时间统计汇总")
    print("="*80)
    
    summary = {}
    for version in sorted(grouped.keys()):
        summary[version] = {}
        print(f"\n🔹 DCT {version}")
        
        for cancer in sorted(grouped[version].keys()):
            files = grouped[version][cancer]
            
            has_time = any(f['has_time_info'] for f in files)
            num_epochs = [f['num_epochs'] for f in files]
            
            best_cindices = [f['best_val_cindex'] for f in files if f['best_val_cindex'] is not None]
            
            cancer_summary = {
                "num_folds": len(files),
                "num_epochs_range": f"{min(num_epochs)}-{max(num_epochs)}" if num_epochs else "N/A",
                "has_time_info": has_time,
                "best_val_cindex_mean": np.mean(best_cindices) if best_cindices else None,
                "best_val_cindex_std": np.std(best_cindices) if best_cindices else None,
            }
            
            summary[version][cancer] = cancer_summary
            
            print(f"  {cancer}: {len(files)} folds, {cancer_summary['num_epochs_range']} epochs")
            if best_cindices:
                print(f"    最佳验证 C-index: {np.mean(best_cindices):.4f} ± {np.std(best_cindices):.4f}")
            
            if has_time:
                print(f"    ✅ 包含时间信息")
            else:
                print(f"    ❌ CSV 中无时间信息")
    
    # 检查日志文件
    print("\n" + "="*80)
    print("📝 检查日志文件中的时间信息")
    print("="*80)
    
    log_time_info = check_log_files_for_time()
    if log_time_info:
        print(f"找到 {len(log_time_info)} 个可能包含时间信息的日志文件:")
        for info in log_time_info[:10]:  # 只显示前10个
            print(f"  - {info['log_file']}")
    else:
        print("❌ 未在日志文件中找到明确的时间信息")
    
    # 保存结果
    output_file = "/data1/DCT-Reg/results/training_time_analysis.json"
    output_data = {
        "summary": summary,
        "total_files_analyzed": len(epoch_files),
        "log_files_with_time": len(log_time_info),
        "note": "CSV文件中不包含训练时间列，需要从日志文件或训练过程中提取"
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")
    
    # 生成建议
    print("\n" + "="*80)
    print("💡 建议")
    print("="*80)
    print("""
1. **CSV 文件中无训练时间**：
   - epoch_curve_fold*.csv 没有包含 training_time 或 epoch_duration 列
   - 需要从其他来源获取时间信息

2. **可能的时间来源**：
   - 训练日志文件（*.log）
   - 训练脚本的 stdout/stderr 输出
   - 系统时间戳（文件创建/修改时间）
   - 重新运行并记录时间

3. **估算方法**：
   - 使用检查点文件的时间戳差异
   - 假设每个 epoch 训练时间相近
   - 参考类似模型的训练时间

4. **实际测量**（推荐）：
   - 在训练脚本中添加时间记录
   - 使用 time 命令包装训练命令
   - 记录每个 epoch 的开始和结束时间
    """)

if __name__ == "__main__":
    main()
