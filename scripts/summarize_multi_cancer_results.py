#!/usr/bin/env python3
"""
汇总所有癌症类型的5折交叉验证结果
计算多癌种的平均C-index和其他指标
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def find_best_epoch_results(csv_path):
    """从epoch_curve文件提取最佳epoch的验证指标"""
    try:
        df = pd.read_csv(csv_path)
        
        # 找到验证C-index最高的epoch
        if 'val_cindex' not in df.columns:
            return None
        
        best_idx = df['val_cindex'].idxmax()
        best_row = df.iloc[best_idx]
        
        return {
            'file': csv_path,
            'best_epoch': int(best_idx),
            'val_cindex': float(best_row.get('val_cindex', np.nan)),
            'val_cindex_ipcw': float(best_row.get('val_cindex_ipcw', np.nan)),
            'val_IBS': float(best_row.get('val_IBS', np.nan)),
            'val_iauc': float(best_row.get('val_iauc', np.nan)),
            'val_loss': float(best_row.get('val_loss', np.nan)),
        }
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def extract_cancer_type(path):
    """从路径提取癌种"""
    path_lower = path.lower()
    cancers = ['blca', 'hnsc', 'kirc', 'lusc', 'skcm', 'brca', 'luad', 'gbmlgg']
    for cancer in cancers:
        if f"/{cancer}/" in path_lower:
            return cancer.upper()
    return "UNKNOWN"

def extract_version(path):
    """从路径提取版本"""
    if "dct_v3.10" in path or "v310" in path:
        return "v3.10"
    elif "dct_v3.2" in path or "v32" in path:
        return "v3.2"
    return "unknown"

def main():
    results_dir = Path("/data1/DCT-Reg/results")
    
    print("="*80)
    print("📊 DCT 多癌种5折验证结果汇总")
    print("="*80)
    print()
    
    # 查找主要的v3.10完整实验结果
    base_path = results_dir / "dct_v3.10" / "robust" / "final_50ep_old"
    
    if not base_path.exists():
        print(f"❌ 目录不存在: {base_path}")
        return
    
    # 收集所有结果
    all_results = defaultdict(lambda: defaultdict(list))
    
    cancers = ['blca', 'hnsc', 'kirc', 'lusc', 'skcm']
    
    for cancer in cancers:
        cancer_path = base_path / cancer / cancer
        if not cancer_path.exists():
            print(f"⚠️  目录不存在: {cancer_path}")
            continue
        
        # 查找所有epoch_curve文件
        pattern = cancer_path / "**/epoch_curve_fold*.csv"
        epoch_files = list(cancer_path.glob("**/epoch_curve_fold*.csv"))
        
        print(f"\n🔹 {cancer.upper()} ({len(epoch_files)} files)")
        
        for epoch_file in epoch_files:
            result = find_best_epoch_results(str(epoch_file))
            if result:
                all_results['v3.10'][cancer.upper()].append(result)
                
                # 提取fold编号
                fold = None
                fname = epoch_file.name
                for i in range(5):
                    if f'fold{i}' in fname:
                        fold = i
                        break
                
                print(f"  fold{fold if fold else '?'}: C-index={result['val_cindex']:.4f}, "
                      f"IBS={result['val_IBS']:.4f}, iAUC={result['val_iauc']:.4f}")
    
    # 汇总统计
    print("\n" + "="*80)
    print("📈 汇总统计")
    print("="*80)
    
    summary = {}
    
    for version in sorted(all_results.keys()):
        summary[version] = {}
        print(f"\n🔸 DCT {version}")
        
        all_cindices = []
        all_ibss = []
        all_iaucs = []
        all_ipcws = []
        
        for cancer in sorted(all_results[version].keys()):
            results = all_results[version][cancer]
            
            if not results:
                continue
            
            cindices = [r['val_cindex'] for r in results if not np.isnan(r['val_cindex'])]
            ibss = [r['val_IBS'] for r in results if not np.isnan(r['val_IBS'])]
            iaucs = [r['val_iauc'] for r in results if not np.isnan(r['val_iauc'])]
            ipcws = [r['val_cindex_ipcw'] for r in results if not np.isnan(r['val_cindex_ipcw'])]
            
            all_cindices.extend(cindices)
            all_ibss.extend(ibss)
            all_iaucs.extend(iaucs)
            all_ipcws.extend(ipcws)
            
            if cindices:
                cancer_stats = {
                    'num_folds': len(cindices),
                    'cindex_mean': np.mean(cindices),
                    'cindex_std': np.std(cindices),
                    'ibs_mean': np.mean(ibss) if ibss else np.nan,
                    'ibs_std': np.std(ibss) if ibss else np.nan,
                    'iauc_mean': np.mean(iaucs) if iaucs else np.nan,
                    'iauc_std': np.std(iaucs) if iaucs else np.nan,
                    'cindex_ipcw_mean': np.mean(ipcws) if ipcws else np.nan,
                    'cindex_ipcw_std': np.std(ipcws) if ipcws else np.nan,
                }
                summary[version][cancer] = cancer_stats
                
                print(f"\n  {cancer} ({len(cindices)} folds):")
                print(f"    C-index: {cancer_stats['cindex_mean']:.4f} ± {cancer_stats['cindex_std']:.4f}")
                print(f"    IBS:     {cancer_stats['ibs_mean']:.4f} ± {cancer_stats['ibs_std']:.4f}")
                print(f"    iAUC:    {cancer_stats['iauc_mean']:.4f} ± {cancer_stats['iauc_std']:.4f}")
                if ipcws:
                    print(f"    C-index (IPCW): {cancer_stats['cindex_ipcw_mean']:.4f} ± {cancer_stats['cindex_ipcw_std']:.4f}")
        
        # 多癌种平均
        if all_cindices:
            overall = {
                'total_folds': len(all_cindices),
                'cindex_mean': np.mean(all_cindices),
                'cindex_std': np.std(all_cindices),
                'ibs_mean': np.mean(all_ibss) if all_ibss else np.nan,
                'ibs_std': np.std(all_ibss) if all_ibss else np.nan,
                'iauc_mean': np.mean(all_iaucs) if all_iaucs else np.nan,
                'iauc_std': np.std(all_iaucs) if all_iaucs else np.nan,
            }
            summary[version]['OVERALL'] = overall
            
            print(f"\n  📊 全部癌种汇总 ({len(all_cindices)} folds):")
            print(f"    C-index: {overall['cindex_mean']:.4f} ± {overall['cindex_std']:.4f}")
            print(f"    IBS:     {overall['ibs_mean']:.4f} ± {overall['ibs_std']:.4f}")
            print(f"    iAUC:    {overall['iauc_mean']:.4f} ± {overall['iauc_std']:.4f}")
    
    # 保存结果
    output_file = results_dir / "multi_cancer_summary_v310.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 结果已保存: {output_file}")
    
    # 生成表格格式
    print("\n" + "="*80)
    print("📋 论文用表格格式")
    print("="*80)
    
    print("""
| 癌种 | Folds | C-index | IBS | iAUC | C-index (IPCW) |
|------|-------|---------|-----|------|----------------|""")
    
    cancer_order = ['BLCA', 'HNSC', 'KIRC', 'LUSC', 'SKCM', 'OVERALL']
    
    for cancer in cancer_order:
        if cancer in summary.get('v3.10', {}):
            s = summary['v3.10'][cancer]
            num_folds = s.get('num_folds', s.get('total_folds', 'N/A'))
            cindex_str = f"{s['cindex_mean']:.4f} ± {s['cindex_std']:.4f}"
            ibs_str = f"{s['ibs_mean']:.4f} ± {s['ibs_std']:.4f}" if not np.isnan(s['ibs_mean']) else "N/A"
            iauc_str = f"{s['iauc_mean']:.4f} ± {s['iauc_std']:.4f}" if not np.isnan(s['iauc_mean']) else "N/A"
            ipcw_str = f"{s['cindex_ipcw_mean']:.4f} ± {s['cindex_ipcw_std']:.4f}" if not np.isnan(s.get('cindex_ipcw_mean', np.nan)) else "N/A"
            marker = "**" if cancer == "OVERALL" else ""
            print(f"| {marker}{cancer}{marker} | {num_folds} | {cindex_str} | {ibs_str} | {iauc_str} | {ipcw_str} |")

if __name__ == "__main__":
    main()
