#!/usr/bin/env python3
"""
提取 BLCA 四臂消融实验的折级 C-index
从 run_manifest.json 中读取 outer_cindex，计算均值和标准差
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

def extract_fold_results(base_path: Path, variant_name: str) -> Dict[int, dict]:
    """从 evidence 目录提取所有折的结果"""
    results = {}
    for fold in range(5):
        manifest_path = base_path / f"evidence/fold_{fold}/run_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
                results[fold] = {
                    'cindex': data['metrics']['outer_cindex'],
                    'ipcw_cindex': data['metrics'].get('outer_cindex_ipcw', None),
                    'ibs': data['metrics'].get('outer_IBS', None),
                    'fixed_epoch': data['metrics'].get('fixed_epoch', None)
                }
        else:
            print(f"Warning: {manifest_path} not found")
    return results

def main():
    results_base = Path("/data1/DCT-Reg/results")
    
    # 定义四个变体的路径
    variants = {
        'Full': results_base / "dct_v3.10_experiments/robust/full/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_full_blca_50ep",
        'Direction-only': results_base / "backups/direction_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep",
        'NLL-only': results_base / "backups/nll_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_nll_only_blca_50ep",
        'IPCW-only': results_base / "backups/ipcw_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_ipcw_only_blca_50ep"
    }
    
    # 提取所有结果
    all_results = {}
    for name, path in variants.items():
        print(f"\n=== {name} ===")
        fold_results = extract_fold_results(path, name)
        all_results[name] = fold_results
        
        if fold_results:
            cindices = [fold_results[f]['cindex'] for f in sorted(fold_results.keys())]
            print(f"Folds found: {sorted(fold_results.keys())}")
            print(f"C-index per fold: {[f'{c:.4f}' for c in cindices]}")
            print(f"Mean ± Std: {np.mean(cindices):.4f} ± {np.std(cindices, ddof=1):.4f}")
    
    # 配对 t 检验
    print("\n" + "="*60)
    print("Paired t-tests (Full vs. Ablations)")
    print("="*60)
    
    full_cindices = [all_results['Full'][f]['cindex'] for f in range(5)]
    
    for variant in ['Direction-only', 'NLL-only', 'IPCW-only']:
        variant_cindices = [all_results[variant][f]['cindex'] for f in range(5)]
        t_stat, p_value = stats.ttest_rel(full_cindices, variant_cindices)
        mean_diff = np.mean(full_cindices) - np.mean(variant_cindices)
        print(f"\nFull vs. {variant}:")
        print(f"  Mean difference: {mean_diff:+.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
    
    # 生成 LaTeX 表格
    print("\n" + "="*60)
    print("LaTeX Table")
    print("="*60)
    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(r"Variant & Fold 0 & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Mean $\pm$ Std & $p$ \\")
    print(r"\midrule")
    
    for variant in ['Full', 'Direction-only', 'NLL-only', 'IPCW-only']:
        cindices = [all_results[variant][f]['cindex'] for f in range(5)]
        row = [variant]
        row.extend([f"{c:.3f}" for c in cindices])
        row.append(f"{np.mean(cindices):.4f} $\\pm$ {np.std(cindices, ddof=1):.4f}")
        
        if variant == 'Full':
            row.append("—")
        else:
            _, p = stats.ttest_rel(full_cindices, cindices)
            row.append(f"{p:.3f}")
        
        print(" & ".join(row) + r" \\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    
    # 保存 JSON 结果
    output_path = results_base / "ablation_summary_blca_5fold.json"
    summary = {}
    for variant, fold_results in all_results.items():
        cindices = [fold_results[f]['cindex'] for f in range(5)]
        summary[variant] = {
            'fold_cindices': cindices,
            'mean': float(np.mean(cindices)),
            'std': float(np.std(cindices, ddof=1)),
            'n_folds': len(cindices)
        }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")

if __name__ == '__main__':
    main()
