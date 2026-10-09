#!/usr/bin/env python3
"""
提取 BLCA 四臂消融实验的 IBS 和 iAUC
从 run_manifest.json 中读取 outer_IBS 和 outer_iauc
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

def extract_survival_metrics(base_path: Path, variant_name: str) -> dict:
    """从 evidence 目录提取所有折的 IBS 和 iAUC"""
    results = {}
    for fold in range(5):
        manifest_path = base_path / f"evidence/fold_{fold}/run_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
                metrics = data['metrics']
                
                # 提取 IBS（可能是数组格式）
                ibs_raw = metrics.get('outer_IBS', None)
                if isinstance(ibs_raw, str):
                    # 解析 "[0.04246352 0.09687068 0.21667703]" 格式
                    ibs_array = np.fromstring(ibs_raw.strip('[]'), sep=' ')
                    ibs = float(np.mean(ibs_array)) if len(ibs_array) > 0 else ibs_raw
                else:
                    ibs = float(ibs_raw) if ibs_raw is not None else None
                
                results[fold] = {
                    'ibs': ibs,
                    'iauc': metrics.get('outer_iauc', None)
                }
        else:
            print(f"Warning: {manifest_path} not found")
    return results

def main():
    results_base = Path("/data1/DCT-Reg/results")
    
    variants = {
        'Full': results_base / "dct_v3.10_experiments/robust/full/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_full_blca_50ep",
        'Direction-only': results_base / "backups/direction_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep",
        'NLL-only': results_base / "backups/nll_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_nll_only_blca_50ep",
        'IPCW-only': results_base / "backups/ipcw_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_ipcw_only_blca_50ep"
    }
    
    all_results = {}
    for name, path in variants.items():
        print(f"\n=== {name} ===")
        fold_results = extract_survival_metrics(path, name)
        all_results[name] = fold_results
        
        if fold_results:
            ibs_values = [fold_results[f]['ibs'] for f in sorted(fold_results.keys()) if fold_results[f]['ibs'] is not None]
            iauc_values = [fold_results[f]['iauc'] for f in sorted(fold_results.keys()) if fold_results[f]['iauc'] is not None]
            
            if ibs_values:
                print(f"IBS per fold:  {[f'{v:.4f}' for v in ibs_values]}")
                print(f"IBS Mean ± Std: {np.mean(ibs_values):.4f} ± {np.std(ibs_values, ddof=1):.4f}")
            
            if iauc_values:
                print(f"iAUC per fold: {[f'{v:.4f}' for v in iauc_values]}")
                print(f"iAUC Mean ± Std: {np.mean(iauc_values):.4f} ± {np.std(iauc_values, ddof=1):.4f}")
    
    # 生成补充表格
    print("\n" + "="*60)
    print("Supplementary Table: IBS and iAUC")
    print("="*60)
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Variant & IBS (↓) & iAUC (↑) \\")
    print(r"\midrule")
    
    for variant in ['Full', 'Direction-only', 'NLL-only', 'IPCW-only']:
        ibs_values = [all_results[variant][f]['ibs'] for f in range(5) if all_results[variant][f]['ibs'] is not None]
        iauc_values = [all_results[variant][f]['iauc'] for f in range(5) if all_results[variant][f]['iauc'] is not None]
        
        ibs_str = f"{np.mean(ibs_values):.4f} $\\pm$ {np.std(ibs_values, ddof=1):.4f}" if ibs_values else "N/A"
        iauc_str = f"{np.mean(iauc_values):.4f} $\\pm$ {np.std(iauc_values, ddof=1):.4f}" if iauc_values else "N/A"
        
        print(f"{variant} & {ibs_str} & {iauc_str} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    
    # 保存 JSON
    output_path = results_base / "survival_metrics_blca_5fold.json"
    summary = {}
    for variant, fold_results in all_results.items():
        ibs_values = [fold_results[f]['ibs'] for f in range(5) if fold_results[f]['ibs'] is not None]
        iauc_values = [fold_results[f]['iauc'] for f in range(5) if fold_results[f]['iauc'] is not None]
        
        summary[variant] = {
            'ibs': {
                'values': ibs_values,
                'mean': float(np.mean(ibs_values)) if ibs_values else None,
                'std': float(np.std(ibs_values, ddof=1)) if ibs_values else None
            },
            'iauc': {
                'values': iauc_values,
                'mean': float(np.mean(iauc_values)) if iauc_values else None,
                'std': float(np.std(iauc_values, ddof=1)) if iauc_values else None
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")

if __name__ == '__main__':
    main()
