#!/usr/bin/env python3
"""
提取 BLCA 主表 C-index（从真实 pkl 文件）
数据来源: results/dct_v3.10/robust/final_50ep_old/<cancer>/split_*_results_final.pkl
使用 sksurv.metrics.concordance_index_censored 计算 C-index
"""
import json
import pickle
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    concordance_index_censored = None


def compute_cindex_from_pkl(pkl_path):
    """从 pkl 预测文件计算 C-index."""
    if concordance_index_censored is None:
        raise RuntimeError("需要 sksurv: pip install scikit-survival")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    risks, times, events = [], [], []
    for case_id, info in data.items():
        risks.append(float(info['risk']))
        times.append(float(info['time']))
        # 数据中 c=0 表示 event (uncensored), c=1 表示 censored
        c = float(info['censor'])
        events.append(1.0 - c)  # 转换为 event=1, censored=0
    risks = np.array(risks)
    times = np.array(times)
    events = np.array(events, dtype=bool)
    if events.sum() < 2:
        return None, len(risks)
    result = concordance_index_censored(events, times, risks)
    return result[0], len(risks)


def extract_from_pkl(cancer):
    """从 pkl 文件提取所有折的 C-index."""
    base = Path("/data1/DCT-Reg/results") / f"dct_v3.10/robust/final_50ep_old/{cancer}/{cancer}"
    if not base.exists():
        return None
    # Find experiment directory (two levels deep)
    exp_dirs = [d for d in base.glob('*/*') if d.is_dir() and d.name != '.gitkeep']
    if not exp_dirs:
        return None
    exp_dir = exp_dirs[0]
    results = {}
    for fold in range(5):
        p = exp_dir / f'split_{fold}_results_final.pkl'
        if not p.exists():
            p = exp_dir / f'split_{fold}_results.pkl'
        if p.exists():
            c, n = compute_cindex_from_pkl(p)
            if c is not None:
                results[fold] = {'cindex': c, 'n_cases': n}
    return results


def extract_from_manifest(base_path):
    """从 evidence/run_manifest.json 提取折 C-index."""
    results = {}
    for fold in range(5):
        manifest_path = base_path / f"evidence/fold_{fold}/run_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
                results[fold] = {
                    'cindex': data['metrics'].get('outer_cindex'),
                    'ipcw_cindex': data['metrics'].get('outer_cindex_ipcw', None),
                }
    return results


def main():
    results_base = Path("/data1/DCT-Reg/results")

    print("=" * 70)
    print("📊 DCT v3.10 主表 C-index（真实数据验证）")
    print("=" * 70)

    cancers = ['blca', 'hnsc', 'lusc', 'skcm', 'kirc']
    all_results = {}

    for cancer in cancers:
        # 主要数据源: dct_v3.10/robust/final_50ep_old (pkl 文件)
        pkl_base = results_base / f"dct_v3.10/robust/final_50ep_old/{cancer}/{cancer}"
        pkl_results = extract_from_pkl(cancer)

        # 备用数据源: dct_v3.10_experiments/robust/full (run_manifest.json)
        exp_path = results_base / f"dct_v3.10_experiments/robust/full/{cancer}/{cancer}"
        manifest_dirs = list(exp_path.glob("*")) if exp_path.exists() else []
        manifest_results = {}
        if manifest_dirs:
            manifest_results = extract_from_manifest(manifest_dirs[0])

        if pkl_results:
            cindices = [pkl_results[f]['cindex'] for f in sorted(pkl_results.keys())]
            ns = [pkl_results[f]['n_cases'] for f in sorted(pkl_results.keys())]
            all_results[cancer] = pkl_results
            print(f"\n{cancer.upper()} (from pkl, n={ns[0]}):")
            fold_vals = [pkl_results[f]["cindex"] for f in sorted(pkl_results.keys())]
            fold_strs = ", ".join(f"{v:.4f}" for v in fold_vals)
            print(f"  Folds: [{fold_strs}]")
            print(f"  Mean: {np.mean(cindices):.4f} ± {np.std(cindices, ddof=1):.4f}")
        elif manifest_results:
            cindices = [manifest_results[f]['cindex'] for f in sorted(manifest_results.keys()) if manifest_results[f]['cindex'] is not None]
            all_results[cancer] = manifest_results
            print(f"\n{cancer.upper()} (from manifest):")
            print(f"  Folds: {[f'{c:.4f}' for c in cindices]}")
            print(f"  Mean: {np.mean(cindices):.4f} ± {np.std(cindices, ddof=1):.4f}")
        else:
            print(f"\n{cancer.upper()}: ❌ 无数据")

    # SlotSPE 对比
    print("\n" + "=" * 70)
    print("SlotSPE 对比（论文报告值）")
    print("=" * 70)
    slotspe = {
        'blca': 0.708, 'hnsc': 0.642, 'kirc': 0.815,
        'lusc': 0.634, 'skcm': 0.688
    }

    print(f"\n{'Cancer':<8} {'DCT':>8} {'SlotSPE':>8} {'Δ':>8}")
    print("-" * 36)
    dct_means = []
    ss_means = []
    for cancer in ['blca', 'hnsc', 'kirc', 'lusc', 'skcm']:
        if cancer in all_results:
            cinds = [all_results[cancer][f]['cindex'] for f in sorted(all_results[cancer].keys())]
            dct_mean = np.mean(cinds)
            dct_means.append(dct_mean)
            ss_means.append(slotspe[cancer])
            delta = dct_mean - slotspe[cancer]
            print(f"{cancer:<8} {dct_mean:>8.4f} {slotspe[cancer]:>8.3f} {delta:>+8.4f}")

    if len(dct_means) == 5:
        print("-" * 36)
        print(f"{'AVG':<8} {np.mean(dct_means):>8.4f} {np.mean(ss_means):>8.4f} {np.mean(dct_means) - np.mean(ss_means):>+8.4f}")

        # 配对 t 检验
        t_stat, t_p = stats.ttest_rel(dct_means, ss_means)
        print(f"\n配对 t 检验: t={t_stat:.3f}, p={t_p:.3f}")

    # 保存结果
    output_path = results_base / "dct_main_table_cindex.json"
    save_data = {}
    for cancer, fold_data in all_results.items():
        cinds = [fold_data[f]['cindex'] for f in sorted(fold_data.keys())]
        save_data[cancer] = {
            'fold_cindices': cinds,
            'mean': float(np.mean(cinds)),
            'std': float(np.std(cinds, ddof=1)),
            'n_folds': len(cinds)
        }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✓ 保存到: {output_path}")


if __name__ == '__main__':
    main()
