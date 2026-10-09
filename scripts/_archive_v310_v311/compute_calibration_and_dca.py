#!/usr/bin/env python3
"""
校准曲线、决策曲线和 IBS/时间依赖 AUC 分析
"""

import os
import sys
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_predictions(predictions_pkl_path):
    """加载预测数据"""
    with open(predictions_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 转换为 DataFrame
    rows = []
    for patient_id, pred in data.items():
        row = {
            'patient_id': patient_id,
            'risk': pred['risk'],
            'time': pred['time'],
            'event': 1 - pred['censor']  # censor=0 表示事件发生
        }
        if 'logits' in pred:
            row['logits'] = pred['logits']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def compute_calibration_curve(df, n_bins=10, time_points=[12, 24, 36, 60]):
    """
    计算校准曲线
    基于风险分数的分位数分箱，比较预测风险与实际事件率
    """
    results = {}
    
    # 按风险分数分箱
    df = df.copy()
    df['risk_bin'] = pd.qcut(df['risk'], n_bins, labels=False, duplicates='drop')
    
    calibration = []
    for bin_idx in df['risk_bin'].unique():
        bin_data = df[df['risk_bin'] == bin_idx]
        
        predicted_risk = bin_data['risk'].mean()
        observed_rate = bin_data['event'].mean()
        n_samples = len(bin_data)
        
        calibration.append({
            'bin': bin_idx,
            'predicted_risk': predicted_risk,
            'observed_rate': observed_rate,
            'n_samples': n_samples
        })
    
    results['calibration'] = pd.DataFrame(calibration)
    
    # 计算校准指标
    cal_df = results['calibration']
    # Brier Score (简化版)
    brier_score = np.mean((df['risk'] - df['event']) ** 2)
    results['brier_score'] = float(brier_score)
    
    # Expected Calibration Error
    ece = np.average(
        np.abs(cal_df['predicted_risk'] - cal_df['observed_rate']),
        weights=cal_df['n_samples']
    )
    results['ece'] = float(ece)
    
    return results

def plot_calibration_curve(calibration_df, output_path, title='Calibration Curve'):
    """绘制校准曲线"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制完美校准线
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # 绘制实际校准
    ax.scatter(
        calibration_df['predicted_risk'],
        calibration_df['observed_rate'],
        s=calibration_df['n_samples'] * 2,
        alpha=0.6,
        label='Observed',
        c='#F97316'
    )
    
    # 连线
    sorted_df = calibration_df.sort_values('predicted_risk')
    ax.plot(
        sorted_df['predicted_risk'],
        sorted_df['observed_rate'],
        'o-',
        alpha=0.5,
        color='#F97316',
        linewidth=2
    )
    
    ax.set_xlabel('Predicted Risk', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Event Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Calibration curve saved to: {output_path}")

def compute_decision_curve(df, threshold_range=np.linspace(0, 1, 101)):
    """
    计算决策曲线分析 (Decision Curve Analysis, DCA)
    """
    results = []
    
    for threshold in threshold_range:
        # 使用模型：根据风险阈值决策
        treat_model = df['risk'] >= threshold
        
        # True Positive: 正确预测高风险并发生事件
        tp = ((treat_model) & (df['event'] == 1)).sum()
        # False Positive: 错误预测高风险但未发生事件
        fp = ((treat_model) & (df['event'] == 0)).sum()
        
        n = len(df)
        prevalence = df['event'].mean()
        
        # Net Benefit = (TP/n) - (FP/n) * (threshold / (1 - threshold))
        if threshold == 1.0:
            net_benefit_model = 0
        else:
            net_benefit_model = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        
        # Treat All: 所有人都治疗
        net_benefit_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold)) if threshold < 1 else 0
        
        # Treat None: 没有人治疗
        net_benefit_none = 0
        
        results.append({
            'threshold': threshold,
            'net_benefit_model': net_benefit_model,
            'net_benefit_all': max(0, net_benefit_all),
            'net_benefit_none': net_benefit_none
        })
    
    return pd.DataFrame(results)

def plot_decision_curve(dca_df, output_path, title='Decision Curve Analysis'):
    """绘制决策曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(
        dca_df['threshold'],
        dca_df['net_benefit_model'],
        label='Model',
        linewidth=2,
        color='#22D3EE'
    )
    ax.plot(
        dca_df['threshold'],
        dca_df['net_benefit_all'],
        label='Treat All',
        linewidth=2,
        linestyle='--',
        color='#F59E0B'
    )
    ax.axhline(
        y=0,
        label='Treat None',
        linewidth=2,
        linestyle=':',
        color='#6B7280'
    )
    
    ax.set_xlabel('Risk Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Net Benefit', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Decision curve saved to: {output_path}")

def extract_ibs_iauc_from_epoch_curves(results_dir):
    """从 epoch_curve_fold*.csv 提取 IBS 和 iAUC"""
    results_path = Path(results_dir)
    curve_files = list(results_path.rglob('epoch_curve_fold*.csv'))
    
    all_metrics = []
    
    for curve_file in curve_files:
        df = pd.read_csv(curve_file)
        
        # 获取最后一个 epoch 的指标（或最佳 epoch）
        if 'val_IBS' in df.columns:
            best_epoch = df['val_cindex'].idxmax()
            
            metrics = {
                'file': str(curve_file.relative_to(results_path)),
                'best_epoch': int(best_epoch),
                'val_IBS': float(df.loc[best_epoch, 'val_IBS']),
                'val_iauc': float(df.loc[best_epoch, 'val_iauc']) if 'val_iauc' in df.columns else None,
                'val_cindex': float(df.loc[best_epoch, 'val_cindex']),
                'val_cindex_ipcw': float(df.loc[best_epoch, 'val_cindex_ipcw']) if 'val_cindex_ipcw' in df.columns else None
            }
            all_metrics.append(metrics)
    
    if not all_metrics:
        return None
    
    metrics_df = pd.DataFrame(all_metrics)
    
    summary = {
        'IBS': {
            'mean': float(metrics_df['val_IBS'].mean()),
            'std': float(metrics_df['val_IBS'].std()),
            'min': float(metrics_df['val_IBS'].min()),
            'max': float(metrics_df['val_IBS'].max())
        },
        'C-index': {
            'mean': float(metrics_df['val_cindex'].mean()),
            'std': float(metrics_df['val_cindex'].std()),
            'min': float(metrics_df['val_cindex'].min()),
            'max': float(metrics_df['val_cindex'].max())
        }
    }
    
    if metrics_df['val_iauc'].notna().any():
        summary['iAUC'] = {
            'mean': float(metrics_df['val_iauc'].mean()),
            'std': float(metrics_df['val_iauc'].std()),
            'min': float(metrics_df['val_iauc'].min()),
            'max': float(metrics_df['val_iauc'].max())
        }
    
    if metrics_df['val_cindex_ipcw'].notna().any():
        summary['C-index (IPCW)'] = {
            'mean': float(metrics_df['val_cindex_ipcw'].mean()),
            'std': float(metrics_df['val_cindex_ipcw'].std()),
            'min': float(metrics_df['val_cindex_ipcw'].min()),
            'max': float(metrics_df['val_cindex_ipcw'].max())
        }
    
    return {
        'fold_metrics': metrics_df.to_dict('records'),
        'summary': summary
    }

def analyze_fold_predictions(results_dir, variant_name='dct_v32'):
    """分析所有折的预测数据"""
    results_path = Path(results_dir)
    
    # 查找所有 predictions.pkl 文件
    pred_files = list(results_path.rglob('split_*_results*.pkl'))
    
    print(f"Found {len(pred_files)} prediction files")
    
    all_calibrations = []
    all_dcas = []
    
    for pred_file in pred_files[:5]:  # 限制到前5个折
        try:
            print(f"\nProcessing: {pred_file.name}")
            df = load_predictions(str(pred_file))
            
            # 校准曲线
            cal_results = compute_calibration_curve(df, n_bins=10)
            all_calibrations.append(cal_results)
            
            # 决策曲线
            dca_df = compute_decision_curve(df)
            all_dcas.append(dca_df)
            
            print(f"  - {len(df)} patients")
            print(f"  - Brier Score: {cal_results['brier_score']:.4f}")
            print(f"  - ECE: {cal_results['ece']:.4f}")
            
        except Exception as e:
            print(f"  - Error: {e}")
            continue
    
    # 汇总校准指标
    if all_calibrations:
        brier_scores = [c['brier_score'] for c in all_calibrations]
        eces = [c['ece'] for c in all_calibrations]
        
        calibration_summary = {
            'brier_score': {
                'mean': float(np.mean(brier_scores)),
                'std': float(np.std(brier_scores)),
                'folds': brier_scores
            },
            'ece': {
                'mean': float(np.mean(eces)),
                'std': float(np.std(eces)),
                'folds': eces
            }
        }
        
        # 绘制第一个折的校准曲线
        if len(all_calibrations) > 0:
            output_dir = Path('/data1/DCT-Reg/results')
            plot_calibration_curve(
                all_calibrations[0]['calibration'],
                output_dir / f'{variant_name}_calibration_curve.png',
                title=f'{variant_name.upper()} Calibration Curve (Fold 0)'
            )
        
        # 绘制第一个折的决策曲线
        if len(all_dcas) > 0:
            plot_decision_curve(
                all_dcas[0],
                output_dir / f'{variant_name}_decision_curve.png',
                title=f'{variant_name.upper()} Decision Curve Analysis (Fold 0)'
            )
        
        return calibration_summary
    
    return None

def main():
    output_dir = Path('/data1/DCT-Reg/results')
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("Calibration & Decision Curve Analysis")
    print("="*80)
    
    # DCT v3.2
    print("\n" + "="*80)
    print("DCT v3.2 Analysis")
    print("="*80)
    
    v32_cal = analyze_fold_predictions(
        '/data1/DCT-Reg/results/dct_v3.2',
        variant_name='dct_v32'
    )
    
    v32_metrics = extract_ibs_iauc_from_epoch_curves('/data1/DCT-Reg/results/dct_v3.2')
    
    # 保存结果
    v32_results = {
        'calibration': v32_cal,
        'survival_metrics': v32_metrics
    }
    
    with open(output_dir / 'calibration_dca_metrics_v32.json', 'w') as f:
        json.dump(v32_results, f, indent=2)
    
    print(f"\nDCT v3.2 Results:")
    if v32_cal:
        print(f"  Brier Score: {v32_cal['brier_score']['mean']:.4f} ± {v32_cal['brier_score']['std']:.4f}")
        print(f"  ECE: {v32_cal['ece']['mean']:.4f} ± {v32_cal['ece']['std']:.4f}")
    
    if v32_metrics:
        print(f"  IBS: {v32_metrics['summary']['IBS']['mean']:.4f} ± {v32_metrics['summary']['IBS']['std']:.4f}")
        if 'iAUC' in v32_metrics['summary']:
            print(f"  iAUC: {v32_metrics['summary']['iAUC']['mean']:.4f} ± {v32_metrics['summary']['iAUC']['std']:.4f}")
    
    print(f"\n✅ Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
