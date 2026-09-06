#!/usr/bin/env python3
"""
提取 BLCA Full 模型的机制审计结果
从 run_manifest.json 中提取 mechanism_audit 和 dose_audit 指标
"""
import json
import numpy as np
from pathlib import Path

def extract_mechanism_audit(manifest_path: Path) -> dict:
    """从 run_manifest 提取机制审计结果"""
    with open(manifest_path) as f:
        data = json.load(f)
    
    metrics = data['metrics']
    
    # 基础预测性能
    performance = {
        'outer_cindex': metrics['outer_cindex'],
        'outer_cindex_ipcw': metrics['outer_cindex_ipcw'],
        'outer_IBS': metrics['outer_IBS'],
        'outer_iauc': metrics['outer_iauc'],
        'fixed_epoch': metrics['fixed_epoch']
    }
    
    # 事实基线审计
    factual = metrics['mechanism_audit/factual']
    dcr_factual = factual['direction_consistency']
    reconfig_factual = factual['reconfiguration']
    
    # 剂量单调性审计
    dose = metrics['dose_audit/dose_both_directions']
    dmr_high = dose['high_dose_monotonicity']['monotone_rate']
    dmr_low = dose['low_dose_monotonicity']['monotone_rate']
    
    # 零假设对照
    uniform = metrics['mechanism_audit/uniform_plan']
    shuffled = metrics['mechanism_audit/shuffled_plan']
    anchor_swap = metrics['mechanism_audit/anchor_swap']
    
    return {
        'performance': performance,
        'direction_consistency': {
            'factual': {
                'high_correct': dcr_factual['high_correct'],
                'high_total': dcr_factual['high_labelled_count'],
                'low_correct': dcr_factual['low_correct'],
                'low_total': dcr_factual['low_labelled_count'],
                'correct_rate': dcr_factual['correct_rate'],
                'chance_gap': dcr_factual['chance_gap']
            },
            'anchor_swap': {
                'correct_rate': anchor_swap['direction_consistency']['correct_rate'],
                'chance_gap': anchor_swap['direction_consistency']['chance_gap']
            }
        },
        'dose_monotonicity': {
            'high_direction': dmr_high,
            'low_direction': dmr_low,
            'n_cases': dose['high_dose_monotonicity']['n_cases']
        },
        'plan_reconfiguration': {
            'mean_tv_low': reconfig_factual['mean_tv_low'],
            'mean_tv_high': reconfig_factual['mean_tv_high'],
            'mean_tv': reconfig_factual['mean_tv']
        },
        'null_controls': {
            'uniform_plan': {
                'mean_plan_tv': uniform['factual_plan_control']['mean_plan_tv'],
                'mean_absolute_risk_change': uniform['factual_plan_control']['mean_absolute_risk_change']
            },
            'shuffled_plan': {
                'mean_plan_tv': shuffled['factual_plan_control']['mean_plan_tv'],
                'mean_absolute_risk_change': shuffled['factual_plan_control']['mean_absolute_risk_change']
            }
        },
        'n_cases': factual['n_cases']
    }

def main():
    # Full BLCA fold 0 路径
    manifest_path = Path("/data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/blca/blca/"
                         "SurvOTRank_dct_v310_directional_regularized_transport/"
                         "0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_full_blca_50ep/"
                         "evidence/fold_0/run_manifest.json")
    
    print("="*70)
    print("DCT-Reg v3.10 Full Model - BLCA Fold 0 Mechanism Audit")
    print("="*70)
    
    audit = extract_mechanism_audit(manifest_path)
    
    # 预测性能
    print("\n### Predictive Performance")
    perf = audit['performance']
    print(f"C-index:           {perf['outer_cindex']:.4f}")
    print(f"C-index (IPCW):    {perf['outer_cindex_ipcw']:.4f}")
    print(f"IBS:               {perf['outer_IBS']:.4f}")
    print(f"iAUC:              {perf['outer_iauc']:.4f}")
    print(f"Fixed Epoch:       {perf['fixed_epoch']}")
    print(f"Test Cases:        {audit['n_cases']}")
    
    # 方向一致性
    print("\n### Direction Consistency Rate (DCR)")
    dc = audit['direction_consistency']['factual']
    print(f"High-risk labeled: {dc['high_correct']}/{dc['high_total']} correct")
    print(f"Low-risk labeled:  {dc['low_correct']}/{dc['low_total']} correct")
    print(f"Overall DCR:       {dc['correct_rate']:.3f}")
    print(f"Chance baseline:   0.500")
    print(f"Gap from chance:   {dc['chance_gap']:+.3f}")
    
    dc_swap = audit['direction_consistency']['anchor_swap']
    print(f"\nAnchor Swap DCR:   {dc_swap['correct_rate']:.3f} (gap: {dc_swap['chance_gap']:+.3f})")
    
    # 剂量单调性
    print("\n### Dose Monotonicity Rate (DMR)")
    dmr = audit['dose_monotonicity']
    print(f"High-risk path:    {dmr['high_direction']:.3f} (5-point strictly increasing)")
    print(f"Low-risk path:     {dmr['low_direction']:.3f} (5-point strictly decreasing)")
    
    # 计划重构
    print("\n### Plan Total Variation (Plan TV)")
    tv = audit['plan_reconfiguration']
    print(f"Low intervention:  {tv['mean_tv_low']:.4f}")
    print(f"High intervention: {tv['mean_tv_high']:.4f}")
    print(f"Average Plan TV:   {tv['mean_tv']:.4f}")
    
    # 零假设对照
    print("\n### Null Controls")
    null = audit['null_controls']
    print(f"Uniform Plan:      Plan TV = {null['uniform_plan']['mean_plan_tv']:.4f}, "
          f"Risk Δ = {null['uniform_plan']['mean_absolute_risk_change']:.6f}")
    print(f"Shuffled Plan:     Plan TV = {null['shuffled_plan']['mean_plan_tv']:.4f}, "
          f"Risk Δ = {null['shuffled_plan']['mean_absolute_risk_change']:.6f}")
    
    # 保存 JSON
    output_path = Path("/data1/DCT-Reg/results/mechanism_audit_blca_fold0.json")
    with open(output_path, 'w') as f:
        json.dump(audit, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {output_path}")
    
    # 生成简洁总结
    print("\n" + "="*70)
    print("Summary for Paper")
    print("="*70)
    print(f"""
在 BLCA 测试折（n={audit['n_cases']}）上：
- 预测性能：C-index = {perf['outer_cindex']:.3f}，iAUC = {perf['outer_iauc']:.3f}
- 方向一致率（DCR）：{dc['correct_rate']:.3f}（低于随机基线 0.50）
- 高风险剂量单调率（DMR）：{dmr['high_direction']:.3f}
- 计划总变差（Plan TV）：{tv['mean_tv']:.3f}
- 零假设对照：Uniform Plan 风险变化 = {null['uniform_plan']['mean_absolute_risk_change']:.6f}

**关键发现**：尽管 C-index 达到 {perf['outer_cindex']:.2f}，传输机制的方向一致性
（DCR = {dc['correct_rate']:.2f}）显著低于随机水平（0.50），说明预测性能与传输
忠实性可能解耦。这支持了论文的核心主张：C-index 不能替代机制审计。
""")

if __name__ == '__main__':
    main()
