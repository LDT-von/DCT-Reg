#!/usr/bin/env python3
"""小规模测试：验证读取器消融和方向审计是否正常工作"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("小规模生死门测试")
print("="*70)
print("测试 1: 读取器消融 (1 fold)")
print("测试 2: 方向一致性审计 (1 fold)")
print("="*70)

# 配置
CHECKPOINT = "results/backups/direction_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep/model_best_s0.pth"
CONFIG = "configs/dct_v310_directional_regularized_transport.yaml"
CANCER = "blca"
FOLD = 0

print(f"\n检查点: {CHECKPOINT}")
print(f"配置: {CONFIG}")
print(f"癌症: {CANCER}, Fold: {FOLD}")

# 测试 1: 读取器消融
print("\n" + "="*70)
print("测试 1: 读取器消融")
print("="*70)

import subprocess
import json
from pathlib import Path

output_dir = Path("results/small_test")
output_dir.mkdir(parents=True, exist_ok=True)

ablation_output = output_dir / "ablation_test.json"

cmd = [
    "python", "scripts/ablate_reader.py",
    "--checkpoint", CHECKPOINT,
    "--cancer", CANCER,
    "--fold", str(FOLD),
    "--modes", "full,pair_context_only,plan_only",
    "--output", str(ablation_output),
    "--gpu", "0"
]

print(f"运行命令: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✓ 读取器消融测试通过")
    
    # 加载并显示结果
    with open(ablation_output) as f:
        data = json.load(f)
    
    print("\n结果:")
    for r in data['results']:
        print(f"  {r['mode']:25s}: C-index = {r['cindex']:.4f}")
    
    # 关键判断
    full_ci = next(r['cindex'] for r in data['results'] if r['mode'] == 'full')
    pair_ci = next(r['cindex'] for r in data['results'] if r['mode'] == 'pair_context_only')
    diff = abs(pair_ci - full_ci)
    
    print(f"\n关键指标:")
    print(f"  Full - Pair_context_only = {full_ci - pair_ci:.4f}")
    
    if diff < 0.01:
        print("  ⚠️  警告: OT计划可能被旁路压制")
        ablation_pass = False
    else:
        print("  ✓ OT计划有贡献")
        ablation_pass = True
else:
    print("✗ 读取器消融测试失败")
    print(result.stderr)
    ablation_pass = False

# 测试 2: 方向一致性审计
print("\n" + "="*70)
print("测试 2: 方向一致性审计")
print("="*70)

audit_output = output_dir / "audit_test.csv"

    cmd = [
    "python", "scripts/e4_audit_adapted.py",
    "--checkpoint", CHECKPOINT,
    "--study", CANCER,
    "--fold", str(FOLD),
    "--output", str(audit_output),
    "--alphas", "0.0,0.5,1.0",  # 只测试 3 个 alpha 值
    "--device", "cuda:0"
]

print(f"运行命令: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✓ 方向审计测试通过")
    
    # 分析结果
    import pandas as pd
    df = pd.read_csv(audit_output)
    
    print(f"\n收集的数据: {len(df)} 条记录")
    print(f"患者数: {len(df['patient_id'].unique())}")
    
    # 检查单调性
    patients = df['patient_id'].unique()
    monotonic_low = 0
    monotonic_high = 0
    
    for pid in patients:
        patient_df = df[df['patient_id'] == pid].sort_values('alpha')
        
        # 低风险方向
        low_df = patient_df[patient_df['direction'] == 'low_risk']
        if len(low_df) > 1:
            risks = low_df['risk_pred'].values
            if all(risks[i] >= risks[i+1] for i in range(len(risks)-1)):
                monotonic_low += 1
        
        # 高风险方向
        high_df = patient_df[patient_df['direction'] == 'high_risk']
        if len(high_df) > 1:
            risks = high_df['risk_pred'].values
            if all(risks[i] <= risks[i+1] for i in range(len(risks)-1)):
                monotonic_high += 1
    
    n_patients = len(patients)
    low_rate = monotonic_low / n_patients
    high_rate = monotonic_high / n_patients
    
    print(f"\n单调性率:")
    print(f"  低风险方向: {monotonic_low}/{n_patients} ({100*low_rate:.1f}%)")
    print(f"  高风险方向: {monotonic_high}/{n_patients} ({100*high_rate:.1f}%)")
    
    if low_rate > 0.8 and high_rate > 0.8:
        print("  ✓ 方向一致性良好")
        audit_pass = True
    elif low_rate > 0.6 and high_rate > 0.6:
        print("  △ 方向一致性中等")
        audit_pass = True
    else:
        print("  ✗ 方向一致性差")
        audit_pass = False
else:
    print("✗ 方向审计测试失败")
    print(result.stderr)
    audit_pass = False

# 最终判断
print("\n" + "="*70)
print("生死门判断")
print("="*70)

if ablation_pass and audit_pass:
    print("✓ 通过生死门")
    print("  → 可以继续大规模实验")
    exit_code = 0
elif not ablation_pass:
    print("✗ 未通过生死门: OT计划被旁路压制")
    print("  → 需要修复融合模块")
    exit_code = 1
elif not audit_pass:
    print("✗ 未通过生死门: 方向一致性不足")
    print("  → 需要调整正则化强度")
    exit_code = 1
else:
    print("△ 部分通过，需要进一步诊断")
    exit_code = 2

print("="*70)
sys.exit(exit_code)
