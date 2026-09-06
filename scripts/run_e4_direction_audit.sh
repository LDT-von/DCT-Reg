#!/bin/bash
# E4 持续干预审计实验
# 测试方向一致性约束是否生效

set -e

REPO_ROOT="/data1/DCT-Reg"
cd "$REPO_ROOT"

# 实验配置
CHECKPOINT_BASE="results/backups/direction_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep"
CONFIG="configs/dct_v310_directional_regularized_transport.yaml"
CANCER="blca"
OUTPUT_DIR="results/e4_audit_direction_only"
ALPHAS="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
DEVICE="cuda:0"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "E4: 持续干预审计 (Continuous Intervention Audit)"
echo "=========================================="
echo "癌症类型: $CANCER"
echo "Alpha 值: $ALPHAS"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 对每个 fold 运行审计
for FOLD in 0 1 2 3 4; do
    CHECKPOINT="$CHECKPOINT_BASE/model_best_s${FOLD}.pth"
    OUTPUT="$OUTPUT_DIR/${CANCER}_fold${FOLD}_audit.csv"
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "⚠️  跳过 fold ${FOLD}: 检查点不存在"
        continue
    fi
    
    echo ""
    echo "运行 fold ${FOLD}..."
    echo "  检查点: $CHECKPOINT"
    echo "  输出: $OUTPUT"
    
    python scripts/e4_continuous_intervention_audit_v2.py \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --study "$CANCER" \
        --fold "$FOLD" \
        --output "$OUTPUT" \
        --alphas "$ALPHAS" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Fold ${FOLD} 完成"
    else
        echo "✗ Fold ${FOLD} 失败"
    fi
done

echo ""
echo "=========================================="
echo "汇总结果"
echo "=========================================="

# 汇总所有 fold 的结果
python -c "
import pandas as pd
import numpy as np
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
cancer = '$CANCER'

all_data = []
for fold in range(5):
    csv_file = output_dir / f'{cancer}_fold{fold}_audit.csv'
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        df['fold'] = fold
        all_data.append(df)

if not all_data:
    print('⚠️  没有找到任何结果')
    exit(1)

df_all = pd.concat(all_data, ignore_index=True)

print('\\n' + '='*60)
print('E4 审计结果汇总')
print('='*60)
print(f'总患者数: {len(df_all[\"patient_id\"].unique())}')
print(f'总测量数: {len(df_all)}')

# 分析单调性
print('\\n' + '='*60)
print('方向一致性分析')
print('='*60)

patients = df_all['patient_id'].unique()
monotonic_low = 0
monotonic_high = 0

for pid in patients:
    patient_df = df_all[df_all['patient_id'] == pid].sort_values('alpha')
    
    # 低风险方向：风险应该下降
    low_df = patient_df[patient_df['direction'] == 'low_risk']
    if len(low_df) > 1:
        risks = low_df['risk_pred'].values
        if all(risks[i] >= risks[i+1] for i in range(len(risks)-1)):
            monotonic_low += 1
    
    # 高风险方向：风险应该上升
    high_df = patient_df[patient_df['direction'] == 'high_risk']
    if len(high_df) > 1:
        risks = high_df['risk_pred'].values
        if all(risks[i] <= risks[i+1] for i in range(len(risks)-1)):
            monotonic_high += 1

n_patients = len(patients)
print(f'\\n低风险方向单调递减率: {monotonic_low}/{n_patients} ({100*monotonic_low/n_patients:.1f}%)')
print(f'高风险方向单调递增率: {monotonic_high}/{n_patients} ({100*monotonic_high/n_patients:.1f}%)')

# 风险变化量
alpha_1 = df_all[df_all['alpha'] == 1.0]
low_changes = alpha_1[alpha_1['direction'] == 'low_risk']['risk_change']
high_changes = alpha_1[alpha_1['direction'] == 'high_risk']['risk_change']

print(f'\\n风险变化 (α=1.0):')
print(f'  低风险方向: {low_changes.mean():.4f} ± {low_changes.std():.4f}')
print(f'  高风险方向: {high_changes.mean():.4f} ± {high_changes.std():.4f}')

# 关键判断
print('\\n' + '='*60)
print('关键结论')
print('='*60)

monotonic_rate_low = monotonic_low / n_patients
monotonic_rate_high = monotonic_high / n_patients

if monotonic_rate_low > 0.8 and monotonic_rate_high > 0.8:
    print('✓ 方向一致性良好 (>80%)')
    print('  → DCT 方向正则化有效')
    print('  → 可以进行大规模实验')
elif monotonic_rate_low > 0.6 and monotonic_rate_high > 0.6:
    print('⚠️  方向一致性中等 (60-80%)')
    print('  → 需要检查超参数')
elif monotonic_rate_low < 0.5 or monotonic_rate_high < 0.5:
    print('✗ 方向一致性差 (<50%)')
    print('  → 方向约束未生效')
    print('  → 不建议继续大规模实验')
else:
    print('△ 方向一致性一般 (50-60%)')
    print('  → 需要进一步诊断')

# 保存汇总
summary_file = output_dir / f'{cancer}_summary.csv'
summary = pd.DataFrame({
    'metric': [
        'n_patients',
        'monotonic_low_rate',
        'monotonic_high_rate',
        'mean_risk_change_low',
        'mean_risk_change_high',
    ],
    'value': [
        n_patients,
        monotonic_rate_low,
        monotonic_rate_high,
        low_changes.mean(),
        high_changes.mean(),
    ]
})
summary.to_csv(summary_file, index=False)
print(f'\\n汇总保存至: {summary_file}')
"

echo ""
echo "=========================================="
echo "实验完成"
echo "=========================================="
echo "结果保存在: $OUTPUT_DIR"
