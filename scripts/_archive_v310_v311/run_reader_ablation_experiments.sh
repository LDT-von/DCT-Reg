#!/bin/bash
# 读取器消融实验：测试 full / pair-context-only / plan-only
# 目标：确认 pair-context 旁路是否压制了 OT 计划

set -e

REPO_ROOT="/data1/DCT-Reg"
cd "$REPO_ROOT"

# 实验配置
CHECKPOINT_BASE="results/backups/direction_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep"
CANCER="blca"
OUTPUT_DIR="results/reader_ablation"
MODES="full,pair_context_only,plan_only"
GPU=0

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "读取器消融实验 (Reader Ablation)"
echo "=========================================="
echo "癌症类型: $CANCER"
echo "测试模式: $MODES"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 对每个 fold 运行消融
for FOLD in 0 1 2 3 4; do
    CHECKPOINT="$CHECKPOINT_BASE/model_best_s${FOLD}.pth"
    OUTPUT="$OUTPUT_DIR/${CANCER}_fold${FOLD}_ablation.json"
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "⚠️  跳过 fold ${FOLD}: 检查点不存在"
        continue
    fi
    
    echo ""
    echo "运行 fold ${FOLD}..."
    echo "  检查点: $CHECKPOINT"
    echo "  输出: $OUTPUT"
    
    python scripts/ablate_reader.py \
        --checkpoint "$CHECKPOINT" \
        --cancer "$CANCER" \
        --fold "$FOLD" \
        --modes "$MODES" \
        --output "$OUTPUT" \
        --gpu "$GPU"
    
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
import json
from pathlib import Path
import numpy as np

output_dir = Path('$OUTPUT_DIR')
cancer = '$CANCER'

all_results = {'full': [], 'pair_context_only': [], 'plan_only': []}

for fold in range(5):
    result_file = output_dir / f'{cancer}_fold{fold}_ablation.json'
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
            for r in data['results']:
                all_results[r['mode']].append(r['cindex'])

print('\\n' + '='*60)
print('读取器消融实验汇总 (5-Fold 平均)')
print('='*60)

for mode in ['full', 'pair_context_only', 'plan_only']:
    scores = all_results[mode]
    if scores:
        mean_ci = np.mean(scores)
        std_ci = np.std(scores)
        print(f'{mode:25s}: C-index = {mean_ci:.4f} ± {std_ci:.4f} (n={len(scores)})')

# 计算相对差异
full_mean = np.mean(all_results['full']) if all_results['full'] else None
if full_mean:
    print('\\n' + '='*60)
    print('相对于完整模型的差异')
    print('='*60)
    for mode in ['pair_context_only', 'plan_only']:
        scores = all_results[mode]
        if scores:
            mode_mean = np.mean(scores)
            diff = mode_mean - full_mean
            pct = 100 * diff / full_mean
            print(f'{mode:25s}: {diff:+.4f} ({pct:+.1f}%)')

# 关键结论
print('\\n' + '='*60)
print('核心判断')
print('='*60)
if all_results['pair_context_only'] and all_results['full']:
    pair_mean = np.mean(all_results['pair_context_only'])
    full_mean = np.mean(all_results['full'])
    diff = abs(pair_mean - full_mean)
    if diff < 0.01:
        print('⚠️  警告: pair_context_only ≈ full')
        print('    → OT 计划可能被旁路压制')
        print('    → 模型主要依赖 slot 对交互，而非传输计划')
    else:
        print('✓ pair_context_only 与 full 有明显差异')
        print('  → OT 计划贡献有效')
"

echo ""
echo "=========================================="
echo "实验完成"
echo "=========================================="
echo "结果保存在: $OUTPUT_DIR"
