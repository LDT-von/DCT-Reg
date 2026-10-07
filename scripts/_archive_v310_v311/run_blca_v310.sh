#!/bin/bash
# ============================================================================
# 重新训练 BLCA (v3.10 / DCT-Reg) — 5 fold × 50 epoch
#
# 目标：复现 results/dct_v3.10/robust/final_50ep_old/blca/.../blca_50ep/
#       里的 5-fold 最佳均值 C-index = 0.7208 ± 0.0162
#
# 注意事项（实测会出现）：
#   * cuDNN 是非确定性（train_runner.py 故意关掉了 deterministic），
#     每次跑的具体数字会有 ±0.01~0.02 浮动。
#   * 每个 fold 跑到 ~50 epoch 需要 1-3 小时（视 GPU 而定）。
#   * 该入口默认把结果写到 results/dct_v3.10/robust/final/blca/，
#     如果不想覆盖原结果，请改 --results-dir 指向新目录（见下方说明）。
#
# 用法：
#   bash scripts/run_blca_v310.sh                # 默认 plan 模式（只打印计划）
#   bash scripts/run_blca_v310.sh run            # 真的开跑（5 fold 顺序）
#   bash scripts/run_blca_v310.sh run --force    # 强制重跑（即使已有 split_*.pkl）
#   bash scripts/run_blca_v310.sh smoke          # 2 epoch 烟雾测试
# ============================================================================

set -euo pipefail

cd /data1/DCT-Reg

MODE="${1:-plan}"
shift || true

# 这些值来自 configs/dct_v310_directional_regularized_transport.yaml
# 如果想换 cancer 或 fold 数，改下面这一行：
CANCERS="blca"
FOLDS="0,1,2,3,4"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"

# 数据路径：默认指 TCGA-UNI2-h 特征 + 5fold_uni2h splits
DATA_ROOT="${UNI2H_ROOT:-/data1/TCGA-UNI2-h-features}"
DATA_CSV_ROOT="${DCT_DATA_CSV_ROOT:-data/dataset_csv}"

echo "============================================================"
echo "BLCA (v3.10 / DCT-Reg) — 5-fold × 50 epoch"
echo "============================================================"
echo "Mode:    $MODE"
echo "Cancer:  $CANCERS"
echo "Folds:   $FOLDS"
echo "GPU:     $GPU"
echo "Python:  $PYTHON_BIN"
echo "Data:    $DATA_ROOT"
echo "Splits:  $DATA_CSV_ROOT/splits/5fold_uni2h"
echo "Config:  configs/dct_v310_directional_regularized_transport.yaml"
echo "------------------------------------------------------------"
echo "Output:  results/dct_v3.10/robust/final/blca/<...>/blca_50ep/"
echo "============================================================"
echo "  保留的 BLCA 实际数据在  results/dct_v3.10/robust/final_50ep_old/blca/"
echo "  本脚本输出会到          results/dct_v3.10/robust/final/blca/"
echo "  （两个目录独立，不冲突；如要替换旧数据请手动移动或删旧）"
echo "============================================================"

python scripts/run_dct_v310_final_cross_cancer.py "$MODE" \
    --cancers "$CANCERS" \
    --folds "$FOLDS" \
    --gpu "$GPU" \
    --python "$PYTHON_BIN" \
    --data-root "$DATA_ROOT" \
    --data-csv-root "$DATA_CSV_ROOT" \
    "$@"
