#!/bin/bash
# 五折顺序训练脚本
# 训练 Fold 1-4 (Fold 0 已完成)

set -e
cd /data1/DCT-Reg

# 模型参数 (从配置中读取)
COMMON_ARGS="--study blca \
  --survot_method dct_v310_fixed_anchors \
  --signature combine \
  --rna_format Pathways \
  --label_col survival_months_dss \
  --max_epochs 30 \
  --batch_size 8 \
  --lr 0.0005 \
  --results_dir results_fixed_anchors \
  --specific_simple dct_v310_fixed_anchors_blca_proof"

for fold in 1 2 3 4; do
    echo "=========================================="
    echo "开始训练 Fold ${fold}"
    echo "=========================================="

    LOG_FILE="/data1/DCT-Reg/logs/fixed_anchors_blca_fold${fold}_20260907.log"

    /home/ubuntu/.conda/envs/trisurv/bin/python survot_rank/training/main.py \
        $COMMON_ARGS \
        --k_start $fold \
        --k_end $((fold + 1)) \
        --seed 3 \
        2>&1 | tee "$LOG_FILE"

    echo "Fold ${fold} 完成"
done

echo "=========================================="
echo "全部五折训练完成!"
echo "=========================================="
