#!/bin/bash
# 顺序训练五折（Fold 1-4，Fold 0 已完成）
# Fold 0 结果: Val C-index = 0.7100

set -e
cd /data1/DCT-Reg

export PYTHONPATH=/data1/DCT-Reg

for fold in 1 2 3 4; do
    echo ""
    echo "==========================================="
    echo "开始训练 Fold ${fold}"
    echo "==========================================="

    LOG_FILE="/data1/DCT-Reg/logs/fixed_anchors_blca_fold${fold}_20260907.log"

    /home/ubuntu/.conda/envs/trisurv/bin/python -m survot_rank.cli train \
        --config configs/fixed_5fold/blca_fixed_anchors_fold${fold}.yaml \
        2>&1 | tee "$LOG_FILE"

    echo ""
    echo "Fold ${fold} 训练完成"

    # 检查结果
    BEST=$(grep "best cindex" "$LOG_FILE" | tail -1)
    if [ -n "$BEST" ]; then
        echo "  Best: $BEST"
    fi
done

echo ""
echo "==========================================="
echo "全部五折训练完成!"
echo "==========================================="
