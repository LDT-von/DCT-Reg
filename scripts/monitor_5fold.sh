#!/bin/bash
echo "==========================================="
echo "五折训练状态 ($(date '+%H:%M:%S'))"
echo "==========================================="

for fold in 0 1 2 3 4; do
    LOG="/data1/DCT-Reg/results/dct_v3.30/prognostic_rank/logs/fold${fold}.log"
    if [ ! -f "$LOG" ]; then
        echo "Fold $fold: ⏳ 等待启动"
        continue
    fi
    
    EPOCH=$(grep -oP 'Epoch \K\d+(?=/)' "$LOG" 2>/dev/null | tail -1)
    BEST=$(grep -oP 'best cindex=\K[0-9.]+' "$LOG" 2>/dev/null | tail -1)
    
    if [ -n "$BEST" ]; then
        echo "Fold $fold: ✅ 完成 (best C-index=$BEST)"
    elif [ -n "$EPOCH" ]; then
        echo "Fold $fold: 🔄 Epoch $EPOCH/50"
    else
        echo "Fold $fold: ⏳ 初始化中"
    fi
done

# 当前训练进程
RUNNING=$(ps aux | grep "survot_rank.cli train" | grep -v grep | wc -l)
echo ""
echo "运行中训练进程: $RUNNING"
