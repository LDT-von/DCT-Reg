#!/bin/bash
# ============================================================================
# 监控 v3.10 训练队列状态
# 用法:
#   bash scripts/monitor_v310_queue.sh                     # 快照
#   bash scripts/monitor_v310_queue.sh watch 30          # 每 30s 刷新
# ============================================================================

set -euo pipefail
cd /data1/DCT-Reg

MODE="${1:-snapshot}"
INTERVAL="${2:-30}"
PYTHON_BIN=/home/ubuntu/.conda/envs/trisurv/bin/python

echo "============================================================"
echo "v3.10 训练队列监控"
echo "模式: $MODE"
echo "刷新间隔: ${INTERVAL}s（watch 模式）"
echo "============================================================"

while true; do
    clear
    echo "============================================================"
    echo "v3.10 训练队列状态  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "============================================================"
    
    echo ""
    echo "【GPU 状态】"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    echo ""
    
    echo "【队列锁状态】"
    ls -la results/.locks/ 2>/dev/null || echo "  (无锁文件)"
    echo ""

    echo "【任务 1: v310_uni2h_extend (4 癌种, UNI2-h)】"
    TASK1_DIR="results/dct_v3.10_uni2h_extend/final"
    if [ -d "$TASK1_DIR" ]; then
        for cancer in coadread luad stad ucec; do
            cancer_dir="$TASK1_DIR/$cancer"
            if [ -d "$cancer_dir" ]; then
                total=5
                completed=$(find "$cancer_dir" -name "split_*_results_final.pkl" 2>/dev/null | wc -l)
                echo "  $cancer: $completed/$total folds 完成"
                # 最新 log
                latest_log=$(find "$cancer_dir" -name "log_start_*.txt" -newer "$cancer_dir/.last_check" 2>/dev/null | head -1)
                if [ -n "$latest_log" ] && [ -f "$latest_log" ]; then
                    last_line=$(tail -1 "$latest_log" 2>/dev/null | head -c 120)
                    echo "    最新: $(basename $latest_log) → $last_line"
                fi
            else
                echo "  $cancer: 目录不存在（未开始）"
            fi
        done
    else
        echo "  (任务 1 目录不存在，未开始)"
    fi

    echo ""
    echo "【任务 2: v310_uni_all10 (10 癌种, UNI)】"
    TASK2_DIR="results/dct_v3.10_uni_all10/final"
    if [ -d "$TASK2_DIR" ]; then
        for cancer in blca brca coadread hnsc kirc luad lusc skcm stad ucec; do
            cancer_dir="$TASK2_DIR/$cancer"
            if [ -d "$cancer_dir" ]; then
                total=5
                completed=$(find "$cancer_dir" -name "split_*_results_final.pkl" 2>/dev/null | wc -l)
                echo "  $cancer: $completed/$total folds 完成"
                latest_log=$(find "$cancer_dir" -name "log_start_*.txt" -newer "$cancer_dir/.last_check" 2>/dev/null | head -1)
                if [ -n "$latest_log" ] && [ -f "$latest_log" ]; then
                    last_line=$(tail -1 "$latest_log" 2>/dev/null | head -c 120)
                    echo "    最新: $(basename $latest_log) → $last_line"
                fi
            else
                echo "  $cancer: 目录不存在（未开始）"
            fi
        done
    else
        echo "  (任务 2 目录不存在，未开始)"
    fi

    if [ "$MODE" != "watch" ]; then
        break
    fi
    echo ""
    echo "($INTERVAL 秒后刷新...)"
    sleep $INTERVAL
done
