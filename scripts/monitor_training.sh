#!/bin/bash
# 监控固定锚点实验的训练进度

LOG_FILE="/data1/DCT-Reg/logs/fixed_anchors_blca_fold0_20260907.log"
PID=1666160

echo "=========================================="
echo "固定锚点实验训练监控"
echo "=========================================="
echo ""

# 检查进程状态
if ps -p $PID > /dev/null 2>&1; then
    echo "✓ 训练进程运行中 (PID: $PID)"
    
    # 显示进程信息
    echo ""
    echo "进程信息:"
    ps -p $PID -o pid,ppid,cmd,%cpu,%mem,etime
    
else
    echo "✗ 训练进程已停止 (PID: $PID)"
    echo ""
    echo "检查日志最后100行:"
    tail -100 "$LOG_FILE"
    exit 1
fi

echo ""
echo "=========================================="

# 提取当前epoch信息
CURRENT_EPOCH=$(grep -oP 'Epoch \K\d+(?=/30)' "$LOG_FILE" | tail -1)
if [ -n "$CURRENT_EPOCH" ]; then
    echo "当前进度: Epoch $CURRENT_EPOCH/30"
    PROGRESS=$((CURRENT_EPOCH * 100 / 30))
    echo "完成度: ${PROGRESS}%"
else
    echo "正在初始化..."
fi

echo ""
echo "=========================================="

# 最新的验证性能
echo "最新验证性能:"
grep "val cindex=" "$LOG_FILE" | tail -5 | while read line; do
    EPOCH_NUM=$(echo "$line" | grep -oP '\[Epoch \K\d+')
    CINDEX=$(echo "$line" | grep -oP 'val cindex=\K[0-9.]+')
    IPCW=$(echo "$line" | grep -oP 'ipcw=\K[0-9.]+')
    echo "  Epoch $EPOCH_NUM: C-index=$CINDEX, IPCW=$IPCW"
done

echo ""
echo "=========================================="

# 显示最新日志
echo "最新日志 (最后20行):"
echo ""
tail -20 "$LOG_FILE"

echo ""
echo "=========================================="
echo "持续监控命令:"
echo "  tail -f $LOG_FILE"
echo ""
echo "GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1
