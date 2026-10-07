#!/bin/bash
# 监控 DCT v3.10 UNI BLCA 训练进度
# 用法: ./monitor_blca.sh

LOG=$(ls -t /data1/DCT-Reg/logs/v310_uni_blca_p2048_30ep_*.log 2>/dev/null | head -1)

if [ -z "$LOG" ]; then
    echo "❌ 未找到日志文件"
    exit 1
fi

echo "=============================================="
echo "Log: $LOG"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits | awk -F', ' '{printf "GPU%s %s: %s/%s MiB  GPU%3s%%  %s°C\n", $1, $2, $3, $4, $5, $6}'

echo ""
echo "--- 进程状态 ---"
PIDS=$(pgrep -f "survot_rank.cli.*blca" 2>/dev/null)
if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        ELAPSED=$(ps -p $pid -o etimes= 2>/dev/null | tr -d ' ')
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_SEC=$((ELAPSED % 60))
        FOLD=$(ps -p $pid -o args= 2>/dev/null | grep -oP 'k_start=\K\d+')
        echo "  PID $pid  Fold $FOLD  运行 ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    done
else
    echo "  (无进程运行)"
fi

echo ""
echo "=============================================="
echo " 各 Fold val c-index (最新 10 epoch)"
echo "=============================================="

# 解析日志：按 Fold 分组 val cindex
# 日志格式: [Fold N] Epoch X/30: ...  (epoch header)
#           [Epoch X] val cindex=Y ... (val metrics after epoch)
# 策略: 追踪当前 fold，上下文行解析
python3 - "$LOG" << 'PYEOF'
import sys, re

log_path = sys.argv[1]
with open(log_path) as f:
    lines = f.readlines()

fold_scores = {}  # fold -> list of (epoch, cindex)
current_fold = None
epoch_pattern = re.compile(r'\[Fold (\d+)\] Epoch (\d+)/30:')
val_pattern = re.compile(r'\[Epoch \d+\] val cindex=([\d.]+)')

for i, line in enumerate(lines):
    m_fold = epoch_pattern.search(line)
    if m_fold:
        current_fold = int(m_fold.group(1))
    
    m_val = val_pattern.search(line)
    if m_val and current_fold is not None:
        cindex = float(m_val.group(1))
        if current_fold not in fold_scores:
            fold_scores[current_fold] = []
        fold_scores[current_fold].append(cindex)

if not fold_scores:
    print("  (暂无 val cindex 数据)")
else:
    for fold in sorted(fold_scores.keys()):
        scores = fold_scores[fold]
        print(f"\n  === Fold {fold} ({len(scores)} epochs) ===")
        # show latest 10
        recent = scores[-10:] if len(scores) > 10 else scores
        best = max(scores)
        for j, s in enumerate(recent):
            epoch_num = len(scores) - len(recent) + j
            marker = " ⭐" if s == best else ""
            print(f"    Epoch {epoch_num:2d}:  cindex = {s:.4f}{marker}")
        print(f"    Best: {best:.4f}")
PYEOF

echo ""
echo "=============================================="
echo " 当前日志末尾"
echo "=============================================="
tail -3 "$LOG" | grep -v "^$" | sed 's/.*\] Epoch \([0-9]*\/30.*\)/  \1/' | head -2

echo ""
echo "--- 5-Fold 整体进度 ---"
# 通过进程数量判断
RUNNING=$(pgrep -f "survot_rank.cli.*blca" | wc -l)
echo "  运行中进程数: $RUNNING"
echo "  Fold 0 ~ Fold 4 顺序执行中"
EST_MIN=$((RUNNING * 30))
echo "  预计剩余: ~${EST_MIN} min"
