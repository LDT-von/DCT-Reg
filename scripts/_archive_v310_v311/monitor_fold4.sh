#!/bin/bash
# 监控 fold4 进度，每 60 秒打印一次最新 epoch + best cindex
# 完成（best 出现且 epoch 停止增长）后自动退出

LOG=/data1/DCT-Reg/logs/v313_abl_double_w/fold4/fold4.log

if [ ! -f "$LOG" ]; then
  echo "❌ $LOG 不存在"
  exit 1
fi

prev_line=""
stable_count=0

while true; do
  # 取最后一条 val cindex 行
  cur=$(grep -E "\[Epoch [0-9]+\] val cindex" "$LOG" 2>/dev/null | tail -1)
  best=$(grep "best cindex" "$LOG" 2>/dev/null | tail -1)

  ts=$(date '+%H:%M:%S')

  if [ -z "$cur" ]; then
    echo "[$ts] ⏳ fold4 还没开始第 1 epoch（可能在初始化 dataloader）"
  else
    # 提取 epoch 号和 cindex
    epoch=$(echo "$cur" | grep -oE "\[Epoch [0-9]+\]" | grep -oE "[0-9]+")
    cindex=$(echo "$cur" | grep -oE "val cindex=[0-9.]+" | grep -oE "[0-9.]+")
    printf "[%s] 🔄 epoch %2s/30 | val cindex=%s | %s\n" "$ts" "$epoch" "$cindex" "${best:-best: --}"

    # 检测完成：best 已出现且 epoch 已停止增长
    if [ -n "$best" ]; then
      # 从日志末尾找 "[stopped @epoch XX]"
      stopped=$(grep -oE "stopped @epoch [0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
      if [ -n "$stopped" ] && [ "$epoch" = "$stopped" ]; then
        echo ""
        echo "[$ts] ✅ fold4 完成！$best"
        echo ""
        # 跑一次汇总
        echo "=== 全部 5 fold 汇总 ==="
        for f in 0 1 2 3 4; do
          flog=/data1/DCT-Reg/logs/v313_abl_double_w/fold$f/fold$f.log
          fb=$(grep "best cindex" "$flog" 2>/dev/null | tail -1)
          echo "fold$f: ${fb:-—还在跑—}"
        done
        exit 0
      fi
    fi
  fi

  # 检查进程是否还在
  if ! pgrep -f "k_start=4" > /dev/null; then
    echo ""
    echo "[$ts] ⚠️ fold4 进程已退出但 best 未出现，检查日志末尾"
    tail -20 "$LOG"
    exit 1
  fi

  sleep 60
done
