#!/bin/bash
# DCT-Reg 新会话启动自检脚本
# 打印：Python / 数据 / 划分 / 最近训练进度 / 关键规则摘要
# 任何一项异常时给出建议，不阻断。

set +e
cd /data1/DCT-Reg
RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; NC='\033[0m'

echo "================================================================"
echo "  DCT-Reg 会话启动自检   $(date '+%F %T')"
echo "================================================================"

# ---- A. Python 环境 ----
echo ""
echo "[A] Python 环境"
DCT_PY=/home/ubuntu/.conda/envs/dct/bin/python3.10
DEFAULT_PY=/data/env/data/env/anaconda/bin/python3

if [ -x "$DCT_PY" ]; then
  DCT_INFO=$("$DCT_PY" -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, arch={torch.cuda.get_arch_list()[-3:]}')" 2>&1 | tail -1)
  echo -e "  ${GRN}OK${NC}   dct env:   $DCT_PY"
  echo "       $DCT_INFO"
else
  echo -e "  ${RED}FAIL${NC} dct env 缺失: $DCT_PY"
fi

DEF_INFO=$("$DEFAULT_PY" -c "import torch; print(f'torch={torch.__version__}, arch_max={torch.cuda.get_arch_list()[-1]}')" 2>&1 | tail -1)
if echo "$DEF_INFO" | grep -q "sm_120"; then
  echo -e "  ${GRN}OK${NC}   default:  $DEF_INFO"
else
  echo -e "  ${YEL}WARN${NC} default python3 不支持 RTX 5090 (sm_120)，跑训练请用 dct env"
fi

# ---- B. 数据路径 ----
echo ""
echo "[B] 数据路径"
for p in /data1/dataset_csv /data1/TCGA-UNI2-h-features; do
  if [ -d "$p" ]; then
    n=$(ls "$p" 2>/dev/null | wc -l)
    echo -e "  ${GRN}OK${NC}   $p   ($n 项)"
  else
    echo -e "  ${RED}FAIL${NC} $p 缺失"
  fi
done

# ---- C. 划分样本 ----
echo ""
echo "[C] 划分样本（首个 cancer 的 fold0.csv）"
SPLIT_LEGACY=/data1/dataset_csv/splits/5fold_uni2h/brca/fold_0.csv
SPLIT_OUTER_DIR=/data1/dataset_csv/splits/5fold_uni2h_outer_test_seed3
[ -f "$SPLIT_LEGACY" ] && echo -e "  ${GRN}OK${NC}   legacy_val: $SPLIT_LEGACY ($(wc -l <"$SPLIT_LEGACY") 行)" || echo -e "  ${RED}FAIL${NC} $SPLIT_LEGACY"
if [ -d "$SPLIT_OUTER_DIR" ]; then
  C=$(ls "$SPLIT_OUTER_DIR" | tr '\n' ' ')
  echo -e "  ${GRN}OK${NC}   outer_test 目录存在，覆盖癌种: $C"
else
  echo -e "  ${RED}FAIL${NC} $SPLIT_OUTER_DIR 缺失"
fi

# ---- D. 最近训练进度 ----
echo ""
echo "[D] 最近训练（v313_three_decoders_six_20261013）"
RES=/data1/DCT-Reg/results/v313_three_decoders_six_20261013
if [ -d "$RES/logs" ]; then
  TOTAL=$(ls "$RES"/logs/three_decoders_*.log 2>/dev/null | wc -l)
  DONE=0; RUNNING=""
  for f in "$RES"/logs/three_decoders_*.log; do
    [ -f "$f" ] || continue
    n=$(grep -c "val cindex=" "$f" 2>/dev/null)
    if [ "$n" -ge 30 ]; then DONE=$((DONE+1));
    else RUNNING="$RUNNING $(basename "$f" .log | sed 's/three_decoders_//')@e$n"; fi
  done
  echo "  完成 folds: $DONE / $TOTAL"
  [ -n "$RUNNING" ] && echo "  进行中:$RUNNING"
  if [ -f /tmp/e049_driver.log ]; then
    PID=$(ps -ef | grep "[r]un_three_decoders_six_cancers" | awk '{print $2}' | head -1)
    [ -n "$PID" ] && echo "  驱动 PID=$PID" || echo "  驱动已退出"
    echo "  最近事件:"; tail -3 /tmp/e049_driver.log | sed 's/^/    /'
  fi
else
  echo "  无 results 目录"
fi

# ---- E. GPU ----
echo ""
echo "[E] GPU"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'

# ---- F. AGENTS.md 摘要 ----
echo ""
echo "[F] AGENTS.md 已登记实验"
grep -oE "E0[0-9]+已[登记记记录]" /data1/DCT-Reg/AGENTS.md | sort -u | tr '\n' ' '
echo ""

echo ""
echo "================================================================"
echo "  自检完成。需要启动训练？ → 复用 dct env + E049 driver 模板"
echo "================================================================"
