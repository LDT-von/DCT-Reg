#!/bin/bash
# 监控两个 DCT 训练 (v3.13 + v3.10 fold 2-4)
# 用法: ./monitor_dual.sh

LOG313=$(ls -t /data1/DCT-Reg/logs/v313_uni_blca_p2048_30ep_*.log 2>/dev/null | head -1)
LOG310=$(ls -t /data1/DCT-Reg/logs/v310_uni_blca_p2048_30ep_*.log 2>/dev/null | grep "$(date +%Y%m%d)" | head -1)

echo "==================================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================="

echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits | awk -F', ' '{printf "  GPU%s %s: %s/%s MiB  GPU%3s%%  %s°C\n", $1, $2, $3, $4, $5, $6}'

echo ""
echo "--- 进程 ---"
pgrep -f "survot_rank.cli" | while read pid; do
    etime=$(ps -p $pid -o etimes= 2>/dev/null | tr -d ' ')
    em=$((etime/60)); es=$((etime%60))
    kstart=$(ps -p $pid -o args= 2>/dev/null | grep -oP 'k_start=\K\d+')
    kend=$(ps -p $pid -o args= 2>/dev/null | grep -oP 'k_end=\K\d+')
    gpu=$(ps -p $pid -o args= 2>/dev/null | grep -oP 'gpu=\K\d+')
    method=$(ps -p $pid -o args= 2>/dev/null | grep -oP 'config configs/\K[^ ]+' | head -1)
    foldrange="$kstart-$(((kend-1)))"
    echo "  PID $pid GPU$gpu  Fold $foldrange  ${em}m${es}s  $method"
done

echo ""
echo "==================================================="
echo " v3.13 训练进度 (transport-aware reconstruction)"
echo "==================================================="
if [ -n "$LOG313" ] && [ -f "$LOG313" ]; then
    python3 - "$LOG313" "v3.13" << 'PYEOF'
import sys, re

log_path, label = sys.argv[1], sys.argv[2]
with open(log_path) as f:
    lines = f.readlines()

fold_scores = {}
current_fold = None
fold_start_pattern = re.compile(r'\[Fold (\d+)\] start')
epoch_pattern = re.compile(r'\[Fold (\d+)\] Epoch (\d+)/30:')
val_pattern = re.compile(r'\[Epoch \d+\] val cindex=([\d.]+)')
finish_pattern = re.compile(r'\[Fold (\d+)\] best cindex=([\d.]+) @epoch (\d+)')

finished_folds = set()
for line in lines:
    m_start = fold_start_pattern.search(line)
    if m_start:
        current_fold = int(m_start.group(1))
    m_epoch = epoch_pattern.search(line)
    if m_epoch:
        current_fold = int(m_epoch.group(1))
    m_val = val_pattern.search(line)
    if m_val and current_fold is not None:
        cindex = float(m_val.group(1))
        if current_fold not in fold_scores:
            fold_scores[current_fold] = []
        fold_scores[current_fold].append(cindex)
    m_finish = finish_pattern.search(line)
    if m_finish:
        finished_folds.add(int(m_finish.group(1)))

if fold_scores:
    for fold in sorted(fold_scores.keys()):
        scores = fold_scores[fold]
        best = max(scores)
        best_ep = scores.index(best)
        status = "[OK]" if fold in finished_folds else "[..]"
        latest = scores[-1]
        print(f"  {status} Fold {fold}: {len(scores)}/30 epochs | Best={best:.4f} @ ep{best_ep} | latest={latest:.4f}")
    
    if len(fold_scores) > 0:
        done = sum(1 for f in fold_scores if f in finished_folds)
        avg_best = sum(max(s) for s in fold_scores.values()) / len(fold_scores)
        print(f"  --> 已完成 fold: {done}/{len(fold_scores)}  |  平均 Best: {avg_best:.4f}")
else:
    print("  (无 val 数据)")
PYEOF
else
    echo "  日志未找到"
fi

echo ""
echo "==================================================="
echo " v3.10 训练进度 (directional regularized transport)"
echo "==================================================="
if [ -n "$LOG310" ] && [ -f "$LOG310" ]; then
    python3 - "$LOG310" "v3.10" << 'PYEOF'
import sys, re

log_path, label = sys.argv[1], sys.argv[2]
with open(log_path) as f:
    lines = f.readlines()

fold_scores = {}
current_fold = None
fold_start_pattern = re.compile(r'\[Fold (\d+)\] start')
epoch_pattern = re.compile(r'\[Fold (\d+)\] Epoch (\d+)/30:')
val_pattern = re.compile(r'\[Epoch \d+\] val cindex=([\d.]+)')
finish_pattern = re.compile(r'\[Fold (\d+)\] best cindex=([\d.]+) @epoch (\d+)')

finished_folds = set()
for line in lines:
    m_start = fold_start_pattern.search(line)
    if m_start:
        current_fold = int(m_start.group(1))
    m_epoch = epoch_pattern.search(line)
    if m_epoch:
        current_fold = int(m_epoch.group(1))
    m_val = val_pattern.search(line)
    if m_val and current_fold is not None:
        cindex = float(m_val.group(1))
        if current_fold not in fold_scores:
            fold_scores[current_fold] = []
        fold_scores[current_fold].append(cindex)
    m_finish = finish_pattern.search(line)
    if m_finish:
        finished_folds.add(int(m_finish.group(1)))

if fold_scores:
    for fold in sorted(fold_scores.keys()):
        scores = fold_scores[fold]
        best = max(scores)
        best_ep = scores.index(best)
        status = "[OK]" if fold in finished_folds else "[..]"
        latest = scores[-1]
        print(f"  {status} Fold {fold}: {len(scores)}/30 epochs | Best={best:.4f} @ ep{best_ep} | latest={latest:.4f}")
    
    if len(fold_scores) > 0:
        done = sum(1 for f in fold_scores if f in finished_folds)
        avg_best = sum(max(s) for s in fold_scores.values()) / len(fold_scores)
        print(f"  --> 已完成 fold: {done}/{len(fold_scores)}  |  平均 Best: {avg_best:.4f}")
else:
    print("  (无 val 数据)")
PYEOF

    echo ""
    echo "  --- 历史 fold (已存在) ---"
    for f in 0 1 2 3 4; do
        CSV="/data1/DCT-Reg/results/dct_v3.10_uni_blca_p2048_30ep/final/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_uni_p2048_blca_fold${f}_30ep/epoch_curve_fold${f}.csv"
        if [ -f "$CSV" ]; then
            echo "  Fold $f: 已存在 ($(basename $(dirname $CSV)))"
        fi
    done
else
    echo "  日志未找到"
fi

echo ""
echo "==================================================="
echo " 当前进度"
echo "==================================================="
echo "  v3.13: $(tail -1 "$LOG313" 2>/dev/null | grep -oP 'Fold \d+.*?batch=\d+/\d+' | head -1)"
echo "  v3.10: $(tail -1 "$LOG310" 2>/dev/null | grep -oP 'Fold \d+.*?batch=\d+/\d+' | head -1)"
