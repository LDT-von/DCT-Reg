#!/bin/bash
# ============================================================================
# BLCA (v3.10) 训练实时监控 — 两个工具打包
#
# 工具 A: monitor_unified.py — 官方 monitor，看 GPU / 锁 / 已完成 fold / 最新日志
# 工具 B: 这个脚本额外加的 tail_blca_curves.py — 看 epoch 实时 C-index 曲线
#
# 用法：
#   bash scripts/monitor_blca_v310.sh                # 一次性快照
#   bash scripts/monitor_blca_v310.sh watch 30       # 每 30 秒刷新一次（清屏）
#   bash scripts/monitor_blca_v310.sh curves         # 看 epoch 实时 C-index 曲线
# ============================================================================

set -euo pipefail
cd /data1/DCT-Reg

MODE="${1:-snapshot}"
INTERVAL="${2:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dct_v3.10/robust/final/blca}"
LOG_DIR="${LOG_DIR:-logs}"

echo "Monitor mode:  $MODE"
echo "Watch interval: ${INTERVAL}s (only used by watch)"
echo "Watching dir:   $OUTPUT_DIR"
echo "============================================================"

# ---------- 工具 A: 官方 monitor（指向最新 rerun 日志而不是历史的 A_queue） ----------
run_official_monitor() {
    # 找最新 rerun 日志
    latest_rerun=$(ls -t ${LOG_DIR}/blca_v310_rerun_*.log 2>/dev/null | head -1)
    if [ -n "$latest_rerun" ]; then
        echo "[rerun log] $latest_rerun"
        # 1. 实时 epoch 进度（从日志里抓）
        latest_ep=$(grep -oE "Epoch [0-9]+/50: *[0-9]+%\|[█▏▎▍▌▋▊▉ ]*\| *[0-9]+/38" "$latest_rerun" 2>/dev/null | tail -1)
        if [ -n "$latest_ep" ]; then
            echo "[progress]  $latest_ep"
        else
            # 兜底：如果日志格式变了，直接 grep Epoch/batch
            echo -n "[progress]  "
            grep -oE "Epoch [0-9]+/50" "$latest_rerun" 2>/dev/null | tail -1
            echo -n "            "
            grep -oE "batch=[0-9]+/38" "$latest_rerun" 2>/dev/null | tail -1
        fi
        # 2. 最近一个 loss 数字
        last_loss=$(grep -oE "loss=[0-9.]+, surv=[0-9.]+" "$latest_rerun" 2>/dev/null | tail -1)
        if [ -n "$last_loss" ]; then
            echo "[last loss] $last_loss"
        fi
        # 3. 估算 fold 已用时间 / 总时间
        last_secs=$(grep -oE "\[[0-9]+:[0-9]+<[0-9:]+, *[0-9.]+s/it" "$latest_rerun" 2>/dev/null | tail -1)
        if [ -n "$last_secs" ]; then
            echo "[timing]    $last_secs"
        fi
        echo
    fi
    python scripts/monitor_unified.py "$@"
}

# ---------- 工具 B: epoch 实时 C-index 曲线 ----------
# 同时看 final/（新跑）和 final_50ep_old/（保留的旧数据）
run_curves_monitor() {
    python - <<PYEOF
import csv
from pathlib import Path

base_dirs = [
    Path("${OUTPUT_DIR}"),                                    # 新跑：final/blca
    Path("${OUTPUT_DIR}").parent / "final_50ep_old" / "blca",  # 保留：final_50ep_old/blca
]

for out_dir in base_dirs:
    if not out_dir.is_dir():
        continue
    label = "新跑" if "final_50ep_old" not in out_dir.as_posix() else "保留"
    curves = sorted(out_dir.rglob("epoch_curve_fold*.csv"))
    if not curves:
        continue
    print(f"=== [{label}] epoch_curve_fold*.csv @ {out_dir} ===\n")
    best_per_fold = {}
    for cf in curves:
        rows = list(csv.DictReader(cf.open()))
        if not rows:
            continue
        fold = cf.stem.replace("epoch_curve_fold", "")
        cindex = [(int(r['epoch']), float(r['val_cindex'])) for r in rows]
        cindex.sort(key=lambda x: -x[1])
        best_ep, best_val = cindex[0]
        last_ep, last_val = cindex[-1]
        best_per_fold[int(fold)] = best_val
        print(f"  fold{fold}: last ep{last_ep:3d}={last_val:.4f}  | best ep{best_ep:3d}={best_val:.4f}  | n_epochs={len(cindex)}")
    if best_per_fold:
        mean = sum(best_per_fold.values()) / len(best_per_fold)
        print(f"\n  mean of best per-fold C-index: {mean:.4f}  ({len(best_per_fold)} folds done)")
        print(f"  reference (recorded):          0.7208  (5 folds)")
    print()

# 兜底：如果两个目录都还没曲线
final_blca = Path("${OUTPUT_DIR}")
if not (final_blca / "epoch_curve_fold0.csv").exists() and not any(final_blca.rglob("epoch_curve_fold*.csv")):
    print(f"[curves] {final_blca} 下还没有 epoch_curve_fold*.csv — fold 0 还在 epoch 1-2")
PYEOF
}

case "$MODE" in
    snapshot)
        run_official_monitor
        echo
        run_curves_monitor
        ;;
    watch)
        run_official_monitor --watch "$INTERVAL" --tail-lines 8
        ;;
    curves)
        run_curves_monitor
        ;;
    *)
        echo "Usage: $0 [snapshot|watch N|curves]"
        exit 1
        ;;
esac
