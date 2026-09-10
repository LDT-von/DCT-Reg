#!/usr/bin/env bash
# Launch a DCT v3.10 final queue, with the lock-free monitor next to it.
#
# This wrapper does NOT start training itself: it prints the exact command
# you need to run, then runs the read-only monitor in --watch mode.
# You copy-paste the command into your shell when ready.
#
# Usage:
#   ./scripts/launch_and_monitor.sh uni2h [cancers csv] [gpu] [refresh s]
#   ./scripts/launch_and_monitor.sh uni    [cancers csv] [gpu] [refresh s]
#
# Examples:
#   ./scripts/launch_and_monitor.sh uni2h "kirc,ucec" 0 5
#   ./scripts/launch_and_monitor.sh uni    "brca,coadread,luad,stad" 0 10

set -euo pipefail

QUEUE="${1:-uni2h}"
CANCERS="${2:-kirc,ucec}"
GPU="${3:-0}"
REFRESH="${4:-5}"

cd "$(dirname "$0")/.."

case "$QUEUE" in
  uni2h)
    SCRIPT=scripts/run_dct_v310_final_cross_cancer.py
    HEADLINE="[QUEUE A] UNI2-h 队列 — $CANCERS — GPU $GPU"
    ;;
  uni)
    SCRIPT=scripts/run_dct_v310_final_uni_queue.py
    HEADLINE="[QUEUE B] UNI 队列 — $CANCERS — GPU $GPU"
    ;;
  *)
    echo "unknown QUEUE='$QUEUE' (use 'uni2h' or 'uni')"
    exit 2
    ;;
esac

cat <<EOF
================================================================
$HEADLINE
================================================================
The script will not start anything yet.  Run the 'PLAN' line below
first to confirm the doctor checks pass, then run 'RUN' to launch.

# 0) sanity (read-only)
python $SCRIPT doctor --cancers $CANCERS --gpu $GPU

# 1) plan (read-only — prints the 5-fold command list)
python $SCRIPT plan --cancers $CANCERS --gpu $GPU

# 2) run (THIS ACTUALLY TRAINS)
mkdir -p logs
nohup python -u $SCRIPT run --cancers $CANCERS --gpu $GPU \\
    > logs/run_dct_v310_${QUEUE}_queue.log 2>&1 &
echo "PID=\$!"

# 3) monitor (read-only)
python scripts/monitor_unified.py --watch $REFRESH
================================================================
EOF
