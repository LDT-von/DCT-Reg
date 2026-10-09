#!/usr/bin/env bash
# Watch v3.11 BLCA UNI2-h v2 5-fold training.
# Auto-slices the single combined log file by [Fold K] markers.
#
# Usage:
#   bash scripts/watch_v311_uni2h_v2.sh              # foreground, refresh 30s
#   INTERVAL=60 bash scripts/watch_v311_uni2h_v2.sh  # custom interval
set -e
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/ubuntu/.conda/envs/trisurv/bin/python}
LOG_FILE=logs/dct_v311_blca_uni2h_v2_5fold.log
INTERVAL=${INTERVAL:-30}
RECENT=${RECENT:-5}

if [ ! -f "$LOG_FILE" ]; then
  echo "[error] log not found: $LOG_FILE"
  exit 1
fi

exec "$PYTHON" scripts/monitor_scores.py \
  --log-file "$LOG_FILE" \
  --label "v3.11 BLCA UNI2-h v2" \
  --interval "$INTERVAL" \
  --recent "$RECENT"
