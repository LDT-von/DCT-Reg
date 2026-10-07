#!/usr/bin/env bash
# Poll until the 20-task controls launcher exits, then run bind + audit.
cd /data1/DCT-Reg
export PYTHONPATH="$PWD"
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
TOOL=scripts/prepare_v313_evidence.py
EVIDENCE="$PWD/results/v313_evidence_v2"
LOG=logs/controls_v2_20261005/monitor.log
mkdir -p logs/controls_v2_20261005

LAUNCHER_PID="${1:-683824}"
echo "[$(date '+%F %T')] monitoring launcher PID=$LAUNCHER_PID" > "$LOG"

while kill -0 "$LAUNCHER_PID" 2>/dev/null; do
  n=$(find results/v313_controls_v2 -name 'epoch_curve_fold*.csv' 2>/dev/null | wc -l)
  done_n=0
  for f in $(find results/v313_controls_v2 -name 'epoch_curve_fold*.csv' 2>/dev/null); do
    rows=$(($(wc -l < "$f") - 1))
    [ "$rows" -ge 30 ] && done_n=$((done_n+1))
  done
  echo "[$(date '+%F %T')] curves=$n  complete(30ep)=$done_n/20  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ',')" >> "$LOG"
  sleep 120
done

echo "[$(date '+%F %T')] launcher exited" >> "$LOG"
echo "=== bind ===" >> "$LOG"
$PY "$TOOL" bind --plan "$EVIDENCE/controls_plan.json" --output "$EVIDENCE/controls.json" >> "$LOG" 2>&1
echo "=== audit ===" >> "$LOG"
$PY "$TOOL" audit --manifest "$EVIDENCE/controls.json" --output "$EVIDENCE/controls_audit.json" >> "$LOG" 2>&1
echo "AUDIT_EXIT=$?" >> "$LOG"
echo "[$(date '+%F %T')] monitor done" >> "$LOG"
