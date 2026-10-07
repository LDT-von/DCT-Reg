#!/usr/bin/env bash
# Monitor fold0 val c-index as training progresses.
# Usage: bash scripts/monitor_fold0.sh [log_file] [poll_seconds]
#
# Prints a table of [Epoch N] val lines, updates in place.
# Press Ctrl+C to stop.

LOG="${1:-/data1/DCT-Reg/logs/v312_fold0_aligned/fold0.log}"
POLL="${2:-60}"

if [[ ! -f "$LOG" ]]; then
    echo "ERROR: log file not found: $LOG"
    exit 1
fi

# Count lines so far
last_count=0

echo "========================================================"
echo "Monitoring: $LOG  (poll every ${POLL}s)"
echo "========================================================"
printf "%-8s %-9s %-8s %-8s %-8s %s\n" \
    "Epoch" "c-index" "ipcw" "IBS" "IAUC" "status"
echo "--------------------------------------------------------"

# Print any existing lines first
grep "^\[Epoch" "$LOG" 2>/dev/null | \
    sed -n 's/.*\[Epoch \([0-9]*\)\] val cindex=\([0-9.]*\) ipcw=\([0-9.]*\) IBS=\([0-9.]*\) iauc=\([0-9.]*\).*/\1 \2 \3 \4 \5/p' \
    2>/dev/null | while read -r ep ci ipcw ibs iauc; do
    printf "%-8s %-9s %-8s %-8s %-8s %s\n" \
        "ep $ep" "$ci" "$ipcw" "$ibs" "$iauc" "known"
done

while true; do
    sleep "$POLL"

    if ! kill -0 "$(ps -o pid= -p $(pgrep -f "dct_v312_slot_interpretable_impute" 2>/dev/null | head -1) 2>/dev/null)" 2>/dev/null; then
        # Check if process is gone (might be finished)
        if ! pgrep -f "dct_v312_slot_interpretable_impute" > /dev/null 2>&1; then
            echo ""
            echo ">>> Training process finished. Final summary:"
            break
        fi
    fi

    current_count=$(wc -l < "$LOG" 2>/dev/null || echo 0)

    if [[ "$current_count" -eq "$last_count" ]]; then
        # Still running but no new output (mid-epoch)
        echo -n ""
    else
        # New epoch lines
        echo ""
        grep "^\[Epoch" "$LOG" 2>/dev/null | tail -5 | \
            sed -n 's/.*\[Epoch \([0-9]*\)\] val cindex=\([0-9.]*\) ipcw=\([0-9.]*\) IBS=\([0-9.]*\) iauc=\([0-9.]*\).*/\1 \2 \3 \4 \5/p' \
            2>/dev/null | while read -r ep ci ipcw ibs iauc; do
            printf "%-8s %-9s %-8s %-8s %-8s %s\n" \
                "ep $ep" "$ci" "$ipcw" "$ibs" "$iauc" ""
        done
        echo "--------------------------------------------------------"
    fi

    last_count="$current_count"
done

# Final full summary
echo ""
echo "=== ALL EPOCHS ==="
grep -E "^\[Epoch|best cindex|FOLD 0 DONE" "$LOG" 2>/dev/null
