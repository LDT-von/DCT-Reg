#!/bin/bash
# E4 Batch Progress Monitor

LOG_FILE=/data1/DCT-Reg/e4_batch.log
RESULTS_DIR=/data1/DCT-Reg/results/e4_audits

echo "========================================="
echo "E4 Batch Progress Monitor"
echo "========================================="
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "ERROR: Log file not found: $LOG_FILE"
    exit 1
fi

# Count completed experiments
completed=$(grep -c "✓ Completed" "$LOG_FILE" 2>/dev/null || echo 0)
total=15

echo "Progress: $completed / $total experiments completed"
echo ""

# Show current activity
echo "Current Activity:"
echo "----------------"
tail -20 "$LOG_FILE" | grep -E "(Running E4 audit|Completed|Processing samples: 100%|Summary:|Anchor distance)" || echo "Waiting for updates..."
echo ""

# List completed results
echo "Completed Results:"
echo "-----------------"
if [ -d "$RESULTS_DIR" ]; then
    ls -lh "$RESULTS_DIR"/*.csv 2>/dev/null | awk '{print $9}' | xargs -I {} basename {} | sort
    echo ""
    echo "Total CSV files: $(ls "$RESULTS_DIR"/*.csv 2>/dev/null | wc -l)"
else
    echo "Results directory not yet created"
fi

echo ""
echo "========================================="
echo "To view live log: tail -f $LOG_FILE"
echo "========================================="
