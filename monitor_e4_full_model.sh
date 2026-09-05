#!/bin/bash
# Monitor E4 Full Model Audit Progress

echo "========================================="
echo "E4 Full Model Audit - Progress Monitor"
echo "========================================="
echo ""

# Check if process is running
if [ -f e4_full_model_run.pid ]; then
    PID=$(cat e4_full_model_run.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ E4 Full Model audit is RUNNING (PID: $PID)"
    else
        echo "✗ E4 Full Model audit process not found (may have completed or crashed)"
    fi
else
    echo "⚠ No PID file found"
fi

echo ""
echo "=== Latest Log Output ==="
echo ""
tail -20 e4_full_model_run.log

echo ""
echo "=== Completed Folds ==="
ls -1 results/e4_audits/e4_audit_full_model_fold*.csv 2>/dev/null | wc -l | xargs -I {} echo "{} / 5 folds completed"

echo ""
echo "=== Results Summary ==="
if [ -f results/e4_audits/e4_audit_full_model_fold0.json ]; then
    echo "Fold 0: ✓"
fi
if [ -f results/e4_audits/e4_audit_full_model_fold1.json ]; then
    echo "Fold 1: ✓"
fi
if [ -f results/e4_audits/e4_audit_full_model_fold2.json ]; then
    echo "Fold 2: ✓"
fi
if [ -f results/e4_audits/e4_audit_full_model_fold3.json ]; then
    echo "Fold 3: ✓"
fi
if [ -f results/e4_audits/e4_audit_full_model_fold4.json ]; then
    echo "Fold 4: ✓"
fi

echo ""
echo "========================================="
echo "Commands:"
echo "  Monitor live: tail -f e4_full_model_run.log"
echo "  Check again:  ./monitor_e4_full_model.sh"
echo "========================================="
