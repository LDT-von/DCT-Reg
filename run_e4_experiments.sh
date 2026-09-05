#!/bin/bash
# One-click runner for all E4 experiments
# This will run E4 intervention audit for all variants and folds

set -e

echo "=========================================="
echo "E4 Experiments One-Click Runner"
echo "=========================================="
echo ""
echo "This will run E4 Continuous Intervention Audit for:"
echo "  - Variants: direction_only, ipcw_only, full"
echo "  - Folds: 0, 1, 2, 3, 4"
echo "  - Study: BLCA"
echo ""
echo "Total: 15 experiments"
echo ""

# Configuration
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
SCRIPT="scripts/run_all_e4_experiments.py"
LOG="results/e4_intervention_audit/e4_experiments.log"

# Create output directory
mkdir -p results/e4_intervention_audit

# Ask for confirmation
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Starting E4 experiments..."
echo "Log file: $LOG"
echo ""

# Run experiments
$PYTHON $SCRIPT \
    --variants direction_only,ipcw_only,full \
    --folds 0,1,2,3,4 \
    2>&1 | tee "$LOG"

echo ""
echo "=========================================="
echo "E4 Experiments Completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Analyze results:"
echo "     python scripts/analyze_e4_results.py"
echo ""
echo "  2. Visualize results:"
echo "     python scripts/visualize_e4_results.py --input results/e4_intervention_audit"
echo ""
