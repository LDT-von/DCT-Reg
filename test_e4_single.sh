#!/bin/bash
# Quick test of E4 experiment on direction_only fold 0

set -e

echo "=========================================="
echo "E4 Test Run: direction_only fold 0"
echo "=========================================="

PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
SCRIPT="scripts/run_all_e4_experiments.py"

$PYTHON $SCRIPT \
    --variants direction_only \
    --folds 0

echo ""
echo "Test completed! Check results in:"
echo "  results/e4_intervention_audit/direction_only_blca_fold0.csv"
