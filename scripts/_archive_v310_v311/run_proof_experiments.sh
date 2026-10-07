#!/usr/bin/env bash
# CPU-only proof-of-idea experiments for v3.11 BLCA.
# No GPU needed; just reads existing epoch_curve_*.csv files.
#
# Usage:
#   bash scripts/run_proof_experiments.sh
set -e
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/ubuntu/.conda/envs/trisurv/bin/python}

echo "================================================================"
echo " Proof A — Component effectiveness (v3.10 vs v3.11)"
echo "================================================================"
$PYTHON scripts/proof_experiments/proof_A_recipe_compare.py

echo ""
echo "================================================================"
echo " Proof B — slot_variance constraint [0.005, 0.050]"
echo "================================================================"
$PYTHON scripts/proof_experiments/proof_B_variance_constraint.py

echo ""
echo "================================================================"
echo " Proof C — IPCW rank loss actually contributes"
echo "================================================================"
$PYTHON scripts/proof_experiments/proof_C_ipcw_rank.py

echo ""
echo "================================================================"
echo " Done. Full report: PROOF_RESULTS.md"
echo " Raw JSON:    results/proof_experiment_{A,B,C}.json"
echo "================================================================"
