#!/bin/bash
# Test E4 script with a single fold

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python3
SCRIPT=/data1/DCT-Reg/scripts/e4_intervention_audit.py

# Find checkpoint for direction_only fold 0
CHECKPOINT=$(find /data1/DCT-Reg/results/dct_v3.10_experiments/robust/direction_only/blca -name "model_best_s0.pth" -type f | head -1)

echo "Testing E4 script..."
echo "Checkpoint: $CHECKPOINT"

$PYTHON $SCRIPT \
    --checkpoint "$CHECKPOINT" \
    --variant direction_only \
    --cancer blca \
    --fold 0 \
    --split test \
    --num_alphas 11 \
    --batch_size 8 \
    --device cuda:0 \
    --output_dir results/e4_audit_test

echo "Test completed!"
