#!/usr/bin/env bash
# Stage B orchestrator: SlotSPE 4-arm comparison on BLCA & KIRC
# - GPU 0: BLCA matched (slotspe_blca_paper) AND BLCA native (slotspe_blca_native) — SEQUENTIAL on GPU 0
# - GPU 1: KIRC matched (slotspe_kirc_paper) AND KIRC native (slotspe_kirc_native) — SEQUENTIAL on GPU 1
# Order: matched first (paper params), then native (no topk), per cancer.
# Designed to be launched AFTER KIRC training (Step A) completes.
set -uo pipefail
cd /data1/DCT-Reg

LOG_DIR=/data1/DCT-Reg/logs/stage_b_chain
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

echo "[$(date '+%F %T')] === Stage B SlotSPE chain starting ===" | tee -a "$MASTER_LOG"

# Run both arms in parallel: GPU 0 does BLCA matched+native; GPU 1 does KIRC matched+native
(
    # GPU 0: BLCA matched → BLCA native
    echo "[$(date '+%F %T')] GPU 0: BLCA matched starting" | tee -a "$MASTER_LOG"
    /data1/DCT-Reg/scripts/run_slotspe_blca_paper.sh >> "$LOG_DIR/gpu0_blca_matched.log" 2>&1
    echo "[$(date '+%F %T')] GPU 0: BLCA matched done" | tee -a "$MASTER_LOG"
    echo "[$(date '+%F %T')] GPU 0: BLCA native starting" | tee -a "$MASTER_LOG"
    /data1/DCT-Reg/scripts/run_slotspe_native_paper.sh blca 0 >> "$LOG_DIR/gpu0_blca_native.log" 2>&1
    echo "[$(date '+%F %T')] GPU 0: BLCA native done" | tee -a "$MASTER_LOG"
) &

(
    # GPU 1: KIRC matched → KIRC native
    echo "[$(date '+%F %T')] GPU 1: KIRC matched starting" | tee -a "$MASTER_LOG"
    /data1/DCT-Reg/scripts/run_slotspe_kirc_paper.sh >> "$LOG_DIR/gpu1_kirc_matched.log" 2>&1
    echo "[$(date '+%F %T')] GPU 1: KIRC matched done" | tee -a "$MASTER_LOG"
    echo "[$(date '+%F %T')] GPU 1: KIRC native starting" | tee -a "$MASTER_LOG"
    /data1/DCT-Reg/scripts/run_slotspe_native_paper.sh kirc 1 >> "$LOG_DIR/gpu1_kirc_native.log" 2>&1
    echo "[$(date '+%F %T')] GPU 1: KIRC native done" | tee -a "$MASTER_LOG"
) &

wait
echo "[$(date '+%F %T')] === Stage B SlotSPE chain DONE ===" | tee -a "$MASTER_LOG"