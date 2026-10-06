#!/usr/bin/env bash
# Supervisor v2: stricter detection
# - "All KIRC tasks complete" line in main log must be AFTER the most recent STARTING entry
# - All 10 fold result dirs must have epoch_curve CSV (>=30 epochs)
set -uo pipefail
KIRC_MAIN_LOG=/data1/DCT-Reg/logs/kirc_arm_2026-10-03_main.log
STAGE_B_SCRIPT=/data1/DCT-Reg/scripts/run_stage_b_chain.sh
SUP_LOG=/data1/DCT-Reg/logs/stage_b_supervisor_v2.log

echo "[$(date '+%F %T')] Supervisor v2 started" | tee -a "$SUP_LOG"

# Helper: count completed KIRC tasks (epoch_curve CSVs with >=30 epochs)
count_completed() {
    local n=0
    for arm in direct independent; do
        for f in 0 1 2 3 4; do
            local csv=$(find /data1/DCT-Reg/results/v313_paper_v1/legacy_val/${arm}/kirc/fold${f} -name "epoch_curve_fold${f}.csv" 2>/dev/null | head -1)
            if [ -n "$csv" ]; then
                local rows=$(awk 'NR>1' "$csv" 2>/dev/null | wc -l)
                if [ "${rows:-0}" -ge 30 ]; then
                    n=$((n+1))
                fi
            fi
        done
    done
    echo $n
}

for i in {1..1800}; do   # up to 15 hours, 30s intervals
    COMPLETED=$(count_completed)
    ACTIVE=$(pgrep -af "survot_rank.cli train" 2>/dev/null | grep -c "kirc" || true)
    if [ "${COMPLETED:-0}" -ge 10 ] && [ "${ACTIVE:-0}" -eq 0 ]; then
        echo "[$(date '+%F %T')] All 10 KIRC tasks done + no active procs; launching Stage B" | tee -a "$SUP_LOG"
        bash "$STAGE_B_SCRIPT" >> "$SUP_LOG" 2>&1
        echo "[$(date '+%F %T')] Stage B returned (rc=$?)" | tee -a "$SUP_LOG"
        exit 0
    fi
    # periodic heartbeat
    if [ $((i % 20)) -eq 0 ]; then
        echo "[$(date '+%F %T')] Heartbeat: completed=$COMPLETED/10, active_kirc_procs=$ACTIVE" | tee -a "$SUP_LOG"
    fi
    sleep 30
done
echo "[$(date '+%F %T')] Supervisor v2 timeout (>15h)" | tee -a "$SUP_LOG"
exit 1
