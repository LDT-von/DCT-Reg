#!/usr/bin/env bash
# Supervisor: wait for KIRC training to fully finish, then launch Stage B
# Detects completion by: (a) "All KIRC tasks complete" line in kirc_arm_2026-10-03_main.log
#                      (b) no survot_rank.cli train processes for kirc results dirs
set -uo pipefail
KIRC_MAIN_LOG=/data1/DCT-Reg/logs/kirc_arm_2026-10-03_main.log
STAGE_B_SCRIPT=/data1/DCT-Reg/scripts/run_stage_b_chain.sh
SUP_LOG=/data1/DCT-Reg/logs/stage_b_supervisor.log

echo "[$(date '+%F %T')] Stage B supervisor started; waiting for KIRC completion signal" | tee -a "$SUP_LOG"

# Wait up to 8 hours
for i in {1..960}; do
    if [ -f "$KIRC_MAIN_LOG" ] && grep -q "All KIRC tasks complete" "$KIRC_MAIN_LOG"; then
        # Double-check no active KIRC processes
        ACTIVE=$(pgrep -af "survot_rank.cli train" 2>/dev/null | grep -c "kirc" || true)
        if [ "${ACTIVE:-0}" -eq 0 ]; then
            echo "[$(date '+%F %T')] KIRC complete + no active procs; launching Stage B" | tee -a "$SUP_LOG"
            bash "$STAGE_B_SCRIPT" >> "$SUP_LOG" 2>&1
            echo "[$(date '+%F %T')] Stage B returned (rc=$?)" | tee -a "$SUP_LOG"
            exit 0
        fi
    fi
    sleep 30
done

echo "[$(date '+%F %T')] Stage B supervisor timeout (>8h)" | tee -a "$SUP_LOG"
exit 1
