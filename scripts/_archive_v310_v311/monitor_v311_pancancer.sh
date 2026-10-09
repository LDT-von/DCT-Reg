#!/bin/bash
# Monitor v3.11 uni Fixed pan-cancer training progress.
# Shows: latest epoch val cindex per fold, best cindex, total time spent.

LOG_ROOT="/data1/DCT-Reg/logs/v311_blca_uni_fixed_pancancer"

echo "================================================================"
echo " v3.11 UNI FIXED — Pan-cancer 5-fold monitor  (GPU 1)"
echo "================================================================"
echo "Started: $(head -1 /data1/DCT-Reg/logs/v311_blca_uni_fixed_pancancer/master.log 2>/dev/null | head -c 60)"
echo ""
echo "Per-fold latest epoch & best cindex:"
printf '%-12s %-10s %-12s %-12s %s\n' "Cancer" "Fold" "Cur epoch" "Best cindex" "Best @epoch"
echo "------------------------------------------------------------------------"

# Iterate over expected cancer/fold combos
for cancer in brca luad coadread lusc skcm hnsc; do
    for fold in 0 1 2 3 4; do
        log_file="${LOG_ROOT}/${cancer}_fold${fold}.log"
        if [ -f "$log_file" ]; then
            cur=$(grep "^\[Epoch" "$log_file" | tail -1 | grep -oE "Epoch [0-9]+" | head -1)
            best=$(grep -E "best cindex" "$log_file" | tail -1 | head -c 80)
            if [ -z "$cur" ]; then
                # 还在初始化
                cur=$(grep "Epoch.*/30:" "$log_file" | tail -1 | grep -oE "Epoch [0-9]+/30" | head -1)
                if [ -z "$cur" ]; then
                    cur="(starting)"
                fi
            fi
            printf '%-12s %-10s %-12s %s\n' "$cancer" "fold$fold" "$cur" "$best"
        fi
    done
done

echo ""
echo "================================================================"
echo " Active fold (latest updated log):"
ls -t ${LOG_ROOT}/*.log 2>/dev/null | head -1 | xargs -I{} basename {}
echo ""
echo " GPU status:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "================================================================"
