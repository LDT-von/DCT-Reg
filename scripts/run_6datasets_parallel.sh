#!/usr/bin/env bash
# Run 5 datasets sequentially on GPU 0 only
# gbmlgg skipped (has duplicate case_ids)
# Order: skcm -> stad -> ucec -> cptac_luad -> cptac_lusc

BASE_LOG="/data1/DCT-Reg/logs/v312_6datasets"
mkdir -p "$BASE_LOG"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Start: $(date)" > "${BASE_LOG}/master.log"

for study in skcm stad ucec cptac_luad cptac_lusc; do
    echo "[$(date)] ===== $study start =====" | tee -a "${BASE_LOG}/master.log"
    for fold in 0 1 2 3 4; do
        bash "${SCRIPT_DIR}/run_one_study.sh" "$study" "$fold" >> "${BASE_LOG}/master.log" 2>&1
        RC=$?
        if [ $RC -eq 0 ]; then
            echo "[$(date)] $study fold $fold OK" | tee -a "${BASE_LOG}/master.log"
        else
            echo "[$(date)] $study fold $fold FAILED (rc=$RC)" | tee -a "${BASE_LOG}/master.log"
        fi
    done
    echo "[$(date)] ===== $study ALL DONE =====" | tee -a "${BASE_LOG}/master.log"
done

echo "ALL DONE: $(date)" >> "${BASE_LOG}/master.log"

# Final summary
python3 -c "
import numpy as np, re, os

studies = ['skcm', 'stad', 'ucec', 'cptac_luad', 'cptac_lusc']
log_dir = '/data1/DCT-Reg/logs/v312_6datasets'
print()
print('=' * 60)
print('FINAL SUMMARY')
print('=' * 60)
all_vals = []
for s in studies:
    vals = []
    for f in range(5):
        log = f'{log_dir}/{s}/fold{f}.log'
        best = None
        if os.path.exists(log):
            for line in open(log):
                if 'best cindex' in line:
                    m = re.search(r'cindex=([0-9.]+)', line)
                    if m: best = float(m.group(1))
        if best is not None:
            vals.append(best)
    if vals:
        vals.sort()
        all_vals.extend(vals)
        print(f'{s:15s}: {[round(v,4) for v in vals]}')
        print(f'               Mean={np.mean(vals):.4f}  Std={np.std(vals,ddof=1):.4f}')
    else:
        print(f'{s:15s}: NO DATA')
if all_vals:
    print()
    print(f'OVERALL MEAN: {np.mean(all_vals):.4f}  Std: {np.std(all_vals,ddof=1):.4f}')
"