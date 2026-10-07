#!/usr/bin/env bash
set -euo pipefail

cd /data1/DCT-Reg
mkdir -p logs results/transport_candidates

run_candidate() {
  local name="$1"
  local config="$2"
  local path_pattern="$3"
  local train_log="logs/${name}_fold0_30ep.log"
  local audit_csv="results/transport_candidates/audit_${name}_fold0.csv"
  local audit_log="logs/audit_${name}_fold0.log"

  echo "[$(date -Is)] START training ${name}"
  python -m survot_rank.cli train --config "${config}" 2>&1 | tee "${train_log}"

  local checkpoint
  checkpoint=$(find results/transport_candidates -type f \
    -path "*${path_pattern}*/model_best_s0.pth" | head -n 1)
  if [[ -z "${checkpoint}" ]]; then
    echo "Missing best checkpoint for ${name}" >&2
    return 1
  fi

  echo "[$(date -Is)] START audit ${name}: ${checkpoint}"
  python scripts/e4_audit_adapted.py \
    --checkpoint "${checkpoint}" \
    --study blca \
    --fold 0 \
    --output "${audit_csv}" \
    --alphas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    2>&1 | tee "${audit_log}"
  echo "[$(date -Is)] DONE ${name}"
}

run_candidate \
  enhanced_direction \
  configs/dct_enhanced_direction_blca_fold0.yaml \
  enhanced_direction_w010_blca_fold0

run_candidate \
  risk_ordering \
  configs/dct_risk_ordering_blca_fold0.yaml \
  risk_ordering_w010_d005_blca_fold0

echo "[$(date -Is)] ALL CANDIDATE GATES COMPLETE"
