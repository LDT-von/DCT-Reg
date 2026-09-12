#!/bin/bash
# Run redesign_audit_blca for folds 1..4 sequentially.
REPO=/data1/DCT-Reg
CKPT_DIR="${REPO}/results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep"
CFG="${REPO}/configs/dct_v310_directional_regularized_transport.yaml"
LOG_DIR="${REPO}/logs/redesign_audit"
mkdir -p "${LOG_DIR}"

folds=("$@")
if [ ${#folds[@]} -eq 0 ]; then
    folds=("1" "2" "3" "4")
fi

for fold in "${folds[@]}"; do
    ckpt="${CKPT_DIR}/model_best_s${fold}.pth"
    if [ ! -f "${ckpt}" ]; then
        echo "[fold ${fold}] MISSING checkpoint, skip"
        continue
    fi
    log_path="${LOG_DIR}/redesign_fold${fold}.log"
    echo "[fold ${fold}] starting, log -> ${log_path}"
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 /home/ubuntu/.conda/envs/trisurv/bin/python \
        "${REPO}/scripts/redesign_audit_blca.py" \
        --config "${CFG}" \
        --checkpoint "${ckpt}" \
        --fold "${fold}" \
        --epoch 50 \
        --output-dir "${REPO}/results/audit_blca_redesign/blca" \
        --set study=blca \
        > "${log_path}" 2>&1
    rc=$?
    t1=$(date +%s)
    elapsed=$((t1 - t0))
    echo "[fold ${fold}] rc=${rc}, ${elapsed}s"
done
