#!/bin/bash
# ============================================================================
# 顺序执行：先 v310_uni2h_extend (4 癌种)，完成后自动启动 v310_uni_all10 (10 癌种)
# 单 GPU 0 上串行运行
# ============================================================================

set -euo pipefail
cd /data1/DCT-Reg

echo "============================================================"
echo "[combined] v3.10 双任务顺序启动器"
echo "[combined] 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo ""
echo "[step 1/2] 启动 v310_uni2h_extend（4 癌种 UNI2-h）..."
bash scripts/launch_v310_uni2h_extend.sh
RC1=$?
echo "[step 1/2] v310_uni2h_extend 退出码: $RC1"
if [ $RC1 -ne 0 ]; then
    echo "[ERROR] 任务 1 失败，跳过任务 2"
    exit $RC1
fi

echo ""
echo "[step 2/2] 启动 v310_uni_all10（10 癌种 UNI）..."
bash scripts/launch_v310_uni_all10.sh
RC2=$?
echo "[step 2/2] v310_uni_all10 退出码: $RC2"

echo ""
echo "============================================================"
echo "[combined] 两个任务都完成"
echo "[combined] 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[combined] 任务 1 (uni2-h): $RC1"
echo "[combined] 任务 2 (uni):    $RC2"
echo "============================================================"
exit $((RC1 + RC2))
