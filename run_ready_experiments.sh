#!/usr/bin/env bash
#
# 一键运行所有已经准备就绪的后续实验
#
# 状态：
# ✅ Lambda扫描实验 - 可直接运行
# ⏳ E4实验 - 需要先调试脚本（模型加载部分）
#
# 使用方式：
#   bash run_ready_experiments.sh
#

set -euo pipefail

PROJECT_ROOT="/data1/DCT-Reg"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python3"
GPU="0"

cd "$PROJECT_ROOT"

echo "========================================"
echo "DCT v3.10 后续实验 - 一键运行"
echo "========================================"
echo ""
echo "📋 实验列表："
echo "  1. Lambda Direction扫描 (ipcw_only基础)"
echo "  2. Lambda IPCW扫描 (direction_only基础)"
echo ""
echo "⚠️  注意: E4实验需要先完成代码调试，暂时不包含在此脚本中"
echo ""

# ============================================================
# 1. Lambda Direction扫描实验 (ipcw_only基础上添加不同强度的direction)
# ============================================================

echo ""
echo "============================================================"
echo "1. Lambda Direction扫描实验"
echo "============================================================"
echo ""
echo "在IPCW Only基础上测试不同强度的方向约束："
echo "  λ_dir ∈ {0.01, 0.025, 0.05, 0.075, 0.10}"
echo ""

LAMBDA_DIR_VALUES="0.01 0.025 0.05 0.075 0.10"

for lambda_dir in $LAMBDA_DIR_VALUES; do
    echo ""
    echo "▶ 运行 λ_direction = $lambda_dir"
    echo ""
    
    nohup $PYTHON scripts/run_dct_v310_experiments.py run \
        --cancers blca \
        --folds 0,1,2,3,4 \
        --variants ipcw_only \
        --python $PYTHON \
        --gpu $GPU \
        --force \
        --set "dct_v38_lambda_direction=$lambda_dir" \
        --set "specific_simple=dct_v310_lambda_dir_${lambda_dir}_blca_50ep" \
        > "lambda_dir_${lambda_dir}.log" 2>&1 &
    
    PID=$!
    echo "  后台运行中, PID: $PID"
    echo "  日志文件: lambda_dir_${lambda_dir}.log"
    
    # 等待一段时间确保启动成功
    sleep 10
done

echo ""
echo "✅ Lambda Direction扫描实验已全部启动！"
echo ""
echo "监控命令："
echo "  tail -f lambda_dir_*.log"
echo ""

# ============================================================
# 2. Lambda IPCW扫描实验 (direction_only基础上添加不同强度的IPCW)
# ============================================================

echo ""
echo "============================================================"
echo "2. Lambda IPCW扫描实验"
echo "============================================================"
echo ""
echo "在Direction Only基础上测试不同强度的IPCW排序："
echo "  λ_ipcw ∈ {0.025, 0.05, 0.10, 0.15, 0.20}"
echo ""

LAMBDA_IPCW_VALUES="0.025 0.05 0.10 0.15 0.20"

for lambda_ipcw in $LAMBDA_IPCW_VALUES; do
    echo ""
    echo "▶ 运行 λ_ipcw = $lambda_ipcw"
    echo ""
    
    nohup $PYTHON scripts/run_dct_v310_experiments.py run \
        --cancers blca \
        --folds 0,1,2,3,4 \
        --variants direction_only \
        --python $PYTHON \
        --gpu $GPU \
        --force \
        --set "dct_lambda_ipcw_rank=$lambda_ipcw" \
        --set "specific_simple=dct_v310_lambda_ipcw_${lambda_ipcw}_blca_50ep" \
        > "lambda_ipcw_${lambda_ipcw}.log" 2>&1 &
    
    PID=$!
    echo "  后台运行中, PID: $PID"
    echo "  日志文件: lambda_ipcw_${lambda_ipcw}.log"
    
    # 等待一段时间确保启动成功
    sleep 10
done

echo ""
echo "✅ Lambda IPCW扫描实验已全部启动！"
echo ""

# ============================================================
# 总结
# ============================================================

echo ""
echo "========================================"
echo "✅ 所有准备就绪的实验已启动"
echo "========================================"
echo ""
echo "📊 实验概览:"
echo "  - Lambda Direction: 5个变体 × 5 folds = 25个训练任务"
echo "  - Lambda IPCW: 5个变体 × 5 folds = 25个训练任务"
echo "  - 总计: 50个训练任务"
echo ""
echo "⏱️  预计时间:"
echo "  - 每个fold: ~1.5小时"
echo "  - 每个变体(5 folds): ~7.5小时"
echo "  - 总计: ~75小时 (串行) 或 ~7.5小时 (10个变体并行)"
echo ""
echo "📝 监控命令:"
echo "  # 查看所有日志"
echo "  tail -f lambda_*.log"
echo ""
echo "  # 查看GPU使用"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "  # 查看运行进程"
echo "  ps aux | grep survot_rank.cli | grep -v grep"
echo ""
echo "  # 检查结果文件"
echo "  find results/dct_v3.10_experiments/robust -name \"*.pkl\" -mtime -1"
echo ""
echo "⚠️  E4实验状态:"
echo "  脚本: scripts/e4_audit_adapted.py"
echo "  状态: 需要先完成代码调试"
echo "  下一步: 修复模型加载和推理部分"
echo ""
echo "📖 详细文档: E4_EXPERIMENTS_ACTION_PLAN.md"
echo ""
