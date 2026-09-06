#!/bin/bash
# 主运行脚本：按正确顺序执行所有实验

set -e

REPO_ROOT="/data1/DCT-Reg"
cd "$REPO_ROOT"

echo "=========================================="
echo "DCT-Reg 实验流程"
echo "=========================================="
echo "步骤 1: 小规模生死门测试"
echo "步骤 2: 读取器消融实验 (如果通过)"
echo "步骤 3: 方向一致性审计 (如果通过)"
echo "=========================================="

# 步骤 1: 小规模生死门测试
echo ""
echo "步骤 1: 运行小规模生死门测试..."
python scripts/run_small_gate_test.py

GATE_RESULT=$?

if [ $GATE_RESULT -eq 0 ]; then
    echo ""
    echo "✓ 通过生死门，继续完整实验"
    
    # 步骤 2: 完整读取器消融
    echo ""
    echo "步骤 2: 运行完整读取器消融实验..."
    bash scripts/run_reader_ablation_experiments.sh
    
    # 步骤 3: 完整方向审计
    echo ""
    echo "步骤 3: 运行完整方向一致性审计..."
    bash scripts/run_e4_direction_audit.sh
    
    echo ""
    echo "=========================================="
    echo "所有实验完成"
    echo "=========================================="
    echo "查看结果:"
    echo "  读取器消融: results/reader_ablation/"
    echo "  方向审计: results/e4_audit_direction_only/"
    echo "=========================================="
    
else
    echo ""
    echo "✗ 未通过生死门 (退出码: $GATE_RESULT)"
    echo "=========================================="
    
    if [ $GATE_RESULT -eq 1 ]; then
        echo "原因: OT计划被旁路压制 或 方向一致性不足"
        echo "建议:"
        echo "  1. 检查融合模块实现"
        echo "  2. 增加方向正则化权重"
        echo "  3. 检查训练日志"
    else
        echo "原因: 部分测试未通过"
        echo "建议: 查看 results/small_test/ 中的详细结果"
    fi
    
    echo "=========================================="
    echo "❌ 停止大规模实验"
    echo "=========================================="
    exit 1
fi
