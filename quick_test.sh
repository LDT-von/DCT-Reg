#!/bin/bash
# 快速启动脚本 - 运行生死门测试

echo "🚀 启动生死门测试..."
echo ""
echo "这个测试将验证:"
echo "  1. OT 计划是否被 pair-context 旁路压制"
echo "  2. 方向一致性约束是否生效"
echo ""
echo "预计耗时: 10-15 分钟"
echo ""

cd /data1/DCT-Reg
python scripts/run_small_gate_test.py

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 通过生死门！"
    echo ""
    echo "下一步: 运行完整实验"
    echo "  bash run_all_experiments.sh"
else
    echo "❌ 未通过生死门 (退出码: $EXIT_CODE)"
    echo ""
    echo "查看详细结果: results/small_test/"
fi
