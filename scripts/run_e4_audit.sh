#!/bin/bash
# 运行E4连续干预审计的辅助脚本

echo "=========================================="
echo "E4: Continuous Intervention Audit"
echo "=========================================="
echo ""
echo "⚠️  注意：此脚本需要手动适配模型加载代码！"
echo ""
echo "在运行之前，需要完成以下步骤："
echo ""
echo "1. 打开 scripts/e4_continuous_intervention_audit.py"
echo "2. 找到标记为 TODO 的部分："
echo "   - 实现从config创建模型"
echo "   - 实现加载test数据集"
echo "   - 实现提取embedding的代码"
echo ""
echo "3. 确认模型中存储预后锚点的位置："
echo "   - model.low_risk_anchor / model.high_risk_anchor"
echo "   - 或 model.anchors"
echo "   - 或 model.dct_module.anchors"
echo ""
echo "=========================================="
echo ""

# 示例使用方法
echo "使用示例："
echo ""
echo "# 单个fold"
echo "python scripts/e4_continuous_intervention_audit.py \\"
echo "    --checkpoint results/dct_v3.10_experiments/robust/full/blca/.../s_0_checkpoint.pt \\"
echo "    --config configs/dct_v310_directional_regularized_transport.yaml \\"
echo "    --study blca \\"
echo "    --fold 0 \\"
echo "    --output results/e4_intervention_audit/blca_fold0.csv \\"
echo "    --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \\"
echo "    --device cuda:0"
echo ""
echo "# 可视化结果"
echo "python scripts/visualize_e4_results.py \\"
echo "    --input results/e4_intervention_audit/blca_fold0.csv \\"
echo "    --output-dir results/e4_intervention_audit/visualizations \\"
echo "    --study blca \\"
echo "    --fold 0 \\"
echo "    --num-examples 20"
echo ""
echo "=========================================="
echo ""

# 提供快速测试命令（需要适配）
if [ "$1" == "--run-test" ]; then
    echo "运行测试（确保已完成TODO部分）..."
    
    # 找到一个checkpoint文件
    CHECKPOINT=$(find results/dct_v3.10_experiments/robust/full -name "s_0_checkpoint.pt" | head -1)
    
    if [ -z "$CHECKPOINT" ]; then
        echo "❌ 错误：找不到checkpoint文件"
        echo "请确保已经训练完成full model的至少一个fold"
        exit 1
    fi
    
    echo "找到checkpoint: $CHECKPOINT"
    echo ""
    
    python scripts/e4_continuous_intervention_audit.py \
        --checkpoint "$CHECKPOINT" \
        --config configs/dct_v310_directional_regularized_transport.yaml \
        --study blca \
        --fold 0 \
        --output results/e4_intervention_audit/blca_fold0_test.csv \
        --alphas 0,0.25,0.5,0.75,1.0 \
        --device cuda:0
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ E4审计完成！正在生成可视化..."
        python scripts/visualize_e4_results.py \
            --input results/e4_intervention_audit/blca_fold0_test.csv \
            --output-dir results/e4_intervention_audit/visualizations \
            --study blca \
            --fold 0 \
            --num-examples 10
    fi
else
    echo "查看此帮助: bash scripts/run_e4_audit.sh"
    echo "运行测试: bash scripts/run_e4_audit.sh --run-test"
fi
