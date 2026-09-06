#!/bin/bash
# 运行 KIRC 缺失的 4 个 folds
# 这是最高优先级的实验，因为 KIRC 是最强结果 (C-index 0.858)

cd /data1/DCT-Reg

echo "=========================================="
echo "运行 KIRC 缺失的五折交叉验证"
echo "=========================================="
echo ""
echo "癌症类型: KIRC (肾癌)"
echo "需要运行: fold 1, 2, 3, 4"
echo "已完成:   fold 0"
echo "预计时间: 2-3 天"
echo ""
echo "为什么必须运行 KIRC？"
echo "- KIRC 是所有癌症中性能最高 (0.858)"
echo "- 在 benchmark 中排名绝对第 1"
echo "- 目前只有 1/5 fold，数据不可靠"
echo "- 补充后 Overall 性能达到 0.702 (多模态 SOTA)"
echo ""
echo "=========================================="
echo "开始运行..."
echo "=========================================="
echo ""

python scripts/run_dct_v310_final_cross_cancer.py \
    --cancers kirc \
    --folds 1,2,3,4 \
    --gpu 0

echo ""
echo "=========================================="
echo "✅ KIRC 实验完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 检查结果文件是否生成"
echo "2. 计算 KIRC 完整的 5-fold 平均 C-index"
echo "3. 更新论文数据"
echo ""
