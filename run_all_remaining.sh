#!/bin/bash
# 运行所有缺失的五折交叉验证实验
# KIRC (4个) + UCEC (2个) = 总计 6 个实验

cd /data1/DCT-Reg

echo "=========================================="
echo "运行所有缺失的五折交叉验证实验"
echo "=========================================="
echo ""
echo "实验计划:"
echo "  KIRC: fold 1, 2, 3, 4 (4个实验)"
echo "  UCEC: fold 0, 3      (2个实验)"
echo "  总计: 6 个实验"
echo ""
echo "预计时间: 3-4 天"
echo ""
echo "=========================================="

# ======================================
# Part 1: KIRC (高优先级)
# ======================================
echo ""
echo "======================================"
echo "Part 1/2: 运行 KIRC (肾癌)"
echo "======================================"
echo "Folds: 1, 2, 3, 4"
echo ""

python scripts/run_dct_v310_final_cross_cancer.py \
    --cancers kirc \
    --folds 1,2,3,4 \
    --gpu 0

echo ""
echo "✅ KIRC 完成！"
echo ""

# ======================================
# Part 2: UCEC (中优先级)
# ======================================
echo ""
echo "======================================"
echo "Part 2/2: 运行 UCEC (子宫内膜癌)"
echo "======================================"
echo "Folds: 0, 3"
echo ""

python scripts/run_dct_v310_final_cross_cancer.py \
    --cancers ucec \
    --folds 0,3 \
    --gpu 0

echo ""
echo "✅ UCEC 完成！"
echo ""

# ======================================
# 总结
# ======================================
echo ""
echo "=========================================="
echo "🎉 所有实验完成！"
echo "=========================================="
echo ""
echo "完成情况:"
echo "  KIRC: 5/5 folds ✅"
echo "  UCEC: 5/5 folds ✅"
echo "  BLCA: 5/5 folds ✅ (已有)"
echo "  SKCM: 5/5 folds ✅ (已有)"
echo "  HNSC: 5/5 folds ✅ (已有)"
echo "  LUSC: 5/5 folds ✅ (已有)"
echo ""
echo "总计: 6 种癌症 × 5 folds = 30 个实验"
echo ""
echo "下一步:"
echo "1. 检查所有结果文件"
echo "2. 计算各癌症的平均 C-index"
echo "3. 更新 benchmark 对比表"
echo "4. 开始撰写论文"
echo ""
