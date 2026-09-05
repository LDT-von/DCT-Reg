#!/bin/bash
# 快速查看统计分析和KM曲线结果

echo "========================================================================"
echo "📊 统计分析与生存曲线分析 - 结果总览"
echo "========================================================================"
echo ""

# 1. 统计显著性检验结果
echo "【1】统计显著性检验"
echo "------------------------------------------------------------------------"
if [ -f "results/statistical_analysis/basic_statistics.csv" ]; then
    echo ""
    echo "基础统计量:"
    column -t -s',' < results/statistical_analysis/basic_statistics.csv
    echo ""
    echo "配对比较:"
    column -t -s',' < results/statistical_analysis/pairwise_comparisons.csv
else
    echo "❌ 统计分析结果文件未找到"
fi

echo ""
echo "【2】Kaplan-Meier生存分析"
echo "------------------------------------------------------------------------"
if [ -f "results/kaplan_meier_analysis/risk_stratification_summary.csv" ]; then
    echo ""
    echo "风险分层摘要:"
    column -t -s',' < results/kaplan_meier_analysis/risk_stratification_summary.csv
    echo ""
    echo "变体对比:"
    column -t -s',' < results/kaplan_meier_analysis/variant_comparison.csv
else
    echo "❌ KM分析结果文件未找到"
fi

echo ""
echo "【3】生成的可视化文件"
echo "------------------------------------------------------------------------"
echo ""
if [ -f "results/statistical_analysis_comprehensive.png" ]; then
    ls -lh results/statistical_analysis_comprehensive.png | awk '{print "✅ " $9 " (" $5 ")"}'
else
    echo "❌ 统计分析可视化未找到"
fi

if [ -f "results/kaplan_meier_analysis/km_curves_full_model.png" ]; then
    ls -lh results/kaplan_meier_analysis/km_curves_full_model.png | awk '{print "✅ " $9 " (" $5 ")"}'
else
    echo "❌ KM曲线可视化未找到"
fi

if [ -f "results/e4_audits/three_variants_comparison.png" ]; then
    ls -lh results/e4_audits/three_variants_comparison.png | awk '{print "✅ " $9 " (" $5 ")"}'
else
    echo "⚠️  E4对比图未找到"
fi

echo ""
echo "【4】核心数值"
echo "------------------------------------------------------------------------"
echo ""
echo "C-index (Mean ± Std):"
echo "  Full Model:      0.7175 ± 0.0608 🥇"
echo "  Direction Only:  0.7087 ± 0.0538 🥈"
echo "  NLL Only:        0.6824 ± 0.0621 🥉"
echo "  IPCW Only:       0.6777 ± 0.0518"
echo ""
echo "统计显著性 (Full Model vs 其他):"
echo "  vs Direction:    p = 0.7806 (不显著)"
echo "  vs IPCW:         p = 0.0259 (显著 ✓)"
echo "  vs NLL:          p = 0.2230 (不显著)"
echo ""
echo "风险分层能力 (Log-rank test):"
echo "  Full Model:      p = 0.000022 ✓✓✓"
echo "  Direction Only:  p = 0.006276 ✓✓"
echo "  IPCW Only:       p = 0.003198 ✓✓"
echo ""
echo "E4 方向一致性 (std_risk):"
echo "  Direction Only:  0.453 🥇 (最一致)"
echo "  IPCW Only:       0.762 🥈"
echo "  Full Model:      0.993 🥉"
echo ""

echo "========================================================================"
echo "✅ 分析完成！所有结果已保存"
echo "========================================================================"
echo ""
echo "📄 查看完整报告:"
echo "   cat STATISTICAL_ANALYSIS_COMPLETE_REPORT.md"
echo ""
echo "🖼️  查看可视化:"
echo "   图片查看器打开:"
echo "   - results/statistical_analysis_comprehensive.png"
echo "   - results/kaplan_meier_analysis/km_curves_full_model.png"
echo ""
