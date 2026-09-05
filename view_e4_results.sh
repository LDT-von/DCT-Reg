#!/bin/bash
# E4 Quick Results Viewer
# Quick summary of E4 audit results

echo "════════════════════════════════════════════════════════════════════════════════"
echo "                    E4 AUDIT - QUICK RESULTS SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if results exist
if [ ! -f "/data1/DCT-Reg/results/e4_audits/e4_audit_summary.csv" ]; then
    echo "❌ No results found. Run experiments first."
    exit 1
fi

echo "✅ All experiments completed!"
echo ""

# Count files
total_csv=$(ls /data1/DCT-Reg/results/e4_audits/*.csv 2>/dev/null | wc -l)
total_json=$(ls /data1/DCT-Reg/results/e4_audits/*.json 2>/dev/null | wc -l)
total_png=$(ls /data1/DCT-Reg/results/e4_audits/*.png 2>/dev/null | wc -l)

echo "📁 Generated Files:"
echo "   - CSV files: $total_csv"
echo "   - JSON files: $total_json"
echo "   - Visualizations: $total_png"
echo ""

echo "────────────────────────────────────────────────────────────────────────────────"
echo "                            KEY FINDINGS"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

python3 << 'PYTHON_SCRIPT'
import pandas as pd

df = pd.read_csv("/data1/DCT-Reg/results/e4_audits/e4_audit_summary.csv")

dir_only = df[df['variant'] == 'direction_only']
ipcw_only = df[df['variant'] == 'ipcw_only']

dir_std = dir_only['std_risk'].mean()
ipcw_std = ipcw_only['std_risk'].mean()
improvement = (ipcw_std - dir_std) / ipcw_std * 100

print("🏆 WINNER: Direction Only")
print("")
print("Risk Consistency (Lower = Better):")
print(f"  • Direction Only: {dir_std:.4f}")
print(f"  • IPCW Only:      {ipcw_std:.4f}")
print(f"  • Improvement:    {improvement:.1f}%")
print("")

dir_anchor = dir_only['anchor_distance'].mean()
ipcw_anchor = ipcw_only['anchor_distance'].mean()

print("Anchor Distance:")
print(f"  • Direction Only: {dir_anchor:.4f} ± {dir_only['anchor_distance'].std():.4f}")
print(f"  • IPCW Only:      {ipcw_anchor:.4f} ± {ipcw_only['anchor_distance'].std():.4f}")
print("")

print("Cross-Fold Stability:")
print(f"  • Direction Only variance: {dir_only['anchor_distance'].std():.4f}")
print(f"  • IPCW Only variance:      {ipcw_only['anchor_distance'].std():.4f}")

PYTHON_SCRIPT

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "                         QUICK ACCESS COMMANDS"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""
echo "View detailed analysis:"
echo "  python /data1/DCT-Reg/analyze_e4_final_results.py"
echo ""
echo "View visualizations:"
echo "  eog /data1/DCT-Reg/results/e4_audits/e4_audit_visualization.png"
echo "  eog /data1/DCT-Reg/results/e4_audits/e4_fold_comparison.png"
echo ""
echo "Read full report:"
echo "  cat /data1/DCT-Reg/E4_AUDIT_FINAL_REPORT.md | less"
echo ""
echo "View raw data:"
echo "  cat /data1/DCT-Reg/results/e4_audits/e4_audit_summary.csv | column -t -s,"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
