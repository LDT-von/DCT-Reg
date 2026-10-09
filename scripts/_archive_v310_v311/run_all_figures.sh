#!/usr/bin/env bash
# Generate all 5 priority interpretability figures.
# 1. Per-fold improvement (fig2_perfold_cindex) — already exists
# 2. KM survival curves (fig5_km_curves) — exists, needs all data
# 3. Transport intervention sweep (fig3_transport_sweep) — re-run from exports
# 4. Cohort pathway heatmap (fig_d_cohort_heatmap)
# 5. Case interpretation figure — generation requires assets
set -euo pipefail
cd /data1/DCT-Reg

echo "=== 1. Per-fold C-index comparison (Fig 2) ==="
python scripts/plot_fig2_perfold_cindex.py 2>&1 | tail -3

echo ""
echo "=== 2. Transport plan replacement sweep (Fig 3) ==="
python scripts/plot_fig3_from_exports.py 2>&1 | tail -3

echo ""
echo "=== 3. Overall KM curves per cancer (Fig 5, Full) ==="
python scripts/plot_fig5_km_curves.py --arm exp6 --cancer blca --cancer kirc 2>&1 | tail -10

echo ""
echo "=== 4. Cohort pathway heatmap (Fig D) ==="
python scripts/plot_fig_d_cohort_heatmap.py 2>&1 | tail -3

echo ""
echo "=== 5. Main plot (scores from manifest) ==="
python scripts/prepare_v313_evidence.py plot \
  --manifest results/v313_evidence_v2/controls.json \
  --output paper/figures/v313_main_plots 2>&1 | tail -3

echo ""
echo "=== Listing all generated figures ==="
find paper/figures -type f \( -name "*.png" -o -name "*.pdf" \) -newer scripts/plot_fig2_perfold_cindex.py 2>&1 | awk 'NR <= 30 {print}'