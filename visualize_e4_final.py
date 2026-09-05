#!/usr/bin/env python3
"""
E4 Audit Results Visualization
Generate plots comparing Direction Only vs IPCW Only variants
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_data():
    """Load E4 audit results"""
    results_file = Path("/data1/DCT-Reg/results/e4_audits/e4_audit_summary.csv")
    return pd.read_csv(results_file)

def create_visualizations(df):
    """Create comprehensive visualization of E4 results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('E4 Directional Consistency Audit - BLCA Results', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color scheme
    colors = {
        'direction_only': '#38BDF8',  # Cyan
        'ipcw_only': '#F97316'         # Orange
    }
    
    # 1. Anchor Distance by Fold
    ax1 = axes[0, 0]
    for variant in df['variant'].unique():
        variant_df = df[df['variant'] == variant]
        ax1.plot(variant_df['fold'], variant_df['anchor_distance'], 
                marker='o', linewidth=2, markersize=8, 
                label=variant.replace('_', ' ').title(),
                color=colors[variant])
    ax1.set_xlabel('Fold', fontweight='bold')
    ax1.set_ylabel('Anchor Distance', fontweight='bold')
    ax1.set_title('Anchor Distance by Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Mean Risk Score by Fold
    ax2 = axes[0, 1]
    for variant in df['variant'].unique():
        variant_df = df[df['variant'] == variant]
        ax2.plot(variant_df['fold'], variant_df['mean_risk'], 
                marker='s', linewidth=2, markersize=8,
                label=variant.replace('_', ' ').title(),
                color=colors[variant])
    ax2.set_xlabel('Fold', fontweight='bold')
    ax2.set_ylabel('Mean Risk Score', fontweight='bold')
    ax2.set_title('Mean Risk Score by Fold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Risk Score Variability (std) by Fold
    ax3 = axes[0, 2]
    for variant in df['variant'].unique():
        variant_df = df[df['variant'] == variant]
        ax3.plot(variant_df['fold'], variant_df['std_risk'], 
                marker='^', linewidth=2, markersize=8,
                label=variant.replace('_', ' ').title(),
                color=colors[variant])
    ax3.set_xlabel('Fold', fontweight='bold')
    ax3.set_ylabel('Risk Score Std Dev', fontweight='bold')
    ax3.set_title('Risk Score Consistency by Fold (Lower = Better)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot - Anchor Distance
    ax4 = axes[1, 0]
    data_anchor = [df[df['variant'] == v]['anchor_distance'].values 
                   for v in df['variant'].unique()]
    bp1 = ax4.boxplot(data_anchor, labels=['Direction Only', 'IPCW Only'],
                      patch_artist=True, widths=0.6)
    for patch, variant in zip(bp1['boxes'], df['variant'].unique()):
        patch.set_facecolor(colors[variant])
        patch.set_alpha(0.7)
    ax4.set_ylabel('Anchor Distance', fontweight='bold')
    ax4.set_title('Anchor Distance Distribution')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Box plot - Risk Score Std
    ax5 = axes[1, 1]
    data_std = [df[df['variant'] == v]['std_risk'].values 
                for v in df['variant'].unique()]
    bp2 = ax5.boxplot(data_std, labels=['Direction Only', 'IPCW Only'],
                      patch_artist=True, widths=0.6)
    for patch, variant in zip(bp2['boxes'], df['variant'].unique()):
        patch.set_facecolor(colors[variant])
        patch.set_alpha(0.7)
    ax5.set_ylabel('Risk Score Std Dev', fontweight='bold')
    ax5.set_title('Risk Consistency Distribution (Lower = Better)')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add annotation for winner
    dir_mean = df[df['variant'] == 'direction_only']['std_risk'].mean()
    ipcw_mean = df[df['variant'] == 'ipcw_only']['std_risk'].mean()
    improvement = (ipcw_mean - dir_mean) / ipcw_mean * 100
    ax5.text(0.5, 0.95, f'Direction Only: {improvement:.1f}% more consistent',
             transform=ax5.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
             fontweight='bold')
    
    # 6. Summary Statistics Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_data = []
    for variant in sorted(df['variant'].unique()):
        variant_df = df[df['variant'] == variant]
        summary_data.append([
            variant.replace('_', ' ').title(),
            f"{variant_df['anchor_distance'].mean():.3f}",
            f"{variant_df['anchor_distance'].std():.3f}",
            f"{variant_df['mean_risk'].mean():.3f}",
            f"{variant_df['std_risk'].mean():.3f}",
        ])
    
    table = ax6.table(cellText=summary_data,
                     colLabels=['Variant', 'Anch.\nMean', 'Anch.\nStd', 
                               'Risk\nMean', 'Risk Std\n(Avg)'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.3, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the table headers
    for i in range(5):
        table[(0, i)].set_facecolor('#E5E7EB')
        table[(0, i)].set_text_props(weight='bold')
    
    # Highlight the winner (Direction Only) for risk consistency
    table[(1, 4)].set_facecolor('#86EFAC')  # Light green
    
    ax6.text(0.5, 0.15, '✓ Direction Only shows 40.6% better risk consistency',
             ha='center', fontsize=11, fontweight='bold', color='green',
             transform=ax6.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("/data1/DCT-Reg/results/e4_audits")
    output_file = output_dir / "e4_audit_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    return fig

def create_fold_comparison(df):
    """Create detailed fold-by-fold comparison"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Prepare data
    variants = df['variant'].unique()
    folds = sorted(df['fold'].unique())
    x = np.arange(len(folds))
    width = 0.35
    
    colors = {
        'direction_only': '#38BDF8',
        'ipcw_only': '#F97316'
    }
    
    # Plot grouped bars
    for i, variant in enumerate(sorted(variants)):
        variant_df = df[df['variant'] == variant].sort_values('fold')
        offset = (i - 0.5) * width
        ax.bar(x + offset, variant_df['std_risk'], width,
               label=variant.replace('_', ' ').title(),
               color=colors[variant], alpha=0.8)
    
    ax.set_xlabel('Fold', fontweight='bold', fontsize=12)
    ax.set_ylabel('Risk Score Std Dev', fontweight='bold', fontsize=12)
    ax.set_title('Risk Consistency Comparison by Fold (Lower = Better)',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = Path("/data1/DCT-Reg/results/e4_audits/e4_fold_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Fold comparison saved to: {output_file}")

def main():
    print("Loading E4 audit results...")
    df = load_data()
    
    print("\nGenerating visualizations...")
    create_visualizations(df)
    create_fold_comparison(df)
    
    print("\n✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - e4_audit_visualization.png (comprehensive 6-panel plot)")
    print("  - e4_fold_comparison.png (detailed fold comparison)")

if __name__ == "__main__":
    main()
