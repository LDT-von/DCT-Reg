#!/usr/bin/env python3
"""
E4 Audit Final Results Analysis
Complete analysis of directional consistency experiments
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_results():
    """Analyze E4 audit results"""
    
    results_file = Path("/data1/DCT-Reg/results/e4_audits/e4_audit_summary.csv")
    df = pd.read_csv(results_file)
    
    print("=" * 80)
    print("E4 AUDIT FINAL RESULTS - DIRECTIONAL CONSISTENCY ANALYSIS")
    print("=" * 80)
    print()
    
    # Group by variant
    variants = df['variant'].unique()
    
    for variant in sorted(variants):
        variant_df = df[df['variant'] == variant]
        
        print(f"\n{'─' * 80}")
        print(f"Variant: {variant.upper().replace('_', ' ')}")
        print(f"{'─' * 80}")
        print()
        
        # Table of all folds
        print(f"{'Fold':<6} {'Samples':<10} {'Anchor Dist':<14} {'Mean Risk':<14} {'Std Risk':<12}")
        print(f"{'-'*6} {'-'*10} {'-'*14} {'-'*14} {'-'*12}")
        
        for _, row in variant_df.iterrows():
            fold = int(row['fold'])
            samples = int(row['n_samples'])
            anchor = row['anchor_distance']
            mean_risk = row['mean_risk']
            std_risk = row['std_risk']
            
            print(f"{fold:<6} {samples:<10} {anchor:<14.4f} {mean_risk:<14.4f} {std_risk:<12.4f}")
        
        # Statistics
        print()
        print(f"Aggregate Statistics:")
        print(f"  Anchor Distance:")
        print(f"    Mean:   {variant_df['anchor_distance'].mean():.4f}")
        print(f"    Std:    {variant_df['anchor_distance'].std():.4f}")
        print(f"    Min:    {variant_df['anchor_distance'].min():.4f}")
        print(f"    Max:    {variant_df['anchor_distance'].max():.4f}")
        print(f"    Range:  {variant_df['anchor_distance'].max() - variant_df['anchor_distance'].min():.4f}")
        print()
        print(f"  Mean Risk Score:")
        print(f"    Mean:   {variant_df['mean_risk'].mean():.4f}")
        print(f"    Std:    {variant_df['mean_risk'].std():.4f}")
        print(f"    Min:    {variant_df['mean_risk'].min():.4f}")
        print(f"    Max:    {variant_df['mean_risk'].max():.4f}")
        print()
        print(f"  Risk Score Variability:")
        print(f"    Mean:   {variant_df['std_risk'].mean():.4f}")
        print(f"    Std:    {variant_df['std_risk'].std():.4f}")
    
    # Comparison between variants
    print()
    print("=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    print()
    
    comparison_data = []
    for variant in sorted(variants):
        variant_df = df[df['variant'] == variant]
        comparison_data.append({
            'Variant': variant.replace('_', ' ').title(),
            'Anchor Dist': f"{variant_df['anchor_distance'].mean():.4f} ± {variant_df['anchor_distance'].std():.4f}",
            'Mean Risk': f"{variant_df['mean_risk'].mean():.4f} ± {variant_df['mean_risk'].std():.4f}",
            'Avg Std Risk': f"{variant_df['std_risk'].mean():.4f}",
        })
    
    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))
    
    # Key findings
    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    
    direction_only = df[df['variant'] == 'direction_only']
    ipcw_only = df[df['variant'] == 'ipcw_only']
    
    print("1. Anchor Distance Comparison:")
    dir_mean = direction_only['anchor_distance'].mean()
    ipcw_mean = ipcw_only['anchor_distance'].mean()
    diff_pct = abs(dir_mean - ipcw_mean) / ipcw_mean * 100
    
    if dir_mean > ipcw_mean:
        print(f"   • Direction Only: {dir_mean:.4f}")
        print(f"   • IPCW Only: {ipcw_mean:.4f}")
        print(f"   • Direction Only has {diff_pct:.1f}% larger anchor distance")
    else:
        print(f"   • IPCW Only: {ipcw_mean:.4f}")
        print(f"   • Direction Only: {dir_mean:.4f}")
        print(f"   • IPCW Only has {diff_pct:.1f}% larger anchor distance")
    
    print()
    print("2. Risk Score Consistency:")
    dir_std = direction_only['std_risk'].mean()
    ipcw_std = ipcw_only['std_risk'].mean()
    
    if dir_std < ipcw_std:
        print(f"   • Direction Only shows MORE consistent risk scores")
        print(f"     (avg std: {dir_std:.4f} vs {ipcw_std:.4f})")
    else:
        print(f"   • IPCW Only shows MORE consistent risk scores")
        print(f"     (avg std: {ipcw_std:.4f} vs {dir_std:.4f})")
    
    print()
    print("3. Cross-Fold Variability:")
    dir_var = direction_only['anchor_distance'].std()
    ipcw_var = ipcw_only['anchor_distance'].std()
    
    print(f"   • Direction Only anchor variance: {dir_var:.4f}")
    print(f"   • IPCW Only anchor variance: {ipcw_var:.4f}")
    
    if dir_var > ipcw_var:
        print(f"   • Direction Only shows {(dir_var/ipcw_var - 1)*100:.1f}% more cross-fold variability")
    else:
        print(f"   • IPCW Only shows {(ipcw_var/dir_var - 1)*100:.1f}% more cross-fold variability")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    analyze_results()
