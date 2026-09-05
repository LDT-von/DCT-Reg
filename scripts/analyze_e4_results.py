#!/usr/bin/env python3
"""Quick analysis script for E4 intervention audit results.

Analyzes direction consistency across all experiments and generates summary statistics.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def analyze_single_experiment(csv_path: str) -> Dict:
    """Analyze a single E4 experiment result."""
    df = pd.read_csv(csv_path)
    
    results = {
        'file': os.path.basename(csv_path),
        'n_patients': len(df['patient_id'].unique()),
        'n_alphas': len(df['alpha'].unique()),
    }
    
    for direction in ['low_risk', 'high_risk']:
        df_dir = df[df['direction'] == direction]
        
        # Risk change at full intervention (α=1.0)
        df_full = df_dir[df_dir['alpha'] == 1.0]
        results[f'{direction}_mean_risk_change'] = df_full['risk_change'].mean()
        results[f'{direction}_std_risk_change'] = df_full['risk_change'].std()
        
        # Check monotonicity for each patient
        n_patients = len(df_dir['patient_id'].unique())
        n_monotonic = 0
        
        for pid in df_dir['patient_id'].unique():
            patient_data = df_dir[df_dir['patient_id'] == pid].sort_values('alpha')
            risks = patient_data['risk_pred'].values
            
            if direction == 'low_risk':
                # Should decrease or stay same
                is_monotonic = all(risks[i] >= risks[i+1] - 1e-6 for i in range(len(risks)-1))
            else:
                # Should increase or stay same
                is_monotonic = all(risks[i] <= risks[i+1] + 1e-6 for i in range(len(risks)-1))
            
            if is_monotonic:
                n_monotonic += 1
        
        results[f'{direction}_monotonic_count'] = n_monotonic
        results[f'{direction}_monotonic_rate'] = n_monotonic / n_patients * 100
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze E4 results")
    parser.add_argument(
        '--input_dir', default='results/e4_intervention_audit',
        help='Directory containing E4 CSV results'
    )
    parser.add_argument(
        '--output', default='results/e4_intervention_audit/summary.csv',
        help='Output summary CSV'
    )
    
    args = parser.parse_args()
    
    # Find all CSV files
    csv_files = sorted(Path(args.input_dir).glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != 'summary.csv']
    
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return
    
    print("=" * 80)
    print("E4 INTERVENTION AUDIT - ANALYSIS")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Found {len(csv_files)} experiment results")
    print("=" * 80)
    
    # Analyze each experiment
    all_results = []
    
    for csv_file in csv_files:
        print(f"\nAnalyzing {csv_file.name}...")
        try:
            results = analyze_single_experiment(str(csv_file))
            all_results.append(results)
            
            # Print summary
            print(f"  Patients: {results['n_patients']}")
            print(f"  Low-risk monotonic: {results['low_risk_monotonic_count']}/{results['n_patients']} ({results['low_risk_monotonic_rate']:.1f}%)")
            print(f"  High-risk monotonic: {results['high_risk_monotonic_count']}/{results['n_patients']} ({results['high_risk_monotonic_rate']:.1f}%)")
            print(f"  Low-risk Δ @ α=1.0: {results['low_risk_mean_risk_change']:.4f} ± {results['low_risk_std_risk_change']:.4f}")
            print(f"  High-risk Δ @ α=1.0: {results['high_risk_mean_risk_change']:.4f} ± {results['high_risk_std_risk_change']:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Save summary
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(args.output, index=False)
    print(f"\n✓ Summary saved to {args.output}")
    
    # Aggregate by variant
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS BY VARIANT")
    print("=" * 80)
    
    df_summary['variant'] = df_summary['file'].str.extract(r'^([^_]+)_')
    
    for variant in df_summary['variant'].unique():
        df_var = df_summary[df_summary['variant'] == variant]
        
        print(f"\n{variant.upper()}:")
        print(f"  Folds: {len(df_var)}")
        print(f"  Low-risk monotonic rate: {df_var['low_risk_monotonic_rate'].mean():.1f}% ± {df_var['low_risk_monotonic_rate'].std():.1f}%")
        print(f"  High-risk monotonic rate: {df_var['high_risk_monotonic_rate'].mean():.1f}% ± {df_var['high_risk_monotonic_rate'].std():.1f}%")
        print(f"  Low-risk Δ @ α=1.0: {df_var['low_risk_mean_risk_change'].mean():.4f} ± {df_var['low_risk_mean_risk_change'].std():.4f}")
        print(f"  High-risk Δ @ α=1.0: {df_var['high_risk_mean_risk_change'].mean():.4f} ± {df_var['high_risk_mean_risk_change'].std():.4f}")


if __name__ == '__main__':
    main()
