#!/usr/bin/env python3
"""
Analyze and visualize intervention audit results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_results(csv_path):
    """Load audit results from CSV."""
    df = pd.read_csv(csv_path)
    return df

def analyze_monotonicity(df):
    """Analyze monotonicity of interventions."""
    results = {}
    
    # Group by patient and direction
    for direction in ['low_risk', 'high_risk']:
        dir_df = df[df['direction'] == direction].copy()
        
        # For each patient, check if risk changes monotonically
        patients = dir_df['patient_id'].unique()
        monotonic_count = 0
        
        for patient in patients:
            patient_df = dir_df[dir_df['patient_id'] == patient].sort_values('alpha')
            risks = patient_df['risk_pred'].values
            
            # Check monotonicity
            if direction == 'low_risk':
                # Risk should decrease as alpha increases
                is_monotonic = all(risks[i] >= risks[i+1] for i in range(len(risks)-1))
            else:
                # Risk should increase as alpha increases
                is_monotonic = all(risks[i] <= risks[i+1] for i in range(len(risks)-1))
            
            if is_monotonic:
                monotonic_count += 1
        
        monotonic_rate = monotonic_count / len(patients) if len(patients) > 0 else 0
        results[f'{direction}_monotonic_rate'] = monotonic_rate
        results[f'{direction}_n_patients'] = len(patients)
    
    return results

def plot_intervention_curves(df, output_dir):
    """Plot intervention curves for sample patients."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select a few patients to visualize
    patients = df['patient_id'].unique()[:12]  # First 12 patients
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, patient in enumerate(patients):
        ax = axes[idx]
        patient_df = df[df['patient_id'] == patient]
        
        # Plot low-risk direction
        low_risk = patient_df[patient_df['direction'] == 'low_risk'].sort_values('alpha')
        ax.plot(low_risk['alpha'], low_risk['risk_pred'], 'b-o', label='→ Low Risk', markersize=4)
        
        # Plot high-risk direction
        high_risk = patient_df[patient_df['direction'] == 'high_risk'].sort_values('alpha')
        ax.plot(high_risk['alpha'], high_risk['risk_pred'], 'r-s', label='→ High Risk', markersize=4)
        
        # Add original risk
        original_risk = patient_df['original_risk'].iloc[0]
        ax.axhline(y=original_risk, color='gray', linestyle='--', alpha=0.5, label='Original')
        
        ax.set_xlabel('Intervention Strength (α)')
        ax.set_ylabel('Risk Score')
        ax.set_title(f'{patient}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'intervention_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved intervention curves to {output_dir / 'intervention_curves.png'}")

def plot_risk_changes(df, output_dir):
    """Plot distribution of risk changes."""
    output_dir = Path(output_dir)
    
    # Get risk changes at maximum alpha
    max_alpha = df['alpha'].max()
    max_alpha_df = df[df['alpha'] == max_alpha].copy()
    max_alpha_df['risk_change'] = max_alpha_df['risk_pred'] - max_alpha_df['original_risk']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Low-risk direction
    low_risk = max_alpha_df[max_alpha_df['direction'] == 'low_risk']
    axes[0].hist(low_risk['risk_change'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    axes[0].set_xlabel('Risk Change (Δ Risk)')
    axes[0].set_ylabel('Number of Patients')
    axes[0].set_title(f'Risk Changes: Intervention → Low Risk (α={max_alpha})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    mean_change = low_risk['risk_change'].mean()
    axes[0].text(0.05, 0.95, f'Mean Δ = {mean_change:.6f}\nExpected: < 0', 
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # High-risk direction
    high_risk = max_alpha_df[max_alpha_df['direction'] == 'high_risk']
    axes[1].hist(high_risk['risk_change'], bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='blue', linestyle='--', linewidth=2, label='No change')
    axes[1].set_xlabel('Risk Change (Δ Risk)')
    axes[1].set_ylabel('Number of Patients')
    axes[1].set_title(f'Risk Changes: Intervention → High Risk (α={max_alpha})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    mean_change = high_risk['risk_change'].mean()
    axes[1].text(0.05, 0.95, f'Mean Δ = {mean_change:.6f}\nExpected: > 0', 
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_change_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved risk change distribution to {output_dir / 'risk_change_distribution.png'}")

def plot_alpha_vs_risk_aggregate(df, output_dir):
    """Plot aggregate risk change vs alpha."""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Aggregate by alpha and direction
    agg_df = df.groupby(['alpha', 'direction']).agg({
        'risk_pred': ['mean', 'std'],
        'original_risk': 'mean'
    }).reset_index()
    
    # Plot for each direction
    for direction, color, marker, label in [
        ('low_risk', 'blue', 'o', 'Toward Low Risk'),
        ('high_risk', 'red', 's', 'Toward High Risk')
    ]:
        dir_df = agg_df[agg_df['direction'] == direction]
        alphas = dir_df['alpha']
        means = dir_df['risk_pred']['mean']
        stds = dir_df['risk_pred']['std']
        
        ax.plot(alphas, means, color=color, marker=marker, linestyle='-', label=label, markersize=6)
        ax.fill_between(alphas, means - stds, means + stds, color=color, alpha=0.2)
    
    # Plot original risk
    original = df['original_risk'].mean()
    ax.axhline(y=original, color='gray', linestyle='--', linewidth=2, label='Original Risk (α=0)')
    
    ax.set_xlabel('Intervention Strength (α)', fontsize=12)
    ax.set_ylabel('Mean Risk Score', fontsize=12)
    ax.set_title('Aggregate Risk Changes with Intervention Strength', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_vs_risk_aggregate.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved aggregate alpha vs risk to {output_dir / 'alpha_vs_risk_aggregate.png'}")

def main():
    # Load results
    results_path = '/data1/DCT-Reg/audit_results/blca_fold0_audit.pkl'  # Actually a CSV
    output_dir = '/data1/DCT-Reg/audit_results/visualizations'
    
    print("Loading audit results...")
    df = load_results(results_path)
    print(f"Loaded {len(df)} intervention records for {df['patient_id'].nunique()} patients")
    
    # Analyze monotonicity
    print("\nAnalyzing monotonicity...")
    monotonicity = analyze_monotonicity(df)
    print(f"  Low-risk direction: {monotonicity['low_risk_monotonic_rate']*100:.2f}% monotonic")
    print(f"  High-risk direction: {monotonicity['high_risk_monotonic_rate']*100:.2f}% monotonic")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_intervention_curves(df, output_dir)
    plot_risk_changes(df, output_dir)
    plot_alpha_vs_risk_aggregate(df, output_dir)
    
    print("\n✅ Analysis complete!")
    print(f"   Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
