#!/usr/bin/env python3
"""
E4 Audit Results Summary
Aggregates all completed E4 experiments and displays key metrics
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

def load_experiment_results(results_dir: Path) -> List[Dict]:
    """Load all JSON metadata files"""
    results = []
    
    for json_file in sorted(results_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results

def print_summary(results: List[Dict]):
    """Print formatted summary of all experiments"""
    
    print("=" * 80)
    print("E4 AUDIT RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Group by cancer type
    by_cancer = {}
    for r in results:
        cancer = r.get('cancer', 'unknown')
        if cancer not in by_cancer:
            by_cancer[cancer] = []
        by_cancer[cancer].append(r)
    
    # Print each cancer type
    for cancer in sorted(by_cancer.keys()):
        cancer_results = by_cancer[cancer]
        print(f"\n{'─' * 80}")
        print(f"Cancer Type: {cancer.upper()}")
        print(f"{'─' * 80}")
        print(f"Completed folds: {len(cancer_results)}")
        print()
        
        # Table header
        print(f"{'Fold':<6} {'Checkpoint':<50} {'Anchor Dist':<14} {'Samples':<10}")
        print(f"{'-'*6} {'-'*50} {'-'*14} {'-'*10}")
        
        # Sort by fold
        cancer_results_sorted = sorted(cancer_results, key=lambda x: x.get('fold', 999))
        
        anchor_dists = []
        
        for r in cancer_results_sorted:
            fold = r.get('fold', '?')
            anchor = r.get('anchor_distance', 0)
            samples = r.get('n_processed', r.get('samples_processed', 0))
            
            # Extract experiment name from checkpoint path
            checkpoint = r.get('checkpoint', '')
            if 'direction_only' in checkpoint:
                exp_name = 'direction_only'
            elif 'magnitude_only' in checkpoint:
                exp_name = 'magnitude_only'
            elif 'no_intervention' in checkpoint:
                exp_name = 'no_intervention'
            else:
                exp_name = 'baseline'
            
            anchor_dists.append(anchor)
            
            print(f"{fold:<6} {exp_name:<50} {anchor:<14.4f} {samples:<10}")
        
        # Statistics across folds
        if len(anchor_dists) > 1:
            print()
            print(f"Cross-fold statistics:")
            print(f"  Anchor distance:  {min(anchor_dists):.4f} - {max(anchor_dists):.4f} (range: {max(anchor_dists)-min(anchor_dists):.4f})")
            print(f"  Mean:             {sum(anchor_dists)/len(anchor_dists):.4f}")
    
    print()
    print("=" * 80)
    print(f"Total experiments completed: {len(results)}")
    print(f"Cancer types: {len(by_cancer)}")
    print("=" * 80)

def main():
    results_dir = Path("/data1/DCT-Reg/results/e4_audits")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    results = load_experiment_results(results_dir)
    
    if not results:
        print("No results found yet")
        return
    
    print_summary(results)
    
    # Also save as CSV
    df = pd.DataFrame(results)
    summary_file = results_dir / "e4_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"\nDetailed summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
